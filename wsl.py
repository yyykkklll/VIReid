import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict


class CMA(nn.Module):
    def __init__(self, args):
        super().__init__()
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_classes = args.num_classes
        self.T = args.temperature
        self.sigma = args.sigma
        self.proto_momentum = getattr(args, 'prototype_momentum', 0.9)
        self.use_ccpa = getattr(args, 'use_cycle_consistency', False)
        
        # State flags
        self.not_saved = True
        self.current_epoch = 0
        
        # Class-level memory banks
        self.register_buffer('vis_memory', torch.zeros(self.num_classes, 2048))
        self.register_buffer('ir_memory', torch.zeros(self.num_classes, 2048))
        
        # Prototype vectors for CCPA
        self.register_buffer('prototypes_v', torch.randn(self.num_classes, 2048))
        self.register_buffer('prototypes_r', torch.randn(self.num_classes, 2048))
        self.prototypes_initialized = False
        
        # CCPA module initialization
        if self.use_ccpa:
            from models.ccpa import CCPA
            self.ccpa = CCPA(
                num_classes=self.num_classes,
                threshold_mode=getattr(args, 'ccpa_threshold_mode', 'hybrid'),
                total_epochs=getattr(args, 'stage2_epoch', 120),
                use_memory_retrieval=True
            )
            self.ccpa.pseudo_momentum = getattr(args, 'pseudo_momentum', 0.9)
        
        self.ragm_module = None
        
        # Debug: store last matching statistics
        self.last_matching_stats = {}
    
    def set_epoch(self, epoch):
        self.current_epoch = epoch
        if self.use_ccpa:
            self.ccpa.set_epoch(epoch)
    
    def set_ragm_module(self, ragm):
        self.ragm_module = ragm
    
    def get_prototypes(self, modality='v'):
        if modality == 'v': return self.prototypes_v
        if modality == 'r': return self.prototypes_r
        raise ValueError(f"Unknown modality: {modality}")

    # =====================================================================
    #  CCPA & Prototype Logic
    # =====================================================================
    @torch.no_grad()
    def prepare_cycle_matching(self, diffusion_bridge):
        """
        Accepts diffusion_bridge as argument instead of storing it.
        """
        if not self.use_ccpa or diffusion_bridge is None:
            return
        
        if not self.prototypes_initialized:
            if self.vis_memory.abs().sum() > 0:
                self.prototypes_v = F.normalize(self.vis_memory.clone(), dim=1)
                self.prototypes_r = F.normalize(self.ir_memory.clone(), dim=1)
                self.prototypes_initialized = True
            else:
                self.prototypes_v = F.normalize(torch.randn_like(self.prototypes_v), dim=1)
                self.prototypes_r = F.normalize(torch.randn_like(self.prototypes_r), dim=1)
                self.prototypes_initialized = True
        
        self.prototypes_v = self.prototypes_v.to(self.device)
        self.prototypes_r = self.prototypes_r.to(self.device)
        
        try:
            self.ccpa.prepare_pseudo_prototypes(
                prototypes_r=self.prototypes_r,
                prototypes_v=self.prototypes_v,
                diffusion_bridge=diffusion_bridge,
                memory_v=self.vis_memory,
                memory_r=self.ir_memory
            )
        except Exception as e:
            # This will be logged by the caller
            raise RuntimeError(f"CCPA preparation failed: {e}")

    @torch.no_grad()
    def compute_cycle_consistent_matching(self, features_v, features_r, labels_v, labels_r):
        """Wrapper for CCPA cycle matching"""
        if not self.use_ccpa or not getattr(self.ccpa, 'pseudo_ready', False):
            return (torch.ones(len(features_v), device=self.device),
                   torch.ones(len(features_r), device=self.device), None, None)
        
        return self.ccpa.compute_cycle_matching(
            features_v, features_r, self.prototypes_v, self.prototypes_r, self.ragm_module
        )
    
    @torch.no_grad()
    def update_prototypes(self, features, labels, modality):
        """EMA update for prototypes"""
        prototypes = self.prototypes_v if modality == 'v' else self.prototypes_r
        for label in torch.unique(labels):
            mask = (labels == label)
            if mask.any():
                batch_proto = F.normalize(features[mask].mean(dim=0), dim=0)
                prototypes[label] = ((1 - self.proto_momentum) * prototypes[label] +
                                    self.proto_momentum * batch_proto)

    @torch.no_grad()
    def update_memory(self, rgb_feats, ir_feats, rgb_labels, ir_labels):
        """EMA update for memory bank"""
        for label in torch.unique(rgb_labels):
            mask = (rgb_labels == label)
            if mask.any():
                self.vis_memory[label] = ((1 - self.sigma) * self.vis_memory[label] +
                                         self.sigma * rgb_feats[mask].mean(dim=0))
        for label in torch.unique(ir_labels):
            mask = (ir_labels == label)
            if mask.any():
                self.ir_memory[label] = ((1 - self.sigma) * self.ir_memory[label] +
                                        self.sigma * ir_feats[mask].mean(dim=0))

    # =====================================================================
    #  Structure-Aware Global Optimal Transport Matching (SA-OT)
    # =====================================================================
    
    def extract(self, args, model, dataset):
        """Extract features and prepare data for matching."""
        model.set_eval()
        rgb_loader, ir_loader = dataset.get_normal_loader()
        
        def _extract(loader, modal):
            feats, labels, cls_outs = [], [], []
            for imgs, infos in loader:
                imgs = imgs.to(self.device)
                _, bn_feats = model.model(imgs)
                
                cls = model.classifier2(bn_feats)[0] if modal == 'rgb' else \
                      model.classifier1(bn_feats)[0]
                
                feats.append(bn_feats)
                labels.append(infos[:, 1].to(self.device))
                cls_outs.append(cls)
            return torch.cat(feats), torch.cat(labels), torch.cat(cls_outs)

        with torch.no_grad():
            rgb_feats, rgb_labels, r2i_cls = _extract(rgb_loader, 'rgb')
            ir_feats, ir_labels, i2r_cls = _extract(ir_loader, 'ir')
        
        self._init_memory_bank(rgb_feats, ir_feats, rgb_labels, ir_labels)
        
        self.vis_scores = F.softmax(self.T * r2i_cls, dim=1)
        self.ir_scores = F.softmax(self.T * i2r_cls, dim=1)
        
        self.rgb_ids = rgb_labels
        self.ir_ids = ir_labels
        self.not_saved = False

    def _init_memory_bank(self, rgb_features, ir_features, rgb_ids, ir_ids):
        """Initialize memory bank with class centers"""
        def _scatter_mean(features, ids, memory):
            numerator = torch.zeros_like(memory)
            counts = torch.zeros(memory.shape[0], 1, device=self.device)
            numerator.index_add_(0, ids, features)
            counts.index_add_(0, ids, torch.ones_like(ids, dtype=torch.float32).unsqueeze(1))
            memory.copy_(numerator / (counts + 1e-6))

        _scatter_mean(rgb_features, rgb_ids, self.vis_memory)
        _scatter_mean(ir_features, ir_ids, self.ir_memory)

    def get_label(self):
        """Generate 1-to-1 matching labels using Global Optimal Transport"""
        if self.not_saved:
            return {}, {}
        return self._optimal_transport_match()

    def _optimal_transport_match(self):
        """
        Structure-Aware Global Matching with Adaptive Threshold
        """
        # 1. Aggregate predictions to Identity-Level
        pred_v2r = torch.zeros(self.num_classes, self.num_classes, device=self.device)
        counts_v = torch.zeros(self.num_classes, 1, device=self.device)
        pred_v2r.index_add_(0, self.rgb_ids, self.vis_scores)
        counts_v.index_add_(0, self.rgb_ids, torch.ones_like(self.rgb_ids, dtype=torch.float).unsqueeze(1))
        pred_v2r = pred_v2r / (counts_v + 1e-6) 

        pred_r2v = torch.zeros(self.num_classes, self.num_classes, device=self.device)
        counts_r = torch.zeros(self.num_classes, 1, device=self.device)
        pred_r2v.index_add_(0, self.ir_ids, self.ir_scores)
        counts_r.index_add_(0, self.ir_ids, torch.ones_like(self.ir_ids, dtype=torch.float).unsqueeze(1))
        pred_r2v = pred_r2v / (counts_r + 1e-6)

        # 2. Build Cost Matrix
        vis_proto = F.normalize(self.vis_memory, dim=1)
        ir_proto = F.normalize(self.ir_memory, dim=1)
        cost_geo = 1.0 - torch.mm(vis_proto, ir_proto.t())
        
        cost_pred = -(torch.log(pred_v2r + 1e-8) + torch.log(pred_r2v.t() + 1e-8)) / 2
        
        alpha = 0.6
        C = alpha * cost_geo + (1 - alpha) * cost_pred

        # 3. Sinkhorn Algorithm
        transport_plan, sinkhorn_iterations = self._sinkhorn(C, epsilon=0.05, max_iter=50)
        
        v2i, i2v = OrderedDict(), OrderedDict()
        
        val_rows, idx_rows = transport_plan.max(dim=1) 
        val_cols, idx_cols = transport_plan.max(dim=0)
        
        safety_threshold = 1e-4
        
        if self.current_epoch < 10:
            max_cost_thresh = 3.5
        elif self.current_epoch < 30:
            max_cost_thresh = 2.8
        else:
            max_cost_thresh = 2.2
        
        # Collect all candidate costs for percentile-based filtering
        candidate_costs = []
        candidate_pairs = []
        
        for rgb_id in range(self.num_classes):
            ir_id = idx_rows[rgb_id].item()
            score = val_rows[rgb_id].item()
            
            if idx_cols[ir_id].item() == rgb_id and score > safety_threshold:
                match_cost = C[rgb_id, ir_id].item()
                candidate_costs.append(match_cost)
                candidate_pairs.append((rgb_id, ir_id, match_cost))
        
        if len(candidate_costs) > 0:
            percentile_thresh = np.percentile(candidate_costs, 75)
            effective_thresh = min(max_cost_thresh, percentile_thresh)
        else:
            effective_thresh = max_cost_thresh
        
        # Accept matches below threshold
        filtered_count = 0
        for rgb_id, ir_id, match_cost in candidate_pairs:
            if match_cost < effective_thresh:
                v2i[int(rgb_id)] = int(ir_id)
                i2v[int(ir_id)] = int(rgb_id)
            else:
                filtered_count += 1
        
        total_matches = len(v2i)
        
        # Store detailed statistics for debugging
        self._store_matching_stats(
            C, cost_geo, cost_pred, candidate_costs, 
            total_matches, filtered_count, effective_thresh,
            percentile_thresh if len(candidate_costs) > 0 else 0.0,
            sinkhorn_iterations, transport_plan
        )
        
        return v2i, i2v

    def _sinkhorn(self, cost_matrix, epsilon=0.05, max_iter=50):
        """Log-domain Sinkhorn-Knopp algorithm with iteration tracking"""
        K = torch.exp(-cost_matrix / epsilon)
        u = torch.ones_like(K[:, 0]) / K.shape[0]
        v = torch.ones_like(K[0, :]) / K.shape[1]
        
        for iteration in range(max_iter):
            u_prev = u.clone()
            u = 1.0 / (torch.matmul(K, v) * K.shape[0])
            v = 1.0 / (torch.matmul(K.t(), u) * K.shape[1])
            
            # Check convergence
            if (u - u_prev).abs().max() < 1e-6:
                break
            
        P = u.unsqueeze(1) * K * v.unsqueeze(0)
        return P, iteration + 1
    
    def _store_matching_stats(self, cost_matrix, cost_geo, cost_pred, 
                              candidate_costs, total_matches, filtered_count,
                              effective_thresh, percentile_thresh, 
                              sinkhorn_iters, transport_plan):
        """Store detailed matching statistics for debugging"""
        self.last_matching_stats = {
            # Cost matrix statistics
            'cost_total_mean': cost_matrix.mean().item(),
            'cost_total_std': cost_matrix.std().item(),
            'cost_total_min': cost_matrix.min().item(),
            'cost_total_max': cost_matrix.max().item(),
            'cost_geo_mean': cost_geo.mean().item(),
            'cost_geo_std': cost_geo.std().item(),
            'cost_pred_mean': cost_pred.mean().item(),
            'cost_pred_std': cost_pred.std().item(),
            
            # Matching results
            'candidates': len(candidate_costs),
            'accepted': total_matches,
            'filtered': filtered_count,
            'acceptance_rate': total_matches / len(candidate_costs) if len(candidate_costs) > 0 else 0.0,
            
            # Thresholds
            'effective_threshold': effective_thresh,
            'percentile_threshold': percentile_thresh,
            
            # Candidate cost statistics
            'candidate_cost_mean': np.mean(candidate_costs) if len(candidate_costs) > 0 else 0.0,
            'candidate_cost_std': np.std(candidate_costs) if len(candidate_costs) > 0 else 0.0,
            'candidate_cost_min': np.min(candidate_costs) if len(candidate_costs) > 0 else 0.0,
            'candidate_cost_max': np.max(candidate_costs) if len(candidate_costs) > 0 else 0.0,
            'candidate_cost_p25': np.percentile(candidate_costs, 25) if len(candidate_costs) > 0 else 0.0,
            'candidate_cost_p50': np.percentile(candidate_costs, 50) if len(candidate_costs) > 0 else 0.0,
            'candidate_cost_p75': np.percentile(candidate_costs, 75) if len(candidate_costs) > 0 else 0.0,
            
            # Sinkhorn convergence
            'sinkhorn_iterations': sinkhorn_iters,
            'transport_plan_sparsity': (transport_plan < 1e-6).float().mean().item(),
        }
    
    def get_matching_diagnostics(self):
        """
        Return formatted string with CMA matching diagnostics.
        """
        if not self.last_matching_stats:
            return "🔀 CMA Matching: No statistics available (not executed yet)"
        
        s = self.last_matching_stats
        lines = []
        lines.append("🔀 CMA Matching Diagnostics:")
        lines.append(f"  ├─ Epoch: {self.current_epoch}")
        lines.append(f"  │")
        lines.append(f"  ├─ Matching Results:")
        lines.append(f"  │  ├─ Candidates: {s['candidates']}")
        lines.append(f"  │  ├─ Accepted: {s['accepted']} ({s['acceptance_rate']*100:.1f}%)")
        lines.append(f"  │  └─ Filtered: {s['filtered']}")
        lines.append(f"  │")
        lines.append(f"  ├─ Cost Thresholds:")
        lines.append(f"  │  ├─ Effective: {s['effective_threshold']:.3f}")
        lines.append(f"  │  └─ Percentile (75th): {s['percentile_threshold']:.3f}")
        lines.append(f"  │")
        lines.append(f"  ├─ Cost Matrix Statistics:")
        lines.append(f"  │  ├─ Total: {s['cost_total_mean']:.3f} ± {s['cost_total_std']:.3f} | Range: [{s['cost_total_min']:.3f}, {s['cost_total_max']:.3f}]")
        lines.append(f"  │  ├─ Geometric: {s['cost_geo_mean']:.3f} ± {s['cost_geo_std']:.3f}")
        lines.append(f"  │  └─ Prediction: {s['cost_pred_mean']:.3f} ± {s['cost_pred_std']:.3f}")
        lines.append(f"  │")
        lines.append(f"  ├─ Candidate Cost Distribution:")
        lines.append(f"  │  ├─ Mean: {s['candidate_cost_mean']:.3f} ± {s['candidate_cost_std']:.3f}")
        lines.append(f"  │  ├─ Range: [{s['candidate_cost_min']:.3f}, {s['candidate_cost_max']:.3f}]")
        lines.append(f"  │  └─ Percentiles: [P25={s['candidate_cost_p25']:.3f}, P50={s['candidate_cost_p50']:.3f}, P75={s['candidate_cost_p75']:.3f}]")
        lines.append(f"  │")
        lines.append(f"  └─ Sinkhorn Algorithm:")
        lines.append(f"     ├─ Iterations: {s['sinkhorn_iterations']}/50")
        lines.append(f"     └─ Transport Sparsity: {s['transport_plan_sparsity']*100:.1f}%")
        
        return '\n'.join(lines)
    
    def get_memory_bank_stats(self):
        """Return memory bank statistics for debugging"""
        vis_occupied = (self.vis_memory.abs().sum(dim=1) > 1e-6).sum().item()
        ir_occupied = (self.ir_memory.abs().sum(dim=1) > 1e-6).sum().item()
        
        vis_norm = torch.norm(self.vis_memory, dim=1)
        ir_norm = torch.norm(self.ir_memory, dim=1)
        
        return {
            'vis_occupied': vis_occupied,
            'ir_occupied': ir_occupied,
            'vis_norm_mean': vis_norm.mean().item(),
            'vis_norm_std': vis_norm.std().item(),
            'ir_norm_mean': ir_norm.mean().item(),
            'ir_norm_std': ir_norm.std().item(),
        }
