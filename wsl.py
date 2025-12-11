"""
Cross-Modal Matching Aggregation Module
========================================
Author: Fixed Version
Date: 2025-12-11
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import Counter, OrderedDict


class CMA(nn.Module):
    """Cross-Modal Matching Aggregator"""
    
    def __init__(self, args):
        super(CMA, self).__init__()
        self.device = torch.device(args.device)
        self.num_classes = args.num_classes
        self.T = args.temperature
        self.sigma = args.sigma
        self.args = args
        
        self.register_buffer('vis_memory', torch.zeros(self.num_classes, 2048))
        self.register_buffer('ir_memory', torch.zeros(self.num_classes, 2048))
        
        self.vis_clip_feats = None
        self.ir_clip_feats = None
        self.not_saved = True
        self.mode = None
    
    def extract_and_match(self, model, dataset, clip_model=None):
        """Extract features and perform matching"""
        self._extract_features(model, dataset, clip_model)
        
        if hasattr(self.args, 'use_sinkhorn') and self.args.use_sinkhorn:
            print("Using Sinkhorn algorithm for global matching...")
            return self._match_sinkhorn()
        else:
            print("Using greedy algorithm for fast matching...")
            return self._match_greedy()
    
    @torch.no_grad()
    def _extract_features(self, model, dataset, clip_model=None):
        """Extract RGB and IR modality features"""
        model.set_eval()
        rgb_loader, ir_loader = dataset.get_normal_loader()
        
        print("Extracting RGB features...")
        rgb_feats, rgb_labels, rgb_cls, rgb_clip = self._extract_single_modal(
            model, rgb_loader, 'rgb', clip_model)
        
        print("Extracting IR features...")
        ir_feats, ir_labels, ir_cls, ir_clip = self._extract_single_modal(
            model, ir_loader, 'ir', clip_model)
        
        self._save_features(
            rgb_cls, ir_cls, rgb_labels, ir_labels, 
            rgb_feats, ir_feats, rgb_clip, ir_clip
        )
    
    @torch.no_grad()
    def _extract_single_modal(self, model, loader, modal, clip_model=None):
        """Extract single modality features"""
        all_features = []
        all_labels = []
        all_cls_scores = []
        all_clip_feats = []
        
        for imgs_list, infos in loader:
            labels = infos[:, 1].to(self.device)
            
            if isinstance(imgs_list, (list, tuple)):
                imgs = imgs_list[0].to(self.device)
            else: 
                imgs = imgs_list.to(self.device)
            
            _, bn_features = model.model(imgs, None) if modal == 'rgb' else model.model(None, imgs)
            
            if modal == 'rgb':
                cls_scores, _ = model.classifier2(bn_features)
            else: 
                cls_scores, _ = model.classifier1(bn_features)
            
            all_features.append(bn_features.cpu())
            all_labels.append(labels.cpu())
            all_cls_scores.append(cls_scores.cpu())
            
            if clip_model is not None: 
                clip_feats = clip_model.encode_image(imgs)
                all_clip_feats.append(clip_feats.detach().cpu())
        
        features = torch.cat(all_features, dim=0)
        labels = torch.cat(all_labels, dim=0)
        cls_scores = torch.cat(all_cls_scores, dim=0)
        clip_feats = torch.cat(all_clip_feats, dim=0) if all_clip_feats else None
        
        return features, labels, cls_scores, clip_feats
    
    @torch.no_grad()
    def _save_features(self, rgb_cls, ir_cls, rgb_labels, ir_labels, 
                       rgb_features, ir_features, clip_rgb, clip_ir):
        """Save features to internal state and update memory bank"""
        self.mode = 'scores'
        self.not_saved = False
        
        self.vis = F.softmax(self.T * rgb_cls, dim=1).cpu().numpy()
        self.ir = F.softmax(self.T * ir_cls, dim=1).cpu().numpy()
        self.rgb_ids = rgb_labels.cpu()
        self.ir_ids = ir_labels.cpu()
        
        self._update_memory(rgb_features.to(self.device), ir_features.to(self.device),
                            rgb_labels.to(self.device), ir_labels.to(self.device))
        
        if clip_rgb is not None and clip_ir is not None: 
            self.vis_clip_feats = clip_rgb
            self.ir_clip_feats = clip_ir
            print(f"CLIP features saved: RGB {clip_rgb.shape}, IR {clip_ir.shape}")
        else:
            self.vis_clip_feats = None
            self.ir_clip_feats = None
    
    @torch.no_grad()
    def _update_memory(self, rgb_feats, ir_feats, rgb_labels, ir_labels):
        """Update feature memory bank using EMA"""
        self.vis_memory = self.vis_memory.to(self.device)
        self.ir_memory = self.ir_memory.to(self.device)
        
        for label in torch.unique(rgb_labels):
            mask = (rgb_labels == label)
            if mask.any():
                new_feat = rgb_feats[mask].mean(dim=0)
                self.vis_memory[label] = (1 - self.sigma) * self.vis_memory[label] + \
                                         self.sigma * new_feat
        
        for label in torch.unique(ir_labels):
            mask = (ir_labels == label)
            if mask.any():
                new_feat = ir_feats[mask].mean(dim=0)
                self.ir_memory[label] = (1 - self.sigma) * self.ir_memory[label] + \
                                        self.sigma * new_feat
    
    def _match_sinkhorn(self):
        """Global optimal matching using Sinkhorn algorithm"""
        score_rgb = torch.from_numpy(self.vis).to(self.device)
        score_ir = torch.from_numpy(self.ir).to(self.device)
        score_rgb = F.normalize(score_rgb, dim=1)
        score_ir = F.normalize(score_ir, dim=1)
        sim_expert = torch.matmul(score_rgb, score_ir.T)
        
        if self.vis_clip_feats is not None and self.ir_clip_feats is not None:
            clip_rgb = F.normalize(self.vis_clip_feats.to(self.device), dim=1)
            clip_ir = F.normalize(self.ir_clip_feats.to(self.device), dim=1)
            sim_clip = torch.matmul(clip_rgb, clip_ir.T)
            
            w_clip = getattr(self.args, 'w_clip', 0.3)
            sim_final = (1 - w_clip) * sim_expert + w_clip * sim_clip
            print(f"CLIP weight: {w_clip:.2f}")
        else:
            sim_final = sim_expert
        
        epsilon = getattr(self.args, 'sinkhorn_reg', 0.05)
        log_Q = sim_final / epsilon
        log_Q = torch.clamp(log_Q, min=-50, max=50)
        
        max_iters = 100
        tolerance = 1e-4
        for iteration in range(max_iters):
            log_Q_prev = log_Q.clone()
            log_Q = log_Q - torch.logsumexp(log_Q, dim=1, keepdim=True)
            log_Q = log_Q - torch.logsumexp(log_Q, dim=0, keepdim=True)
            
            if torch.abs(log_Q - log_Q_prev).max() < tolerance:
                print(f"Sinkhorn converged at iteration {iteration}")
                break
        
        Q = torch.exp(log_Q).cpu().numpy()
        
        # 修复：使用更合理的阈值策略
        v2i, i2v = self._generate_matches_from_Q_fixed(Q)
        
        print(f"Matching results: RGB->IR {len(v2i)}/{len(self.rgb_ids)}, "
              f"IR->RGB {len(i2v)}/{len(self.ir_ids)}")
        
        return v2i, i2v
    
    def _generate_matches_from_Q_fixed(self, Q):
        """
        Fixed matching generation with adaptive threshold
        """
        v2i = OrderedDict()
        i2v = OrderedDict()
        
        # 计算每行和每列的最大值
        max_rgb = np.max(Q, axis=1)
        max_ir = np.max(Q, axis=0)
        
        # 使用Top-K策略而非固定阈值
        top_k = min(5, Q.shape[1])  # 每个RGB最多匹配5个IR
        
        for i in range(Q.shape[0]):
            # 获取Top-K匹配
            top_indices = np.argsort(Q[i])[-top_k:][::-1]
            
            for j in top_indices:
                # 双向验证：必须互为Top-K
                if i in np.argsort(Q[:, j])[-top_k:]:
                    rgb_id = self.rgb_ids[i].item()
                    ir_id = self.ir_ids[j].item()
                    
                    if rgb_id not in v2i:
                        v2i[rgb_id] = ir_id
                    if ir_id not in i2v:
                        i2v[ir_id] = rgb_id
                    break  # 只保留最佳匹配
        
        return v2i, i2v
    
    def _match_greedy(self):
        """Greedy matching algorithm"""
        dists = np.matmul(self.vis, self.ir.T)
        
        sorted_indices = np.argsort(-dists, axis=None)
        sorted_2d = np.unravel_index(sorted_indices, dists.shape)
        idx_rgb, idx_ir = sorted_2d[0], sorted_2d[1]
        
        pairs = [(self.rgb_ids[i].item(), self.ir_ids[j].item()) 
                 for i, j in zip(idx_rgb, idx_ir)]
        pair_counts = Counter(pairs)
        
        v2i, i2v = OrderedDict(), OrderedDict()
        matched_rgb, matched_ir = set(), set()
        
        for (rgb_id, ir_id), count in pair_counts.most_common():
            if rgb_id not in matched_rgb and ir_id not in matched_ir:
                v2i[rgb_id] = ir_id
                i2v[ir_id] = rgb_id
                matched_rgb.add(rgb_id)
                matched_ir.add(ir_id)
        
        print(f"Greedy matching results: {len(v2i)} pairs")
        return v2i, i2v
    
    def get_memory_features(self):
        """Get memory bank features"""
        return self.vis_memory, self.ir_memory
    
    def reset(self):
        """Reset internal state"""
        self.not_saved = True
        self.vis_clip_feats = None
        self.ir_clip_feats = None
