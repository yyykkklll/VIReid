import torch
import torch.nn as nn
import torch.nn.functional as F


class MemoryBank(nn.Module):
    def __init__(self, num_classes=206, feat_dim=2048,
                 slots_per_class=5, device='cuda'):
        super().__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.slots_per_class = slots_per_class
        self.device = device
        
        # Buffers (Auto-saved)
        self.register_buffer('memory_keys_v', torch.zeros(num_classes, slots_per_class, feat_dim))
        self.register_buffer('memory_values_v', torch.zeros(num_classes, slots_per_class, feat_dim))
        self.register_buffer('quality_scores_v', torch.zeros(num_classes, slots_per_class))
        
        self.register_buffer('memory_keys_r', torch.zeros(num_classes, slots_per_class, feat_dim))
        self.register_buffer('memory_values_r', torch.zeros(num_classes, slots_per_class, feat_dim))
        self.register_buffer('quality_scores_r', torch.zeros(num_classes, slots_per_class))
        
        self.register_buffer('slot_usage_v', torch.zeros(num_classes, slots_per_class))
        self.register_buffer('slot_usage_r', torch.zeros(num_classes, slots_per_class))
        
        # Temporary counters for epoch statistics (Not saved in state_dict)
        self.update_count_v = 0
        self.update_count_r = 0
        self.rejected_count_v = 0
        self.rejected_count_r = 0
        
        # Last retrieval statistics
        self.last_retrieval_stats = {}
    
    def sync_from_cma(self, cma_vis_memory, cma_ir_memory):
        """Initialize empty slots 0 using CMA global class centers"""
        if self.quality_scores_v.sum() == 0:
            self.memory_keys_v[:, 0] = cma_vis_memory
            self.memory_values_v[:, 0] = cma_ir_memory
            self.quality_scores_v[:, 0] = 0.5
        
        if self.quality_scores_r.sum() == 0:
            self.memory_keys_r[:, 0] = cma_ir_memory
            self.memory_values_r[:, 0] = cma_vis_memory
            self.quality_scores_r[:, 0] = 0.5
    
    @torch.no_grad()
    def update(self, keys, values, labels, quality_scores, modality='v', threshold=0.5):
        """
        Update memory slots with high-quality samples.
        """
        # Select buffers
        if modality == 'v':
            mem_keys = self.memory_keys_v
            mem_vals = self.memory_values_v
            mem_scores = self.quality_scores_v
            mem_usage = self.slot_usage_v
        else:
            mem_keys = self.memory_keys_r
            mem_vals = self.memory_values_r
            mem_scores = self.quality_scores_r
            mem_usage = self.slot_usage_r
        
        # Filter high quality samples
        mask = quality_scores > threshold
        
        # Track rejected samples
        if modality == 'v':
            self.rejected_count_v += (~mask).sum().item()
        else:
            self.rejected_count_r += (~mask).sum().item()
        
        if not mask.any():
            return
        
        # Log Update Count
        if modality == 'v':
            self.update_count_v += mask.sum().item()
        else:
            self.update_count_r += mask.sum().item()
        
        valid_indices = torch.nonzero(mask).squeeze(1)
        
        for idx in valid_indices:
            lbl = labels[idx].item()
            score = quality_scores[idx].item()
            key_vec = keys[idx]
            val_vec = values[idx]
            
            # Find worst slot for this class
            min_score, min_idx = torch.min(mem_scores[lbl], dim=0)
            
            if score > min_score:
                mem_keys[lbl, min_idx] = key_vec
                mem_vals[lbl, min_idx] = val_vec
                mem_scores[lbl, min_idx] = score
                mem_usage[lbl, min_idx] += 1
    
    @torch.no_grad()
    def retrieve(self, query, labels, modality='v', top_k=3):
        """
        Retrieve best matching generated features from memory.
        """
        if modality == 'v':
            mem_keys = self.memory_keys_v
            mem_vals = self.memory_values_v
            mem_scores = self.quality_scores_v
        else:
            mem_keys = self.memory_keys_r
            mem_vals = self.memory_values_r
            mem_scores = self.quality_scores_r
        
        B = query.size(0)
        retrieved = torch.zeros(B, self.feat_dim, device=self.device)
        weights = torch.zeros(B, device=self.device)
        
        # Statistics
        hit_count = 0
        miss_count = 0
        
        for i, lbl in enumerate(labels):
            lbl = lbl.item()
            class_keys = mem_keys[lbl]
            class_vals = mem_vals[lbl]
            class_scores = mem_scores[lbl]
            
            if class_scores.sum() == 0:
                miss_count += 1
                continue
            
            hit_count += 1
            sim = F.cosine_similarity(query[i].unsqueeze(0), class_keys)
            combined = sim * class_scores
            
            k = min(top_k, self.slots_per_class)
            top_scores, top_idx = torch.topk(combined, k)
            
            attn = F.softmax(top_scores / 0.1, dim=0)
            retrieved_vec = (class_vals[top_idx] * attn.unsqueeze(1)).sum(dim=0)
            
            retrieved[i] = retrieved_vec
            weights[i] = top_scores.mean()
        
        # Store retrieval stats
        self.last_retrieval_stats[modality] = {
            'hit_rate': hit_count / B if B > 0 else 0.0,
            'avg_weight': weights[weights > 0].mean().item() if (weights > 0).any() else 0.0
        }
        
        return retrieved, weights
    
    def get_statistics(self):
        """Return usage stats for logging and reset epoch counters"""
        # Static stats
        occupied_v = (self.quality_scores_v.sum(dim=1) > 0).sum().item()
        occupied_r = (self.quality_scores_r.sum(dim=1) > 0).sum().item()
        
        # Safe average calculation
        valid_v = self.quality_scores_v[self.quality_scores_v > 0]
        valid_r = self.quality_scores_r[self.quality_scores_r > 0]
        
        avg_v = valid_v.mean().item() if valid_v.numel() > 0 else 0.0
        avg_r = valid_r.mean().item() if valid_r.numel() > 0 else 0.0
        
        stats = {
            'occupied_classes_v': occupied_v,
            'occupied_classes_r': occupied_r,
            'avg_quality_v': avg_v,
            'avg_quality_r': avg_r,
            'updates_v': self.update_count_v,
            'updates_r': self.update_count_r,
            'rejected_v': self.rejected_count_v,
            'rejected_r': self.rejected_count_r,
        }
        
        # Reset counters for next epoch
        self.update_count_v = 0
        self.update_count_r = 0
        self.rejected_count_v = 0
        self.rejected_count_r = 0
        
        return stats
    
    def get_detailed_diagnostics(self, current_threshold=0.5):
        """
        Comprehensive diagnostics for logger output.
        Returns formatted string with memory bank states.
        """
        # Get percentile statistics
        valid_v = self.quality_scores_v[self.quality_scores_v > 0]
        valid_r = self.quality_scores_r[self.quality_scores_r > 0]
        
        if valid_v.numel() > 0:
            q_v_p25 = torch.quantile(valid_v, 0.25).item()
            q_v_p50 = torch.quantile(valid_v, 0.50).item()
            q_v_p75 = torch.quantile(valid_v, 0.75).item()
        else:
            q_v_p25 = q_v_p50 = q_v_p75 = 0.0
        
        if valid_r.numel() > 0:
            q_r_p25 = torch.quantile(valid_r, 0.25).item()
            q_r_p50 = torch.quantile(valid_r, 0.50).item()
            q_r_p75 = torch.quantile(valid_r, 0.75).item()
        else:
            q_r_p25 = q_r_p50 = q_r_p75 = 0.0
        
        # Slot utilization
        occupied_v = (self.quality_scores_v.sum(dim=1) > 0).sum().item()
        occupied_r = (self.quality_scores_r.sum(dim=1) > 0).sum().item()
        
        # Slots per class stats
        slots_used_v = (self.quality_scores_v > 0).sum(dim=1).float()
        slots_used_r = (self.quality_scores_r > 0).sum(dim=1).float()
        
        avg_slots_v = slots_used_v[slots_used_v > 0].mean().item() if (slots_used_v > 0).any() else 0.0
        avg_slots_r = slots_used_r[slots_used_r > 0].mean().item() if (slots_used_r > 0).any() else 0.0
        
        lines = []
        lines.append("💾 Memory Bank Diagnostics:")
        lines.append(f"  ├─ Threshold: {current_threshold:.3f}")
        lines.append(f"  │")
        lines.append(f"  ├─ Visible→IR Memory:")
        lines.append(f"  │  ├─ Occupied Classes: {occupied_v}/{self.num_classes} ({occupied_v/self.num_classes*100:.1f}%)")
        lines.append(f"  │  ├─ Avg Slots/Class: {avg_slots_v:.2f}/{self.slots_per_class}")
        lines.append(f"  │  ├─ Quality Percentiles: [P25={q_v_p25:.3f}, P50={q_v_p50:.3f}, P75={q_v_p75:.3f}]")
        lines.append(f"  │  ├─ Updates: +{self.update_count_v} | Rejected: {self.rejected_count_v}")
        
        if 'v' in self.last_retrieval_stats:
            ret_v = self.last_retrieval_stats['v']
            lines.append(f"  │  └─ Retrieval Hit Rate: {ret_v['hit_rate']*100:.1f}% | Avg Weight: {ret_v['avg_weight']:.3f}")
        
        lines.append(f"  │")
        lines.append(f"  └─ IR→Visible Memory:")
        lines.append(f"     ├─ Occupied Classes: {occupied_r}/{self.num_classes} ({occupied_r/self.num_classes*100:.1f}%)")
        lines.append(f"     ├─ Avg Slots/Class: {avg_slots_r:.2f}/{self.slots_per_class}")
        lines.append(f"     ├─ Quality Percentiles: [P25={q_r_p25:.3f}, P50={q_r_p50:.3f}, P75={q_r_p75:.3f}]")
        lines.append(f"     ├─ Updates: +{self.update_count_r} | Rejected: {self.rejected_count_r}")
        
        if 'r' in self.last_retrieval_stats:
            ret_r = self.last_retrieval_stats['r']
            lines.append(f"     └─ Retrieval Hit Rate: {ret_r['hit_rate']*100:.1f}% | Avg Weight: {ret_r['avg_weight']:.3f}")
        
        return '\n'.join(lines)
