import torch
import torch.nn as nn
import torch.nn.functional as F

class CCPA(nn.Module):
    def __init__(self, num_classes=206, threshold_mode='hybrid',
                 total_epochs=120, use_memory_retrieval=True):
        super().__init__()
        self.num_classes = num_classes
        self.threshold_mode = threshold_mode
        self.total_epochs = total_epochs
        
        # Pseudo prototypes (generated via diffusion)
        self.register_buffer('pseudo_prototypes_v2r', torch.zeros(num_classes, 2048))
        self.register_buffer('pseudo_prototypes_r2v', torch.zeros(num_classes, 2048))
        
        # Quality scores (for fallback decision)
        self.register_buffer('quality_v2r', torch.zeros(num_classes))
        self.register_buffer('quality_r2v', torch.zeros(num_classes))
        
        # EMA smoothing for pseudo prototypes
        self.pseudo_momentum = 0.9
        
        # State
        self.current_epoch = 0
        self.pseudo_ready = False
        self.use_fallback = False
        
        # Debug statistics
        self.last_threshold = 0.0
        self.last_quality_stats = {}
    
    def set_epoch(self, epoch):
        """Update current epoch for dynamic thresholding"""
        self.current_epoch = epoch
    
    @torch.no_grad()
    def prepare_pseudo_prototypes(self, prototypes_r, prototypes_v,
                                   diffusion_bridge, memory_v=None, memory_r=None):
        device = prototypes_r.device
        
        # Create labels for all classes
        all_labels = torch.arange(self.num_classes, device=device)
        
        # Unpack 4 values: f_r, f_v, unc_v, unc_r
        _, pseudo_v, _, unc_r2v = diffusion_bridge.generate(
            f_r=prototypes_r,
            labels_r=all_labels
        )
        
        pseudo_r, _, unc_v2r, _ = diffusion_bridge.generate(
            f_v=prototypes_v,
            labels_v=all_labels
        )
        
        if pseudo_v is None or pseudo_r is None:
            self.use_fallback = True
            self.pseudo_ready = False
            return
        
        # Normalize
        pseudo_v = F.normalize(pseudo_v, dim=1)
        pseudo_r = F.normalize(pseudo_r, dim=1)
        
        # ✅ 修改1: 添加质量崩塌检测
        raw_qual_r2v = F.cosine_similarity(pseudo_v, prototypes_v, dim=1)
        raw_qual_v2r = F.cosine_similarity(pseudo_r, prototypes_r, dim=1)
        
        # 检查是否出现崩塌
        if raw_qual_r2v.mean() < 0.1 or raw_qual_v2r.mean() < 0.1:
            print(f"⚠️ WARNING: CCPA Quality Collapse Detected!")
            print(f"   Raw V→R: {raw_qual_v2r.mean():.4f} | Raw R→V: {raw_qual_r2v.mean():.4f}")
            # 使用fallback机制
            self.use_fallback = True
            self.pseudo_ready = False
            return
        
        # ReLU确保非负
        raw_qual_r2v = F.relu(raw_qual_r2v)
        raw_qual_v2r = F.relu(raw_qual_v2r)
        
        # ✅ 修改2: 降低uncertainty惩罚强度（从0.2→0.05）
        lambda_pen = 0.05
        
        if unc_r2v is not None:
            unc_r2v = torch.nan_to_num(unc_r2v, nan=10.0).clamp(min=0.0, max=2.0)  # ✅ 添加上限
            self.quality_r2v = raw_qual_r2v * torch.exp(-lambda_pen * unc_r2v)
        else:
            self.quality_r2v = raw_qual_r2v
        
        if unc_v2r is not None:
            unc_v2r = torch.nan_to_num(unc_v2r, nan=10.0).clamp(min=0.0, max=2.0)  # ✅ 添加上限
            self.quality_v2r = raw_qual_v2r * torch.exp(-lambda_pen * unc_v2r)
        else:
            self.quality_v2r = raw_qual_v2r
        
        # Update buffers (EMA)
        if self.pseudo_ready:
            self.pseudo_prototypes_r2v = (1 - self.pseudo_momentum) * self.pseudo_prototypes_r2v + \
                                          self.pseudo_momentum * pseudo_v
            self.pseudo_prototypes_v2r = (1 - self.pseudo_momentum) * self.pseudo_prototypes_v2r + \
                                          self.pseudo_momentum * pseudo_r
        else:
            self.pseudo_prototypes_r2v = pseudo_v
            self.pseudo_prototypes_v2r = pseudo_r
            self.pseudo_ready = True
        
        self.use_fallback = False
        
        # Store quality statistics for debugging
        self._update_quality_stats(unc_v2r, unc_r2v, raw_qual_v2r, raw_qual_r2v)
    
    @torch.no_grad()
    def compute_cycle_matching(self, features_v, features_r, prototypes_v, prototypes_r, ragm_module=None):
        """Compute reliability masks based on cycle consistency"""
        
        # Threshold calculation
        thresh_v = self._compute_threshold(self.quality_v2r, self.quality_r2v)
        thresh_r = thresh_v
        self.last_threshold = thresh_v
        
        # Predict labels from real prototypes
        sim_v2real_v = torch.mm(features_v, prototypes_v.t())
        pred_labels_v = sim_v2real_v.argmax(dim=1)
        
        sim_r2real_r = torch.mm(features_r, prototypes_r.t())
        pred_labels_r = sim_r2real_r.argmax(dim=1)
        
        # Check against Pseudo Prototypes
        target_pseudo_v = self.pseudo_prototypes_r2v[pred_labels_v]
        consistency_v = F.cosine_similarity(features_v, target_pseudo_v, dim=1)
        
        target_pseudo_r = self.pseudo_prototypes_v2r[pred_labels_r]
        consistency_r = F.cosine_similarity(features_r, target_pseudo_r, dim=1)
        
        # Generate Masks
        mask_v = (consistency_v > thresh_v).float()
        mask_r = (consistency_r > thresh_r).float()
        
        return mask_v, mask_r, consistency_v, consistency_r
    
    def _compute_threshold(self, qual_v2r, qual_r2v):
        """✅ 修改3: 动态threshold，适应低质量状态"""
        avg_quality = (qual_v2r.mean() + qual_r2v.mean()) / 2
        
        # 质量自适应base threshold
        if avg_quality < 0.05:  # 质量极低时
            base_thresh = 0.01  # 降低阈值，避免全部拒绝
        elif avg_quality < 0.2:
            base_thresh = 0.1
        else:
            base_thresh = 0.2
        
        if self.threshold_mode == 'fixed':
            return base_thresh
        elif self.threshold_mode == 'adaptive':
            progress = self.current_epoch / self.total_epochs
            return base_thresh + 0.2 * progress * avg_quality
        else:  # hybrid/default
            return base_thresh * 0.8 + 0.2 * avg_quality.item()
    
    def _update_quality_stats(self, unc_v2r, unc_r2v, raw_qual_v2r, raw_qual_r2v):
        """Internal: update quality statistics for debugging"""
        self.last_quality_stats = {
            'raw_qual_v2r_mean': raw_qual_v2r.mean().item(),
            'raw_qual_r2v_mean': raw_qual_r2v.mean().item(),
            'calibrated_qual_v2r_mean': self.quality_v2r.mean().item(),
            'calibrated_qual_r2v_mean': self.quality_r2v.mean().item(),
            'qual_v2r_min': self.quality_v2r.min().item(),
            'qual_v2r_max': self.quality_v2r.max().item(),
            'qual_r2v_min': self.quality_r2v.min().item(),
            'qual_r2v_max': self.quality_r2v.max().item(),
        }
        
        if unc_v2r is not None:
            self.last_quality_stats.update({
                'unc_v2r_mean': unc_v2r.mean().item(),
                'unc_v2r_std': unc_v2r.std().item(),
            })
        if unc_r2v is not None:
            self.last_quality_stats.update({
                'unc_r2v_mean': unc_r2v.mean().item(),
                'unc_r2v_std': unc_r2v.std().item(),
            })
    
    def get_pseudo_quality_info(self):
        """Enhanced info dict for debugging"""
        return {
            'quality_v2r': self.quality_v2r.mean().item(),
            'quality_r2v': self.quality_r2v.mean().item(),
            'use_fallback': self.use_fallback,
            'pseudo_ready': self.pseudo_ready,
            'current_threshold': self.last_threshold,
            'quality_stats': self.last_quality_stats.copy() if self.last_quality_stats else {}
        }
    
    def get_detailed_diagnostics(self):
        """
        Comprehensive diagnostics for logger output.
        Returns formatted string with all CCPA module states.
        """
        if not self.pseudo_ready:
            return "🔄 CCPA: Not Ready (Pseudo prototypes not generated)"
        
        info = self.get_pseudo_quality_info()
        stats = info.get('quality_stats', {})
        
        lines = []
        lines.append("🔄 CCPA Module Diagnostics:")
        lines.append(f"   ├─ Status: {'✓ Ready' if self.pseudo_ready else '⏳ Preparing'}")
        lines.append(f"   ├─ Fallback Mode: {self.use_fallback}")
        lines.append(f"   ├─ Threshold Mode: {self.threshold_mode}")
        lines.append(f"   ├─ Current Threshold: {self.last_threshold:.4f}")
        lines.append(f"   │")
        lines.append(f"   ├─ Quality Scores (Calibrated):")
        lines.append(f"   │   ├─ V→R: {info['quality_v2r']:.4f}")
        lines.append(f"   │   └─ R→V: {info['quality_r2v']:.4f}")
        
        if stats:
            lines.append(f"   │")
            lines.append(f"   ├─ Raw Quality (Before Uncertainty Penalty):")
            lines.append(f"   │   ├─ V→R: {stats.get('raw_qual_v2r_mean', 0):.4f}")
            lines.append(f"   │   └─ R→V: {stats.get('raw_qual_r2v_mean', 0):.4f}")
            lines.append(f"   │")
            lines.append(f"   ├─ Quality Range:")
            lines.append(f"   │   ├─ V→R: [{stats.get('qual_v2r_min', 0):.3f}, {stats.get('qual_v2r_max', 0):.3f}]")
            lines.append(f"   │   └─ R→V: [{stats.get('qual_r2v_min', 0):.3f}, {stats.get('qual_r2v_max', 0):.3f}]")
            
            if 'unc_v2r_mean' in stats:
                lines.append(f"   │")
                lines.append(f"   └─ Uncertainty:")
                lines.append(f"       ├─ V→R: {stats['unc_v2r_mean']:.4f} ± {stats.get('unc_v2r_std', 0):.4f}")
                lines.append(f"       └─ R→V: {stats.get('unc_r2v_mean', 0):.4f} ± {stats.get('unc_r2v_std', 0):.4f}")
        
        return '\n'.join(lines)
