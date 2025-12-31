import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import weakref
from .diffusion_denoiser import ImprovedDiffusionUNet
from .memory_bank import MemoryBank
from .prototype_guidance import PrototypeGuidance


# ==================== Single-Layer Diffusion Module ====================
class DiffusionModule(nn.Module):
    def __init__(self, feat_dim=2048, hidden_dim=1024, num_steps=10,
                 num_heads=4, num_res_blocks=2, dropout=0.1, device='cuda'):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_steps = num_steps
        self.device = device
        
        self.denoiser = ImprovedDiffusionUNet(
            feat_dim=feat_dim,
            hidden_dim=hidden_dim,
            num_steps=num_steps,
            num_heads=num_heads,
            num_res_blocks=num_res_blocks,
            dropout=dropout
        )
        
        self._init_diffusion_params()
        
        # Debug statistics
        self.last_loss_stats = {}
        self.last_sample_stats = {}
    
    def _init_diffusion_params(self):
        """Cosine noise schedule"""
        betas = self._cosine_beta_schedule(self.num_steps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas', torch.sqrt(1.0 / alphas))
        
        alphas_cumprod_prev = torch.cat([alphas_cumprod[:1], alphas_cumprod[:-1]])
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)
    
    def _cosine_beta_schedule(self, num_steps):
        """Cosine schedule for stable few-step diffusion"""
        s = 0.008
        steps = num_steps + 1
        x = torch.linspace(0, num_steps, steps)
        alphas_cumprod = torch.cos(((x / num_steps) + s) / (1 + s) * np.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)
    
    def q_sample(self, x_start, t, noise=None):
        """Forward diffusion: add noise"""
        if noise is None:
            noise = torch.randn_like(x_start)
        
        batch_size = x_start.size(0)
        if t.dim() == 0:
            t = t.unsqueeze(0).expand(batch_size)
        elif t.size(0) != batch_size:
            t = t[:batch_size]
        
        sqrt_alpha_t = self.sqrt_alphas_cumprod[t].view(batch_size, 1)
        sqrt_one_minus_alpha_t = self.sqrt_one_minus_alphas_cumprod[t].view(batch_size, 1)
        
        return sqrt_alpha_t * x_start + sqrt_one_minus_alpha_t * noise
    
    def _heteroscedastic_loss(self, pred_noise, pred_log_var, target_noise):
        # Clamp log_var to prevent extreme values
        pred_log_var = torch.clamp(pred_log_var, min=-3, max=2)
        
        # Compute variance and precision
        variance = torch.exp(pred_log_var)
        precision = 1.0 / (variance + 1e-6)
        
        # MSE between predicted and target noise
        mse = (pred_noise - target_noise) ** 2
        
        # Heteroscedastic loss (correct formula)
        loss_hetero = 0.5 * (mse * precision + pred_log_var)
        
        # Variance regularization to prevent collapse
        target_log_var = -0.5  # Encourage variance ~ 0.6
        variance_reg = 0.1 * (pred_log_var - target_log_var) ** 2
        
        loss = loss_hetero + variance_reg
        
        return loss
    
    @torch.no_grad()
    def sample(self, f_cond, prototype_guide=None, guide_weight=0.3):
        """
        Single-pass sampling that returns both features and learned uncertainty.
        """
        B = f_cond.size(0)
        z_t = torch.randn(B, self.feat_dim, device=self.device)
        
        accumulated_uncertainty = 0.0
        log_var_history = []
        
        for t in reversed(range(self.num_steps)):
            t_batch = torch.full((B,), t, dtype=torch.long, device=self.device)
            
            model_out = self.denoiser(z_t, f_cond, t_batch)
            pred_noise, pred_log_var = torch.chunk(model_out, 2, dim=1)
            
            pred_log_var = torch.clamp(pred_log_var, min=-3, max=2)
            log_var_history.append(pred_log_var.mean().item())
            
            accumulated_uncertainty += torch.exp(pred_log_var).mean(dim=1)
            
            beta_t = self.betas[t_batch].view(B, 1)
            sqrt_recip_alpha_t = self.sqrt_recip_alphas[t_batch].view(B, 1)
            sqrt_one_minus_alpha_t = self.sqrt_one_minus_alphas_cumprod[t_batch].view(B, 1)
            
            mean = sqrt_recip_alpha_t * (z_t - beta_t / sqrt_one_minus_alpha_t * pred_noise)
            
            if t > 0:
                variance = self.posterior_variance[t_batch].view(B, 1)
                noise = torch.randn_like(z_t)
                z_t = mean + torch.sqrt(variance) * noise
            else:
                z_t = mean
            
            if prototype_guide is not None:
                proto_offset = prototype_guide - z_t
                z_t = z_t + guide_weight * proto_offset
        
        avg_uncertainty = accumulated_uncertainty / self.num_steps
        
        # Store sampling statistics
        self.last_sample_stats = {
            'log_var_mean': np.mean(log_var_history),
            'log_var_std': np.std(log_var_history),
            'log_var_min': np.min(log_var_history),
            'log_var_max': np.max(log_var_history),
            'avg_uncertainty_mean': avg_uncertainty.mean().item(),
            'avg_uncertainty_std': avg_uncertainty.std().item(),
        }
        
        return F.normalize(z_t, dim=1), avg_uncertainty
    
    def compute_loss(self, f_cond, f_target):
        """
        Compute heteroscedastic loss with balanced regularization
        """
        if f_cond.size(0) != f_target.size(0):
            min_batch = min(f_cond.size(0), f_target.size(0))
            f_cond = f_cond[:min_batch]
            f_target = f_target[:min_batch]
        
        B = f_cond.size(0)
        # ✅ [FIX 1] 放大信号尺度，匹配高斯噪声 (sqrt(2048) ≈ 45)
        # z_0 = F.normalize(f_target, dim=1) # ❌ Old
        z_0 = F.normalize(f_target, dim=1) * np.sqrt(self.feat_dim) # ✅ Fixed
        
        t = torch.randint(0, self.num_steps, (B,), device=self.device)
        noise = torch.randn_like(z_0)
        z_t = self.q_sample(z_0, t, noise)
        
        model_out = self.denoiser(z_t, f_cond, t)
        pred_noise, pred_log_var = torch.chunk(model_out, 2, dim=1)
        
        pred_log_var = torch.clamp(pred_log_var, min=-3, max=2)
        
        loss_per_pixel = self._heteroscedastic_loss(pred_noise, pred_log_var, noise)
        loss_batch = loss_per_pixel.mean(dim=1)
        
        with torch.no_grad():
            predicted_uncertainty = torch.exp(pred_log_var).mean(dim=1)
            
            # Store loss statistics
            self.last_loss_stats = {
                'loss_mean': loss_batch.mean().item(),
                'loss_std': loss_batch.std().item(),
                'log_var_mean': pred_log_var.mean().item(),
                'log_var_std': pred_log_var.std().item(),
                'log_var_min': pred_log_var.min().item(),
                'log_var_max': pred_log_var.max().item(),
                'uncertainty_mean': predicted_uncertainty.mean().item(),
                'uncertainty_std': predicted_uncertainty.std().item(),
                'noise_mse': F.mse_loss(pred_noise, noise).item(),
            }
        
        return loss_batch.mean(), z_0, predicted_uncertainty


# ==================== Hierarchical Diffusion Bridge ====================
class HierarchicalDiffusionBridge(nn.Module):
    def __init__(self, feat_dim=2048, hidden_dim=1024,
                 feature_steps=5, semantic_steps=10,
                 num_heads=4, dropout=0.1, device='cuda',
                 use_memory=True, memory_slots=5,
                 num_classes=206):
        super().__init__()
        self.feat_dim = feat_dim
        self.device = device
        self.use_memory = use_memory
        self.num_classes = num_classes
        
        diff_args = {
            'feat_dim': feat_dim, 'hidden_dim': hidden_dim,
            'num_heads': num_heads, 'dropout': dropout, 'device': device
        }
        
        self.feature_diffusion_v2r = DiffusionModule(
            num_steps=feature_steps, num_res_blocks=2, **diff_args
        )
        self.feature_diffusion_r2v = DiffusionModule(
            num_steps=feature_steps, num_res_blocks=2, **diff_args
        )
        self.semantic_diffusion_v2r = DiffusionModule(
            num_steps=semantic_steps, num_res_blocks=3, **diff_args
        )
        self.semantic_diffusion_r2v = DiffusionModule(
            num_steps=semantic_steps, num_res_blocks=3, **diff_args
        )
        
        if use_memory:
            self.memory_bank = MemoryBank(
                num_classes=num_classes, feat_dim=feat_dim,
                slots_per_class=memory_slots, device=device
            )
        
        self.proto_guidance = PrototypeGuidance(
            num_classes=num_classes, feat_dim=feat_dim, device=device
        )
        
        self.fusion = nn.Sequential(
            nn.Linear(feat_dim * 2, feat_dim),
            nn.LayerNorm(feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim, feat_dim)
        )
        
        self.cma = None
        self.current_epoch = 0
        self.register_buffer('denoiser_health_status', torch.tensor(1.0))
    
    def set_cma(self, cma_module):
        self.cma = weakref.proxy(cma_module)
        if self.use_memory:
            self.memory_bank.sync_from_cma(cma_module.vis_memory, cma_module.ir_memory)
    
    def set_epoch(self, epoch):
        self.current_epoch = epoch
        self.proto_guidance.set_epoch(epoch)
    
    def forward(self, f_v, f_r, labels_v=None, labels_r=None, mode='train'):
        if mode == 'train':
            return self.compute_loss(f_v, f_r, labels_v, labels_r)
        else:
            return self.generate(f_v, f_r, labels_v, labels_r)
    
    def compute_loss(self, f_v, f_r, labels_v=None, labels_r=None):
        loss_v2r_feat, feat_v2r, unc_v2r = self.feature_diffusion_v2r.compute_loss(f_v, f_r)
        loss_v2r_sem, sem_v2r, unc_v2r_sem = self.semantic_diffusion_v2r.compute_loss(f_v, f_r)
        
        loss_r2v_feat, feat_r2v, unc_r2v = self.feature_diffusion_r2v.compute_loss(f_r, f_v)
        loss_r2v_sem, sem_r2v, unc_r2v_sem = self.semantic_diffusion_r2v.compute_loss(f_r, f_v)
        
        with torch.no_grad():
            uncertainty_v = (unc_v2r + unc_v2r_sem) / 2.0
            uncertainty_r = (unc_r2v + unc_r2v_sem) / 2.0
        
        memory_loss = 0.0
        if self.use_memory and labels_v is not None:
            memory_loss = self._compute_memory_loss(f_v, feat_v2r, labels_v, 'v') + \
                         self._compute_memory_loss(f_r, feat_r2v, labels_r, 'r')
            memory_loss = memory_loss / 2
        
        proto_loss_v = torch.tensor(0.0, device=f_v.device)
        proto_loss_r = torch.tensor(0.0, device=f_r.device)
        
        if self.cma is not None and labels_v is not None:
            try:
                proto_weight = self.proto_guidance.get_guidance_weight(self.current_epoch)
                proto_loss_v = proto_weight * self.proto_guidance.compute_guidance_loss(
                    sem_v2r, self.cma.get_prototypes('r'), labels_v
                )
                proto_loss_r = proto_weight * self.proto_guidance.compute_guidance_loss(
                    sem_r2v, self.cma.get_prototypes('v'), labels_r
                )
            except ReferenceError:
                pass
        
        # ✅ [FIX 2] 彻底删除权重惩罚
        # ❌ 已移除：weight_reg = 0.0 ... weight_penalty = 1e-4 * weight_reg
        
        total_loss = (loss_v2r_feat + loss_r2v_feat +
                    loss_v2r_sem + loss_r2v_sem +
                    0.3 * memory_loss +
                    proto_loss_v + proto_loss_r)
                    # + weight_penalty) ❌ 移除这里
        
        loss_dict = {
            'v2r_feat_mse': loss_v2r_feat,
            'r2v_feat_mse': loss_r2v_feat,
            'v2r_sem_mse': loss_v2r_sem,
            'r2v_sem_mse': loss_r2v_sem,
            'memory_align': memory_loss,
            'proto_align_v': proto_loss_v,
            'proto_align_r': proto_loss_r,
            'weight_reg': 0.0, # 保持 0.0 以避免报错
            'total': total_loss
        }
        
        f_r_fake, f_v_fake, _, _ = self.generate(f_v, f_r, labels_v, labels_r)
        return loss_dict, f_r_fake, f_v_fake, uncertainty_v, uncertainty_r
    
    def _compute_memory_loss(self, f_src, f_generated, labels, modality='v'):
        """Align generated features with retrieved memories"""
        retrieved_samples, weights = self.memory_bank.retrieve(
            f_src, labels, modality=modality, top_k=3
        )
        
        valid_mask = (weights > 0.1)
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=f_src.device)
        
        return F.mse_loss(f_generated[valid_mask], retrieved_samples[valid_mask])
    
    @torch.no_grad()
    def generate(self, f_v=None, f_r=None, labels_v=None, labels_r=None):
        f_r_fake, f_v_fake = None, None
        unc_v2r, unc_r2v = None, None
        
        guide_weight = self.proto_guidance.get_guidance_weight(self.current_epoch)
        
        if f_v is not None:
            proto_guide_r = None
            if self.cma is not None and labels_v is not None:
                try:
                    proto_guide_r = self.cma.get_prototypes('r')[labels_v]
                except ReferenceError:
                    pass
            
            feat_r, unc_feat = self.feature_diffusion_v2r.sample(f_v)
            sem_r, unc_sem = self.semantic_diffusion_v2r.sample(f_v, proto_guide_r, guide_weight)
            
            unc_v2r = (unc_feat + unc_sem) / 2.0
            
            memory_r = torch.zeros_like(feat_r)
            valid_memory_mask = torch.zeros(len(f_v), dtype=torch.bool, device=f_v.device)
            
            if self.use_memory and labels_v is not None:
                retrieved_r, weights_r = self.memory_bank.retrieve(f_v, labels_v, 'v', top_k=3)
                memory_r = retrieved_r
                valid_memory_mask = (weights_r > 0.1)
            
            f_r_fake = self.fusion(torch.cat([feat_r, sem_r], dim=1))
            
            if valid_memory_mask.any():
                f_r_fake[valid_memory_mask] = (0.9 * f_r_fake[valid_memory_mask] +
                                               0.1 * memory_r[valid_memory_mask])
        
        if f_r is not None:
            proto_guide_v = None
            if self.cma is not None and labels_r is not None:
                try:
                    proto_guide_v = self.cma.get_prototypes('v')[labels_r]
                except ReferenceError:
                    pass
            
            feat_v, unc_feat_v = self.feature_diffusion_r2v.sample(f_r)
            sem_v, unc_sem_v = self.semantic_diffusion_r2v.sample(f_r, proto_guide_v, guide_weight)
            
            unc_r2v = (unc_feat_v + unc_sem_v) / 2.0
            
            memory_v = torch.zeros_like(feat_v)
            valid_memory_mask = torch.zeros(len(f_r), dtype=torch.bool, device=f_r.device)
            
            if self.use_memory and labels_r is not None:
                retrieved_v, weights_v = self.memory_bank.retrieve(f_r, labels_r, 'r', top_k=3)
                memory_v = retrieved_v
                valid_memory_mask = (weights_v > 0.1)
            
            f_v_fake = self.fusion(torch.cat([feat_v, sem_v], dim=1))
            
            if valid_memory_mask.any():
                f_v_fake[valid_memory_mask] = (0.9 * f_v_fake[valid_memory_mask] +
                                               0.1 * memory_v[valid_memory_mask])
        
        return f_r_fake, f_v_fake, unc_v2r, unc_r2v
    
    def update_memory(self, f_v, f_r, labels_v, labels_r, f_r_fake, f_v_fake):
        """Update memory bank with high-quality transformations"""
        if not self.use_memory:
            return
        
        quality_v2r = F.cosine_similarity(f_r_fake, f_r, dim=1)
        quality_r2v = F.cosine_similarity(f_v_fake, f_v, dim=1)
        
        quality_v2r = F.relu(quality_v2r)
        quality_r2v = F.relu(quality_r2v)
        
        threshold = 0.0
        
        self.memory_bank.update(f_v, f_r_fake, labels_v, quality_v2r, 'v', threshold)
        self.memory_bank.update(f_r, f_v_fake, labels_r, quality_r2v, 'r', threshold)
    
    def get_detailed_diagnostics(self):
        """
        Comprehensive diagnostics for diffusion bridge.
        Returns formatted string with all module states.
        """
        lines = []
        lines.append("🔬 Hierarchical Diffusion Bridge Diagnostics:")
        lines.append(f"  ├─ Current Epoch: {self.current_epoch}")
        lines.append(f"  │")
        
        # Feature-level diffusion V2R
        lines.append(f"  ├─ Feature Diffusion (V→R) [{self.feature_diffusion_v2r.num_steps} steps]:")
        if self.feature_diffusion_v2r.last_loss_stats:
            s = self.feature_diffusion_v2r.last_loss_stats
            lines.append(f"  │  ├─ Loss: {s['loss_mean']:.4f} ± {s['loss_std']:.4f}")
            lines.append(f"  │  ├─ Noise MSE: {s['noise_mse']:.4f}")
            lines.append(f"  │  ├─ Log-Var: {s['log_var_mean']:.3f} ± {s['log_var_std']:.3f} | Range: [{s['log_var_min']:.2f}, {s['log_var_max']:.2f}]")
            lines.append(f"  │  └─ Uncertainty: {s['uncertainty_mean']:.4f} ± {s['uncertainty_std']:.4f}")
        
        lines.append(f"  │")
        
        # Semantic-level diffusion V2R
        lines.append(f"  ├─ Semantic Diffusion (V→R) [{self.semantic_diffusion_v2r.num_steps} steps]:")
        if self.semantic_diffusion_v2r.last_loss_stats:
            s = self.semantic_diffusion_v2r.last_loss_stats
            lines.append(f"  │  ├─ Loss: {s['loss_mean']:.4f} ± {s['loss_std']:.4f}")
            lines.append(f"  │  ├─ Log-Var: {s['log_var_mean']:.3f} ± {s['log_var_std']:.3f}")
            lines.append(f"  │  └─ Uncertainty: {s['uncertainty_mean']:.4f} ± {s['uncertainty_std']:.4f}")
        
        lines.append(f"  │")
        
        # Denoiser weight diagnostics
        denoiser_stats = self.feature_diffusion_v2r.denoiser.get_weight_diagnostics()
        lines.append(f"  ├─ Denoiser Network Weights:")
        lines.append(f"  │  ├─ Output Layer Norm: {denoiser_stats['output_weight_norm']:.2f}")
        lines.append(f"  │  ├─ Encoder Avg Norm: {denoiser_stats['encoder_avg_norm']:.2f} | Max: {denoiser_stats['encoder_max_norm']:.2f}")
        lines.append(f"  │  └─ Decoder Avg Norm: {denoiser_stats['decoder_avg_norm']:.2f} | Max: {denoiser_stats['decoder_max_norm']:.2f}")
        
        lines.append(f"  │")
        
        # Fusion layer
        fusion_weight_norm = torch.norm(self.fusion[0].weight).item()
        lines.append(f"  └─ Fusion Layer Weight Norm: {fusion_weight_norm:.2f}")
        
        return '\n'.join(lines)


# Aliases for backward compatibility
BidirectionalDiffusionBridge = HierarchicalDiffusionBridge
FeatureDiffusion = HierarchicalDiffusionBridge