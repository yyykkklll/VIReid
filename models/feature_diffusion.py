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
    
    def _init_diffusion_params(self):
        """Cosine noise schedule"""
        betas = self._cosine_beta_schedule(self.num_steps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))

    def _cosine_beta_schedule(self, timesteps, s=0.008):
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0, 0.999)

    def compute_loss(self, x_start, condition):
        """
        x_start: Target features to reconstruct (e.g., IR features)
        condition: Source features to condition on (e.g., RGB features)
        """
        B = x_start.size(0)
        t = torch.randint(0, self.num_steps, (B,), device=self.device).long()
        
        noise = torch.randn_like(x_start)
        x_noisy = (
            self.sqrt_alphas_cumprod[t].view(-1, 1) * x_start +
            self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1) * noise
        )
        
        # forward(self, x_noisy, x_cond, t)
        model_output = self.denoiser(x_noisy, condition, t)
        
        # Heteroscedastic Loss (Predict noise + uncertainty)
        pred_noise, pred_log_var = torch.chunk(model_output, 2, dim=1)
        
        # Loss calculation (NLL style)
        mse = (noise - pred_noise) ** 2
        loss = torch.mean(torch.exp(-pred_log_var) * mse + pred_log_var)
        
        # Stats
        with torch.no_grad():
            self.last_loss_stats = {
                'loss_mean': loss.item(),
                'loss_std': loss.std().item() if B > 1 else 0.0,
                'log_var_mean': pred_log_var.mean().item(),
                'uncertainty_mean': torch.exp(pred_log_var).mean().item(),
            }
            
        return loss, pred_noise, pred_log_var

    @torch.no_grad()
    def sample(self, shape, condition):
        B = shape[0]
        img = torch.randn(shape, device=self.device)
        
        for i in reversed(range(0, self.num_steps)):
            t = torch.full((B,), i, device=self.device, dtype=torch.long)
            
            model_output = self.denoiser(img, condition, t)
            
            pred_noise, pred_log_var = torch.chunk(model_output, 2, dim=1)
            
            # Reparameterization trick inverse
            alpha = self.alphas[i]
            alpha_cumprod = self.alphas_cumprod[i]
            beta = self.betas[i]
            
            if i > 0:
                noise = torch.randn_like(img)
            else:
                noise = torch.zeros_like(img)
                
            pred_mean = (1 / torch.sqrt(alpha)) * (
                img - (beta / torch.sqrt(1 - alpha_cumprod)) * pred_noise
            )
            
            std = torch.exp(0.5 * pred_log_var)
            img = pred_mean + std * noise
            
        return img, pred_log_var


# ==================== Hierarchical Diffusion Bridge ====================
class HierarchicalDiffusionBridge(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Config
        self.feat_dim = 2048
        self.hidden_dim = getattr(args, 'diffusion_hidden', 1024)
        self.feature_steps = getattr(args, 'feature_diffusion_steps', 5)
        self.semantic_steps = getattr(args, 'semantic_diffusion_steps', 10)
        
        # 1. Feature-level Diffusion
        self.feature_diffusion_v2r = DiffusionModule(
            feat_dim=self.feat_dim, hidden_dim=self.hidden_dim, 
            num_steps=self.feature_steps, device=self.device
        )
        self.feature_diffusion_r2v = DiffusionModule(
            feat_dim=self.feat_dim, hidden_dim=self.hidden_dim, 
            num_steps=self.feature_steps, device=self.device
        )
        
        # 2. Semantic-level Diffusion
        self.semantic_diffusion_v2r = DiffusionModule(
            feat_dim=self.feat_dim, hidden_dim=self.hidden_dim, 
            num_steps=self.semantic_steps, device=self.device
        )
        self.semantic_diffusion_r2v = DiffusionModule(
            feat_dim=self.feat_dim, hidden_dim=self.hidden_dim, 
            num_steps=self.semantic_steps, device=self.device
        )
        
        # Memory Bank
        self.memory_bank = MemoryBank(
            num_classes=args.num_classes, 
            feat_dim=self.feat_dim,
            slots_per_class=getattr(args, 'memory_size_per_class', 5)
        )
        
        # Prototype Guidance Link
        self.cma = None 
        
        self.current_epoch = 0

    def set_epoch(self, epoch):
        self.current_epoch = epoch

    def set_cma(self, cma_module):
        self.cma = weakref.proxy(cma_module)

    def forward(self, f_v, f_r, labels_v, labels_r, mode='train'):
        if mode == 'train':
            return self.compute_loss(f_v, f_r, labels_v, labels_r)
        else:
            return self.generate(f_v=f_v, f_r=f_r, labels_v=labels_v, labels_r=labels_r)

    def compute_loss(self, f_v, f_r, labels_v, labels_r):
        """
        Compute total diffusion loss (Cross-Modal Conditional)
        V->R: Target=f_r, Condition=f_v
        R->V: Target=f_v, Condition=f_r
        """
        
        # Feature Diffusion (Texture/Low-level)
        l_v2r, _, unc_v2r = self.feature_diffusion_v2r.compute_loss(x_start=f_r, condition=f_v)
        l_r2v, _, unc_r2v = self.feature_diffusion_r2v.compute_loss(x_start=f_v, condition=f_r)
        
        # Semantic Diffusion (High-level)
        l_sem_v2r, _, _ = self.semantic_diffusion_v2r.compute_loss(x_start=f_r, condition=f_v)
        l_sem_r2v, _, _ = self.semantic_diffusion_r2v.compute_loss(x_start=f_v, condition=f_r)
        
        total_loss = l_v2r + l_r2v + 0.1 * (l_sem_v2r + l_sem_r2v)
        
        # Generate samples for downstream use (Consistency/Augmentation)
        with torch.no_grad():
             # Generate V from R
             f_v_fake, _ = self.feature_diffusion_r2v.sample(f_r.shape, condition=f_r)
             # Generate R from V
             f_r_fake, _ = self.feature_diffusion_v2r.sample(f_v.shape, condition=f_v)

        loss_dict = {
            'total': total_loss,
            'v2r': l_v2r.item(),
            'r2v': l_r2v.item()
        }
        
        # ✅ FIX: Return per-sample uncertainty (Shape: [B])
        # unc_v2r is log_var [B, D]. We take mean over D, then exp.
        # RAGM expects [B] tensor to unsqueeze(1).
        
        # Uncertainty of "Generating V" (from R2V model)
        uncertainty_v = torch.exp(unc_r2v).mean(dim=1) 
        
        # Uncertainty of "Generating R" (from V2R model)
        uncertainty_r = torch.exp(unc_v2r).mean(dim=1)
        
        return loss_dict, f_r_fake, f_v_fake, uncertainty_v, uncertainty_r

    @torch.no_grad()
    def generate(self, f_v=None, f_r=None, labels_v=None, labels_r=None):
        """
        External generation call (e.g. for PRUD/CCPA)
        """
        cma_ref = getattr(self, 'cma', None)
        
        f_v_fake = None
        f_r_fake = None
        unc_v = None
        unc_r = None
        
        # ============================================
        # 1. V->R Generation (Need Condition: f_v)
        # ============================================
        cond_v = None
        if f_v is not None:
            cond_v = f_v
        elif labels_v is not None and cma_ref is not None:
            try:
                # [B, 2048]
                cond_v = cma_ref.get_prototypes('v')[labels_v]
            except:
                cond_v = None
        
        if cond_v is not None:
             f_r_fake, log_var_r = self.feature_diffusion_v2r.sample(cond_v.shape, condition=cond_v)
             # ✅ FIX: Per-sample uncertainty
             unc_r = torch.exp(log_var_r).mean(dim=1)

        # ============================================
        # 2. R->V Generation (Need Condition: f_r)
        # ============================================
        cond_r = None
        if f_r is not None:
            cond_r = f_r
        elif labels_r is not None and cma_ref is not None:
            try:
                cond_r = cma_ref.get_prototypes('r')[labels_r]
            except:
                cond_r = None

        if cond_r is not None:
             f_v_fake, log_var_v = self.feature_diffusion_r2v.sample(cond_r.shape, condition=cond_r)
             # ✅ FIX: Per-sample uncertainty
             unc_v = torch.exp(log_var_v).mean(dim=1)
            
        return f_r_fake, f_v_fake, unc_v, unc_r

    def get_detailed_diagnostics(self):
        lines = []
        lines.append(f"🔬 Hierarchical Diffusion Bridge Diagnostics:")
        lines.append(f"  ├─ Current Epoch: {self.current_epoch}")
        
        if self.feature_diffusion_v2r.last_loss_stats:
            s = self.feature_diffusion_v2r.last_loss_stats
            lines.append(f"  │  ├─ Loss: {s['loss_mean']:.4f} ± {s['loss_std']:.4f}")
            lines.append(f"  │  ├─ Uncertainty: {s['uncertainty_mean']:.4f}")

        # Memory Bank Stats
        if hasattr(self, 'memory_bank'):
             mb_stats = self.memory_bank.get_statistics()
             lines.append(f"  │")
             lines.append(f"  └─ Memory Bank: {mb_stats['occupied_classes_v']} classes occupied")

        return '\n'.join(lines)