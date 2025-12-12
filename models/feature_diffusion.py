"""
Feature-level Diffusion Bridge for VI-ReID
特征级扩散桥：从可见光特征生成红外特征（无配对训练）
"""
import torch
import torch.nn as nn
import numpy as np
from .diffusion_denoiser import DenoiseMLP
from .diffusion_loss import DiffusionLoss


class FeatureDiffusionBridge(nn.Module):
    def __init__(self, feat_dim=2048, hidden_dim=1024, num_steps=10, 
                 beta_schedule='linear', device='cuda'):
        """
        Args:
            feat_dim: 特征维度（默认 ResNet50 输出 2048）
            hidden_dim: 去噪网络隐藏层维度
            num_steps: 扩散步数 T
            beta_schedule: 噪声调度策略 ('linear' or 'cosine')
            device: 计算设备
        """
        super().__init__()
        self.feat_dim = feat_dim
        self.num_steps = num_steps
        self.device = device
        
        # 去噪网络：预测噪声 ϵ̂
        self.denoiser = DenoiseMLP(feat_dim, hidden_dim, num_steps)
        
        # 损失计算模块
        self.loss_fn = DiffusionLoss()
        
        # 注册扩散过程的超参数（不可训练）
        self.register_buffer('betas', self._get_beta_schedule(beta_schedule, num_steps))
        self.register_buffer('alphas', 1.0 - self.betas)
        self.register_buffer('alphas_cumprod', torch.cumprod(self.alphas, dim=0))
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(self.alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', 
                            torch.sqrt(1.0 - self.alphas_cumprod))
        
        # 用于反向采样
        self.register_buffer('sqrt_recip_alphas', torch.sqrt(1.0 / self.alphas))
        self.register_buffer('posterior_variance', 
                            self.betas * (1.0 - torch.cat([self.alphas_cumprod[:1], 
                            self.alphas_cumprod[:-1]])) / (1.0 - self.alphas_cumprod))
    
    def _get_beta_schedule(self, schedule_type, num_steps):
        """生成噪声调度表"""
        if schedule_type == 'linear':
            # 线性调度：β_1=0.0001 到 β_T=0.02
            return torch.linspace(1e-4, 0.02, num_steps)
        elif schedule_type == 'cosine':
            # 余弦调度（适合少步数）
            s = 0.008
            steps = num_steps + 1
            x = torch.linspace(0, num_steps, steps)
            alphas_cumprod = torch.cos(((x / num_steps) + s) / (1 + s) * np.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            return torch.clip(betas, 0.0001, 0.9999)
        else:
            raise ValueError(f"Unknown schedule: {schedule_type}")
    
    def q_sample(self, x_start, t, noise=None):
        """
        前向扩散：q(z_t | z_0) = N(√ᾱ_t·z_0, (1-ᾱ_t)I)
        Args:
            x_start: 初始特征 f_r（仅用于训练时模拟，推理时无需真实 f_r）
            t: 时间步 [batch_size]
            noise: 标准高斯噪声（可选）
        Returns:
            加噪后的特征 z_t
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alpha_t = self.sqrt_alphas_cumprod[t].view(-1, 1)
        sqrt_one_minus_alpha_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1)
        
        return sqrt_alpha_t * x_start + sqrt_one_minus_alpha_t * noise
    
    def p_sample(self, z_t, f_cond, t):
        """
        反向去噪单步：p(z_{t-1} | z_t, f_cond)
        Args:
            z_t: 当前噪声特征
            f_cond: 条件特征（可见光 f_v）
            t: 当前时间步
        Returns:
            z_{t-1}: 去噪一步后的特征
        """
        # 预测噪声
        pred_noise = self.denoiser(z_t, f_cond, t)
        
        # 计算均值：μ = 1/√α_t · (z_t - β_t/√(1-ᾱ_t)·ϵ̂)
        beta_t = self.betas[t].view(-1, 1)
        sqrt_recip_alpha_t = self.sqrt_recip_alphas[t].view(-1, 1)
        sqrt_one_minus_alpha_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1)
        
        mean = sqrt_recip_alpha_t * (z_t - beta_t / sqrt_one_minus_alpha_t * pred_noise)
        
        # 添加方差（t>0 时）
        if t[0] > 0:
            variance = self.posterior_variance[t].view(-1, 1)
            noise = torch.randn_like(z_t)
            return mean + torch.sqrt(variance) * noise
        else:
            return mean
    
    @torch.no_grad()
    def generate(self, f_v, num_samples=None):
        """
        推理模式：从可见光特征生成红外特征
        Args:
            f_v: 可见光特征 [batch_size, feat_dim]
            num_samples: 生成样本数（默认与 batch_size 相同）
        Returns:
            f_r_fake: 生成的红外特征
        """
        batch_size = f_v.shape[0]
        if num_samples is None:
            num_samples = batch_size
        
        # 从纯噪声开始 z_T ~ N(0, I)
        z_t = torch.randn(num_samples, self.feat_dim).to(self.device)
        
        # 逐步去噪 T → 0
        for t in reversed(range(self.num_steps)):
            t_batch = torch.full((num_samples,), t, dtype=torch.long).to(self.device)
            z_t = self.p_sample(z_t, f_v, t_batch)
        
        return z_t  # z_0 即为生成的 f_r_fake
    
    def forward(self, f_v, W_r=None, mode='train'):
        """
        主函数：支持训练和推理两种模式
        Args:
            f_v: 可见光特征 [batch_size, feat_dim]
            W_r: 红外分类器权重（用于置信度引导）
            mode: 'train' 或 'inference'
        Returns:
            train 模式: (loss_dict, f_r_fake)
            inference 模式: f_r_fake
        """
        if mode == 'train':
            return self.compute_loss(f_v, W_r)
        elif mode == 'inference':
            return self.generate(f_v)
        else:
            raise ValueError(f"Invalid mode: {mode}")
    
    def compute_loss(self, f_v, W_r=None):
        """
        训练损失计算（无配对训练）
        策略：对随机采样的 z_0 加噪到 z_t，然后预测噪声
        Args:
            f_v: 可见光特征
            W_r: 红外分类器（可选，用于置信度引导）
        Returns:
            loss_dict: {'mse_loss': ..., 'conf_loss': ..., 'total_loss': ...}
            f_r_fake: 生成的红外特征（用于后续训练）
        """
        batch_size = f_v.shape[0]
        
        # 1. 采样初始点 z_0（从标准正态分布，代表目标红外特征空间）
        z_0 = torch.randn(batch_size, self.feat_dim).to(self.device)
        
        # 2. 随机采样时间步 t ~ Uniform(0, T-1)
        t = torch.randint(0, self.num_steps, (batch_size,)).to(self.device)
        
        # 3. 前向加噪 q(z_t | z_0)
        noise = torch.randn_like(z_0)
        z_t = self.q_sample(z_0, t, noise)
        
        # 4. 预测噪声
        pred_noise = self.denoiser(z_t, f_v, t)
        
        # 5. 计算损失
        loss_dict = self.loss_fn(pred_noise, noise, z_0, f_v, W_r)
        
        # 6. 生成一个伪红外特征用于后续训练（采样少量步数以节省时间）
        with torch.no_grad():
            f_r_fake = self.generate(f_v)
        
        return loss_dict, f_r_fake


# 辅助函数：提取指定时间步的参数
def extract(a, t, x_shape):
    """从 a 中提取 t 对应的值，并广播到 x_shape"""
    batch_size = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))
