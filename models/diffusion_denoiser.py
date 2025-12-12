"""
Denoising MLP Network for Feature Diffusion
去噪网络：输入 [z_t, f_cond, t] → 输出预测噪声 ϵ̂
"""
import torch
import torch.nn as nn
import math


class DenoiseMLP(nn.Module):
    def __init__(self, feat_dim=2048, hidden_dim=1024, num_steps=10, 
                 num_layers=4, dropout=0.1):
        """
        Args:
            feat_dim: 特征维度
            hidden_dim: 隐藏层维度
            num_steps: 总扩散步数（用于时间嵌入）
            num_layers: MLP 层数
            dropout: Dropout 概率
        """
        super().__init__()
        self.feat_dim = feat_dim
        
        # 时间步嵌入（Sinusoidal Position Embedding）
        self.time_embed = TimeEmbedding(num_steps, hidden_dim // 4)
        
        # 输入投影：[z_t (2048) + f_cond (2048) + t_emb (256)] → hidden_dim
        input_dim = feat_dim * 2 + hidden_dim // 4
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU()
        )
        
        # 中间 MLP 层
        self.layers = nn.ModuleList([
            MLPBlock(hidden_dim, dropout) for _ in range(num_layers)
        ])
        
        # 输出投影：hidden_dim → feat_dim（预测噪声）
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, feat_dim)
        )
    
    def forward(self, z_t, f_cond, t):
        """
        Args:
            z_t: 当前噪声特征 [batch_size, feat_dim]
            f_cond: 条件特征（可见光） [batch_size, feat_dim]
            t: 时间步 [batch_size]
        Returns:
            pred_noise: 预测的噪声 [batch_size, feat_dim]
        """
        # 时间步嵌入
        t_emb = self.time_embed(t)  # [batch_size, hidden_dim//4]
        
        # 拼接输入
        x = torch.cat([z_t, f_cond, t_emb], dim=1)  # [batch_size, input_dim]
        
        # 输入投影
        x = self.input_proj(x)
        
        # 通过 MLP 层
        for layer in self.layers:
            x = layer(x)
        
        # 输出噪声预测
        pred_noise = self.output_proj(x)
        
        return pred_noise


class MLPBlock(nn.Module):
    """单个 MLP Block：Linear + LayerNorm + SiLU + Dropout + Residual"""
    def __init__(self, hidden_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout)
        )
        self.norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, x):
        return self.norm(x + self.net(x))  # 残差连接


class TimeEmbedding(nn.Module):
    """
    Sinusoidal Time Step Embedding（类似 Transformer 位置编码）
    将离散时间步 t ∈ [0, T-1] 编码为连续向量
    """
    def __init__(self, num_steps, embed_dim):
        super().__init__()
        self.num_steps = num_steps
        self.embed_dim = embed_dim
        
        # 预计算位置编码
        position = torch.arange(num_steps).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * 
                            (-math.log(10000.0) / embed_dim))
        
        pe = torch.zeros(num_steps, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
    
    def forward(self, t):
        """
        Args:
            t: 时间步 [batch_size] (整数)
        Returns:
            嵌入向量 [batch_size, embed_dim]
        """
        return self.pe[t]
