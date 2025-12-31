import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class CrossAttentionBlock(nn.Module):
    def __init__(
        self, 
        hidden_dim, 
        num_heads=4, 
        dropout=0.1,
        ffn_ratio=4,
        use_bias=True,
        activation='gelu'
    ):
        super().__init__()
        assert hidden_dim % num_heads == 0, f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})"
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Cross-Attention (PyTorch Native)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            bias=use_bias,
            batch_first=True  # [B, L, D] format
        )
        
        # Layer Normalization
        self.norm1 = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_dim, eps=1e-6)
        
        # Feed-Forward Network
        self.ffn = self._build_ffn(hidden_dim, ffn_ratio, dropout, activation)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._reset_parameters()
    
    def _build_ffn(self, hidden_dim, ffn_ratio, dropout, activation):
        """构建 Feed-Forward Network"""
        ffn_hidden = hidden_dim * ffn_ratio
        
        if activation == 'gelu':
            act_fn = nn.GELU()
        elif activation == 'relu':
            act_fn = nn.ReLU(inplace=True)
        elif activation == 'silu':
            act_fn = nn.SiLU(inplace=True)
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        return nn.Sequential(
            nn.Linear(hidden_dim, ffn_hidden),
            act_fn,
            nn.Dropout(dropout),
            nn.Linear(ffn_hidden, hidden_dim),
            nn.Dropout(dropout)
        )
    
    def _reset_parameters(self):
        """Xavier Uniform 初始化（增强训练稳定性）"""
        for module in self.ffn:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
    
    def forward(self, query, key_value, attn_mask=None):
        # 1. 处理输入维度
        is_1d = (query.dim() == 2)  # [B, D]
        if is_1d:
            query = query.unsqueeze(1)  # [B, 1, D]
            key_value = key_value.unsqueeze(1)
        
        B, L, D = query.shape
        
        # 2. Cross-Attention
        # Q from noisy features, K/V from condition
        attn_out, attn_weights = self.cross_attn(
            query=query,
            key=key_value,
            value=key_value,
            attn_mask=attn_mask,
            need_weights=True,
            average_attn_weights=False  # 返回每个头的权重
        )
        
        # 3. Residual Connection + LayerNorm
        h = self.norm1(query + self.dropout(attn_out))
        
        # 4. Feed-Forward Network
        ffn_out = self.ffn(h)
        output = self.norm2(h + ffn_out)
        
        # 5. 恢复原始维度
        if is_1d:
            output = output.squeeze(1)  # [B, D]
        
        return output, attn_weights
    
# Alias for backward compatibility
CrossModalAttention = CrossAttentionBlock
