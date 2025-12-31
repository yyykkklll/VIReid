import torch
import torch.nn as nn
import math
from models.cross_attention import CrossAttentionBlock


class ImprovedDiffusionUNet(nn.Module):
    def __init__(self, feat_dim=2048, hidden_dim=1024, num_steps=10,
                 num_heads=8, num_res_blocks=3, dropout=0.1):
        super().__init__()
        self.feat_dim = feat_dim
        self.hidden_dim = hidden_dim
        self.num_res_blocks = num_res_blocks
        
        # Sinusoidal time embedding
        self.time_embed = SinusoidalTimeEmbedding(num_steps, hidden_dim // 4)
        
        # Condition projection (source modality feature)
        self.cond_proj = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU()
        )
        
        # Input projection (noisy target feature)
        self.input_proj = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU()
        )
        
        # Encoder with residual blocks
        self.encoder = nn.ModuleList([
            ResidualBlock(hidden_dim, hidden_dim // 4, dropout)
            for _ in range(num_res_blocks)
        ])
        
        # Cross-modal attention
        self.cross_attention = CrossAttentionBlock(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            ffn_ratio=4,
            activation='gelu'
        )
        
        # Decoder with residual blocks
        self.decoder = nn.ModuleList([
            ResidualBlock(hidden_dim, hidden_dim // 4, dropout)
            for _ in range(num_res_blocks)
        ])
        
        # Output projection: [B, D] -> [B, 2*D]
        # Split: [0:D] = Predicted Noise, [D:2D] = Predicted Log-Variance
        self.output_proj = nn.Linear(hidden_dim, feat_dim * 2)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Xavier initialization for stability"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
    
    def forward(self, x_noisy, x_cond, t):
        # Time embedding
        t_emb = self.time_embed(t)  # [B, hidden_dim//4]
        
        # Condition encoding
        cond = self.cond_proj(x_cond)  # [B, hidden_dim]
        
        # Input encoding
        h = self.input_proj(x_noisy)  # [B, hidden_dim]
        
        # Encoder with skip connections
        skips = []
        for block in self.encoder:
            h = block(h, t_emb)
            skips.append(h)
        
        # Cross-modal attention (fuse condition)
        h, attn_weights = self.cross_attention(h, cond)
        
        # Decoder with skip connections (U-Net style)
        for block, skip in zip(self.decoder, reversed(skips)):
            h = h + skip  # Skip connection
            h = block(h, t_emb)
        
        # Predict noise and uncertainty (concatenated)
        out = self.output_proj(h)
        
        return out
    
    def get_weight_diagnostics(self):
        """
        Return weight norm statistics for debugging.
        """
        stats = {}
        
        # Output layer (critical for log_var prediction)
        stats['output_weight_norm'] = torch.norm(self.output_proj.weight).item()
        stats['output_bias_norm'] = torch.norm(self.output_proj.bias).item() if self.output_proj.bias is not None else 0.0
        
        # Encoder norms
        encoder_norms = [torch.norm(block.mlp[1].weight).item() for block in self.encoder]
        stats['encoder_avg_norm'] = sum(encoder_norms) / len(encoder_norms)
        stats['encoder_max_norm'] = max(encoder_norms)
        
        # Decoder norms
        decoder_norms = [torch.norm(block.mlp[1].weight).item() for block in self.decoder]
        stats['decoder_avg_norm'] = sum(decoder_norms) / len(decoder_norms)
        stats['decoder_max_norm'] = max(decoder_norms)
        
        return stats
    
    def compute_weight_reg(self):
        """计算权重正则化损失(无截断)"""
        total_norm = 0.0
        for param in self.parameters():
            if param.requires_grad:
                param_norm = torch.norm(param.data, p=2)
                total_norm += param_norm ** 2
        total_norm = torch.sqrt(total_norm)
        reg_loss = 0.1 * total_norm
        return reg_loss, total_norm

class ResidualBlock(nn.Module):
    """Residual block with FiLM time modulation"""
    def __init__(self, hidden_dim, time_dim, dropout=0.1):
        super().__init__()
        
        # Main MLP
        self.mlp = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout)
        )
        
        # Time modulation (FiLM: scale & shift)
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, hidden_dim * 2)
        )
        
        self.norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, x, t_emb):
        # Time-conditional modulation (FiLM)
        scale, shift = self.time_mlp(t_emb).chunk(2, dim=1)
        
        # MLP with modulation
        h = self.mlp(x)
        h = h * (1 + scale) + shift  # FiLM modulation
        
        # Residual connection
        return self.norm(x + h)


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal positional encoding for timesteps"""
    def __init__(self, num_steps, embed_dim):
        super().__init__()
        position = torch.arange(num_steps).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() *
                            (-math.log(10000.0) / embed_dim))
        
        pe = torch.zeros(num_steps, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
    
    def forward(self, t):
        if t.dtype != torch.long:
            t = t.long()
        if t.dim() > 1:
            t = t.squeeze(-1)
        return self.pe[t]

