"""
ISG-DM v3 - Optimized for BNNeck
[修改] 移除了 id_fc 层的 BatchNorm，以便在主模型中实现 BNNeck 策略
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ISG_DM(nn.Module):
    def __init__(self, input_dim=2048, id_dim=256, mod_dim=256):
        super(ISG_DM, self).__init__()
        self.input_dim = input_dim
        self.id_dim = id_dim
        self.mod_dim = mod_dim
        
        # A. 模态/风格特征提取器
        self.style_mlp = nn.Sequential(
            nn.Linear(input_dim * 2, input_dim),
            nn.BatchNorm1d(input_dim),
            nn.ReLU(inplace=True),
            nn.Linear(input_dim, mod_dim),
            nn.BatchNorm1d(mod_dim)
        )
        
        # B. 身份/内容特征提取器
        # B.1 SE-Gate
        reduction_dim = input_dim // 16
        self.se_gate = nn.Sequential(
            nn.Linear(input_dim, reduction_dim),
            nn.ReLU(inplace=True),
            nn.Linear(reduction_dim, input_dim),
            nn.Sigmoid()
        )
        
        # B.2 身份特征映射
        # [关键修改] 移除最后的 BatchNorm1d
        # Raw features (No BN) -> Triplet Loss
        # Features -> BN -> FC -> ID Loss (在外部实现)
        self.id_fc = nn.Sequential(
            nn.Linear(input_dim, id_dim)
            # Removed BN here
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, part_feature):
        B, C, H, W = part_feature.size()
        eps = 1e-5
        
        # 1. Style Extraction
        x_var, x_mean = torch.var_mean(part_feature, dim=(2, 3), keepdim=True, unbiased=False)
        x_std = torch.sqrt(x_var + eps)
        
        mean_flat = x_mean.view(B, C)
        std_flat = x_std.view(B, C)
        style_input = torch.cat([mean_flat, std_flat], dim=1)
        f_mod = self.style_mlp(style_input)
        
        # 2. Content Extraction
        f_norm = (part_feature - x_mean) / x_std
        f_norm_gap = F.adaptive_avg_pool2d(f_norm, 1).view(B, C)
        w = self.se_gate(f_norm_gap)
        
        # Residual correction
        f_id_raw = part_feature + (f_norm * w.view(B, C, 1, 1))
        f_id_vec = F.adaptive_avg_pool2d(f_id_raw, 1).view(B, C)
        f_id = self.id_fc(f_id_vec)
        
        return f_id, f_mod


class MultiPartISG_DM(nn.Module):
    def __init__(self, num_parts=6, input_dim=2048, id_dim=256, mod_dim=256):
        super(MultiPartISG_DM, self).__init__()
        self.num_parts = num_parts
        self.isg_modules = nn.ModuleList([
            ISG_DM(input_dim, id_dim, mod_dim) for _ in range(num_parts)
        ])
    
    def forward(self, part_features):
        id_features = []
        mod_features = []
        for i, part in enumerate(part_features):
            f_id, f_mod = self.isg_modules[i](part)
            id_features.append(f_id)
            mod_features.append(f_mod)
        return id_features, mod_features