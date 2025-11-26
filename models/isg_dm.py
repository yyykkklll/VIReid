"""
ISG-DM v2 - Enhanced Instance Statistics Guided Disentanglement Module
改进版实例统计引导解耦模块：增强特征判别性
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ISG_DM(nn.Module):
    """改进版ISG-DM：多层特征提取 + 轻量级注意力"""
    
    def __init__(self, input_dim=2048, id_dim=256, mod_dim=256):
        super(ISG_DM, self).__init__()
        self.input_dim = input_dim
        self.id_dim = id_dim
        self.mod_dim = mod_dim
        
        # [改进1] 多层身份特征提取（渐进式降维）
        self.identity_fc = nn.Sequential(
            # 第一层：2048 -> 1024
            nn.Linear(input_dim, input_dim // 2),
            nn.BatchNorm1d(input_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),  # 降低dropout
            
            # 第二层：1024 -> 512
            nn.Linear(input_dim // 2, input_dim // 4),
            nn.BatchNorm1d(input_dim // 4),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            
            # 第三层：512 -> 256
            nn.Linear(input_dim // 4, id_dim),
            nn.BatchNorm1d(id_dim)
        )
        
        # [改进2] 轻量级通道注意力（类似SENet但更简单）
        self.channel_attention = nn.Sequential(
            nn.Linear(input_dim, input_dim // 16),
            nn.ReLU(inplace=True),
            nn.Linear(input_dim // 16, input_dim),
            nn.Sigmoid()
        )
        
        # 模态特征提取（保持简单）
        self.modality_fc = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.BatchNorm1d(input_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(input_dim // 2, mod_dim),
            nn.BatchNorm1d(mod_dim)
        )
        
        # 显式权重初始化
        self._init_weights()
    
    def _init_weights(self):
        """Kaiming初始化，确保训练稳定"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, part_feature):
        """
        Args:
            part_feature: [B, C, H, W]
        Returns:
            f_id: 身份特征 [B, id_dim]
            f_mod: 模态特征 [B, mod_dim]
        """
        B, C, H, W = part_feature.size()
        
        # 全局平均池化
        pooled = F.adaptive_avg_pool2d(part_feature, 1).view(B, C)  # [B, C]
        
        # [改进] 通道注意力加权
        att_weights = self.channel_attention(pooled)  # [B, C]
        pooled_weighted = pooled * att_weights
        
        # 提取身份特征和模态特征
        f_id = self.identity_fc(pooled_weighted)
        f_mod = self.modality_fc(pooled)
        
        return f_id, f_mod


class MultiPartISG_DM(nn.Module):
    """多部件ISG-DM模块"""
    
    def __init__(self, num_parts=6, input_dim=2048, id_dim=256, mod_dim=256):
        super(MultiPartISG_DM, self).__init__()
        self.num_parts = num_parts
        
        # 为每个部件创建独立的ISG-DM
        self.isg_modules = nn.ModuleList([
            ISG_DM(input_dim, id_dim, mod_dim) for _ in range(num_parts)
        ])
    
    def forward(self, part_features):
        """
        Args:
            part_features: List of K个部件特征 [B, C, H_k, W]
        Returns:
            id_features: List of K个身份特征 [B, id_dim]
            mod_features: List of K个模态特征 [B, mod_dim]
        """
        id_features = []
        mod_features = []
        
        for i, part in enumerate(part_features):
            f_id, f_mod = self.isg_modules[i](part)
            id_features.append(f_id)
            mod_features.append(f_mod)
        
        return id_features, mod_features
