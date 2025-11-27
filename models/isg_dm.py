"""
ISG-DM v2 - Enhanced Instance Statistics Guided Disentanglement Module
改进版实例统计引导解耦模块：基于物理统计量的解耦
[修复版] 添加残差连接，防止特征在小分辨率下被过度归一化破坏
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ISG_DM(nn.Module):
    """
    改进版ISG-DM (带有残差修正)
    1. 模态/风格提取：基于Instance Norm统计量 (Mean, Std)
    2. 身份/内容提取：Instance Norm去风格 + SE-Gate注意力补偿 + 原始特征残差
    """
    
    def __init__(self, input_dim=2048, id_dim=256, mod_dim=256):
        super(ISG_DM, self).__init__()
        self.input_dim = input_dim
        self.id_dim = id_dim
        self.mod_dim = mod_dim
        
        # A. 模态/风格特征提取器
        # 输入：[Mean, Std] -> 2 * input_dim
        self.style_mlp = nn.Sequential(
            nn.Linear(input_dim * 2, input_dim),
            nn.BatchNorm1d(input_dim),
            nn.ReLU(inplace=True),
            nn.Linear(input_dim, mod_dim),
            nn.BatchNorm1d(mod_dim)
        )
        
        # B. 身份/内容特征提取器
        # B.1 SE-Gate 注意力 (输入是GAP后的特征)
        reduction_dim = input_dim // 16
        self.se_gate = nn.Sequential(
            nn.Linear(input_dim, reduction_dim),
            nn.ReLU(inplace=True),
            nn.Linear(reduction_dim, input_dim),
            nn.Sigmoid()
        )
        
        # B.2 身份特征映射 (降维)
        self.id_fc = nn.Sequential(
            nn.Linear(input_dim, id_dim),
            nn.BatchNorm1d(id_dim)
        )
        
        # 显式权重初始化
        self._init_weights()
    
    def _init_weights(self):
        """Kaiming初始化"""
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
        eps = 1e-5
        
        # 1. 提取统计量 (Modality/Style Extraction)
        # 计算每个实例每个通道的均值和方差 (在H, W维度上)
        # x_var, x_mean: [B, C, 1, 1]
        x_var, x_mean = torch.var_mean(part_feature, dim=(2, 3), keepdim=True, unbiased=False)
        x_std = torch.sqrt(x_var + eps)
        
        # 构建模态特征输入：拼接 Mean 和 Std
        mean_flat = x_mean.view(B, C)
        std_flat = x_std.view(B, C)
        style_input = torch.cat([mean_flat, std_flat], dim=1) # [B, 2C]
        
        # 生成模态特征
        f_mod = self.style_mlp(style_input)
        
        # 2. 提取身份特征 (Identity/Content Extraction)
        # 2.1 Instance Normalization 去除风格
        # 注意：在极小分辨率(如4x8)下，单纯的IN会破坏纹理结构
        f_norm = (part_feature - x_mean) / x_std # [B, C, H, W]
        
        # 2.2 SE-Gate 通道注意力
        # 对归一化后的特征做GAP，作为SE模块输入
        f_norm_gap = F.adaptive_avg_pool2d(f_norm, 1).view(B, C)
        w = self.se_gate(f_norm_gap) # [B, C]
        
        # 2.3 [关键修复] 残差连接 + 注意力加权
        # 原始方案：f_id_raw = f_norm * w
        # 修复方案：将去风格化的特征作为“修正项”加回到原始特征上，防止信息丢失
        # 或者理解为：我们希望 f_norm 指导网络关注哪些通道(w)，但保留原始 spatial info
        f_id_raw = part_feature + (f_norm * w.view(B, C, 1, 1))
        
        # 2.4 最终池化与降维
        f_id_vec = F.adaptive_avg_pool2d(f_id_raw, 1).view(B, C)
        f_id = self.id_fc(f_id_vec)
        
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