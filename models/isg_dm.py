"""
ISG-DM - Instance Statistics Guided Disentanglement Module
实例统计引导解耦模块: 将特征分离为身份特征和模态特征
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ISG_DM(nn.Module):
    """
    实例统计引导解耦模块
    利用Instance Normalization的统计量提取模态特征
    利用归一化后的内容提取身份特征
    """
    
    def __init__(self, input_dim=2048, id_dim=256, mod_dim=256):
        """
        Args:
            input_dim: 输入特征维度 (来自PCB的每个部件)
            id_dim: 身份特征输出维度
            mod_dim: 模态特征输出维度
        """
        super(ISG_DM, self).__init__()
        self.input_dim = input_dim
        self.id_dim = id_dim
        self.mod_dim = mod_dim
        
        # ===== 模态特征提取 (Style/Modality) =====
        # 统计量 (均值+标准差) 的维度是 input_dim * 2
        self.style_mlp = nn.Sequential(
            nn.Linear(input_dim * 2, input_dim),
            nn.ReLU(inplace=True),
            nn.Linear(input_dim, mod_dim),
            nn.BatchNorm1d(mod_dim)
        )
        
        # ===== 身份特征提取 (Content/Identity) =====
        # SE-Gate 通道注意力
        self.se_reduce = nn.Linear(input_dim, input_dim // 16)
        self.se_expand = nn.Linear(input_dim // 16, input_dim)
        
        # 身份特征降维
        self.identity_fc = nn.Sequential(
            nn.Linear(input_dim, id_dim),
            nn.BatchNorm1d(id_dim),
            nn.ReLU(inplace=True)
        )
        
        # Instance Normalization的epsilon
        self.eps = 1e-5
    
    def forward(self, part_feature):
        """
        Args:
            part_feature: 单个部件特征 [B, C, H, W]
        Returns:
            f_id: 身份特征 [B, id_dim]
            f_mod: 模态特征 [B, mod_dim]
        """
        B, C, H, W = part_feature.size()
        
        # ===== Step 1: 提取统计量 (模态特征) =====
        f_mod = self._extract_modality_feature(part_feature)
        
        # ===== Step 2: Instance Normalization (去除风格) =====
        f_norm = self._instance_normalize(part_feature)
        
        # ===== Step 3: SE-Gate通道注意力 (强化身份信息) =====
        f_id_raw = self._apply_se_gate(f_norm)
        
        # ===== Step 4: 降维输出身份特征 =====
        # 先做全局平均池化
        f_id_pooled = F.adaptive_avg_pool2d(f_id_raw, 1).view(B, C)
        f_id = self.identity_fc(f_id_pooled)
        
        return f_id, f_mod
    
    def _extract_modality_feature(self, x):
        """
        提取模态特征: 利用IN的统计量 (均值和标准差)
        Args:
            x: [B, C, H, W]
        Returns:
            f_mod: [B, mod_dim]
        """
        B, C, H, W = x.size()
        
        # 计算每个实例的均值和标准差 (在空间维度上)
        mu = x.view(B, C, -1).mean(dim=2)  # [B, C]
        var = x.view(B, C, -1).var(dim=2) + self.eps  # [B, C]
        sigma = torch.sqrt(var)  # [B, C]
        
        # 拼接统计量
        style_stats = torch.cat([mu, sigma], dim=1)  # [B, 2C]
        
        # 通过MLP压缩
        f_mod = self.style_mlp(style_stats)  # [B, mod_dim]
        
        return f_mod
    
    def _instance_normalize(self, x):
        """
        Instance Normalization: 去除风格/模态信息
        Args:
            x: [B, C, H, W]
        Returns:
            x_norm: [B, C, H, W]
        """
        B, C, H, W = x.size()
        
        # 计算均值和标准差
        mu = x.view(B, C, -1).mean(dim=2, keepdim=True)  # [B, C, 1]
        var = x.view(B, C, -1).var(dim=2, keepdim=True) + self.eps  # [B, C, 1]
        sigma = torch.sqrt(var)
        
        # 归一化
        x_flat = x.view(B, C, -1)
        x_norm = (x_flat - mu) / sigma  # [B, C, H*W]
        x_norm = x_norm.view(B, C, H, W)
        
        return x_norm
    
    def _apply_se_gate(self, x):
        """
        SE-Gate 通道注意力机制
        Args:
            x: [B, C, H, W]
        Returns:
            x_weighted: [B, C, H, W]
        """
        B, C, H, W = x.size()
        
        # Global Average Pooling
        squeeze = F.adaptive_avg_pool2d(x, 1).view(B, C)  # [B, C]
        
        # FC layers
        excitation = F.relu(self.se_reduce(squeeze))  # [B, C/16]
        excitation = torch.sigmoid(self.se_expand(excitation))  # [B, C]
        
        # 通道加权
        excitation = excitation.view(B, C, 1, 1)
        x_weighted = x * excitation
        
        return x_weighted


class MultiPartISG_DM(nn.Module):
    """
    多部件ISG-DM模块
    对K个部件分别应用ISG-DM解耦
    """
    
    def __init__(self, num_parts=6, input_dim=2048, id_dim=256, mod_dim=256):
        """
        Args:
            num_parts: 部件数量 K
            input_dim: 输入特征维度
            id_dim: 身份特征维度
            mod_dim: 模态特征维度
        """
        super(MultiPartISG_DM, self).__init__()
        self.num_parts = num_parts
        
        # 为每个部件创建独立的ISG-DM模块
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
