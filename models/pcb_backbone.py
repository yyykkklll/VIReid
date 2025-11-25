"""
PCB Backbone - Part-based Convolutional Baseline
实现ResNet50骨干网络 + 水平切分 (K个部件)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
try:
    from torchvision.models import ResNet50_Weights
    WEIGHTS_AVAILABLE = True
except ImportError:
    WEIGHTS_AVAILABLE = False


class PCBBackbone(nn.Module):
    """
    PCB特征提取器
    - 使用ResNet50作为骨干网络
    - 去除Layer4的stride=2，保留更多空间分辨率
    - 水平切分为K个部件
    """
    
    def __init__(self, num_parts=6, pretrained=True):
        """
        Args:
            num_parts: 水平切分的部件数量 (K)
            pretrained: 是否使用预训练权重
        """
        super(PCBBackbone, self).__init__()
        self.num_parts = num_parts
        
        # 加载ResNet50 - 使用新的权重API
        if pretrained:
            if WEIGHTS_AVAILABLE:
                # 新版torchvision (>=0.13)
                resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
            else:
                # 旧版torchvision
                resnet = models.resnet50(pretrained=True)
        else:
            if WEIGHTS_AVAILABLE:
                resnet = models.resnet50(weights=None)
            else:
                resnet = models.resnet50(pretrained=False)
        
        # 提取前面的层 (去掉avgpool和fc)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        
        self.layer1 = resnet.layer1  # /4
        self.layer2 = resnet.layer2  # /8
        self.layer3 = resnet.layer3  # /16
        
        # 关键修改: 去除Layer4的stride=2，保持分辨率
        self.layer4 = self._modify_layer4_stride(resnet.layer4)  # /16 (不再是/32)
        
        # 输出通道数
        self.feature_dim = 2048
        
    def _modify_layer4_stride(self, layer4):
        """
        修改Layer4的第一个bottleneck的stride从2改为1
        同时修改downsample的stride
        """
        # Layer4的第一个block
        layer4[0].conv2.stride = (1, 1)  # 原本是(2, 2)
        layer4[0].downsample[0].stride = (1, 1)  # 原本是(2, 2)
        return layer4
    
    def forward(self, x):
        """
        前向传播
        Args:
            x: 输入图像 [B, 3, H, W] (H=384, W=128)
        Returns:
            part_features: List of K个部件特征 [B, 2048, h_k, w]
            global_feature: 全局特征图 [B, 2048, H_f, W_f]
        """
        # ResNet特征提取
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)  # [B, 256, H/4, W/4]
        x = self.layer2(x)  # [B, 512, H/8, W/8]
        x = self.layer3(x)  # [B, 1024, H/16, W/16]
        x = self.layer4(x)  # [B, 2048, H/16, W/16]
        
        # 保存全局特征图 (用于可视化或其他用途)
        global_feature = x  # [B, 2048, 24, 8] (假设输入384x128)
        
        # 水平切分为K个部件
        part_features = self._horizontal_split(global_feature)
        
        return part_features, global_feature
    
    def _horizontal_split(self, feature_map):
        """
        将特征图在高度方向均匀切分为K个部件
        Args:
            feature_map: [B, C, H, W]
        Returns:
            parts: List of K个张量，每个为 [B, C, H/K, W]
        """
        B, C, H, W = feature_map.size()
        
        # 确保高度可以被K整除
        assert H % self.num_parts == 0, \
            f"Feature height {H} must be divisible by num_parts {self.num_parts}"
        
        part_height = H // self.num_parts
        parts = []
        
        for i in range(self.num_parts):
            start = i * part_height
            end = (i + 1) * part_height
            part = feature_map[:, :, start:end, :]  # [B, C, H/K, W]
            parts.append(part)
        
        return parts


class PartPooling(nn.Module):
    """
    部件池化模块
    对每个部件特征进行平均池化，得到固定维度的向量
    """
    
    def __init__(self, input_dim=2048, output_dim=256):
        """
        Args:
            input_dim: 输入特征维度 (ResNet50为2048)
            output_dim: 输出特征维度 (降维后)
        """
        super(PartPooling, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)  # Global Average Pooling
        
        # 降维层 (可选)
        self.reduction = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, part_features):
        """
        Args:
            part_features: List of K个部件特征 [B, C, H_k, W]
        Returns:
            pooled_features: List of K个向量 [B, output_dim]
        """
        pooled_features = []
        
        for part in part_features:
            # 平均池化
            pooled = self.gap(part)  # [B, C, 1, 1]
            pooled = pooled.view(pooled.size(0), -1)  # [B, C]
            
            # 降维
            pooled = self.reduction(pooled)  # [B, output_dim]
            pooled_features.append(pooled)
        
        return pooled_features