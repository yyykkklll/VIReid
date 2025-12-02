"""
models/pcb_backbone.py - PCB Backbone with IBN-Net (API 修复版)

修复日志:
1. [Fixed] 修复 _load_resnet 中 torchvision 新旧 API 混用导致的 TypeError
2. 包含 IBN (Instance-Batch Normalization) 模块
3. 包含 GeMPooling 和 PartPooling 模块
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# 尝试导入新版 torchvision 权重 API
try:
    from torchvision.models import (
        ResNet50_Weights, 
        ResNet101_Weights, 
        ResNet152_Weights
    )
    WEIGHTS_AVAILABLE = True
except ImportError:
    WEIGHTS_AVAILABLE = False


class IBN(nn.Module):
    """
    IBN-Net: Instance-Batch Normalization
    将通道分为两半，一半用 IN (去除风格差异)，一半用 BN (保留内容特征)
    """
    def __init__(self, planes):
        super(IBN, self).__init__()
        half1 = int(planes / 2)
        self.half = half1
        half2 = planes - half1
        self.IN = nn.InstanceNorm2d(half1, affine=True)
        self.BN = nn.BatchNorm2d(half2)

    def forward(self, x):
        split = torch.split(x, self.half, 1)
        out1 = self.IN(split[0].contiguous())
        out2 = self.BN(split[1].contiguous())
        out = torch.cat((out1, out2), 1)
        return out


class GeMPooling(nn.Module):
    """
    广义平均池化 (Generalized Mean Pooling)
    公式: f = (mean(x^p))^(1/p)
    """
    def __init__(self, p=3.0, eps=1e-6):
        super(GeMPooling, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        x = x.clamp(min=self.eps).pow(self.p)
        x = F.avg_pool2d(x, (x.size(-2), x.size(-1)))
        return x.pow(1.0 / self.p)
    
    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'


class PCBBackbone(nn.Module):
    """
    PCB 特征提取器 - 支持 IBN-Net 增强
    """
    def __init__(self, num_parts=6, pretrained=True, backbone='resnet50', use_ibn=True):
        """
        Args:
            num_parts: 水平切分的部件数量 (K)
            pretrained: 是否使用 ImageNet 预训练权重
            backbone: 骨干网络类型
            use_ibn: 是否启用 IBN-Net (默认 True)
        """
        super(PCBBackbone, self).__init__()
        self.num_parts = num_parts
        self.backbone_name = backbone
        
        # 加载对应的 ResNet 骨干
        resnet = self._load_resnet(backbone, pretrained)
        
        # [策略一] IBN-Net 核心修改
        # 将 layer1 和 layer2 中的 BN 替换为 IBN
        if use_ibn:
            self._replace_ibn(resnet.layer1)
            self._replace_ibn(resnet.layer2)
            print("✅ IBN-Net enabled: IBN blocks injected into Layer 1 & 2")
        
        # 提取 ResNet 层
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        
        self.layer1 = resnet.layer1  # /4, 输出 256 通道
        self.layer2 = resnet.layer2  # /8, 输出 512 通道
        self.layer3 = resnet.layer3  # /16, 输出 1024 通道
        
        # 将 Layer4 的 stride 设为 1，保持特征图空间分辨率
        self.layer4 = self._modify_layer4_stride(resnet.layer4)  # /16, 输出 2048 通道
        
        self.feature_dim = 2048
    
    def _replace_ibn(self, layer):
        """将 ResNet Layer 中的 BN 替换为 IBN"""
        for i, block in enumerate(layer):
            if hasattr(block, 'bn1'):
                planes = block.bn1.num_features
                ibn = IBN(planes)
                # 替换 bn1
                block.bn1 = ibn

    def _load_resnet(self, backbone, pretrained):
        """
        加载 ResNet 模型，严格区分新旧 API
        """
        if WEIGHTS_AVAILABLE:
            # === 新版 API (0.13+) ===
            # 仅使用 weights 参数，严禁传递 pretrained
            if backbone == 'resnet50':
                weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
                return models.resnet50(weights=weights)
            elif backbone == 'resnet101':
                weights = ResNet101_Weights.IMAGENET1K_V1 if pretrained else None
                return models.resnet101(weights=weights)
            elif backbone == 'resnet152':
                weights = ResNet152_Weights.IMAGENET1K_V1 if pretrained else None
                return models.resnet152(weights=weights)
        else:
            # === 旧版 API ===
            # 使用 pretrained 参数
            if backbone == 'resnet50':
                return models.resnet50(pretrained=pretrained)
            elif backbone == 'resnet101':
                return models.resnet101(pretrained=pretrained)
            elif backbone == 'resnet152':
                return models.resnet152(pretrained=pretrained)
        
        raise ValueError(f"Unsupported backbone: {backbone}")
    
    def _modify_layer4_stride(self, layer4):
        layer4[0].conv2.stride = (1, 1)
        if layer4[0].downsample is not None:
            layer4[0].downsample[0].stride = (1, 1)
        return layer4
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)  # [B, 2048, H, W]
        
        # 返回切分后的部件特征 和 全局特征图
        global_feature = x
        part_features = self._horizontal_split(global_feature)
        
        return part_features, global_feature
    
    def _horizontal_split(self, feature_map):
        B, C, H, W = feature_map.size()
        if H % self.num_parts != 0:
            target_h = (H // self.num_parts) * self.num_parts
            feature_map = F.interpolate(feature_map, size=(target_h, W), mode='bilinear', align_corners=True)
            H = target_h
        
        part_height = H // self.num_parts
        parts = []
        for i in range(self.num_parts):
            start = i * part_height
            end = (i + 1) * part_height
            part = feature_map[:, :, start:end, :]
            parts.append(part)
        return parts


class PartPooling(nn.Module):
    """
    部件池化模块
    使用 GeM Pooling 替代普通的 GAP
    """
    def __init__(self, input_dim=2048, output_dim=256):
        super(PartPooling, self).__init__()
        self.gap = GeMPooling(p=3.0)
        self.reduction = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, part_features):
        pooled_features = []
        for part in part_features:
            pooled = self.gap(part)
            pooled = pooled.view(pooled.size(0), -1)
            pooled = self.reduction(pooled)
            pooled_features.append(pooled)
        return pooled_features