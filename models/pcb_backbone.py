"""
PCB Backbone - Part-based Convolutional Baseline
支持ResNet50/101/152多种骨干网络
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# 尝试导入新版权重API
try:
    from torchvision.models import (
        ResNet50_Weights, 
        ResNet101_Weights, 
        ResNet152_Weights
    )
    WEIGHTS_AVAILABLE = True
except ImportError:
    WEIGHTS_AVAILABLE = False


class PCBBackbone(nn.Module):
    """
    PCB特征提取器 - 支持多种ResNet骨干
    
    支持的骨干网络:
    - ResNet50 (默认): 25.6M params, 4.1G FLOPs
    - ResNet101: 44.5M params, 7.8G FLOPs (推荐)
    - ResNet152: 60.2M params, 11.5G FLOPs
    """
    
    def __init__(self, num_parts=6, pretrained=True, backbone='resnet50'):
        """
        Args:
            num_parts: 水平切分的部件数量 (K)
            pretrained: 是否使用ImageNet预训练权重
            backbone: 骨干网络类型 ['resnet50', 'resnet101', 'resnet152']
        """
        super(PCBBackbone, self).__init__()
        self.num_parts = num_parts
        self.backbone_name = backbone
        
        # 加载对应的ResNet骨干
        resnet = self._load_resnet(backbone, pretrained)
        
        # 提取ResNet层 (去掉avgpool和fc)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        
        self.layer1 = resnet.layer1  # /4, 输出256通道
        self.layer2 = resnet.layer2  # /8, 输出512通道
        self.layer3 = resnet.layer3  # /16, 输出1024通道
        
        # [关键修改] 去除Layer4的stride=2，保持空间分辨率
        self.layer4 = self._modify_layer4_stride(resnet.layer4)  # /16, 输出2048通道
        
        # 输出特征维度
        self.feature_dim = 2048
        
        print(f"PCB Backbone initialized:")
        print(f"  Architecture: {backbone.upper()}")
        print(f"  Output channels: {self.feature_dim}")
        print(f"  Num parts: {num_parts}")
        print(f"  Pretrained: {pretrained}")
    
    def _load_resnet(self, backbone, pretrained):
        """加载指定的ResNet模型"""
        
        if backbone == 'resnet50':
            if pretrained:
                if WEIGHTS_AVAILABLE:
                    return models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
                else:
                    return models.resnet50(pretrained=True)
            else:
                return models.resnet50(pretrained=False)
        
        elif backbone == 'resnet101':
            if pretrained:
                if WEIGHTS_AVAILABLE:
                    return models.resnet101(weights=ResNet101_Weights.IMAGENET1K_V1)
                else:
                    return models.resnet101(pretrained=True)
            else:
                return models.resnet101(pretrained=False)
        
        elif backbone == 'resnet152':
            if pretrained:
                if WEIGHTS_AVAILABLE:
                    return models.resnet152(weights=ResNet152_Weights.IMAGENET1K_V1)
                else:
                    return models.resnet152(pretrained=True)
            else:
                return models.resnet152(pretrained=False)
        
        else:
            raise ValueError(f"Unsupported backbone: {backbone}. "
                           f"Choose from ['resnet50', 'resnet101', 'resnet152']")
    
    def _modify_layer4_stride(self, layer4):
        """
        修改Layer4的stride，保持特征图分辨率
        
        对于ReID任务，保持更高的空间分辨率很重要，
        因此将Layer4的stride从2改为1
        """
        # Layer4的第一个bottleneck
        layer4[0].conv2.stride = (1, 1)  # 原本是(2, 2)
        
        # 修改downsample的stride（如果存在）
        if layer4[0].downsample is not None:
            layer4[0].downsample[0].stride = (1, 1)  # 原本是(2, 2)
        
        return layer4
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入图像 [B, 3, H, W]
               典型尺寸: [B, 3, 288, 144]
        
        Returns:
            part_features: List of K个部件特征 [B, 2048, h_k, w]
                          对于288x144输入，每个部件约为 [B, 2048, 3, 9]
            global_feature: 全局特征图 [B, 2048, 18, 9]
        """
        # ResNet特征提取
        x = self.conv1(x)      # [B, 64, H/2, W/2]
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)    # [B, 64, H/4, W/4]
        
        x = self.layer1(x)     # [B, 256, H/4, W/4]
        x = self.layer2(x)     # [B, 512, H/8, W/8]
        x = self.layer3(x)     # [B, 1024, H/16, W/16]
        x = self.layer4(x)     # [B, 2048, H/16, W/16]
        
        # 保存全局特征图
        global_feature = x
        
        # 水平切分为K个部件
        part_features = self._horizontal_split(global_feature)
        
        return part_features, global_feature
    
    def _horizontal_split(self, feature_map):
        """
        在高度方向均匀切分特征图
        
        Args:
            feature_map: [B, C, H, W]
        
        Returns:
            parts: List of K个张量 [B, C, H/K, W]
        """
        B, C, H, W = feature_map.size()
        
        # 检查高度是否可被K整除
        if H % self.num_parts != 0:
            # 如果不能整除，进行插值调整
            target_h = (H // self.num_parts) * self.num_parts
            feature_map = F.interpolate(
                feature_map, 
                size=(target_h, W), 
                mode='bilinear', 
                align_corners=True
            )
            H = target_h
        
        part_height = H // self.num_parts
        parts = []
        
        for i in range(self.num_parts):
            start = i * part_height
            end = (i + 1) * part_height
            part = feature_map[:, :, start:end, :]  # [B, C, H/K, W]
            parts.append(part)
        
        return parts
    
    def get_params_count(self):
        """获取参数量统计"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total': total_params,
            'trainable': trainable_params,
            'frozen': total_params - trainable_params
        }


class PartPooling(nn.Module):
    """
    部件池化模块
    对每个部件特征进行全局平均池化 + 降维
    """
    
    def __init__(self, input_dim=2048, output_dim=256):
        """
        Args:
            input_dim: 输入特征维度 (ResNet输出2048)
            output_dim: 输出特征维度 (降维后)
        """
        super(PartPooling, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # 降维层
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
            # 全局平均池化
            pooled = self.gap(part)  # [B, C, 1, 1]
            pooled = pooled.view(pooled.size(0), -1)  # [B, C]
            
            # 降维
            pooled = self.reduction(pooled)  # [B, output_dim]
            pooled_features.append(pooled)
        
        return pooled_features


def test_backbone():
    """测试不同backbone的输出"""
    print("=" * 70)
    print("Testing PCB Backbones")
    print("=" * 70)
    
    x = torch.randn(2, 3, 288, 144)  # [B, C, H, W]
    
    for backbone in ['resnet50', 'resnet101', 'resnet152']:
        print(f"\n{'='*70}")
        print(f"Testing {backbone.upper()}")
        print(f"{'='*70}")
        
        model = PCBBackbone(num_parts=6, pretrained=False, backbone=backbone)
        
        # 参数统计
        params = model.get_params_count()
        print(f"\nParameters:")
        print(f"  Total: {params['total']/1e6:.2f}M")
        print(f"  Trainable: {params['trainable']/1e6:.2f}M")
        
        # 前向传播
        part_features, global_feature = model(x)
        
        print(f"\nOutput shapes:")
        print(f"  Global feature: {global_feature.shape}")
        print(f"  Num parts: {len(part_features)}")
        print(f"  Each part: {part_features[0].shape}")
        
        # 估算FLOPs (简单估计)
        from thop import profile
        try:
            flops, _ = profile(model, inputs=(x,), verbose=False)
            print(f"  FLOPs: {flops/1e9:.2f}G")
        except:
            print(f"  FLOPs: N/A (install thop: pip install thop)")


if __name__ == '__main__':
    test_backbone()
