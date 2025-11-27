"""
Teacher Network - Pre-trained frozen networks for memory initialization
教师网络: 用于记忆库初始化的预训练冻结网络
(修改版：支持PCB切分，输出多粒度特征以匹配记忆库)
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


class TeacherNetwork(nn.Module):
    """
    教师网络 (PCB架构)
    - 使用预训练的ResNet50
    - 参数冻结
    - 输出 K 个部件特征，用于初始化多粒度记忆库
    """
    
    def __init__(self, feature_dim=256, num_parts=6, pretrained=True, freeze=True):
        """
        Args:
            feature_dim: 输出特征维度
            num_parts: 部件数量
            pretrained: 是否使用预训练权重
            freeze: 是否冻结参数
        """
        super(TeacherNetwork, self).__init__()
        self.num_parts = num_parts
        
        # 加载预训练的ResNet50
        if pretrained:
            if WEIGHTS_AVAILABLE:
                resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
            else:
                resnet = models.resnet50(pretrained=True)
        else:
            if WEIGHTS_AVAILABLE:
                resnet = models.resnet50(weights=None)
            else:
                resnet = models.resnet50(pretrained=False)
        
        # 提取ResNet层
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        
        # [关键修改] 修改Layer4 stride=1，保持空间分辨率 (同PCBBackbone)
        self.layer4 = resnet.layer4
        self.layer4[0].conv2.stride = (1, 1)
        if self.layer4[0].downsample is not None:
            self.layer4[0].downsample[0].stride = (1, 1)
        
        # 全局平均池化 (用于每个部件)
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # 降维层 (共享权重或独立权重? 这里使用共享权重以保持简单)
        self.reduction = nn.Sequential(
            nn.Linear(2048, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True)
        )
        
        # 冻结参数
        if freeze:
            self.freeze_parameters()
    
    def freeze_parameters(self):
        for param in self.parameters():
            param.requires_grad = False
        print("Teacher network parameters frozen.")
    
    def _horizontal_split(self, feature_map):
        """水平切分特征图"""
        B, C, H, W = feature_map.size()
        if H % self.num_parts != 0:
            target_h = (H // self.num_parts) * self.num_parts
            feature_map = F.interpolate(feature_map, size=(target_h, W), 
                                      mode='bilinear', align_corners=True)
            H = target_h
        
        part_height = H // self.num_parts
        parts = []
        for i in range(self.num_parts):
            part = feature_map[:, :, i*part_height:(i+1)*part_height, :]
            parts.append(part)
        return parts

    def forward(self, x):
        """
        Returns:
            features_list: List of [B, feature_dim] (K个)
        """
        # ResNet特征提取
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x) # [B, 2048, H, W]
        
        # 切分
        parts = self._horizontal_split(x)
        
        # 池化与降维
        features_list = []
        for part in parts:
            pooled = self.gap(part) # [B, 2048, 1, 1]
            pooled = pooled.view(pooled.size(0), -1)
            feat = self.reduction(pooled) # [B, feature_dim]
            features_list.append(feat)
            
        return features_list


class DualModalityTeacher(nn.Module):
    """双模态教师网络 (PCB版)"""
    
    def __init__(self, feature_dim=256, num_parts=6, pretrained=True, freeze=True):
        super(DualModalityTeacher, self).__init__()
        self.visible_teacher = TeacherNetwork(feature_dim, num_parts, pretrained, freeze)
        self.infrared_teacher = TeacherNetwork(feature_dim, num_parts, pretrained, freeze)
    
    def forward(self, x, modality_labels):
        """
        Returns:
            features_list: List of [B, feature_dim]
        """
        # 假设所有Teacher输出形状一致，我们可以先运行一个来获取列表结构
        # 但由于Batch中可能混合了模态，我们需要分别计算再组合
        
        B = x.size(0)
        # 临时运行一次获取部件数量和维度 (为了初始化容器)
        # 注意：这里假设 visible_teacher 和 infrared_teacher 结构一致
        num_parts = self.visible_teacher.num_parts
        out_dim = self.visible_teacher.reduction[0].out_features
        
        # 初始化输出容器: K个 [B, D] 的Tensor
        features_list = [torch.zeros(B, out_dim, device=x.device) for _ in range(num_parts)]
        
        visible_mask = (modality_labels == 0)
        infrared_mask = (modality_labels == 1)
        
        # 可见光前向
        if visible_mask.sum() > 0:
            vis_feats = self.visible_teacher(x[visible_mask]) # List of [B_vis, D]
            for k in range(num_parts):
                features_list[k][visible_mask] = vis_feats[k]
        
        # 红外前向
        if infrared_mask.sum() > 0:
            ir_feats = self.infrared_teacher(x[infrared_mask]) # List of [B_ir, D]
            for k in range(num_parts):
                features_list[k][infrared_mask] = ir_feats[k]
        
        return features_list


def create_teacher_network(teacher_type='single', **kwargs):
    if teacher_type == 'single':
        return TeacherNetwork(**kwargs)
    elif teacher_type == 'dual':
        return DualModalityTeacher(**kwargs)
    else:
        raise ValueError(f"Unknown teacher type: {teacher_type}")