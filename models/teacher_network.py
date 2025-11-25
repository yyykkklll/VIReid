"""
Teacher Network - Pre-trained frozen networks for memory initialization
教师网络: 用于记忆库初始化的预训练冻结网络
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
    教师网络
    - 使用预训练的ResNet50
    - 参数冻结，仅用于特征提取
    - 用于初始化记忆库
    """
    
    def __init__(self, feature_dim=256, pretrained=True, freeze=True):
        """
        Args:
            feature_dim: 输出特征维度
            pretrained: 是否使用预训练权重
            freeze: 是否冻结参数
        """
        super(TeacherNetwork, self).__init__()
        
        # 加载预训练的ResNet50 - 使用新的权重API
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
        
        # 提取特征提取部分（去掉最后的FC层）
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool
        
        # 添加降维层
        self.reduction = nn.Sequential(
            nn.Linear(2048, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True)
        )
        
        # 冻结参数
        if freeze:
            self.freeze_parameters()
    
    def freeze_parameters(self):
        """冻结所有参数"""
        for param in self.parameters():
            param.requires_grad = False
        print("Teacher network parameters frozen.")
    
    def unfreeze_parameters(self):
        """解冻所有参数"""
        for param in self.parameters():
            param.requires_grad = True
        print("Teacher network parameters unfrozen.")
    
    def forward(self, x):
        """
        前向传播
        Args:
            x: 输入图像 [B, 3, H, W]
        Returns:
            features: 特征向量 [B, feature_dim]
        """
        # ResNet特征提取
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # 全局平均池化
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        # 降维
        features = self.reduction(x)
        
        return features


class DualModalityTeacher(nn.Module):
    """
    双模态教师网络
    - 分别为可见光和红外模态创建独立的教师网络
    - 用于更好地初始化跨模态记忆库
    """
    
    def __init__(self, feature_dim=256, pretrained=True, freeze=True):
        """
        Args:
            feature_dim: 输出特征维度
            pretrained: 是否使用预训练权重
            freeze: 是否冻结参数
        """
        super(DualModalityTeacher, self).__init__()
        
        # 可见光教师网络
        self.visible_teacher = TeacherNetwork(feature_dim, pretrained, freeze)
        
        # 红外教师网络（共享大部分权重，仅调整输入层）
        self.infrared_teacher = TeacherNetwork(feature_dim, pretrained, freeze)
    
    def forward(self, x, modality_labels):
        """
        前向传播
        Args:
            x: 输入图像 [B, 3, H, W]
            modality_labels: 模态标签 [B] (0=可见光, 1=红外)
        Returns:
            features: 特征向量 [B, feature_dim]
        """
        B = x.size(0)
        features = torch.zeros(B, self.visible_teacher.reduction[0].out_features, 
                              device=x.device)
        
        # 分别处理不同模态
        visible_mask = (modality_labels == 0)
        infrared_mask = (modality_labels == 1)
        
        if visible_mask.sum() > 0:
            features[visible_mask] = self.visible_teacher(x[visible_mask])
        
        if infrared_mask.sum() > 0:
            features[infrared_mask] = self.infrared_teacher(x[infrared_mask])
        
        return features


def create_teacher_network(teacher_type='single', **kwargs):
    """
    工厂函数: 创建教师网络
    Args:
        teacher_type: 教师网络类型 ('single' or 'dual')
        **kwargs: 其他参数
    Returns:
        teacher: 教师网络实例
    """
    if teacher_type == 'single':
        return TeacherNetwork(**kwargs)
    elif teacher_type == 'dual':
        return DualModalityTeacher(**kwargs)
    else:
        raise ValueError(f"Unknown teacher type: {teacher_type}")