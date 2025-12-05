"""
models/pfmgcd_model.py - Simple Strong Baseline
减少 Dropout，保持简洁
"""

import torch
import torch.nn as nn
import torch.nn. functional as F

from .pcb_backbone import PCBBackbone


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname. find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.01)
        nn. init.constant_(m.bias, 0.0)


class PF_MGCD(nn.Module):
    """
    Simple Strong Baseline
    - PCB + IBN Backbone
    - 6 部件 + BNNeck
    - 单层 Dropout
    """
    def __init__(self, num_parts=6, num_identities=395, feature_dim=512,
                 pretrained=True, backbone='resnet50', use_ibn=True,
                 dropout=0.5, **kwargs):
        super().__init__()
        
        self.num_parts = num_parts
        self. feature_dim = feature_dim
        self. num_identities = num_identities
        
        # 1.  Backbone
        self.backbone = PCBBackbone(num_parts, pretrained, backbone, use_ibn)
        
        # 2. 部件降维 (无 Dropout)
        self.reducers = nn.ModuleList()
        for _ in range(num_parts):
            self.reducers.append(nn.Sequential(
                nn. Conv2d(2048, feature_dim, 1, bias=False),
                nn.BatchNorm2d(feature_dim),
                nn.ReLU(inplace=True)
            ))
        
        # 3. BNNeck
        self.bottlenecks = nn.ModuleList()
        for _ in range(num_parts):
            bn = nn.BatchNorm1d(feature_dim)
            bn.bias.requires_grad_(False)
            self.bottlenecks.append(bn)
        
        # 4. 分类器
        self.classifiers = nn.ModuleList()
        for _ in range(num_parts):
            self. classifiers.append(nn.Linear(feature_dim, num_identities, bias=False))
        
        # 5. 单层 Dropout (仅分类器前)
        self. dropout = nn. Dropout(p=dropout)
        
        # 初始化
        self.reducers. apply(weights_init_kaiming)
        self.bottlenecks.apply(weights_init_kaiming)
        self.classifiers.apply(weights_init_kaiming)
        
        # 占位
        self.memory_bank = None

    def forward(self, x, labels=None, **kwargs):
        # 1.  Backbone
        part_features_raw, _ = self.backbone(x)
        
        # 2.  降维 + 池化
        id_features = []
        for k in range(self.num_parts):
            feat = self.reducers[k](part_features_raw[k])
            feat = F.adaptive_avg_pool2d(feat, 1). flatten(1)
            id_features.append(feat)
        
        # 3. BNNeck + 分类
        logits = []
        bn_features = []
        for k in range(self.num_parts):
            feat_bn = self.bottlenecks[k](id_features[k])
            bn_features.append(feat_bn)
            
            if self.training:
                logits.append(self.classifiers[k](self.dropout(feat_bn)))
            else:
                logits.append(self.classifiers[k](feat_bn))
        
        return {
            'id_logits': logits,
            'id_features': id_features,
            'bn_features': bn_features,
            'gray_features': None,
            'adv_logits': None,
            'soft_labels': None,
            'entropy_weights': None
        }

    def extract_features(self, x, pool_parts=True):
        with torch.no_grad():
            outputs = self.forward(x)
            bn_features = outputs['bn_features']
            norm_feats = [F.normalize(f, p=2, dim=1) for f in bn_features]
            
            if pool_parts:
                return torch.cat(norm_feats, dim=1)
            else:
                return torch.stack(norm_feats, dim=1). mean(dim=1)

    def initialize_memory(self, *args, **kwargs):
        pass