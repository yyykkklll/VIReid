"""
PF-MGCD Model v2 - Enhanced with Part Weighting and Better Feature Normalization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .pcb_backbone import PCBBackbone
    from .isg_dm import MultiPartISG_DM
    from .memory_bank import MultiPartMemoryBank
    from .graph_propagation import AdaptiveGraphPropagation
except ImportError:
    from pcb_backbone import PCBBackbone
    from isg_dm import MultiPartISG_DM
    from memory_bank import MultiPartMemoryBank
    from graph_propagation import AdaptiveGraphPropagation


class PF_MGCD(nn.Module):
    """PF-MGCD完整模型 - 改进版"""
    
    def __init__(self, 
                 num_parts=6,
                 num_identities=395,
                 feature_dim=256,
                 memory_momentum=0.9,
                 temperature=3.0,
                 top_k=5,
                 pretrained=True,
                 backbone='resnet50'):
        super(PF_MGCD, self).__init__()
        self.num_parts = num_parts
        self.num_identities = num_identities
        self.feature_dim = feature_dim
        
        # PCB Backbone（支持多种ResNet）
        self.backbone = PCBBackbone(
            num_parts=num_parts, 
            pretrained=pretrained,
            backbone=backbone
        )
        
        # ISG-DM 解耦模块（改进版）
        self.isg_dm = MultiPartISG_DM(
            num_parts=num_parts,
            input_dim=2048,
            id_dim=feature_dim,
            mod_dim=feature_dim
        )
        
        # Memory Bank
        self.memory_bank = MultiPartMemoryBank(
            num_parts=num_parts,
            num_identities=num_identities,
            feature_dim=feature_dim,
            momentum=memory_momentum
        )
        
        # Graph Propagation
        self.graph_propagation = AdaptiveGraphPropagation(
            temperature=temperature,
            top_k=top_k,
            use_entropy_weight=True
        )
        
        # Multi-Part Classifiers
        self.id_classifiers = nn.ModuleList([
            nn.Linear(feature_dim, num_identities) for _ in range(num_parts)
        ])
        
        self.mod_classifiers = nn.ModuleList([
            nn.Linear(feature_dim, 2) for _ in range(num_parts)
        ])
        
        # 可学习的部件权重
        self.part_weights = nn.Parameter(torch.ones(num_parts))
        
        # 温度参数
        self.register_buffer('temperature_scale', torch.tensor(1.0))
        
        # Label映射（用于memory bank初始化）
        self.train_label_mapping = None
    
    def forward(self, x, labels=None, modality_labels=None, update_memory=False):
        """前向传播"""
        B = x.size(0)
        
        part_features, global_feature = self.backbone(x)
        id_features, mod_features = self.isg_dm(part_features)
        
        id_logits = []
        for k in range(self.num_parts):
            logit = self.id_classifiers[k](id_features[k])
            id_logits.append(logit)
        
        mod_logits = []
        for k in range(self.num_parts):
            logit = self.mod_classifiers[k](mod_features[k])
            mod_logits.append(logit)
        
        soft_labels, similarities, entropy_weights = self.graph_propagation(
            id_features, self.memory_bank
        )
        
        outputs = {
            'id_features': id_features,
            'mod_features': mod_features,
            'id_logits': id_logits,
            'mod_logits': mod_logits,
            'soft_labels': soft_labels,
            'similarities': similarities,
            'entropy_weights': entropy_weights,
            'global_feature': global_feature
        }
        
        return outputs
    
    def extract_features(self, x, pool_parts=True):
        """提取测试特征（改进版）"""
        with torch.no_grad():
            part_features, _ = self.backbone(x)
            id_features, _ = self.isg_dm(part_features)
            
            if pool_parts:
                id_features_norm = [F.normalize(f, p=2, dim=1) for f in id_features]
                weights = F.softmax(self.part_weights, dim=0)
                weighted_features = [f * weights[i] for i, f in enumerate(id_features_norm)]
                features = torch.cat(weighted_features, dim=1)
                features = F.normalize(features, p=2, dim=1)
            else:
                features = [F.normalize(f, p=2, dim=1) for f in id_features]
            
            return features
    
    def initialize_memory(self, dataloader, device):
        """
        使用数据加载器初始化记忆库
        自动处理label映射（原始ID -> 连续索引）
        """
        print("Initializing memory bank...")
        self.eval()
        
        all_id_features = [[] for _ in range(self.num_parts)]
        all_labels = []
        
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(dataloader):
                # 识别batch格式
                if len(batch_data) == 2:
                    images, info = batch_data
                    images = images.to(device)
                    
                    if isinstance(info, torch.Tensor):
                        labels = info[:, 1].long()
                    else:
                        labels = torch.from_numpy(info[:, 1]).long()
                
                elif len(batch_data) == 3:
                    images, labels, cams = batch_data
                    images = images.to(device)
                    if not isinstance(labels, torch.Tensor):
                        labels = torch.tensor(labels).long()
                
                else:
                    raise ValueError(f"Unexpected batch format: {len(batch_data)} elements")
                
                # 前向传播
                outputs = self.forward(images)
                id_features = outputs['id_features']
                
                # 收集特征
                for k in range(self.num_parts):
                    all_id_features[k].append(id_features[k].cpu())
                all_labels.append(labels.cpu())
                
                if (batch_idx + 1) % 50 == 0:
                    print(f"  Processed {batch_idx + 1} batches")
        
        # 合并特征
        for k in range(self.num_parts):
            all_id_features[k] = torch.cat(all_id_features[k], dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        # [关键修复] 创建label映射
        unique_labels = torch.unique(all_labels)
        print(f"  Original label range: [{unique_labels.min().item()}, {unique_labels.max().item()}]")
        print(f"  Unique IDs: {len(unique_labels)}")
        
        # 创建映射：原始ID -> 连续索引
        label_mapping = {old_id.item(): new_id for new_id, old_id in enumerate(unique_labels)}
        
        # 应用映射
        mapped_labels = torch.tensor([label_mapping[label.item()] for label in all_labels])
        
        print(f"  Mapped label range: [{mapped_labels.min().item()}, {mapped_labels.max().item()}]")
        
        # 安全检查
        if mapped_labels.max().item() >= self.num_identities:
            raise ValueError(
                f"Label mapping failed! Max mapped label: {mapped_labels.max().item()}, "
                f"but memory bank size is {self.num_identities}"
            )
        
        # 初始化记忆库
        all_id_features_device = [f.to(device) for f in all_id_features]
        mapped_labels_device = mapped_labels.to(device)
        
        self.memory_bank.initialize_memory(all_id_features_device, mapped_labels_device)
        
        initialized_count = self.memory_bank.initialized.sum().item()
        print(f"Memory bank initialized! {initialized_count}/{self.num_identities} IDs")
        
        # 保存映射（可能需要用于训练时的label转换）
        self.train_label_mapping = label_mapping
        
        self.train()
