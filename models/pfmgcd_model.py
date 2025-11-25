"""
PF-MGCD Model - Part-based Fine-grained Multi-Granularity Cross-modal Distillation
完整的PF-MGCD模型: 整合所有模块
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# 尝试相对导入，如果失败则使用绝对导入
try:
    from .pcb_backbone import PCBBackbone, PartPooling
    from .isg_dm import MultiPartISG_DM
    from .memory_bank import MultiPartMemoryBank
    from .graph_propagation import AdaptiveGraphPropagation
except ImportError:
    # 直接运行此文件时使用绝对导入
    from pcb_backbone import PCBBackbone, PartPooling
    from isg_dm import MultiPartISG_DM
    from memory_bank import MultiPartMemoryBank
    from graph_propagation import AdaptiveGraphPropagation


class PF_MGCD(nn.Module):
    """
    PF-MGCD完整模型
    包含:
    1. PCB Backbone (部件特征提取)
    2. ISG-DM (解耦模块)
    3. Memory Bank (记忆库)
    4. Graph Propagation (图传播)
    5. Multi-Part Classifiers (多部件分类器)
    """
    
    def __init__(self, 
                 num_parts=6,
                 num_identities=395,  # SYSU-MM01的ID数
                 feature_dim=256,
                 memory_momentum=0.9,
                 temperature=3.0,
                 top_k=5,
                 pretrained=True):
        """
        Args:
            num_parts: 部件数量 K
            num_identities: 身份数量 N
            feature_dim: 解耦后的特征维度
            memory_momentum: 记忆库动量系数
            temperature: Softmax温度参数
            top_k: 图传播的Top-K邻居
            pretrained: 是否使用预训练的ResNet50
        """
        super(PF_MGCD, self).__init__()
        self.num_parts = num_parts
        self.num_identities = num_identities
        self.feature_dim = feature_dim
        
        # ===== 1. PCB Backbone =====
        self.backbone = PCBBackbone(num_parts=num_parts, pretrained=pretrained)
        
        # ===== 2. Part Pooling =====
        self.part_pooling = PartPooling(input_dim=2048, output_dim=feature_dim)
        
        # ===== 3. ISG-DM 解耦模块 =====
        self.isg_dm = MultiPartISG_DM(
            num_parts=num_parts,
            input_dim=2048,  # ResNet50输出
            id_dim=feature_dim,
            mod_dim=feature_dim
        )
        
        # ===== 4. Memory Bank =====
        self.memory_bank = MultiPartMemoryBank(
            num_parts=num_parts,
            num_identities=num_identities,
            feature_dim=feature_dim,
            momentum=memory_momentum
        )
        
        # ===== 5. Graph Propagation =====
        self.graph_propagation = AdaptiveGraphPropagation(
            temperature=temperature,
            top_k=top_k,
            use_entropy_weight=True
        )
        
        # ===== 6. Multi-Part Classifiers =====
        # 为每个部件创建独立的分类器
        self.id_classifiers = nn.ModuleList([
            nn.Linear(feature_dim, num_identities) for _ in range(num_parts)
        ])
        
        # 模态分类器 (二分类: 可见光/红外)
        self.mod_classifiers = nn.ModuleList([
            nn.Linear(feature_dim, 2) for _ in range(num_parts)
        ])
    
    def forward(self, x, labels=None, modality_labels=None, update_memory=False):
        """
        前向传播
        Args:
            x: 输入图像 [B, 3, H, W]
            labels: 身份标签 [B] (训练时需要)
            modality_labels: 模态标签 [B] (0=可见光, 1=红外)
            update_memory: 是否更新记忆库 (已废弃，由外部控制)
        Returns:
            outputs: 字典，包含各种特征和预测
        """
        B = x.size(0)
        
        # ===== Step 1: 部件特征提取 =====
        part_features, global_feature = self.backbone(x)  # List of K × [B, 2048, H_k, W]
        
        # ===== Step 2: ISG-DM 解耦 =====
        id_features, mod_features = self.isg_dm(part_features)  # List of K × [B, 256]
        
        # ===== Step 3: 身份分类 =====
        id_logits = []
        for k in range(self.num_parts):
            logit = self.id_classifiers[k](id_features[k])  # [B, N]
            id_logits.append(logit)
        
        # ===== Step 4: 模态分类 =====
        mod_logits = []
        for k in range(self.num_parts):
            logit = self.mod_classifiers[k](mod_features[k])  # [B, 2]
            mod_logits.append(logit)
        
        # ===== Step 5: 图传播 (生成软标签) =====
        soft_labels, similarities, entropy_weights = self.graph_propagation(
            id_features, self.memory_bank
        )
        
        # ===== 返回结果 =====
        outputs = {
            'id_features': id_features,          # List of K × [B, D]
            'mod_features': mod_features,        # List of K × [B, D]
            'id_logits': id_logits,              # List of K × [B, N]
            'mod_logits': mod_logits,            # List of K × [B, 2]
            'soft_labels': soft_labels,          # List of K × [B, N]
            'similarities': similarities,        # List of K × [B, N]
            'entropy_weights': entropy_weights,  # List of K × [B]
            'global_feature': global_feature     # [B, 2048, H, W]
        }
        
        return outputs
    
    def extract_features(self, x, pool_parts=True):
        """
        提取测试特征 (推理阶段)
        Args:
            x: 输入图像 [B, 3, H, W]
            pool_parts: 是否合并部件特征
        Returns:
            features: 特征向量 [B, D] or [B, K*D]
        """
        with torch.no_grad():
            # 提取部件特征
            part_features, _ = self.backbone(x)
            
            # ISG-DM解耦 (仅使用身份特征)
            id_features, _ = self.isg_dm(part_features)
            
            # 归一化
            id_features = [F.normalize(f, dim=1) for f in id_features]
            
            if pool_parts:
                # 合并所有部件特征
                features = torch.cat(id_features, dim=1)  # [B, K*D]
            else:
                # 返回列表
                features = id_features
            
            return features
    
    def initialize_memory(self, dataloader, device):
        """
        使用数据加载器初始化记忆库
        Args:
            dataloader: 数据加载器
            device: 设备
        """
        print("Initializing memory bank...")
        self.eval()
        
        all_id_features = [[] for _ in range(self.num_parts)]
        all_labels = []
        
        with torch.no_grad():
            for batch_idx, (images, labels, _) in enumerate(dataloader):
                images = images.to(device)
                labels = labels.to(device)
                
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
        
        # 初始化记忆库
        all_id_features_device = [f.to(device) for f in all_id_features]
        all_labels_device = all_labels.to(device)
        
        self.memory_bank.initialize_memory(all_id_features_device, all_labels_device)
        print(f"Memory bank initialized! Initialized IDs: {self.memory_bank.initialized.sum().item()}")
        
        self.train()

