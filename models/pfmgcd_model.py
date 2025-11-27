"""
PF-MGCD Model v2
基于部件的细粒度多粒度跨模态蒸馏模型
包含：PCB骨干网、ISG-DM解耦模块、记忆库、图传播模块
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
    """
    PF-MGCD 完整模型架构
    
    结构:
    1. Backbone: PCB切分的ResNet (支持 ResNet50/101/152)
    2. ISG-DM: 实例统计引导的特征解耦
    3. Memory: 多粒度模态无关记忆库
    4. Graph: 自适应图传播与蒸馏
    """
    
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
        
        # 1. PCB 骨干网络 (支持 ResNet101 等)
        self.backbone = PCBBackbone(
            num_parts=num_parts, 
            pretrained=pretrained,
            backbone=backbone
        )
        
        # 2. ISG-DM 解耦模块 (物理统计解耦)
        self.isg_dm = MultiPartISG_DM(
            num_parts=num_parts,
            input_dim=2048,
            id_dim=feature_dim,
            mod_dim=feature_dim
        )
        
        # 3. 多粒度记忆库
        self.memory_bank = MultiPartMemoryBank(
            num_parts=num_parts,
            num_identities=num_identities,
            feature_dim=feature_dim,
            momentum=memory_momentum
        )
        
        # 4. 自适应图传播模块
        self.graph_propagation = AdaptiveGraphPropagation(
            temperature=temperature,
            top_k=top_k,
            use_entropy_weight=True
        )
        
        # 身份分类器 (每个部件一个)
        self.id_classifiers = nn.ModuleList([
            nn.Linear(feature_dim, num_identities) for _ in range(num_parts)
        ])
        
        # 模态分类器 (每个部件一个)
        self.mod_classifiers = nn.ModuleList([
            nn.Linear(feature_dim, 2) for _ in range(num_parts)
        ])
        
        # 可学习的部件权重 (用于测试阶段特征融合)
        self.part_weights = nn.Parameter(torch.ones(num_parts))
        
        # 注册缓冲区
        self.register_buffer('temperature_scale', torch.tensor(1.0))
        
        # Label映射缓存 (用于Debug和记录)
        self.train_label_mapping = None
    
    def forward(self, x, labels=None, modality_labels=None, update_memory=False):
        """
        前向传播流程
        Args:
            x: 输入图像 [B, 3, H, W]
            labels: 身份标签 [B]
            modality_labels: 模态标签 [B]
            update_memory: 是否更新记忆库 (通常在外部控制)
        """
        # 1. 骨干提取与切分
        part_features, global_feature = self.backbone(x)
        
        # 2. ISG-DM 解耦
        id_features, mod_features = self.isg_dm(part_features)
        
        # 3. 计算分类 Logits
        id_logits = []
        for k in range(self.num_parts):
            logit = self.id_classifiers[k](id_features[k])
            id_logits.append(logit)
        
        mod_logits = []
        for k in range(self.num_parts):
            logit = self.mod_classifiers[k](mod_features[k])
            mod_logits.append(logit)
        
        # 4. 图传播与软标签生成
        # 利用当前 batch 的特征去查询记忆库，获取软标签
        soft_labels, similarities, entropy_weights = self.graph_propagation(
            id_features, self.memory_bank
        )
        
        outputs = {
            'id_features': id_features,      # 用于 Triplet Loss, Orth Loss
            'mod_features': mod_features,    # 用于 Orth Loss
            'id_logits': id_logits,          # 用于 ID Loss
            'mod_logits': mod_logits,        # 用于 Modality Loss
            'soft_labels': soft_labels,      # 用于 Graph Distillation Loss
            'similarities': similarities,
            'entropy_weights': entropy_weights,
            'global_feature': global_feature
        }
        
        return outputs
    
    def extract_features(self, x, pool_parts=True):
        """
        测试阶段特征提取
        Args:
            pool_parts: 是否融合所有部件特征
        """
        with torch.no_grad():
            part_features, _ = self.backbone(x)
            id_features, _ = self.isg_dm(part_features)
            
            # L2 归一化
            if pool_parts:
                id_features_norm = [F.normalize(f, p=2, dim=1) for f in id_features]
                # 使用学习到的部件权重进行加权融合
                weights = F.softmax(self.part_weights, dim=0)
                weighted_features = [f * weights[i] for i, f in enumerate(id_features_norm)]
                features = torch.cat(weighted_features, dim=1)
                features = F.normalize(features, p=2, dim=1)
            else:
                features = [F.normalize(f, p=2, dim=1) for f in id_features]
            
            return features
    
    def initialize_memory(self, dataloader, device, teacher_model=None):
        """
        初始化记忆库
        
        功能：
        1. 遍历指定的数据加载器 (通常是 RGB 训练集)
        2. 提取特征 (使用 Student 自己或 Teacher 网络)
        3. 计算每个 ID 的特征中心并存入记忆库
        
        Args:
            teacher_model: 可选，如果提供，则使用教师网络提取高置信度特征
        """
        print("Initializing memory bank...")
        
        # 设置模式
        if teacher_model is not None:
            print("  [Init Strategy] Using Teacher Network (High Confidence).")
            teacher_model.eval()
        else:
            print("  [Init Strategy] Using Student Network (Self-Clustering).")
            self.eval()
        
        all_id_features = [[] for _ in range(self.num_parts)]
        all_labels = []
        
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(dataloader):
                # 解析 Batch 数据并推断模态
                # 这里主要处理 SYSU/RegDB 的数据格式差异
                modality_labels = None
                
                if len(batch_data) == 2:
                    # 格式: (images, info) - 通常来自 SYSU_train 的 __getitem__
                    images, info = batch_data
                    images = images.to(device)
                    
                    if isinstance(info, torch.Tensor):
                        labels = info[:, 1].long()
                        cam_ids = info[:, 2].long()
                    else:
                        labels = torch.from_numpy(info[:, 1]).long()
                        cam_ids = torch.from_numpy(info[:, 2]).long()
                    
                    # SYSU 推断: Cam 3, 6 为红外(1)，其余为可见光(0)
                    modality_labels = (cam_ids == 3) | (cam_ids == 6)
                    modality_labels = modality_labels.long().to(device)
                    
                elif len(batch_data) == 3:
                    # 格式: (images, labels, cams) - 通用格式
                    images, labels, cams = batch_data
                    images = images.to(device)
                    if not isinstance(labels, torch.Tensor):
                        labels = torch.tensor(labels).long()
                    
                    # 推断模态
                    modality_labels = ((cams == 3) | (cams == 6)).long().to(device)
                    
                else:
                    raise ValueError(f"Unexpected batch format: {len(batch_data)} elements")
                
                # 特征提取
                if teacher_model is not None:
                    # 使用教师网络 (需要传入模态标签以选择 T_vis 或 T_ir)
                    # 注意：如果是单模态Loader(如只传了RGB)，modality_labels全0，会自动只用T_vis
                    if hasattr(teacher_model, 'visible_teacher'): # 双模态教师
                        id_features = teacher_model(images, modality_labels)
                    else: # 单模态教师
                        id_features = teacher_model(images)
                else:
                    # 使用学生网络 (自己)
                    outputs = self.forward(images)
                    id_features = outputs['id_features']
                
                # 收集所有特征
                for k in range(self.num_parts):
                    all_id_features[k].append(id_features[k].cpu())
                all_labels.append(labels.cpu())
                
                if (batch_idx + 1) % 100 == 0:
                    print(f"  Processed {batch_idx + 1} batches...")
        
        # 拼接所有收集到的数据
        for k in range(self.num_parts):
            all_id_features[k] = torch.cat(all_id_features[k], dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        # 处理标签映射 (Original ID -> 0..N-1)
        unique_labels = torch.unique(all_labels)
        print(f"  Label Statistics:")
        print(f"    Range: [{unique_labels.min().item()}, {unique_labels.max().item()}]")
        print(f"    Unique IDs: {len(unique_labels)}")
        
        label_mapping = {old_id.item(): new_id for new_id, old_id in enumerate(unique_labels)}
        mapped_labels = torch.tensor([label_mapping[label.item()] for label in all_labels])
        
        # 执行初始化
        all_id_features_device = [f.to(device) for f in all_id_features]
        mapped_labels_device = mapped_labels.to(device)
        
        self.memory_bank.initialize_memory(all_id_features_device, mapped_labels_device)
        
        print(f"  Initialization Complete: {self.memory_bank.initialized.sum().item()} IDs updated.")
        
        self.train_label_mapping = label_mapping
        
        # 恢复训练模式
        self.train()