"""
PF-MGCD Model v5 (Stable Baseline)
[关键修复]
1. 分类器启用 Bias=True (配合 CrossEntropyLoss)
2. 保留 BNNeck 结构 (Tri-Loss与ID-Loss特征分离)
3. 集成 Bi-Mamba 上下文融合
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# 尝试导入 Mamba
try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False

# 尝试导入子模块
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


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        nn.init.normal_(m.weight, 1.0, 0.01)
        nn.init.constant_(m.bias, 0.0)


class PartContextMamba(nn.Module):
    """双向 Mamba 部件上下文融合模块"""
    def __init__(self, dim, d_state=16, d_conv=4, expand=2, dropout=0.1):
        super(PartContextMamba, self).__init__()
        if not MAMBA_AVAILABLE:
            # 如果没有 Mamba，此模块退化为 Identity，不报错以便调试
            print("Warning: Mamba not installed, skipping context fusion.")
            self.mamba_ok = False
        else:
            self.mamba_ok = True
            self.fwd_mamba = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)
            self.bwd_mamba = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)
            self.norm = nn.LayerNorm(dim)
            self.dropout = nn.Dropout(dropout)

    def forward(self, part_features_list):
        if not self.mamba_ok:
            return part_features_list
            
        x = torch.stack(part_features_list, dim=1) # [B, K, D]
        x_fwd = self.fwd_mamba(x)
        x_bwd = self.bwd_mamba(x.flip(1)).flip(1)
        # 残差 + Dropout + Norm
        x_enhanced = self.norm(x + self.dropout(x_fwd + x_bwd))
        return [x_enhanced[:, i, :] for i in range(x.size(1))]


class PF_MGCD(nn.Module):
    def __init__(self, num_parts=6, num_identities=395, feature_dim=256,
                 memory_momentum=0.9, temperature=3.0, top_k=5,
                 pretrained=True, backbone='resnet50'):
        super(PF_MGCD, self).__init__()
        self.num_parts = num_parts
        self.feature_dim = feature_dim
        
        # 1. Backbone
        self.backbone = PCBBackbone(num_parts, pretrained, backbone)
        
        # 2. ISG-DM (无 BN)
        self.isg_dm = MultiPartISG_DM(num_parts, 2048, feature_dim, feature_dim)
        
        # 3. Bi-Mamba (Dropout=0.2 防止 RegDB 过拟合)
        self.part_mamba = PartContextMamba(feature_dim, dropout=0.2)
        
        # 4. BNNeck 层 (Bottlenecks)
        # 将特征归一化后再送入分类器
        self.bottlenecks = nn.ModuleList([
            nn.BatchNorm1d(feature_dim) for _ in range(num_parts)
        ])
        self.bottlenecks.apply(weights_init_kaiming)
        
        # 5. Classifiers
        # [关键修复] bias=True (配合标准 CrossEntropyLoss)
        self.id_classifiers = nn.ModuleList([
            nn.Linear(feature_dim, num_identities, bias=True) for _ in range(num_parts)
        ])
        self.id_classifiers.apply(weights_init_kaiming)
        
        self.mod_classifiers = nn.ModuleList([
            nn.Linear(feature_dim, 2, bias=True) for _ in range(num_parts)
        ])
        
        # 6. Memory & Graph
        self.memory_bank = MultiPartMemoryBank(num_parts, num_identities, feature_dim, memory_momentum)
        self.graph_propagation = AdaptiveGraphPropagation(temperature, top_k)
        
        self.part_weights = nn.Parameter(torch.ones(num_parts))
        self.register_buffer('temperature_scale', torch.tensor(1.0))

    def forward(self, x, labels=None, modality_labels=None, update_memory=False):
        # 1. 提取基础特征
        part_features, global_feature = self.backbone(x)
        id_features_raw, mod_features = self.isg_dm(part_features)
        
        # 2. Mamba 上下文增强
        # 得到的 id_features 是 BN 之前的特征 (Pre-BN)
        id_features = self.part_mamba(id_features_raw)
        
        # 3. BNNeck & 分类
        id_logits = []
        bn_features = [] # BN 之后的特征
        
        for k in range(self.num_parts):
            # 通过 BNNeck
            f_bn = self.bottlenecks[k](id_features[k])
            bn_features.append(f_bn)
            
            # 分类器使用 BN 后的特征
            logit = self.id_classifiers[k](f_bn)
            id_logits.append(logit)
        
        mod_logits = []
        for k in range(self.num_parts):
            logit = self.mod_classifiers[k](mod_features[k])
            mod_logits.append(logit)
        
        # 4. 图传播
        # 使用 Pre-BN 特征查询记忆库，保持几何结构一致性
        soft_labels, similarities, entropy_weights = self.graph_propagation(
            id_features, self.memory_bank
        )
        
        outputs = {
            'id_features': id_features,      # [BN 前] -> Triplet Loss & Memory
            'bn_features': bn_features,      # [BN 后] -> (备用)
            'mod_features': mod_features,
            'id_logits': id_logits,          # [BN 后] -> ID Loss
            'mod_logits': mod_logits,
            'soft_labels': soft_labels,
            'similarities': similarities,
            'entropy_weights': entropy_weights,
            'global_feature': global_feature
        }
        
        return outputs
    
    def extract_features(self, x, pool_parts=True):
        """测试阶段特征提取"""
        with torch.no_grad():
            part_features, _ = self.backbone(x)
            id_features_raw, _ = self.isg_dm(part_features)
            
            # Mamba 增强
            id_features = self.part_mamba(id_features_raw)
            
            # 通过 BNNeck (测试时必须用 BN 后的特征，这是 BNNeck 的标准用法)
            bn_features = []
            for k in range(self.num_parts):
                f_bn = self.bottlenecks[k](id_features[k])
                bn_features.append(f_bn)
            
            # L2 归一化并融合
            if pool_parts:
                norm_features = [F.normalize(f, p=2, dim=1) for f in bn_features]
                weights = F.softmax(self.part_weights, dim=0)
                weighted = [f * weights[i] for i, f in enumerate(norm_features)]
                final = torch.cat(weighted, dim=1)
                final = F.normalize(final, p=2, dim=1)
            else:
                final = [F.normalize(f, p=2, dim=1) for f in bn_features]
            
            return final
    
    def initialize_memory(self, dataloader, device, teacher_model=None):
        """初始化记忆库"""
        self.eval()
        if teacher_model: teacher_model.eval()
        
        all_features = [[] for _ in range(self.num_parts)]
        all_labels = []
        
        print("Initializing memory (using pre-BN features)...")
        with torch.no_grad():
            for batch in dataloader:
                # 适配不同的 dataloader 返回格式
                if len(batch) == 4:
                    imgs, _, pids, cams = batch
                elif len(batch) == 3:
                    imgs, pids, cams = batch
                else:
                    imgs, info = batch
                    pids = info[:, 1]
                
                imgs = imgs.to(device)
                
                if teacher_model:
                    # Teacher 返回 dict, 取 id_features (Pre-BN)
                    out = teacher_model(imgs)
                    feats = out['id_features']
                else:
                    out = self.forward(imgs)
                    feats = out['id_features']
                
                for k in range(self.num_parts):
                    all_features[k].append(feats[k].cpu())
                all_labels.append(pids)
                
        # 拼接与映射
        for k in range(self.num_parts):
            all_features[k] = torch.cat(all_features[k], dim=0).to(device)
        all_labels = torch.cat(all_labels, dim=0).long().to(device)
        
        uq_labels = torch.unique(all_labels)
        mapping = {old.item(): new for new, old in enumerate(uq_labels)}
        mapped_labels = torch.tensor([mapping[x.item()] for x in all_labels]).to(device)
        
        self.memory_bank.initialize_memory(all_features, mapped_labels)
        print(f"Initialized {len(uq_labels)} identities.")
        self.train_label_mapping = mapping
        self.train()