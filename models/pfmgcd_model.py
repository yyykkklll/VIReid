"""
PF-MGCD Model (Full Version)
[集成]
1. Backbone: ResNet50
2. Decoupling: ISG-DM (提取纯净 ID 特征)
3. Interaction: Transformer (部件上下文交互)
4. Head: BNNeck + Dropout + Classifier
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .pcb_backbone import PCBBackbone
from .isg_dm import MultiPartISG_DM  # 恢复导入
from .memory_bank import MultiPartMemoryBank
from .graph_propagation import AdaptiveGraphPropagation

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        nn.init.normal_(m.weight, 1.0, 0.01)
        nn.init.constant_(m.bias, 0.0)

class PartContextTransformer(nn.Module):
    def __init__(self, feature_dim, nhead=8, num_layers=1, dropout=0.1):
        super(PartContextTransformer, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim, 
            nhead=nhead, 
            dim_feedforward=feature_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

    def forward(self, part_features_list):
        x = torch.stack(part_features_list, dim=1) # [B, K, D]
        x = self.transformer(x)
        return [x[:, i, :] for i in range(x.size(1))]

class PF_MGCD(nn.Module):
    def __init__(self, num_parts=6, num_identities=395, feature_dim=512, 
                 memory_momentum=0.9, temperature=3.0, top_k=5, **kwargs):
        super(PF_MGCD, self).__init__()
        self.num_parts = num_parts
        self.feature_dim = feature_dim
        
        # 1. Backbone
        self.backbone = PCBBackbone(num_parts, pretrained=True, backbone='resnet50')
        
        # 2. [恢复] ISG-DM 解耦模块
        # 输入 2048，输出 feature_dim (512)
        self.isg_dm = MultiPartISG_DM(num_parts, 2048, feature_dim, feature_dim)
        
        # 3. [保留] Transformer 上下文交互
        self.part_context = PartContextTransformer(feature_dim, nhead=8, num_layers=1)
        
        # 4. BNNeck & Dropout
        self.bottlenecks = nn.ModuleList([
            nn.BatchNorm1d(feature_dim) for _ in range(num_parts)
        ])
        self.bottlenecks.apply(weights_init_kaiming)
        self.dropout = nn.Dropout(p=0.5)
        
        # 5. Classifiers
        self.id_classifiers = nn.ModuleList([
            nn.Linear(feature_dim, num_identities, bias=False) for _ in range(num_parts)
        ])
        self.id_classifiers.apply(weights_init_kaiming)
        
        # 6. Memory & Graph
        self.memory_bank = MultiPartMemoryBank(num_parts, num_identities, feature_dim, memory_momentum)
        self.graph_propagation = AdaptiveGraphPropagation(temperature, top_k, scale=30.0)
        
        self.part_weights = nn.Parameter(torch.ones(num_parts))

    def forward(self, x, labels=None, **kwargs):
        # 1. Backbone
        part_features, _ = self.backbone(x) # List of [B, 2048, H, W]
        
        # 2. ISG-DM 解耦
        # id_features_raw: 去除了风格的纯内容特征
        # mod_features: 风格/模态特征 (可用于 Orthogonal Loss)
        id_features_raw, mod_features = self.isg_dm(part_features)
        
        # 3. Transformer 交互 (增强特征)
        id_features = self.part_context(id_features_raw)
        
        # 4. 后续流程
        id_logits = []
        graph_logits = []
        
        for k in range(self.num_parts):
            feat_bn = self.bottlenecks[k](id_features[k])
            
            if self.training:
                # Clean -> Graph Loss
                logit_clean = self.id_classifiers[k](feat_bn)
                graph_logits.append(logit_clean)
                
                # Drop -> ID Loss
                feat_drop = self.dropout(feat_bn)
                logit_drop = self.id_classifiers[k](feat_drop)
                id_logits.append(logit_drop)
            else:
                logit = self.id_classifiers[k](feat_bn)
                id_logits.append(logit)
                graph_logits.append(logit)
        
        # Graph Propagation
        soft_labels, _, _ = self.graph_propagation(id_features, self.memory_bank)
            
        outputs = {
            'id_features': id_features, 
            'mod_features': mod_features, # 保留接口
            'id_logits': id_logits,
            'graph_logits': graph_logits,
            'soft_labels': soft_labels,
        }
        return outputs

    def extract_features(self, x, pool_parts=True):
        with torch.no_grad():
            part_features, _ = self.backbone(x)
            
            # 1. ISG-DM
            id_features_raw, _ = self.isg_dm(part_features)
            
            # 2. Transformer
            id_features = self.part_context(id_features_raw)
            
            # 3. BNNeck
            bn_features = []
            for k in range(self.num_parts):
                feat_bn = self.bottlenecks[k](id_features[k])
                bn_features.append(feat_bn)
            
            norm_features = [F.normalize(f, p=2, dim=1) for f in bn_features]
            return torch.cat(norm_features, dim=1)

    def initialize_memory(self, dataloader, device, teacher_model=None):
        self.eval()
        print("Initializing memory bank...")
        all_features = [[] for _ in range(self.num_parts)]
        all_labels = []
        with torch.no_grad():
            for batch in dataloader:
                if len(batch) == 3: imgs, pids, _ = batch
                else: imgs, info = batch; pids = info[:, 1]
                imgs = imgs.to(device)
                
                # 完整 Forward 流程提取特征
                part_features, _ = self.backbone(imgs)
                id_features_raw, _ = self.isg_dm(part_features)
                id_features = self.part_context(id_features_raw)
                
                for k in range(self.num_parts):
                    all_features[k].append(id_features[k].cpu())
                all_labels.append(pids)
                
        for k in range(self.num_parts):
            all_features[k] = torch.cat(all_features[k], dim=0).to(device)
        all_labels = torch.cat(all_labels, dim=0).long().to(device)
        self.memory_bank.initialize_memory(all_features, all_labels)
        self.train()