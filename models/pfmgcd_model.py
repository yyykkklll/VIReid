"""
models/pfmgcd_model.py - PF-MGCD Ultimate (Modular Architecture)

功能模块:
1. Backbone: IBN-Net + GeM Pooling (基础底座)
2. ISG-DM: 解耦模块
3. [策略三] Modality Adversarial: 梯度反转层 (GRL) + 模态判别器
4. [策略四] Graph Reasoning: 基于记忆库的 GCN 特征增强
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

from .pcb_backbone import PCBBackbone, GeMPooling
from .isg_dm import MultiPartISG_DM
from .memory_bank import MultiPartMemoryBank
from .graph_propagation import AdaptiveGraphPropagation

# ==================== 基础组件 ====================

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        nn.init.normal_(m.weight, 1.0, 0.01)
        nn.init.constant_(m.bias, 0.0)

class GradientReversalFunction(Function):
    """梯度反转层 (GRL) 的核心实现"""
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        # 反向传播时，梯度取反并乘以 lambda
        return grad_output.neg() * ctx.lambda_, None

class GradientReversal(nn.Module):
    def __init__(self, lambda_=1.0):
        super(GradientReversal, self).__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)

# ==================== 策略三：模态判别器 ====================

class ModalityDiscriminator(nn.Module):
    """
    模态判别器: 试图分辨特征是来自 RGB 还是 IR
    对抗目标: 提取器生成的特征让判别器分不清 (Prob -> 0.5)
    """
    def __init__(self, input_dim):
        super(ModalityDiscriminator, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.BatchNorm1d(input_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(input_dim // 2, 2)  # 2类: RGB vs IR
        )
        self.classifier.apply(weights_init_kaiming)

    def forward(self, x):
        return self.classifier(x)

# ==================== 策略四：图推理模块 ====================

class GraphReasoning(nn.Module):
    """
    图推理模块 (GCN on Memory)
    利用记忆库中的 Top-K 邻居特征来增强当前查询特征
    """
    def __init__(self, feature_dim, top_k=5):
        super(GraphReasoning, self).__init__()
        self.top_k = top_k
        self.feature_dim = feature_dim
        
        # GCN 权重
        self.gcn_weight = nn.Linear(feature_dim, feature_dim)
        self.relu = nn.ReLU(inplace=True)
        # 融合门控
        self.fusion = nn.Linear(feature_dim * 2, feature_dim)
        
        self._init_weights()

    def _init_weights(self):
        self.gcn_weight.apply(weights_init_kaiming)
        self.fusion.apply(weights_init_kaiming)

    def forward(self, part_feature, memory):
        """
        Args:
            part_feature: [B, D]
            memory: [N, D] 记忆库特征
        Returns:
            enhanced_feature: [B, D]
        """
        B, D = part_feature.size()
        
        # 1. 检索 Top-K 邻居
        # 归一化
        feat_norm = F.normalize(part_feature, p=2, dim=1)
        mem_norm = F.normalize(memory, p=2, dim=1)
        
        # 计算相似度 [B, N]
        sim = torch.mm(feat_norm, mem_norm.t())
        
        # 获取 Top-K [B, K]
        topk_val, topk_idx = torch.topk(sim, k=self.top_k, dim=1)
        
        # 2. 构建局部图特征
        # 收集邻居特征 [B, K, D]
        neighbor_feats = F.embedding(topk_idx, memory)
        
        # 3. 简化的 GCN 聚合
        # A_ij = softmax(sim_ij)
        affinity = F.softmax(topk_val * 10, dim=1).unsqueeze(2) # [B, K, 1]
        
        # Aggregation: sum(A * W * X_neighbor)
        weighted_neighbors = (neighbor_feats * affinity).sum(dim=1) # [B, D]
        gcn_out = self.relu(self.gcn_weight(weighted_neighbors))
        
        # 4. 残差融合
        # concat [original, gcn] -> fuse
        fused = torch.cat([part_feature, gcn_out], dim=1)
        out = self.fusion(fused)
        
        return out + part_feature # Residual connection

# ==================== 主模型 ====================

class PF_MGCD(nn.Module):
    def __init__(self, num_parts=6, num_identities=395, feature_dim=512,
                 memory_momentum=0.9, temperature=3.0, top_k=5, 
                 pretrained=True, backbone='resnet50', use_ibn=True,
                 # 新增开关参数
                 use_adversarial=False, use_graph_reasoning=False):
        super(PF_MGCD, self).__init__()
        self.num_parts = num_parts
        self.feature_dim = feature_dim
        self.use_adversarial = use_adversarial
        self.use_graph_reasoning = use_graph_reasoning
        
        # 1. Backbone
        self.backbone = PCBBackbone(
            num_parts=num_parts, 
            pretrained=pretrained, 
            backbone=backbone,
            use_ibn=use_ibn
        )
        
        # 2. ISG-DM
        self.isg_dm = MultiPartISG_DM(
            num_parts=num_parts,
            input_dim=2048,
            id_dim=feature_dim,
            mod_dim=feature_dim
        )
        
        # 3. GeM Pooling
        self.gem_poolings = nn.ModuleList([
            GeMPooling(p=3.0) for _ in range(num_parts)
        ])
        
        # --- 策略三：模态对抗 ---
        if self.use_adversarial:
            self.grl = GradientReversal(lambda_=0.1) # 梯度反转
            self.mod_discriminators = nn.ModuleList([
                ModalityDiscriminator(feature_dim) for _ in range(num_parts)
            ])
            print("✅ Modality Adversarial Learning Enabled.")
            
        # --- 策略四：图推理 ---
        if self.use_graph_reasoning:
            self.graph_reasoning_modules = nn.ModuleList([
                GraphReasoning(feature_dim, top_k=top_k) for _ in range(num_parts)
            ])
            print("✅ Graph Reasoning (GCN) Enabled.")

        # 4. BNNeck & Classifiers
        self.bottlenecks = nn.ModuleList([
            nn.BatchNorm1d(feature_dim) for _ in range(num_parts)
        ])
        self.id_classifiers = nn.ModuleList([
            nn.Linear(feature_dim, num_identities, bias=False) for _ in range(num_parts)
        ])
        
        self.bottlenecks.apply(weights_init_kaiming)
        self.id_classifiers.apply(weights_init_kaiming)
        self.dropout = nn.Dropout(p=0.5)
        
        # 5. Memory Bank
        # 如果使用图推理，必须启用记忆库
        self.memory_bank = MultiPartMemoryBank(
            num_parts=num_parts,
            num_identities=num_identities,
            feature_dim=feature_dim,
            momentum=memory_momentum
        )
        
        # 6. Graph Propagation (Loss)
        # 即使使用了 GCN，这个模块也可以保留用于计算 Soft Labels 损失
        self.graph_propagation = AdaptiveGraphPropagation(
            temperature=temperature,
            top_k=top_k,
            use_entropy_weight=True,
            scale=30.0
        )

    def forward(self, x, labels=None, **kwargs):
        # 1. Backbone
        output = self.backbone(x)
        if isinstance(output, tuple): part_features = output[0]
        else: part_features = output
        
        # 2. ISG-DM
        id_features_raw, mod_features = self.isg_dm(part_features)
        
        id_features = [] # 最终用于分类的特征
        adv_logits = []  # 对抗判别器输出
        
        # 3. 逐部件处理
        for k in range(self.num_parts):
            feat = id_features_raw[k] # [B, D]
            
            # [策略四] 图推理增强
            if self.use_graph_reasoning and self.memory_bank.initialized.sum() > 0:
                # 使用记忆库中的特征进行 GCN 更新
                # 注意：这里使用 detach 的 memory 防止梯度回传到 memory (它是 buffer)
                mem_k = self.memory_bank.get_part_memory(k).detach()
                feat = self.graph_reasoning_modules[k](feat, mem_k)
            
            # [策略三] 模态对抗
            if self.training and self.use_adversarial:
                # 梯度反转 -> 判别器
                feat_rev = self.grl(feat)
                mod_logit = self.mod_discriminators[k](feat_rev)
                adv_logits.append(mod_logit)
            
            id_features.append(feat)

        # 4. BNNeck + Classifier
        id_logits = []
        graph_logits = []
        
        for k in range(self.num_parts):
            feat_bn = self.bottlenecks[k](id_features[k])
            
            if self.training:
                graph_logits.append(self.id_classifiers[k](feat_bn))
                id_logits.append(self.id_classifiers[k](self.dropout(feat_bn)))
            else:
                logit = self.id_classifiers[k](feat_bn)
                id_logits.append(logit)
                graph_logits.append(logit)
        
        # 5. Graph Propagation (Loss Calculation)
        # 如果启用了图推理，这里的 id_features 已经是增强过的
        if self.training and self.memory_bank.initialized.sum() > 0:
            soft_labels, _, entropy_weights = self.graph_propagation(id_features, self.memory_bank)
        else:
            soft_labels, entropy_weights = None, None
        
        outputs = {
            'id_features': id_features,
            'id_logits': id_logits, 
            'graph_logits': graph_logits,
            'soft_labels': soft_labels,
            'entropy_weights': entropy_weights,
            'adv_logits': adv_logits if self.use_adversarial else None # 返回对抗Logits
        }
        return outputs
    
    def extract_features(self, x, pool_parts=True):
        with torch.no_grad():
            output = self.backbone(x)
            if isinstance(output, tuple): part_features = output[0]
            else: part_features = output
            id_features_raw, _ = self.isg_dm(part_features)
            
            bn_features = []
            for k in range(self.num_parts):
                feat = id_features_raw[k]
                # 测试时通常不开启 GCN 推理，以保持高效和稳定
                # 如果想极致性能，可以开启，但需要加载训练好的 Memory
                # 这里为了简单，只用 Backbone 特征
                feat_bn = self.bottlenecks[k](feat)
                bn_features.append(feat_bn)
            
            norm_features = [F.normalize(f, p=2, dim=1) for f in bn_features]
            
            if pool_parts:
                return torch.cat(norm_features, dim=1)
            else:
                return torch.stack(norm_features, dim=1).mean(dim=1)

    def initialize_memory(self, dataloader, device, teacher_model=None):
        self.eval()
        print("🔄 正在初始化记忆库...")
        all_features = [[] for _ in range(self.num_parts)]
        all_labels = []
        with torch.no_grad():
            for batch in dataloader:
                if len(batch) == 3: imgs, pids, _ = batch
                else: imgs, info = batch; pids = info[:, 1]
                imgs = imgs.to(device)
                
                output = self.backbone(imgs)
                if isinstance(output, tuple): part_features = output[0]
                else: part_features = output
                id_features_raw, _ = self.isg_dm(part_features)
                
                for k in range(self.num_parts):
                    all_features[k].append(id_features_raw[k].cpu())
                all_labels.append(pids)
        
        for k in range(self.num_parts):
            all_features[k] = torch.cat(all_features[k], dim=0).to(device)
        all_labels = torch.cat(all_labels, dim=0).long().to(device)
        self.memory_bank.initialize_memory(all_features, all_labels)
        self.train()
        print("✅ 记忆库初始化完成!")