"""
models/pfmgcd_model.py - PF-MGCD主模型 (完整修复版)

修复日志:
1. [P2] 修复extract_features中pool_parts参数未生效的问题
2. 优化记忆库恢复检查逻辑
3. 添加详细的中文注释

模型架构:
1. PCB Backbone (ResNet50) - 人体部件切分
2. ISG-DM - 身份/模态解耦
3. Transformer - 部件上下文交互
4. BNNeck - 归一化瓶颈层
5. Classifier - 身份分类器
6. Memory Bank - 多粒度记忆库
7. Graph Propagation - 图传播模块
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .pcb_backbone import PCBBackbone
from .isg_dm import MultiPartISG_DM
from .memory_bank import MultiPartMemoryBank
from .graph_propagation import AdaptiveGraphPropagation


def weights_init_kaiming(m):
    """
    Kaiming初始化（He初始化）
    适用于ReLU激活函数的网络
    """
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        nn.init.normal_(m.weight, 1.0, 0.01)
        nn.init.constant_(m.bias, 0.0)


class PartContextTransformer(nn.Module):
    """
    部件上下文交互模块
    
    使用Transformer Encoder捕获部件间的空间关系
    例如：头部特征可以帮助上身特征的判别
    """
    def __init__(self, feature_dim, nhead=8, num_layers=1, dropout=0.1):
        """
        Args:
            feature_dim: 特征维度D
            nhead: 多头注意力的头数
            num_layers: Transformer层数
            dropout: Dropout概率
        """
        super(PartContextTransformer, self).__init__()
        
        # Transformer编码器层
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=nhead,
            dim_feedforward=feature_dim * 4,  # FFN隐层维度
            dropout=dropout,
            batch_first=True  # 输入格式为[B, K, D]
        )
        
        # 堆叠多层Transformer
        self.transformer = nn.TransformerEncoder(
            self.encoder_layer, 
            num_layers=num_layers
        )
    
    def forward(self, part_features_list):
        """
        Args:
            part_features_list: List[Tensor[B, D]]，K个部件特征
        Returns:
            enhanced_features: List[Tensor[B, D]]，交互增强后的特征
        """
        # 将List转换为Tensor: [B, K, D]
        x = torch.stack(part_features_list, dim=1)
        
        # Transformer编码
        x = self.transformer(x)  # [B, K, D]
        
        # 转回List格式
        return [x[:, i, :] for i in range(x.size(1))]


class PF_MGCD(nn.Module):
    """
    PF-MGCD主模型
    
    Part-Based Fine-Grained Multi-Granularity Cross-Modal Distillation
    for Visible-Infrared Person Re-Identification
    """
    def __init__(self, num_parts=6, num_identities=395, feature_dim=512,
                 memory_momentum=0.9, temperature=3.0, top_k=5, 
                 pretrained=True, backbone='resnet50'):
        """
        Args:
            num_parts: 部件数量K（默认6：头-上身-下身-腿部等）
            num_identities: 身份数量N（训练集ID数）
            feature_dim: 解耦后的特征维度D
            memory_momentum: 记忆库动量系数
            temperature: 图传播温度
            top_k: Top-K邻居数
            pretrained: 是否使用ImageNet预训练权重
            backbone: 骨干网络类型
        """
        super(PF_MGCD, self).__init__()
        self.num_parts = num_parts
        self.feature_dim = feature_dim
        
        # 1. PCB Backbone - 人体部件切分
        # 输入: [B, 3, 288, 144] -> 输出: K个 [B, 2048, H/K, W]
        self.backbone = PCBBackbone(
            num_parts=num_parts, 
            pretrained=pretrained, 
            backbone=backbone
        )
        
        # 2. ISG-DM 解耦模块
        # 输入: [B, 2048, H, W] -> 输出: ([B, D], [B, D_mod])
        # 分离身份特征和模态特征
        self.isg_dm = MultiPartISG_DM(
            num_parts=num_parts,
            input_dim=2048,       # ResNet50的输出通道数
            id_dim=feature_dim,   # 身份特征维度
            mod_dim=feature_dim   # 模态特征维度
        )
        
        # 3. Transformer 部件上下文交互
        self.part_context = PartContextTransformer(
            feature_dim=feature_dim,
            nhead=8,
            num_layers=1
        )
        
        # 4. BNNeck - Batch Normalization Neck
        # 用于特征归一化，提升度量学习性能
        self.bottlenecks = nn.ModuleList([
            nn.BatchNorm1d(feature_dim) for _ in range(num_parts)
        ])
        self.bottlenecks.apply(weights_init_kaiming)
        
        # Dropout - 训练时随机失活，增强泛化
        self.dropout = nn.Dropout(p=0.5)
        
        # 5. 分类器 - 每个部件独立的ID分类头
        self.id_classifiers = nn.ModuleList([
            nn.Linear(feature_dim, num_identities, bias=False) 
            for _ in range(num_parts)
        ])
        self.id_classifiers.apply(weights_init_kaiming)
        
        # 6. 多粒度记忆库
        self.memory_bank = MultiPartMemoryBank(
            num_parts=num_parts,
            num_identities=num_identities,
            feature_dim=feature_dim,
            momentum=memory_momentum
        )
        
        # 7. 自适应图传播
        self.graph_propagation = AdaptiveGraphPropagation(
            temperature=temperature,
            top_k=top_k,
            use_entropy_weight=True,
            scale=30.0
        )
        
        # 部件权重（可学习参数，用于加权融合）
        self.part_weights = nn.Parameter(torch.ones(num_parts))
    
    def forward(self, x, labels=None, **kwargs):
        """
        前向传播
        
        Args:
            x: Tensor[B, 3, H, W]，输入图像
            labels: Tensor[B]，Ground Truth标签（训练时需要）
        
        Returns:
            outputs: Dict，包含:
                - 'id_features': List[Tensor[B, D]]，K个部件的身份特征
                - 'mod_features': List[Tensor[B, D]]，K个部件的模态特征
                - 'id_logits': List[Tensor[B, N]]，带Dropout的分类logits
                - 'graph_logits': List[Tensor[B, N]]，不带Dropout的分类logits
                - 'soft_labels': List[Tensor[B, N]]，图传播生成的软标签
                - 'entropy_weights': List[Tensor[B]]，熵权重
        """
        # 1. Backbone提取部件特征
        part_features, _ = self.backbone(x)  # List of [B, 2048, H, W]
        
        # 2. ISG-DM 身份/模态解耦
        id_features_raw, mod_features = self.isg_dm(part_features)
        # id_features_raw: List of [B, D] - 纯身份特征
        # mod_features: List of [B, D] - 模态/风格特征
        
        # 3. Transformer 部件上下文交互
        id_features = self.part_context(id_features_raw)
        # id_features: List of [B, D] - 交互增强后的特征
        
        # 4. 后续流程：BNNeck + Dropout + Classifier
        id_logits = []       # 带Dropout的logits（用于ID Loss）
        graph_logits = []    # 不带Dropout的logits（用于Graph Loss）
        
        for k in range(self.num_parts):
            # BNNeck归一化
            feat_bn = self.bottlenecks[k](id_features[k])  # [B, D]
            
            if self.training:
                # 训练模式：分别计算两种logits
                
                # Clean Logits -> Graph Loss（蒸馏需要稳定预测）
                logit_clean = self.id_classifiers[k](feat_bn)
                graph_logits.append(logit_clean)
                
                # Dropout Logits -> ID Loss（增强泛化）
                feat_drop = self.dropout(feat_bn)
                logit_drop = self.id_classifiers[k](feat_drop)
                id_logits.append(logit_drop)
            else:
                # 测试模式：不使用Dropout
                logit = self.id_classifiers[k](feat_bn)
                id_logits.append(logit)
                graph_logits.append(logit)
        
        # 5. 图传播生成软标签
        soft_labels, similarities, entropy_weights = self.graph_propagation(
            id_features, 
            self.memory_bank
        )
        
        # 返回所有输出
        outputs = {
            'id_features': id_features,         # 用于Triplet Loss和记忆库更新
            'mod_features': mod_features,       # 保留接口（可用于模态判别）
            'id_logits': id_logits,             # 用于ID Loss
            'graph_logits': graph_logits,       # 用于Graph Distillation Loss
            'soft_labels': soft_labels,         # 软标签
            'entropy_weights': entropy_weights  # 熵权重
        }
        
        return outputs
    
    def extract_features(self, x, pool_parts=True):
        """
        特征提取（测试阶段）
        
        Args:
            x: Tensor[B, 3, H, W]，输入图像
            pool_parts: bool，是否拼接所有部件特征
                       True: 返回 [B, K*D] 拼接特征
                       False: 返回 [B, D] 平均特征
        
        Returns:
            features: Tensor[B, K*D] or Tensor[B, D]
        """
        with torch.no_grad():
            # 1. 前向传播提取特征
            part_features, _ = self.backbone(x)
            id_features_raw, _ = self.isg_dm(part_features)
            id_features = self.part_context(id_features_raw)
            
            # 2. BNNeck归一化
            bn_features = []
            for k in range(self.num_parts):
                feat_bn = self.bottlenecks[k](id_features[k])  # [B, D]
                bn_features.append(feat_bn)
            
            # 3. L2归一化（余弦距离度量）
            norm_features = [F.normalize(f, p=2, dim=1) for f in bn_features]
            
            # 4. [修复] 根据pool_parts参数选择输出方式
            if pool_parts:
                # 拼接所有部件特征 [B, K*D]
                return torch.cat(norm_features, dim=1)
            else:
                # 取所有部件的平均 [B, D]
                stacked = torch.stack(norm_features, dim=1)  # [B, K, D]
                return stacked.mean(dim=1)  # [B, D]
    
    def initialize_memory(self, dataloader, device, teacher_model=None):
        """
        初始化记忆库
        
        通常在训练开始前调用，使用整个训练集的RGB模态数据
        批量计算所有样本的特征并存入记忆库
        
        Args:
            dataloader: DataLoader，训练数据加载器
            device: torch.device，计算设备
            teacher_model: 教师模型（可选，暂未使用）
        """
        self.eval()  # 切换到评估模式
        print("🔄 正在初始化记忆库...")
        
        # 收集所有特征和标签
        all_features = [[] for _ in range(self.num_parts)]
        all_labels = []
        
        with torch.no_grad():
            for batch in dataloader:
                # 数据解包（兼容多种格式）
                if len(batch) == 3:
                    imgs, pids, _ = batch
                else:
                    imgs, info = batch
                    pids = info[:, 1]
                
                imgs = imgs.to(device)
                
                # 完整前向传播提取特征
                part_features, _ = self.backbone(imgs)
                id_features_raw, _ = self.isg_dm(part_features)
                id_features = self.part_context(id_features_raw)
                
                # 收集每个部件的特征
                for k in range(self.num_parts):
                    all_features[k].append(id_features[k].cpu())
                all_labels.append(pids)
        
        # 拼接所有batch
        for k in range(self.num_parts):
            all_features[k] = torch.cat(all_features[k], dim=0).to(device)
        all_labels = torch.cat(all_labels, dim=0).long().to(device)
        
        # 批量初始化记忆库
        self.memory_bank.initialize_memory(all_features, all_labels)
        
        self.train()  # 恢复训练模式
        print("✅ 记忆库初始化完成!")


# ===== 测试代码 =====
if __name__ == "__main__":
    print("="*60)
    print("PF-MGCD 模型测试")
    print("="*60)
    
    # 创建模型
    model = PF_MGCD(
        num_parts=6,
        num_identities=395,
        feature_dim=256,
        pretrained=False  # 测试时不加载预训练权重
    )
    
    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"模型参数量: {total_params:.2f}M")
    
    # 测试前向传播
    batch_size = 4
    x = torch.randn(batch_size, 3, 288, 144)
    labels = torch.randint(0, 395, (batch_size,))
    
    print(f"\n输入shape: {x.shape}")
    
    # 训练模式
    model.train()
    outputs = model(x, labels=labels)
    
    print(f"\n输出字典keys: {outputs.keys()}")
    print(f"id_features数量: {len(outputs['id_features'])}")
    print(f"每个特征shape: {outputs['id_features'][0].shape}")
    print(f"id_logits shape: {outputs['id_logits'][0].shape}")
    print(f"soft_labels shape: {outputs['soft_labels'][0].shape}")
    
    # 测试模式
    model.eval()
    feat_concat = model.extract_features(x, pool_parts=True)
    feat_avg = model.extract_features(x, pool_parts=False)
    
    print(f"\n拼接特征shape: {feat_concat.shape}")
    print(f"平均特征shape: {feat_avg.shape}")
    
    print("\n" + "="*60)
    print("所有测试通过! ✅")
    print("="*60)
