"""
models/memory_bank.py - 多粒度记忆库模块 (完整修复版)

修复日志:
1. [P0] 添加update_memory()中的label范围检查
2. 优化initialize_memory()的错误提示信息
3. 添加详细的中文注释

功能:
维护一个三维记忆库 M ∈ R^(K × N × D)
- K: 部件数量
- N: 身份数量  
- D: 特征维度
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiPartMemoryBank(nn.Module):
    """
    多粒度张量记忆库
    
    存储结构:
    - memory: Tensor[K, N, D]，每个部件-身份组合的特征原型
    - initialized: Tensor[N]，标记每个身份是否已初始化
    
    更新策略:
    动量更新: M[k, y] ← m * M[k, y] + (1-m) * f_new
    """
    def __init__(self, num_parts, num_identities, feature_dim, momentum=0.9):
        """
        Args:
            num_parts: 部件数量 K
            num_identities: 身份数量 N
            feature_dim: 特征维度 D
            momentum: 动量更新系数 m，范围[0, 1]
        """
        super(MultiPartMemoryBank, self).__init__()
        self.num_parts = num_parts
        self.num_identities = num_identities
        self.feature_dim = feature_dim
        self.momentum = momentum
        
        # 初始化记忆库 [K, N, D]
        # 使用register_buffer使其作为模型状态保存，但不参与梯度更新
        self.register_buffer(
            'memory',
            F.normalize(torch.randn(num_parts, num_identities, feature_dim), dim=2)
        )
        
        # 记录每个ID是否已初始化（首次出现）
        self.register_buffer(
            'initialized',
            torch.zeros(num_identities, dtype=torch.bool)
        )
    
    @torch.no_grad()
    def initialize_memory(self, part_features, labels):
        """
        使用初始数据批量初始化记忆库
        通常在训练开始前用整个训练集的RGB模态数据初始化
        
        Args:
            part_features: List[Tensor[B, D]]，K个部件特征
            labels: Tensor[B]，样本标签
        """
        K = len(part_features)
        assert K == self.num_parts, f"Expected {self.num_parts} parts, got {K}"
        
        # [修复] 检查label范围，提供详细错误信息
        unique_labels = labels.unique()
        max_label = unique_labels.max().item()
        min_label = unique_labels.min().item()
        
        print(f"  📊 Label range in batch: [{min_label}, {max_label}]")
        print(f"  📊 Memory bank size: [K={K}, N={self.num_identities}, D={self.feature_dim}]")
        
        # 如果label超出范围，抛出详细错误
        if max_label >= self.num_identities:
            raise ValueError(
                f"\n{'='*60}\n"
                f"❌ ERROR: Label out of range!\n"
                f"   Max label in data: {max_label}\n"
                f"   Memory bank size: {self.num_identities}\n"
                f"   This error occurs because the dataloader did not properly\n"
                f"   map the original IDs to continuous indices [0, N-1].\n"
                f"   Please check 'dataloader_adapter.py' for label mapping.\n"
                f"{'='*60}"
            )
        
        # 对每个部件分别初始化
        for k in range(K):
            features = part_features[k]  # [B, D]
            features = F.normalize(features, dim=1)  # L2归一化
            
            # 按标签聚合特征（取同ID样本的均值）
            for label in unique_labels:
                label_idx = label.item()
                mask = (labels == label)  # 当前ID的样本mask
                
                if mask.sum() > 0:
                    # 计算该ID所有样本的平均特征
                    mean_feature = features[mask].mean(dim=0)  # [D]
                    
                    # 归一化后存入记忆库（使用clone避免inplace警告）
                    normalized_feature = F.normalize(mean_feature.unsqueeze(0), dim=1).squeeze(0)
                    self.memory[k, label_idx] = normalized_feature.clone()
                    self.initialized[label_idx] = True
        
        num_initialized = self.initialized.sum().item()
        print(f"  ✅ Initialized: {num_initialized}/{self.num_identities} identities")
    
    @torch.no_grad()
    def update_memory(self, part_features, labels):
        """
        动量更新记忆库
        在每个训练batch后调用，更新当前batch涉及的ID的记忆
        
        公式: M[k, y] ← m * M[k, y] + (1-m) * f_id^(k)
        
        Args:
            part_features: List[Tensor[B, D]]，K个部件特征
            labels: Tensor[B]，样本标签
        """
        K = len(part_features)
        m = self.momentum
        
        # [修复] 检查label范围，防止IndexError
        max_label = labels.max().item()
        if max_label >= self.num_identities:
            raise ValueError(
                f"❌ Label {max_label} exceeds memory bank size {self.num_identities}. "
                f"Check your dataloader label mapping!"
            )
        
        # 创建新的记忆库副本，避免inplace操作导致的梯度问题
        new_memory = self.memory.clone()
        
        for k in range(K):
            features = part_features[k]  # [B, D]
            features = F.normalize(features, dim=1)  # L2归一化
            
            # 更新每个样本对应的记忆
            for i, label in enumerate(labels):
                label_idx = label.item()
                old_memory = self.memory[k, label_idx]  # [D] 旧记忆
                new_feature = features[i]  # [D] 新特征
                
                # 动量更新
                if self.initialized[label_idx]:
                    # 已初始化，使用动量融合
                    updated = m * old_memory + (1 - m) * new_feature
                else:
                    # 首次出现，直接赋值
                    updated = new_feature
                    self.initialized[label_idx] = True
                
                # 归一化并更新到新记忆库
                new_memory[k, label_idx] = F.normalize(updated.unsqueeze(0), dim=1).squeeze(0)
        
        # 一次性更新整个记忆库（避免多次写入）
        self.memory.copy_(new_memory)
    
    def get_memory(self, part_idx=None):
        """
        获取记忆库
        
        Args:
            part_idx: 部件索引 (0 ~ K-1)，None表示返回全部
        Returns:
            memory: Tensor[N, D] or Tensor[K, N, D]
        """
        if part_idx is not None:
            return self.memory[part_idx]  # [N, D]
        return self.memory  # [K, N, D]
    
    def get_part_memory(self, part_idx):
        """
        获取指定部件的记忆库
        
        Args:
            part_idx: 部件索引 (0 ~ K-1)
        Returns:
            memory: Tensor[N, D]
        """
        return self.memory[part_idx]
    
    def forward(self, part_features):
        """
        计算特征与记忆库的余弦相似度
        
        Args:
            part_features: List[Tensor[B, D]]，K个部件特征
        Returns:
            similarities: List[Tensor[B, N]]，K个相似度矩阵
        """
        similarities = []
        for k in range(self.num_parts):
            features = F.normalize(part_features[k], dim=1)  # [B, D]
            memory = self.memory[k]  # [N, D]
            
            # 余弦相似度: sim(f, m) = f · m^T
            sim = torch.mm(features, memory.t())  # [B, N]
            similarities.append(sim)
        
        return similarities


class AdaptiveMemoryBank(MultiPartMemoryBank):
    """
    自适应记忆库 (高级版本)
    
    在基础记忆库上增加了置信度过滤机制:
    - 低置信度样本不参与记忆库更新
    - 维护每个ID的全局置信度分数
    """
    def __init__(self, num_parts, num_identities, feature_dim,
                 momentum=0.9, confidence_threshold=0.5):
        """
        Args:
            confidence_threshold: 置信度阈值，低于此值的样本被过滤
        """
        super().__init__(num_parts, num_identities, feature_dim, momentum)
        self.confidence_threshold = confidence_threshold
        
        # 记录每个ID的置信度
        self.register_buffer(
            'confidence',
            torch.zeros(num_identities)
        )
    
    @torch.no_grad()
    def update_memory(self, part_features, labels, confidences=None):
        """
        带置信度的动量更新
        
        Args:
            part_features: List[Tensor[B, D]]，K个部件特征
            labels: Tensor[B]，样本标签
            confidences: Tensor[B] (可选)，置信度分数
        """
        K = len(part_features)
        m = self.momentum
        
        # 创建新的记忆库和置信度副本
        new_memory = self.memory.clone()
        new_confidence = self.confidence.clone()
        
        for k in range(K):
            features = part_features[k]  # [B, D]
            features = F.normalize(features, dim=1)
            
            for i, label in enumerate(labels):
                label_idx = label.item()
                
                # 检查置信度（如果提供）
                if confidences is not None:
                    conf = confidences[i].item()
                    if conf < self.confidence_threshold:
                        continue  # 跳过低置信度样本
                    
                    # 更新全局置信度（动量平均）
                    new_confidence[label_idx] = m * self.confidence[label_idx] + (1 - m) * conf
                
                # 更新记忆
                old_memory = self.memory[k, label_idx]
                new_feature = features[i]
                
                if self.initialized[label_idx]:
                    updated = m * old_memory + (1 - m) * new_feature
                else:
                    updated = new_feature
                    self.initialized[label_idx] = True
                
                new_memory[k, label_idx] = F.normalize(updated.unsqueeze(0), dim=1).squeeze(0)
        
        # 一次性更新
        self.memory.copy_(new_memory)
        self.confidence.copy_(new_confidence)
