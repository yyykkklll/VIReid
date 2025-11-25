"""
Memory Bank - Multi-Part Modality-Agnostic Memory
多粒度模态无关记忆库: 存储K个部件 × N个身份的纯净原型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiPartMemoryBank(nn.Module):
    """
    多粒度张量记忆库
    维护一个三维张量 M ∈ R^(K × N × D)
    - K: 部件数量
    - N: 身份数量
    - D: 特征维度
    """
    
    def __init__(self, num_parts, num_identities, feature_dim, momentum=0.9):
        """
        Args:
            num_parts: 部件数量 K
            num_identities: 身份数量 N
            feature_dim: 特征维度 D
            momentum: 动量更新系数 m
        """
        super(MultiPartMemoryBank, self).__init__()
        self.num_parts = num_parts
        self.num_identities = num_identities
        self.feature_dim = feature_dim
        self.momentum = momentum
        
        # 初始化记忆库 [K, N, D]
        # 使用register_buffer使其不被当作参数训练，但会保存到state_dict
        self.register_buffer(
            'memory',
            F.normalize(torch.randn(num_parts, num_identities, feature_dim), dim=2)
        )
        
        # 记录每个ID是否已初始化
        self.register_buffer(
            'initialized',
            torch.zeros(num_identities, dtype=torch.bool)
        )
    
    @torch.no_grad()
    def initialize_memory(self, part_features, labels):
        """
        使用教师网络或初始特征初始化记忆库
        Args:
            part_features: List of K个部件特征 [B, D]
            labels: 样本标签 [B]
        """
        K = len(part_features)
        assert K == self.num_parts, f"Expected {self.num_parts} parts, got {K}"
        
        for k in range(K):
            features = part_features[k]  # [B, D]
            features = F.normalize(features, dim=1)
            
            # 按标签聚合特征 (取平均)
            for label in labels.unique():
                mask = (labels == label)
                if mask.sum() > 0:
                    mean_feature = features[mask].mean(dim=0)  # [D]
                    # 使用 clone() 避免 inplace 操作
                    normalized_feature = F.normalize(mean_feature.unsqueeze(0), dim=1).squeeze(0)
                    self.memory[k, label] = normalized_feature.clone()
                    self.initialized[label] = True
    
    @torch.no_grad()
    def update_memory(self, part_features, labels):
        """
        动量更新记忆库
        M[k, y] ← m * M[k, y] + (1-m) * f_id^(k)
        
        Args:
            part_features: List of K个部件特征 [B, D]
            labels: 样本标签 [B]
        """
        K = len(part_features)
        m = self.momentum
        
        # 创建新的记忆库副本，避免 inplace 操作
        new_memory = self.memory.clone()
        
        for k in range(K):
            features = part_features[k]  # [B, D]
            features = F.normalize(features, dim=1)
            
            # 更新每个样本对应的记忆
            for i, label in enumerate(labels):
                label_idx = label.item()
                old_memory = self.memory[k, label_idx]  # [D]
                new_feature = features[i]  # [D]
                
                # 动量更新
                if self.initialized[label_idx]:
                    updated = m * old_memory + (1 - m) * new_feature
                else:
                    # 首次出现，直接赋值
                    updated = new_feature
                    self.initialized[label_idx] = True
                
                # 归一化并更新到新的记忆库
                new_memory[k, label_idx] = F.normalize(updated.unsqueeze(0), dim=1).squeeze(0)
        
        # 一次性更新整个记忆库
        self.memory.copy_(new_memory)
    
    def get_memory(self, part_idx=None):
        """
        获取记忆库
        Args:
            part_idx: 部件索引，None表示返回全部
        Returns:
            memory: [N, D] or [K, N, D]
        """
        if part_idx is not None:
            return self.memory[part_idx]  # [N, D]
        return self.memory  # [K, N, D]
    
    def get_part_memory(self, part_idx):
        """
        获取指定部件的记忆库
        Args:
            part_idx: 部件索引 (0 to K-1)
        Returns:
            memory: [N, D]
        """
        return self.memory[part_idx]
    
    def forward(self, part_features):
        """
        计算特征与记忆库的相似度
        Args:
            part_features: List of K个部件特征 [B, D]
        Returns:
            similarities: List of K个相似度矩阵 [B, N]
        """
        similarities = []
        
        for k in range(self.num_parts):
            features = F.normalize(part_features[k], dim=1)  # [B, D]
            memory = self.memory[k]  # [N, D]
            
            # 余弦相似度
            sim = torch.mm(features, memory.t())  # [B, N]
            similarities.append(sim)
        
        return similarities


class AdaptiveMemoryBank(MultiPartMemoryBank):
    """
    自适应记忆库
    增加了自动过滤低置信度样本的功能
    """
    
    def __init__(self, num_parts, num_identities, feature_dim, 
                 momentum=0.9, confidence_threshold=0.5):
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
            part_features: List of K个部件特征 [B, D]
            labels: 样本标签 [B]
            confidences: 置信度分数 [B] (可选)
        """
        K = len(part_features)
        m = self.momentum
        
        # 创建新的记忆库副本
        new_memory = self.memory.clone()
        new_confidence = self.confidence.clone()
        
        for k in range(K):
            features = part_features[k]  # [B, D]
            features = F.normalize(features, dim=1)
            
            for i, label in enumerate(labels):
                label_idx = label.item()
                
                # 检查置信度
                if confidences is not None:
                    conf = confidences[i].item()
                    if conf < self.confidence_threshold:
                        continue  # 跳过低置信度样本
                    
                    # 更新全局置信度 (动量)
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


# 测试代码
if __name__ == "__main__":
    print("="*50)
    print("Memory Bank Test")
    print("="*50)
    
    # 参数
    num_parts = 6
    num_identities = 100
    feature_dim = 256
    batch_size = 8
    
    # 创建记忆库
    memory_bank = MultiPartMemoryBank(
        num_parts=num_parts,
        num_identities=num_identities,
        feature_dim=feature_dim,
        momentum=0.9
    )
    
    print(f"Memory shape: {memory_bank.memory.shape}")
    print(f"Expected: [{num_parts}, {num_identities}, {feature_dim}]")
    
    # 模拟特征和标签
    part_features = [torch.randn(batch_size, feature_dim) for _ in range(num_parts)]
    labels = torch.randint(0, num_identities, (batch_size,))
    
    print(f"\nBatch size: {batch_size}")
    print(f"Labels: {labels}")
    
    # 初始化记忆库
    print("\n--- Initializing Memory ---")
    memory_bank.initialize_memory(part_features, labels)
    print(f"Initialized IDs: {memory_bank.initialized.sum().item()}/{num_identities}")
    
    # 计算相似度
    print("\n--- Computing Similarities ---")
    similarities = memory_bank(part_features)
    print(f"Number of similarity matrices: {len(similarities)}")
    print(f"Each similarity shape: {similarities[0].shape}")
    print(f"Sample similarities (Part 0, Sample 0): {similarities[0][0][:5]}")
    
    # 更新记忆库
    print("\n--- Updating Memory ---")
    new_features = [torch.randn(batch_size, feature_dim) for _ in range(num_parts)]
    memory_bank.update_memory(new_features, labels)
    print("Memory updated successfully")
    
    # 测试自适应记忆库
    print("\n" + "="*50)
    print("Adaptive Memory Bank Test")
    print("="*50)
    
    adaptive_bank = AdaptiveMemoryBank(
        num_parts=num_parts,
        num_identities=num_identities,
        feature_dim=feature_dim,
        confidence_threshold=0.5
    )
    
    confidences = torch.rand(batch_size)  # 模拟置信度
    print(f"Confidences: {confidences}")
    
    adaptive_bank.update_memory(part_features, labels, confidences)
    print(f"Confidence scores: {adaptive_bank.confidence[labels]}")
    
    # 测试梯度问题
    print("\n" + "="*50)
    print("Gradient Test (simulating training)")
    print("="*50)
    
    # 创建需要梯度的特征
    test_features = [torch.randn(batch_size, feature_dim, requires_grad=True) for _ in range(num_parts)]
    test_labels = torch.randint(0, num_identities, (batch_size,))
    
    # 计算相似度（前向传播）
    sims = memory_bank(test_features)
    loss = sum([s.sum() for s in sims])
    
    print(f"Forward pass successful, loss: {loss.item():.4f}")
    
    # 反向传播
    loss.backward()
    print("Backward pass successful")
    
    # 更新记忆库（使用detach的特征）
    memory_bank.update_memory([f.detach() for f in test_features], test_labels)
    print("Memory update successful (no gradient error)")
    
    print("="*50)
    print("All tests passed! ✅")
    print("="*50)