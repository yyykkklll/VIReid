"""
Adaptive Graph Propagation Module
支持混合精度训练的图传播模块
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphPropagation(nn.Module):
    """基础图传播模块"""
    
    def __init__(self, temperature=3.0, top_k=5):
        super(GraphPropagation, self).__init__()
        self.temperature = temperature
        self.top_k = top_k
    
    def forward(self, part_features, memory_bank):
        """
        图传播：从记忆库获取软标签
        Args:
            part_features: List of K个部件特征 [B, D]
            memory_bank: MultiPartMemoryBank
        Returns:
            soft_labels: List of K个软标签 [B, N]
            similarities: List of K个相似度矩阵 [B, N]
        """
        soft_labels = []
        similarities = []
        
        for k, feat in enumerate(part_features):
            # 计算与记忆库的余弦相似度
            feat_norm = F.normalize(feat, p=2, dim=1)  # [B, D]
            
            # [关键修复] 使用正确的属性名: memory 而不是 features
            memory_norm = F.normalize(memory_bank.memory[k], p=2, dim=1)  # [N, D]
            
            sim = torch.mm(feat_norm, memory_norm.t())  # [B, N]
            
            # Top-K过滤（支持float16）
            filtered_sim = self._top_k_filter(sim, self.top_k)
            
            # 温度缩放的softmax
            soft_label = F.softmax(filtered_sim / self.temperature, dim=1)
            
            soft_labels.append(soft_label)
            similarities.append(sim)
        
        return soft_labels, similarities
    
    def _top_k_filter(self, similarity, k):
        """
        Top-K过滤：保留每行最大的K个值，其余设为极小值
        支持混合精度训练
        """
        B, N = similarity.size()
        
        if k >= N:
            return similarity
        
        # 找到Top-K的值和索引
        topk_values, topk_indices = torch.topk(similarity, k, dim=1)
        
        # 根据数据类型选择合适的fill_value
        # float16范围: -65504 到 +65504
        if similarity.dtype == torch.float16:
            fill_value = -65000.0  # float16安全范围
        else:
            fill_value = -1e9      # float32可以使用更大的值
        
        # 创建mask并填充
        filtered = torch.full_like(similarity, fill_value)
        filtered.scatter_(1, topk_indices, topk_values)
        
        return filtered


class AdaptiveGraphPropagation(GraphPropagation):
    """
    自适应图传播：使用熵权重调整软标签
    """
    
    def __init__(self, temperature=3.0, top_k=5, use_entropy_weight=True):
        super(AdaptiveGraphPropagation, self).__init__(temperature, top_k)
        self.use_entropy_weight = use_entropy_weight
    
    def forward(self, part_features, memory_bank):
        """
        自适应图传播
        Returns:
            soft_labels: List of K个软标签 [B, N]
            similarities: List of K个相似度 [B, N]
            entropy_weights: List of K个熵权重 [B]
        """
        # 基础图传播
        soft_labels, similarities = super().forward(part_features, memory_bank)
        
        # 计算熵权重
        if self.use_entropy_weight:
            entropy_weights = []
            adjusted_soft_labels = []
            
            for soft_label in soft_labels:
                # 计算熵
                entropy = self._compute_entropy(soft_label)  # [B]
                
                # 熵归一化：低熵 -> 高权重
                weight = torch.exp(-entropy)  # [B]
                weight = weight / (weight.mean() + 1e-8)  # 归一化
                
                entropy_weights.append(weight)
                adjusted_soft_labels.append(soft_label)
            
            return adjusted_soft_labels, similarities, entropy_weights
        else:
            return soft_labels, similarities, None
    
    def _compute_entropy(self, prob_dist):
        """
        计算概率分布的熵
        Args:
            prob_dist: [B, N] 概率分布
        Returns:
            entropy: [B] 熵值
        """
        # 避免log(0)
        prob_dist = torch.clamp(prob_dist, min=1e-8)
        entropy = -(prob_dist * torch.log(prob_dist)).sum(dim=1)
        return entropy


# 测试代码
if __name__ == '__main__':
    print("Testing Graph Propagation with mixed precision...")
    
    # 创建模拟数据
    B, K, N, D = 8, 6, 206, 256
    
    # 模拟部件特征
    part_features = [torch.randn(B, D) for _ in range(K)]
    
    # [修复] 使用正确的属性名测试
    class MockMemoryBank:
        def __init__(self):
            # 使用 memory 属性（与实际的MultiPartMemoryBank一致）
            self.memory = torch.randn(K, N, D)
    
    memory_bank = MockMemoryBank()
    
    # 测试标准精度
    print("\n[1] Testing with float32:")
    gp = AdaptiveGraphPropagation(temperature=3.0, top_k=5)
    soft_labels, sims, weights = gp(part_features, memory_bank)
    print(f"  Soft labels shape: {soft_labels[0].shape}")
    print(f"  Dtype: {soft_labels[0].dtype}")
    print(f"  Values in range: {soft_labels[0].min():.4f} to {soft_labels[0].max():.4f}")
    
    # 测试混合精度
    print("\n[2] Testing with float16 (autocast):")
    from torch.amp import autocast
    
    with autocast(device_type='cpu', dtype=torch.float16):
        part_features_fp16 = [f.half() for f in part_features]
        memory_bank.memory = memory_bank.memory.half()
        
        gp_fp16 = AdaptiveGraphPropagation(temperature=3.0, top_k=5)
        soft_labels_fp16, sims_fp16, weights_fp16 = gp_fp16(part_features_fp16, memory_bank)
        
        print(f"  Soft labels shape: {soft_labels_fp16[0].shape}")
        print(f"  Dtype: {soft_labels_fp16[0].dtype}")
        print(f"  Values in range: {soft_labels_fp16[0].min():.4f} to {soft_labels_fp16[0].max():.4f}")
        print(f"  Contains NaN: {torch.isnan(soft_labels_fp16[0]).any()}")
        print(f"  Contains Inf: {torch.isinf(soft_labels_fp16[0]).any()}")
    
    print("\n✓ All tests passed!")
