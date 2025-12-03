"""
models/graph_propagation.py - 自适应图传播模块 (Stable Version)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaptiveGraphPropagation(nn.Module):
    """
    自适应图传播模块
    
    关键改进:
    - 降低 scale factor (30.0 -> 16.0)，避免 Softmax 输出过于尖锐
    """
    def __init__(self, temperature=3.0, top_k=5, use_entropy_weight=True, scale=16.0):
        """
        Args:
            temperature: 温度参数
            scale: 相似度缩放因子，推荐 16.0 以保持训练稳定
        """
        super(AdaptiveGraphPropagation, self).__init__()
        self.temperature = temperature
        self.top_k = top_k
        self.use_entropy_weight = use_entropy_weight
        self.scale = scale
    
    def forward(self, part_features, memory_bank):
        soft_labels = []
        similarities = []
        entropy_weights = [] if self.use_entropy_weight else None
        
        for k, feat in enumerate(part_features):
            # 1. 计算余弦相似度 [-1, 1]
            feat_norm = F.normalize(feat, p=2, dim=1)
            memory_norm = F.normalize(memory_bank.memory[k], p=2, dim=1)
            sim = torch.mm(feat_norm, memory_norm.t())
            
            # 2. 缩放相似度
            scaled_sim = sim * self.scale
            
            # 3. 生成软标签 (带温度缩放)
            # 移除 Top-K 硬截断，使用全量 Softmax
            soft_label = F.softmax(scaled_sim / self.temperature, dim=1)
            soft_labels.append(soft_label)
            similarities.append(sim)
            
            # 4. 计算熵权重
            if self.use_entropy_weight:
                entropy = self._compute_entropy(soft_label)
                weight = torch.exp(-entropy)
                weight = weight / (weight.mean() + 1e-8)
                entropy_weights.append(weight)
        
        if self.use_entropy_weight:
            adjusted_soft_labels = []
            for i, (sl, weight) in enumerate(zip(soft_labels, entropy_weights)):
                weighted_sl = sl * weight.unsqueeze(1)
                weighted_sl = weighted_sl / (weighted_sl.sum(dim=1, keepdim=True) + 1e-8)
                adjusted_soft_labels.append(weighted_sl)
            
            return adjusted_soft_labels, similarities, entropy_weights
        else:
            return soft_labels, similarities, None
    
    def _compute_entropy(self, prob_dist):
        prob_dist = torch.clamp(prob_dist, min=1e-8)
        entropy = -(prob_dist * torch.log(prob_dist)).sum(dim=1)
        return entropy