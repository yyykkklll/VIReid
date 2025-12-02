"""
Adaptive Graph Propagation Module (Fixed)
[修复日志]
1. 移除 Top-K 硬截断：防止将 Ground Truth 误杀为 0，避免 Loss 冲突。
2. 引入 Scale Factor：将余弦相似度放大，解决 Softmax 分布过平（Entropy 过高）的问题。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaptiveGraphPropagation(nn.Module):
    """
    自适应图传播模块
    """
    
    def __init__(self, temperature=3.0, top_k=5, use_entropy_weight=True, scale=30.0):
        super(AdaptiveGraphPropagation, self).__init__()
        self.temperature = temperature
        self.top_k = top_k # 保留参数定义以兼容接口，但在forward中不再做硬截断
        self.use_entropy_weight = use_entropy_weight
        self.scale = scale # [新增] 缩放因子，ReID 中通常为 30~64
    
    def forward(self, part_features, memory_bank):
        """
        Args:
            part_features: List of K个部件特征 [B, D]
            memory_bank: MultiPartMemoryBank
        Returns:
            soft_labels: List of K个软标签 [B, N]
            similarities: List of K个相似度矩阵 [B, N]
            entropy_weights: List of K个熵权重 [B]
        """
        soft_labels = []
        similarities = []
        entropy_weights = []
        
        for k, feat in enumerate(part_features):
            # 1. 计算余弦相似度 [-1, 1]
            feat_norm = F.normalize(feat, p=2, dim=1)  # [B, D]
            memory_norm = F.normalize(memory_bank.memory[k], p=2, dim=1)  # [N, D]
            sim = torch.mm(feat_norm, memory_norm.t())  # [B, N]
            
            # [关键修复] 2. 缩放相似度
            # 将 [-1, 1] 放大到 [-30, 30]，这样 Softmax 才能生成尖锐的分布
            scaled_sim = sim * self.scale
            
            # [关键修复] 3. 移除 Top-K 硬截断
            # 直接对全量类别做 Softmax。由于 Scale 很大，非相似类的概率会自动趋近于 0，
            # 但 Ground Truth 即使排在第 20 名，也能保留梯度，不会被一刀切死。
            
            # 4. 生成软标签 (带温度 T)
            # Logits = scaled_sim / T
            soft_label = F.softmax(scaled_sim / self.temperature, dim=1)
            
            soft_labels.append(soft_label)
            similarities.append(sim)
            
            # 5. 计算熵权重 (可选)
            if self.use_entropy_weight:
                entropy = self._compute_entropy(soft_label)
                weight = torch.exp(-entropy)
                weight = weight / (weight.mean() + 1e-8)
                entropy_weights.append(weight)
        
        if self.use_entropy_weight:
            # 应用熵权重调整
            adjusted_soft_labels = []
            for i, sl in enumerate(soft_labels):
                # 简单的权重应用逻辑，或者仅返回权重供 Loss 使用
                # 这里我们保持 soft_labels 原样，返回 weights 让外部决定
                adjusted_soft_labels.append(sl)
            return adjusted_soft_labels, similarities, entropy_weights
        else:
            return soft_labels, similarities, None
    
    def _compute_entropy(self, prob_dist):
        """
        计算概率分布的熵
        """
        prob_dist = torch.clamp(prob_dist, min=1e-8)
        entropy = -(prob_dist * torch.log(prob_dist)).sum(dim=1)
        return entropy