"""
Graph Propagation - Fine-Grained Graph Propagation and Distillation
细粒度图传播与蒸馏: 通过记忆库邻居生成软标签
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphPropagation(nn.Module):
    """
    细粒度图传播模块
    1. 计算特征与记忆库的亲和力
    2. Top-K邻居过滤
    3. 生成软标签
    """
    
    def __init__(self, temperature=3.0, top_k=5):
        """
        Args:
            temperature: Softmax温度参数 τ
            top_k: 保留的Top-K邻居数量
        """
        super(GraphPropagation, self).__init__()
        self.temperature = temperature
        self.top_k = top_k
    
    def forward(self, part_features, memory_bank):
        """
        Args:
            part_features: List of K个部件特征 [B, D]
            memory_bank: 记忆库对象
        Returns:
            soft_labels: List of K个软标签分布 [B, N]
            similarities: List of K个相似度矩阵 [B, N]
        """
        K = len(part_features)
        soft_labels = []
        similarities = []
        
        for k in range(K):
            features = F.normalize(part_features[k], dim=1)  # [B, D]
            memory = memory_bank.get_part_memory(k)  # [N, D]
            
            # 计算余弦相似度
            sim = torch.mm(features, memory.t())  # [B, N]
            
            # Top-K过滤
            filtered_sim = self._top_k_filter(sim, self.top_k)
            
            # 生成软标签
            soft_label = F.softmax(filtered_sim / self.temperature, dim=1)
            
            soft_labels.append(soft_label)
            similarities.append(sim)
        
        return soft_labels, similarities
    
    def _top_k_filter(self, similarities, k):
        """
        仅保留Top-K个最相似的邻居，其余置为-inf
        Args:
            similarities: [B, N]
            k: Top-K数量
        Returns:
            filtered: [B, N]
        """
        B, N = similarities.size()
        
        # 找到Top-K的值和索引
        topk_values, topk_indices = torch.topk(similarities, k, dim=1)  # [B, K]
        
        # 创建mask
        mask = torch.zeros_like(similarities).bool()
        mask.scatter_(1, topk_indices, True)
        
        # 应用mask (非Top-K设为-inf)
        filtered = similarities.clone()
        filtered[~mask] = float('-inf')
        
        return filtered


class AdaptiveGraphPropagation(GraphPropagation):
    """
    自适应图传播
    根据软标签的熵动态调整权重
    """
    
    def __init__(self, temperature=3.0, top_k=5, use_entropy_weight=True):
        super().__init__(temperature, top_k)
        self.use_entropy_weight = use_entropy_weight
    
    def forward(self, part_features, memory_bank):
        """
        Returns:
            soft_labels: List of K个软标签 [B, N]
            similarities: List of K个相似度 [B, N]
            weights: List of K个权重 [B] (基于熵)
        """
        soft_labels, similarities = super().forward(part_features, memory_bank)
        
        if self.use_entropy_weight:
            weights = self._compute_entropy_weights(soft_labels)
        else:
            K = len(soft_labels)
            B = soft_labels[0].size(0)
            weights = [torch.ones(B, device=soft_labels[0].device) for _ in range(K)]
        
        return soft_labels, similarities, weights
    
    def _compute_entropy_weights(self, soft_labels):
        """
        计算软标签的熵，并转换为权重
        熵越大 → 不确定性越高 → 权重越低
        Args:
            soft_labels: List of K个软标签 [B, N]
        Returns:
            weights: List of K个权重 [B]
        """
        weights = []
        
        for soft_label in soft_labels:
            # 计算熵 H = -Σ p*log(p)
            entropy = -(soft_label * torch.log(soft_label + 1e-8)).sum(dim=1)  # [B]
            
            # 转换为权重 w = exp(-H)
            weight = torch.exp(-entropy)
            weights.append(weight)
        
        return weights


class GraphDistillationLoss(nn.Module):
    """
    图蒸馏损失
    让学生网络的预测分布拟合图传播的软标签
    """
    
    def __init__(self, use_adaptive_weight=True):
        super().__init__()
        self.use_adaptive_weight = use_adaptive_weight
    
    def forward(self, logits_list, soft_labels_list, weights_list=None):
        """
        Args:
            logits_list: List of K个学生网络的logits [B, N]
            soft_labels_list: List of K个图传播的软标签 [B, N]
            weights_list: List of K个自适应权重 [B] (可选)
        Returns:
            loss: 标量损失
        """
        K = len(logits_list)
        total_loss = 0.0
        
        for k in range(K):
            logits = logits_list[k]  # [B, N]
            soft_labels = soft_labels_list[k]  # [B, N]
            
            # KL散度损失
            log_probs = F.log_softmax(logits, dim=1)
            kl_loss = F.kl_div(log_probs, soft_labels, reduction='none').sum(dim=1)  # [B]
            
            # 应用自适应权重
            if self.use_adaptive_weight and weights_list is not None:
                weights = weights_list[k]  # [B]
                kl_loss = kl_loss * weights
            
            total_loss += kl_loss.mean()
        
        return total_loss / K