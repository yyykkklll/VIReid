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
    """
    
    def __init__(self, temperature=3.0, top_k=5):
        super(GraphPropagation, self).__init__()
        self.temperature = temperature
        self.top_k = top_k
    
    def forward(self, part_features, memory_bank):
        K = len(part_features)
        soft_labels = []
        similarities = []
        
        for k in range(K):
            features = F.normalize(part_features[k], dim=1)
            # 获取记忆库 (detach状态，不更新记忆库的梯度)
            memory = memory_bank.get_part_memory(k).detach()
            
            # 计算余弦相似度
            sim = torch.mm(features, memory.t())
            
            # Top-K过滤
            filtered_sim = self._top_k_filter(sim, self.top_k)
            
            # 生成软标签
            soft_label = F.softmax(filtered_sim / self.temperature, dim=1)
            
            # [关键修复] Detach 软标签，将其作为固定的 Target，阻断梯度回传
            soft_labels.append(soft_label.detach())
            similarities.append(sim)
        
        return soft_labels, similarities
    
    def _top_k_filter(self, similarities, k):
        B, N = similarities.size()
        k = min(k, N)
        topk_values, topk_indices = torch.topk(similarities, k, dim=1)
        
        mask = torch.zeros_like(similarities, dtype=torch.bool)
        mask.scatter_(1, topk_indices, True)
        
        # 使用 -1e9 替代 -inf，数值更稳定
        filtered = similarities.clone()
        filtered[~mask] = -1e9
        
        return filtered


class AdaptiveGraphPropagation(GraphPropagation):
    """
    自适应图传播
    """
    
    def __init__(self, temperature=3.0, top_k=5, use_entropy_weight=True):
        super().__init__(temperature, top_k)
        self.use_entropy_weight = use_entropy_weight
    
    def forward(self, part_features, memory_bank):
        # 获取 soft_labels (已经是 detached 的)
        soft_labels, similarities = super().forward(part_features, memory_bank)
        
        if self.use_entropy_weight:
            weights = self._compute_entropy_weights(soft_labels)
        else:
            K = len(soft_labels)
            B = soft_labels[0].size(0)
            weights = [torch.ones(B, device=soft_labels[0].device) for _ in range(K)]
        
        return soft_labels, similarities, weights
    
    def _compute_entropy_weights(self, soft_labels):
        weights = []
        for soft_label in soft_labels:
            # 计算熵 H = -Σ p*log(p)
            entropy = -(soft_label * torch.log(soft_label + 1e-8)).sum(dim=1)
            
            # 转换为权重 w = exp(-H)
            weight = torch.exp(-entropy)
            # [关键修复] 权重也需要 detach
            weights.append(weight.detach())
        
        return weights


class GraphDistillationLoss(nn.Module):
    """
    图蒸馏损失
    """
    
    def __init__(self, use_adaptive_weight=True):
        super().__init__()
        self.use_adaptive_weight = use_adaptive_weight
    
    def forward(self, logits_list, soft_labels_list, weights_list=None):
        K = len(logits_list)
        total_loss = 0.0
        
        for k in range(K):
            logits = logits_list[k]      # Student Logits
            soft_labels = soft_labels_list[k]  # Teacher Targets (Detached)
            
            # Log Softmax
            log_probs = F.log_softmax(logits, dim=1)
            
            # 使用软标签交叉熵: -sum(target * log(pred))
            # 避免 KLDiv 在 target=0 时的 NaN 问题
            ce_loss = -torch.sum(soft_labels * log_probs, dim=1)
            
            # 应用自适应权重
            if self.use_adaptive_weight and weights_list is not None:
                weights = weights_list[k]
                ce_loss = ce_loss * weights
            
            total_loss += ce_loss.mean()
        
        return total_loss / K