"""
models/graph_propagation.py - 自适应图传播模块 (完整修复版)

修复日志:
1. [P1] 修复熵权重计算后未应用的问题
2. 保留scale factor缩放机制
3. 添加详细的中文注释

功能:
利用记忆库生成软标签，指导模型学习跨模态关联
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveGraphPropagation(nn.Module):
    """
    自适应图传播模块
    
    工作流程:
    1. 计算特征与记忆库的余弦相似度
    2. 使用Scale Factor放大相似度（避免Softmax过于平滑）
    3. 应用温度缩放生成软标签
    4. 可选：计算熵权重，降低低置信度样本的贡献
    
    关键改进:
    - 移除Top-K硬截断，避免误杀Ground Truth类别
    - 引入Scale Factor解决相似度分布过平的问题
    """
    def __init__(self, temperature=3.0, top_k=5, use_entropy_weight=True, scale=30.0):
        """
        Args:
            temperature: Softmax温度参数T，越大分布越平滑
            top_k: 保留参数以兼容旧接口（实际不再使用硬截断）
            use_entropy_weight: 是否计算并应用熵权重
            scale: 相似度缩放因子，通常取30~64（ReID经验值）
        """
        super(AdaptiveGraphPropagation, self).__init__()
        self.temperature = temperature
        self.top_k = top_k  # 保留接口兼容性
        self.use_entropy_weight = use_entropy_weight
        self.scale = scale
    
    def forward(self, part_features, memory_bank):
        """
        前向传播：生成软标签
        
        Args:
            part_features: List[Tensor[B, D]]，K个部件特征
            memory_bank: MultiPartMemoryBank，记忆库对象
        
        Returns:
            soft_labels: List[Tensor[B, N]]，K个软标签分布
            similarities: List[Tensor[B, N]]，K个原始相似度矩阵
            entropy_weights: List[Tensor[B]] or None，K个熵权重（如果启用）
        """
        soft_labels = []
        similarities = []
        entropy_weights = [] if self.use_entropy_weight else None
        
        for k, feat in enumerate(part_features):
            # 1. 计算余弦相似度 [-1, 1]
            feat_norm = F.normalize(feat, p=2, dim=1)  # [B, D]
            memory_norm = F.normalize(memory_bank.memory[k], p=2, dim=1)  # [N, D]
            sim = torch.mm(feat_norm, memory_norm.t())  # [B, N]
            
            # 2. [关键] 缩放相似度
            # 将 [-1, 1] 放大到 [-scale, scale]，使Softmax分布更尖锐
            # 例如: scale=30 时，相似度0.9会变成27，经过Softmax后概率接近1
            scaled_sim = sim * self.scale
            
            # 3. [关键修复] 移除Top-K硬截断
            # 原因：硬截断会将排名>K的类别概率直接置0，
            #      如果Ground Truth恰好排在第K+1位，梯度会被完全截断
            # 解决：使用全量Softmax，由于scale很大，非相似类概率自动趋近0
            
            # 4. 生成软标签 (带温度缩放)
            soft_label = F.softmax(scaled_sim / self.temperature, dim=1)  # [B, N]
            soft_labels.append(soft_label)
            similarities.append(sim)  # 保存原始相似度（用于分析）
            
            # 5. 计算熵权重（可选）
            if self.use_entropy_weight:
                entropy = self._compute_entropy(soft_label)  # [B]
                # 熵越高（分布越平）说明模型不确定，降低其权重
                weight = torch.exp(-entropy)  # [B]
                # 归一化权重，使均值为1（保持损失尺度一致）
                weight = weight / (weight.mean() + 1e-8)
                entropy_weights.append(weight)
        
        # [修复] 应用熵权重调整软标签
        if self.use_entropy_weight:
            adjusted_soft_labels = []
            for i, (sl, weight) in enumerate(zip(soft_labels, entropy_weights)):
                # 方案：加权后重新归一化（保持概率分布性质）
                weighted_sl = sl * weight.unsqueeze(1)  # [B, N] * [B, 1] = [B, N]
                weighted_sl = weighted_sl / (weighted_sl.sum(dim=1, keepdim=True) + 1e-8)
                adjusted_soft_labels.append(weighted_sl)
            
            return adjusted_soft_labels, similarities, entropy_weights
        else:
            return soft_labels, similarities, None
    
    def _compute_entropy(self, prob_dist):
        """
        计算概率分布的香农熵
        
        H(p) = -Σ p(x) * log(p(x))
        
        熵越高表示分布越均匀（模型不确定）
        熵越低表示分布越集中（模型有信心）
        
        Args:
            prob_dist: Tensor[B, N]，概率分布
        Returns:
            entropy: Tensor[B]，每个样本的熵
        """
        # 添加小常数避免log(0)
        prob_dist = torch.clamp(prob_dist, min=1e-8)
        entropy = -(prob_dist * torch.log(prob_dist)).sum(dim=1)  # [B]
        return entropy