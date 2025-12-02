"""
models/loss.py - 损失函数模块 (完整修复版)

修复日志:
1. [P0] 修复 TripletLoss 中的距离计算错误
2. 优化 GraphDistillationLoss 支持熵权重
3. 添加详细的中文注释
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiPartIDLoss(nn.Module):
    """
    多部件身份损失
    对K个部件的分类logits分别计算交叉熵损失后取平均
    """
    def __init__(self, label_smoothing=0.1):
        """
        Args:
            label_smoothing: 标签平滑系数，范围[0, 1]
                            0表示无平滑，0.1表示10%的标签被平滑到其他类别
        """
        super(MultiPartIDLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    
    def forward(self, logits_list, labels):
        """
        Args:
            logits_list: List[Tensor]，K个部件的logits，每个shape为[B, N]
            labels: Tensor[B]，Ground Truth身份标签
        Returns:
            loss: Tensor，标量损失值
        """
        loss = 0
        for logits in logits_list:
            loss += self.criterion(logits, labels)
        return loss / len(logits_list)


class TripletLoss(nn.Module):
    """
    三元组损失 (Batch Hard Mining)
    
    修复说明:
    - 原实现中dist_ap计算错误，会将非正样本的0距离选入
    - 修复后正确选择最难正样本(同类中最远)和最难负样本(异类中最近)
    """
    def __init__(self, margin=0.3):
        """
        Args:
            margin: 三元组边界，正负样本距离差的最小阈值
        """
        super(TripletLoss, self).__init__()
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)
    
    def forward(self, features, labels):
        """
        Args:
            features: Tensor[B, D]，归一化后的特征向量
            labels: Tensor[B]，样本标签
        Returns:
            loss: Tensor，三元组损失
        """
        # 计算欧氏距离矩阵 [B, B]
        dist_mat = torch.cdist(features, features)
        N = dist_mat.size(0)
        
        # 构建正负样本mask
        is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())  # 同类别mask
        is_neg = ~is_pos  # 异类别mask
        
        # 排除对角线（自己与自己的距离为0）
        mask_diag = torch.eye(N, dtype=torch.bool, device=features.device)
        is_pos = is_pos & ~mask_diag
        
        # [关键修复] 正样本对：选择同类别中距离最远的样本 (Hardest Positive)
        dist_ap = []
        for i in range(N):
            pos_dists = dist_mat[i][is_pos[i]]  # 当前样本的所有正样本距离
            if pos_dists.numel() > 0:
                dist_ap.append(pos_dists.max())  # 取最大距离
            else:
                # 如果没有正样本（batch中只有一个此类别），则距离设为0
                dist_ap.append(torch.tensor(0.0, device=features.device))
        dist_ap = torch.stack(dist_ap)
        
        # [关键修复] 负样本对：选择异类别中距离最近的样本 (Hardest Negative)
        dist_an = []
        for i in range(N):
            neg_dists = dist_mat[i][is_neg[i]]  # 当前样本的所有负样本距离
            if neg_dists.numel() > 0:
                dist_an.append(neg_dists.min())  # 取最小距离
            else:
                # 理论上不会发生（除非batch中只有一个类别）
                dist_an.append(torch.tensor(1e6, device=features.device))
        dist_an = torch.stack(dist_an)
        
        # 优化目标: dist_an > dist_ap + margin
        # 即 dist_an - dist_ap > margin
        y = torch.ones_like(dist_an)  # 目标方向
        return self.ranking_loss(dist_an, dist_ap, y)


class GraphDistillationLoss(nn.Module):
    """
    图蒸馏损失 (Knowledge Distillation from Memory Bank)
    使用记忆库生成的软标签指导模型学习
    """
    def __init__(self, temperature=3.0):
        """
        Args:
            temperature: 蒸馏温度，越大分布越平滑
        """
        super(GraphDistillationLoss, self).__init__()
        self.temperature = temperature
        self.kl_div = nn.KLDivLoss(reduction='batchmean')
    
    def forward(self, logits_list, soft_labels_list, entropy_weights=None):
        """
        Args:
            logits_list: List[Tensor[B, N]]，K个部件的分类logits
            soft_labels_list: List[Tensor[B, N]]，记忆库生成的软标签
            entropy_weights: List[Tensor[B]] (可选)，熵权重，用于降低低置信度样本的贡献
        Returns:
            loss: Tensor，KL散度损失
        """
        loss = 0
        T = self.temperature
        
        for k, (logits, soft_labels) in enumerate(zip(logits_list, soft_labels_list)):
            # 计算学生网络的对数概率分布 (需要温度缩放)
            log_probs = F.log_softmax(logits / T, dim=1)
            
            # KL散度: D_KL(soft_labels || log_probs)
            kl_loss = self.kl_div(log_probs, soft_labels)
            
            # 应用熵权重（如果提供）
            if entropy_weights is not None:
                # entropy_weights[k]: [B]
                # 高熵（低置信度）样本权重较小
                kl_loss = (kl_loss * entropy_weights[k]).mean()
            
            # 温度平方补偿 (Hinton's KD 论文)
            loss += kl_loss * (T * T)
        
        return loss / len(logits_list)


class TotalLoss(nn.Module):
    """
    PF-MGCD 总损失
    
    损失组成:
    1. ID Loss (带Dropout的logits) - 主要优化目标
    2. Triplet Loss - 度量学习
    3. Graph Distillation Loss (Warmup后启用) - 记忆库知识蒸馏
    """
    def __init__(self, num_parts=6, lambda_graph=0.1, lambda_triplet=1.0,
                 label_smoothing=0.1, start_epoch=20):
        """
        Args:
            num_parts: 部件数量K
            lambda_graph: 图蒸馏损失权重
            lambda_triplet: 三元组损失权重
            label_smoothing: 标签平滑系数
            start_epoch: 图蒸馏损失开始的epoch（前期记忆库不稳定）
        """
        super(TotalLoss, self).__init__()
        self.num_parts = num_parts
        self.id_loss = MultiPartIDLoss(label_smoothing=label_smoothing)
        self.tri_loss = TripletLoss(margin=0.3)
        self.graph_loss = GraphDistillationLoss(temperature=3.0)
        
        self.lambda_graph = lambda_graph
        self.lambda_triplet = lambda_triplet
        self.start_epoch = start_epoch
    
    def forward(self, outputs, labels, current_epoch=0, **kwargs):
        """
        Args:
            outputs: Dict，模型输出字典，包含:
                - 'id_features': List[Tensor[B, D]]，K个部件的身份特征
                - 'id_logits': List[Tensor[B, N]]，带Dropout的分类logits
                - 'graph_logits': List[Tensor[B, N]]，不带Dropout的分类logits
                - 'soft_labels': List[Tensor[B, N]]，记忆库生成的软标签
            labels: Tensor[B]，Ground Truth标签
            current_epoch: int，当前训练轮数
        Returns:
            total_loss: Tensor，总损失
            loss_dict: Dict，各子损失的字典（用于日志记录）
        """
        # 1. ID分类损失 (使用带Dropout的logits，增强泛化)
        loss_id = self.id_loss(outputs['id_logits'], labels)
        
        # 2. 三元组度量损失 (拼接所有部件特征)
        full_feat = torch.cat(outputs['id_features'], dim=1)  # [B, K*D]
        full_feat = F.normalize(full_feat, p=2, dim=1)  # L2归一化
        loss_tri = self.tri_loss(full_feat, labels)
        
        # 3. 图蒸馏损失 (Warmup后启用)
        loss_graph = torch.tensor(0.0, device=labels.device)
        if current_epoch >= self.start_epoch and 'soft_labels' in outputs:
            # 使用Clean Logits（不带Dropout）计算蒸馏损失
            # 因为蒸馏需要稳定的预测，而Dropout会引入随机性
            logits_for_graph = outputs.get('graph_logits', outputs['id_logits'])
            
            # 如果graph_propagation返回了熵权重，则传入
            entropy_weights = outputs.get('entropy_weights', None)
            loss_graph = self.graph_loss(logits_for_graph, outputs['soft_labels'], entropy_weights)
        
        # 总损失加权组合
        total = loss_id + self.lambda_triplet * loss_tri + self.lambda_graph * loss_graph
        
        # 返回总损失和各子损失字典（.item()转为Python float，避免内存泄漏）
        return total, {
            'loss_total': total.item(),
            'loss_id': loss_id.item(),
            'loss_triplet': loss_tri.item(),
            'loss_graph': loss_graph.item()
        }
