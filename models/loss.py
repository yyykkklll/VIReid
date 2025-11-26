"""
Loss Functions for PF-MGCD
包含: ID Loss, Triplet Loss, Graph Loss, Orthogonal Loss, Modality Loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiPartIDLoss(nn.Module):
    """多部件ID分类损失"""
    
    def __init__(self, label_smoothing=0.1):
        super(MultiPartIDLoss, self).__init__()
        self.label_smoothing = label_smoothing
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    
    def forward(self, logits_list, labels):
        """
        Args:
            logits_list: List of K个分类器输出 [B, N]
            labels: [B]
        """
        loss = 0
        for logits in logits_list:
            loss += self.criterion(logits, labels)
        return loss / len(logits_list)


class TripletLoss(nn.Module):
    """Hard Triplet Loss - 使用batch内最难样本"""
    
    def __init__(self, margin=0.3):
        super(TripletLoss, self).__init__()
        self.margin = margin
    
    def forward(self, features, labels):
        """
        Args:
            features: [B, D] 归一化后的特征
            labels: [B]
        Returns:
            loss: scalar
        """
        # 计算距离矩阵
        dist_mat = torch.cdist(features, features, p=2)  # [B, B]
        
        # 为每个anchor找最难的正负样本
        losses = []
        for i in range(len(labels)):
            anchor_label = labels[i]
            
            # 找到所有正样本（同ID但不是自己）
            pos_mask = (labels == anchor_label)
            pos_mask[i] = False  # 排除自己
            
            if pos_mask.sum() > 0:
                # 最难正样本：距离最远的正样本
                pos_dists = dist_mat[i][pos_mask]
                hardest_pos_dist = pos_dists.max()
                
                # 找到所有负样本（不同ID）
                neg_mask = (labels != anchor_label)
                
                if neg_mask.sum() > 0:
                    # 最难负样本：距离最近的负样本
                    neg_dists = dist_mat[i][neg_mask]
                    hardest_neg_dist = neg_dists.min()
                    
                    # Triplet loss
                    loss = F.relu(hardest_pos_dist - hardest_neg_dist + self.margin)
                    losses.append(loss)
        
        if len(losses) > 0:
            return torch.stack(losses).mean()
        else:
            return torch.tensor(0.0, device=features.device)


class GraphDistillationLoss(nn.Module):
    """图传播蒸馏损失"""
    
    def __init__(self, temperature=3.0):
        super(GraphDistillationLoss, self).__init__()
        self.temperature = temperature
        self.kl_div = nn.KLDivLoss(reduction='batchmean')
    
    def forward(self, logits_list, soft_labels_list):
        """
        Args:
            logits_list: List of K个分类器输出 [B, N]
            soft_labels_list: List of K个软标签 [B, N]
        """
        loss = 0
        T = self.temperature
        
        for logits, soft_labels in zip(logits_list, soft_labels_list):
            # 温度缩放
            log_probs = F.log_softmax(logits / T, dim=1)
            soft_probs = F.softmax(soft_labels / T, dim=1)
            
            # KL散度
            loss += self.kl_div(log_probs, soft_probs) * (T * T)
        
        return loss / len(logits_list)


class OrthogonalLoss(nn.Module):
    """正交损失：鼓励不同部件学习不同特征"""
    
    def __init__(self):
        super(OrthogonalLoss, self).__init__()
    
    def forward(self, features_list):
        """
        Args:
            features_list: List of K个特征 [B, D]
        """
        K = len(features_list)
        if K < 2:
            return torch.tensor(0.0, device=features_list[0].device)
        
        loss = 0
        count = 0
        
        # 计算所有特征对的余弦相似度
        for i in range(K):
            for j in range(i + 1, K):
                f1 = F.normalize(features_list[i], p=2, dim=1)
                f2 = F.normalize(features_list[j], p=2, dim=1)
                
                # 余弦相似度的平方（鼓励正交）
                sim = (f1 * f2).sum(dim=1).pow(2).mean()
                loss += sim
                count += 1
        
        return loss / count if count > 0 else torch.tensor(0.0, device=features_list[0].device)


class ModalityLoss(nn.Module):
    """模态分类损失"""
    
    def __init__(self):
        super(ModalityLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()
    
    def forward(self, logits_list, modality_labels):
        """
        Args:
            logits_list: List of K个模态分类器输出 [B, 2]
            modality_labels: [B] (0=可见光, 1=红外)
        """
        loss = 0
        for logits in logits_list:
            loss += self.criterion(logits, modality_labels)
        return loss / len(logits_list)


class TotalLoss(nn.Module):
    """
    总损失函数
    支持动态权重调整和所有损失项
    """
    
    def __init__(self, 
                 num_parts=6, 
                 label_smoothing=0.1, 
                 lambda_id=1.0, 
                 lambda_triplet=0.5, 
                 lambda_graph=1.0, 
                 lambda_orth=0.1, 
                 lambda_mod=0.5,
                 use_adaptive_weight=False,  # [修复] 添加这个参数
                 triplet_margin=0.3):
        super(TotalLoss, self).__init__()
        self.num_parts = num_parts
        self.use_adaptive_weight = use_adaptive_weight
        
        # 各个损失函数
        self.id_loss = MultiPartIDLoss(label_smoothing=label_smoothing)
        self.triplet_loss = TripletLoss(margin=triplet_margin)
        self.graph_loss = GraphDistillationLoss(temperature=3.0)
        self.orth_loss = OrthogonalLoss()
        self.mod_loss = ModalityLoss()
        
        # 损失权重（如果使用自适应权重，这些会被动态调整）
        if use_adaptive_weight:
            # 可学习的权重参数
            self.lambda_id = nn.Parameter(torch.tensor(lambda_id))
            self.lambda_triplet = nn.Parameter(torch.tensor(lambda_triplet))
            self.lambda_graph = nn.Parameter(torch.tensor(lambda_graph))
            self.lambda_orth = nn.Parameter(torch.tensor(lambda_orth))
            self.lambda_mod = nn.Parameter(torch.tensor(lambda_mod))
        else:
            # 固定权重
            self.lambda_id = lambda_id
            self.lambda_triplet = lambda_triplet
            self.lambda_graph = lambda_graph
            self.lambda_orth = lambda_orth
            self.lambda_mod = lambda_mod
    
    def get_lambda(self, param):
        """获取lambda值（支持固定和可学习两种模式）"""
        if self.use_adaptive_weight:
            return torch.clamp(param, min=0.0)  # 确保非负
        else:
            return param
    
    def forward(self, outputs, labels, modality_labels, current_epoch=0):
        """
        Args:
            outputs: 模型输出字典
            labels: [B]
            modality_labels: [B]
            current_epoch: 当前epoch（用于动态调整权重）
        Returns:
            total_loss: 总损失
            loss_dict: 各项损失的字典
        """
        # 1. ID分类损失
        loss_id = self.id_loss(outputs['id_logits'], labels)
        
        # 2. Triplet损失（使用拼接后的特征）
        concat_features = torch.cat(outputs['id_features'], dim=1)
        concat_features = F.normalize(concat_features, p=2, dim=1)
        loss_triplet = self.triplet_loss(concat_features, labels)
        
        # 3. 图传播蒸馏损失（warmup后开始）
        warmup_epochs = 5  # 可以作为参数传入
        if current_epoch >= warmup_epochs and 'soft_labels' in outputs:
            loss_graph = self.graph_loss(outputs['id_logits'], outputs['soft_labels'])
        else:
            loss_graph = torch.tensor(0.0, device=labels.device)
        
        # 4. 正交损失
        loss_orth = self.orth_loss(outputs['id_features'])
        
        # 5. 模态分类损失
        loss_mod = self.mod_loss(outputs['mod_logits'], modality_labels)
        
        # 获取当前权重
        lambda_id = self.get_lambda(self.lambda_id)
        lambda_triplet = self.get_lambda(self.lambda_triplet)
        lambda_graph = self.get_lambda(self.lambda_graph)
        lambda_orth = self.get_lambda(self.lambda_orth)
        lambda_mod = self.get_lambda(self.lambda_mod)
        
        # 总损失
        total_loss = (
            lambda_id * loss_id +
            lambda_triplet * loss_triplet +
            lambda_graph * loss_graph +
            lambda_orth * loss_orth +
            lambda_mod * loss_mod
        )
        
        # 返回详细损失信息
        loss_dict = {
            'loss_total': total_loss.item(),
            'loss_id': loss_id.item(),
            'loss_triplet': loss_triplet.item(),
            'loss_graph': loss_graph.item(),
            'loss_orth': loss_orth.item(),
            'loss_mod': loss_mod.item(),
            # 如果使用自适应权重，记录当前权重值
            'lambda_id': lambda_id.item() if self.use_adaptive_weight else lambda_id,
            'lambda_triplet': lambda_triplet.item() if self.use_adaptive_weight else lambda_triplet,
            'lambda_graph': lambda_graph.item() if self.use_adaptive_weight else lambda_graph,
            'lambda_orth': lambda_orth.item() if self.use_adaptive_weight else lambda_orth,
            'lambda_mod': lambda_mod.item() if self.use_adaptive_weight else lambda_mod,
        }
        
        return total_loss, loss_dict
