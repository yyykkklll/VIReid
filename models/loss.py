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
        dist_mat = torch.cdist(features, features, p=2)  # [B, B]
        losses = []
        for i in range(len(labels)):
            anchor_label = labels[i]
            pos_mask = (labels == anchor_label)
            pos_mask[i] = False
            
            if pos_mask.sum() > 0:
                pos_dists = dist_mat[i][pos_mask]
                hardest_pos_dist = pos_dists.max()
                
                neg_mask = (labels != anchor_label)
                if neg_mask.sum() > 0:
                    neg_dists = dist_mat[i][neg_mask]
                    hardest_neg_dist = neg_dists.min()
                    
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
        loss = 0
        T = self.temperature
        for logits, soft_labels in zip(logits_list, soft_labels_list):
            log_probs = F.log_softmax(logits / T, dim=1)
            soft_probs = F.softmax(soft_labels / T, dim=1)
            loss += self.kl_div(log_probs, soft_probs) * (T * T)
        return loss / len(logits_list)


class OrthogonalLoss(nn.Module):
    """
    正交损失：强制身份特征和模态特征正交 (解耦)
    [改进] 增加数值稳定性，防止梯度爆炸
    """
    
    def __init__(self):
        super(OrthogonalLoss, self).__init__()
    
    def forward(self, id_features_list, mod_features_list):
        K = len(id_features_list)
        loss = 0
        eps = 1e-8
        
        for f_id, f_mod in zip(id_features_list, mod_features_list):
            # 归一化 (加 eps 防止除零)
            f_id_norm = f_id / (f_id.norm(p=2, dim=1, keepdim=True) + eps)
            f_mod_norm = f_mod / (f_mod.norm(p=2, dim=1, keepdim=True) + eps)
            
            # 计算余弦相似度的平方 (target=0)
            # Dot product: [B, D] * [B, D] -> sum(dim=1) -> [B]
            sim = (f_id_norm * f_mod_norm).sum(dim=1).pow(2).mean()
            loss += sim
            
        return loss / K


class ModalityLoss(nn.Module):
    """模态分类损失"""
    
    def __init__(self):
        super(ModalityLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()
    
    def forward(self, logits_list, modality_labels):
        loss = 0
        for logits in logits_list:
            loss += self.criterion(logits, modality_labels)
        return loss / len(logits_list)


class TotalLoss(nn.Module):
    """
    总损失函数
    """
    
    def __init__(self, 
                 num_parts=6, 
                 label_smoothing=0.1, 
                 lambda_id=1.0, 
                 lambda_triplet=0.5, 
                 lambda_graph=1.0, 
                 lambda_orth=0.1, 
                 lambda_mod=0.5,
                 use_adaptive_weight=False,
                 triplet_margin=0.3):
        super(TotalLoss, self).__init__()
        self.num_parts = num_parts
        self.use_adaptive_weight = use_adaptive_weight
        
        self.id_loss = MultiPartIDLoss(label_smoothing=label_smoothing)
        self.triplet_loss = TripletLoss(margin=triplet_margin)
        self.graph_loss = GraphDistillationLoss(temperature=3.0)
        self.orth_loss = OrthogonalLoss()
        self.mod_loss = ModalityLoss()
        
        # 损失权重
        if use_adaptive_weight:
            self.lambda_id = nn.Parameter(torch.tensor(lambda_id))
            self.lambda_triplet = nn.Parameter(torch.tensor(lambda_triplet))
            self.lambda_graph = nn.Parameter(torch.tensor(lambda_graph))
            self.lambda_orth = nn.Parameter(torch.tensor(lambda_orth))
            self.lambda_mod = nn.Parameter(torch.tensor(lambda_mod))
        else:
            self.lambda_id = lambda_id
            self.lambda_triplet = lambda_triplet
            self.lambda_graph = lambda_graph
            self.lambda_orth = lambda_orth
            self.lambda_mod = lambda_mod
    
    def get_lambda(self, param):
        if self.use_adaptive_weight:
            return torch.clamp(param, min=0.0)
        else:
            return param
    
    def forward(self, outputs, labels, modality_labels, current_epoch=0):
        loss_id = self.id_loss(outputs['id_logits'], labels)
        
        concat_features = torch.cat(outputs['id_features'], dim=1)
        concat_features = F.normalize(concat_features, p=2, dim=1)
        loss_triplet = self.triplet_loss(concat_features, labels)
        
        warmup_epochs = 5
        if current_epoch >= warmup_epochs and 'soft_labels' in outputs:
            loss_graph = self.graph_loss(outputs['id_logits'], outputs['soft_labels'])
        else:
            loss_graph = torch.tensor(0.0, device=labels.device)
        
        if 'mod_features' in outputs:
            loss_orth = self.orth_loss(outputs['id_features'], outputs['mod_features'])
        else:
            loss_orth = torch.tensor(0.0, device=labels.device)
        
        loss_mod = self.mod_loss(outputs['mod_logits'], modality_labels)
        
        # 获取权重
        lambda_id = self.get_lambda(self.lambda_id)
        lambda_triplet = self.get_lambda(self.lambda_triplet)
        lambda_graph = self.get_lambda(self.lambda_graph)
        lambda_orth = self.get_lambda(self.lambda_orth)
        lambda_mod = self.get_lambda(self.lambda_mod)
        
        total_loss = (
            lambda_id * loss_id +
            lambda_triplet * loss_triplet +
            lambda_graph * loss_graph +
            lambda_orth * loss_orth +
            lambda_mod * loss_mod
        )
        
        loss_dict = {
            'loss_total': total_loss.item(),
            'loss_id': loss_id.item(),
            'loss_triplet': loss_triplet.item(),
            'loss_graph': loss_graph.item(),
            'loss_orth': loss_orth.item(),
            'loss_mod': loss_mod.item(),
            'lambda_id': lambda_id.item() if self.use_adaptive_weight else lambda_id,
            'lambda_triplet': lambda_triplet.item() if self.use_adaptive_weight else lambda_triplet,
            'lambda_graph': lambda_graph.item() if self.use_adaptive_weight else lambda_graph,
            'lambda_orth': lambda_orth.item() if self.use_adaptive_weight else lambda_orth,
            'lambda_mod': lambda_mod.item() if self.use_adaptive_weight else lambda_mod,
        }
        
        return total_loss, loss_dict