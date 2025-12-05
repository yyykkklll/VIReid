"""
models/loss.py - Simple & Effective Baseline Loss
回归简单有效的设计
"""
import torch
import torch. nn as nn
import torch.nn.functional as F


class MultiPartIDLoss(nn. Module):
    """多部件交叉熵损失"""
    def __init__(self, label_smoothing=0.1):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    
    def forward(self, logits_list, labels):
        loss = sum(self.criterion(logits, labels) for logits in logits_list)
        return loss / len(logits_list)


class TripletLossHardMining(nn. Module):
    """
    简单有效的 Batch Hard Triplet Loss
    - 使用 Soft Margin 避免 loss 归零
    - 不使用复杂的加权策略
    """
    def __init__(self, margin=0.3):
        super().__init__()
        self.margin = margin
    
    def forward(self, feat, labels):
        # L2 归一化
        feat = F.normalize(feat, p=2, dim=1)
        
        # 欧氏距离矩阵
        dist = torch.cdist(feat, feat, p=2)
        
        N = dist.size(0)
        is_pos = labels.unsqueeze(0).eq(labels. unsqueeze(1))
        is_neg = ~is_pos
        
        # 排除对角线
        mask_diag = torch.eye(N, dtype=torch.bool, device=feat.device)
        is_pos = is_pos & ~mask_diag
        
        # Batch Hard Mining
        # 最难正样本: 同类中最远的
        dist_ap = torch.where(is_pos, dist, torch.zeros_like(dist)). max(dim=1)[0]
        # 最难负样本: 异类中最近的
        dist_an = torch.where(is_neg, dist, torch.full_like(dist, float('inf'))).min(dim=1)[0]
        
        # Soft Margin Loss: log(1 + exp(ap - an + margin))
        loss = F.softplus(dist_ap - dist_an + self.margin).mean()
        
        return loss


class TotalLoss(nn. Module):
    """
    简洁版总损失 = ID Loss + Triplet Loss
    不使用复杂的跨模态损失
    """
    def __init__(self, num_parts=6, num_classes=206, feature_dim=512,
                 lambda_triplet=1.0, label_smoothing=0.1, **kwargs):
        super().__init__()
        
        self.id_loss = MultiPartIDLoss(label_smoothing)
        self.tri_loss = TripletLossHardMining(margin=0.3)
        self.lambda_triplet = lambda_triplet

    def forward(self, outputs, labels, current_epoch=0, **kwargs):
        # 1. ID Loss
        L_id = self. id_loss(outputs['id_logits'], labels)
        
        # 2.  Triplet Loss
        feat = torch.cat(outputs['id_features'], dim=1)
        L_tri = self.tri_loss(feat, labels)
        
        # 总损失
        total = L_id + self.lambda_triplet * L_tri
        
        return total, {
            'loss_total': total. item(),
            'loss_id': L_id.item(),
            'loss_triplet': L_tri.item(),
        }


# 兼容别名
TripletLoss = TripletLossHardMining
WeightedRegularizationTriplet = TripletLossHardMining