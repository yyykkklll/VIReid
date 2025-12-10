import torch
import torch.nn as nn
import torch.nn.functional as F


def normalize(x, axis=-1):
    norm = torch.norm(x, 2, axis, keepdim=True)
    return x / (norm + 1e-12)


def euclidean_dist(x, y):
    m, n = x. size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy - 2 * torch.matmul(x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()
    return dist


class ConsistencyLoss(nn.Module):
    def __init__(self):
        super(ConsistencyLoss, self).__init__()

    def forward(self, feat_real, feat_aug):
        feat_real = F.normalize(feat_real, p=2, dim=1)
        feat_aug = F.normalize(feat_aug, p=2, dim=1)
        return 1 - (feat_real * feat_aug).sum(dim=1).mean()


class CrossEntropyLabelSmooth(nn. Module):
    def __init__(self, num_classes, epsilon=0.1, use_gpu=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        targets_onehot = torch.zeros_like(log_probs).scatter_(1, targets. unsqueeze(1), 1)
        targets_smooth = (1 - self. epsilon) * targets_onehot + self.epsilon / self.num_classes
        loss = (-targets_smooth * log_probs).mean(0).sum()
        return loss


class TripletLoss_WRT(nn.Module):
    """Triplet Loss with Hard Mining"""
    def __init__(self, margin=0.3):
        super(TripletLoss_WRT, self).__init__()
        self.margin = margin
        self.ranking_loss = nn. MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets, normalize_feature=True):
        if normalize_feature: 
            inputs = normalize(inputs, axis=-1)
        
        n = inputs.size(0)
        dist = euclidean_dist(inputs, inputs)
        
        # Mask
        is_same = targets.unsqueeze(0).eq(targets.unsqueeze(1))
        is_diff = ~is_same
        is_same. fill_diagonal_(False)  # 排除自己
        
        # Hard mining
        dist_ap, dist_an = [], []
        for i in range(n):
            # Hardest positive (最远的正样本)
            if is_same[i].sum() > 0:
                dist_ap.append(dist[i][is_same[i]].max())
            else:
                dist_ap.append(torch.tensor(0.0, device=inputs.device))
            
            # Hardest negative (最近的负样本)
            if is_diff[i].sum() > 0:
                dist_an.append(dist[i][is_diff[i]].min())
            else:
                dist_an.append(torch. tensor(self.margin + 1.0, device=inputs.device))
        
        dist_ap = torch. stack(dist_ap)
        dist_an = torch.stack(dist_an)
        
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        
        return loss


class Weak_loss(nn.Module):
    def __init__(self, tau=0.05, q=0.5, ratio=0):
        super(Weak_loss, self).__init__()

    def forward(self, scores, labels):
        eps = 1e-10
        n, c = scores.shape
        probs = F.softmax(scores, dim=1)
        
        label_mask = torch.zeros_like(probs, dtype=torch.bool)
        label_mask. scatter_(1, labels. unsqueeze(1), 1)
        
        label_probs = probs[label_mask]. view(n, -1).min(dim=1, keepdim=True)[0]
        mask = (probs < label_probs).float()
        
        loss = -(torch.log(1 - probs + eps) * mask).sum(1).mean()
        return loss