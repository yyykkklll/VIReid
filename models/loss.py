import torch 
import torch.nn as nn
def normalize(x, axis=-1):
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x

def pdist_torch(emb1, emb2):
    m, n = emb1.shape[0], emb2.shape[0]
    emb1_pow = torch.pow(emb1, 2).sum(dim=1, keepdim=True).expand(m, n)
    emb2_pow = torch.pow(emb2, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_mtx = emb1_pow + emb2_pow
    dist_mtx = dist_mtx.addmm_(1, -2, emb1, emb2.t())
    dist_mtx = dist_mtx.clamp(min=1e-12).sqrt()
    return dist_mtx

def softmax_weights(dist, mask):
    max_v = torch.max(dist * mask, dim=1, keepdim=True)[0]
    diff = dist - max_v
    Z = torch.sum(torch.exp(diff) * mask, dim=1, keepdim=True) + 1e-6 # avoid division by zero
    W = torch.exp(diff) * mask / Z
    return W

class TripletLoss_WRT(nn.Module):

    def __init__(self):
        super(TripletLoss_WRT, self).__init__()
        self.ranking_loss = nn.SoftMarginLoss()

    def forward(self, inputs, targets, normalize_feature=False):
        if normalize_feature:
            inputs = normalize(inputs, axis=-1)
        dist_mat = pdist_torch(inputs, inputs)

        N = dist_mat.size(0)
        is_pos = targets.expand(N, N).eq(targets.expand(N, N).t()).float()
        is_neg = targets.expand(N, N).ne(targets.expand(N, N).t()).float()

        dist_ap = dist_mat * is_pos
        dist_an = dist_mat * is_neg

        weights_ap = softmax_weights(dist_ap, is_pos)
        weights_an = softmax_weights(-dist_an, is_neg)
        furthest_positive = torch.sum(dist_ap * weights_ap, dim=1)
        closest_negative = torch.sum(dist_an * weights_an, dim=1)

        y = furthest_positive.new().resize_as_(furthest_positive).fill_(1)
        loss = self.ranking_loss(closest_negative - furthest_positive, y)

        return loss

class Weak_loss(nn.Module):
    """
    Compute contrastive loss
    """
    def __init__(self, tau=0.05, q=0.5, ratio=0):
        super(Weak_loss, self).__init__()
        self.tau = tau
        self.q = q
        self.ratio = ratio

    def forward(self, scores, labels):
        eps = 1e-10
        n = scores.shape[0]
        scores = scores.exp()
        scores = scores/((scores.sum(1,keepdim=True))+eps)
        # TODO 
        # torch.nn.functional.softmax()
        label_mask = labels.bool()
        label_probs = torch.zeros((n,1),device=scores.device)
        for i in range(n):
            min_prob = scores[i][label_mask[i]].min()
            label_probs[i] = min_prob
        mask = (scores < label_probs).int()    
        criterion = lambda x: -((1. - x + eps).log() * mask).sum(1).mean()
        return criterion(scores)

class WeightedContrastiveLoss(nn.Module):
    def __init__(self, margin=0.5):
        super(WeightedContrastiveLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin, reduction='none')

    def forward(self, x1, x2, weights):
        """
        x1: feat 1 (N, D)
        x2: feat 2 (N, D)
        weights: (N,) confidence scores. Higher means more likely positive.
        
        We treat these as positive pairs weighted by confidence.
        Standard Contrastive Loss for positive pair: D(x1, x2)^2
        Here we can use: weight * D(x1, x2)^2
        
        Or if we consider them as "soft" positives, maybe we just want to minimize distance proportional to weight.
        Let's use Euclidean distance squared weighted.
        """
        dist_sq = torch.sum(torch.pow(x1 - x2, 2), dim=1)
        loss = weights * dist_sq
        return loss.mean()