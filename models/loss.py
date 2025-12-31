import torch
import torch.nn as nn
import torch.nn.functional as F


def normalize(x, axis=-1):
    """L2 normalization"""
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x


def pdist_torch(emb1, emb2):
    """Compute pairwise distance matrix efficiently"""
    m, n = emb1.shape[0], emb2.shape[0]
    emb1_pow = torch.pow(emb1, 2).sum(dim=1, keepdim=True).expand(m, n)
    emb2_pow = torch.pow(emb2, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_mtx = emb1_pow + emb2_pow
    dist_mtx = dist_mtx.addmm_(emb1, emb2.t(), beta=1, alpha=-2)
    dist_mtx = dist_mtx.clamp(min=1e-12).sqrt()
    return dist_mtx


def softmax_weights(dist, mask):
    """Compute softmax weights for weighted triplet loss"""
    max_v = torch.max(dist * mask, dim=1, keepdim=True)[0]
    diff = dist - max_v
    Z = torch.sum(torch.exp(diff) * mask, dim=1, keepdim=True) + 1e-6
    W = torch.exp(diff) * mask / Z
    return W


class TripletLoss_WRT(nn.Module):
    """Weighted Regularized Triplet Loss"""
    def __init__(self):
        super(TripletLoss_WRT, self).__init__()
        self.ranking_loss = nn.SoftMarginLoss()

    def forward(self, inputs, targets, normalize_feature=False, weights=None):
        if normalize_feature:
            inputs = normalize(inputs, axis=-1)
            
        dist_mat = pdist_torch(inputs, inputs)

        # For distributed training, gathering implies full batch handling
        # Here assuming single GPU or gathered inputs
        N = dist_mat.size(0)
        
        # Shape: [N, N]
        is_pos = targets.expand(N, N).eq(targets.expand(N, N).t())
        is_neg = targets.expand(N, N).ne(targets.expand(N, N).t())

        # Compute weights for hard mining
        dist_ap = dist_mat * is_pos.float()
        dist_an = dist_mat * is_neg.float()
        
        weights_ap = softmax_weights(dist_ap, is_pos)
        weights_an = softmax_weights(-dist_an, is_neg)
        
        furthest_positive = torch.sum(dist_ap * weights_ap, dim=1)
        closest_negative = torch.sum(dist_an * weights_an, dim=1)
        
        # Compute ranking loss
        y = furthest_positive.new().resize_as_(furthest_positive).fill_(1)
        loss = self.ranking_loss(closest_negative - furthest_positive, y)
        
        # Apply reliability weighting if provided
        if weights is not None:
            if weights.device != loss.device:
                weights = weights.to(loss.device)
            # Weights shape usually [N], loss is scalar (mean).
            # To apply sample weights, we need reduction='none' in SoftMarginLoss ideally,
            # but since standard implementation uses mean, we apply heuristic scaling or
            # would need to reimplement ranking_loss manually.
            # Assuming 'weights' here refers to RAGM sample reliability:
            # Re-calculating unreduced loss for weighting:
            loss_per_sample = torch.log1p(torch.exp(furthest_positive - closest_negative))
            loss = (loss_per_sample * weights).mean()

        return loss


class Weak_loss(nn.Module):
    """
    Weak Supervision Loss optimized with vectorization.
    """
    def __init__(self):
        super(Weak_loss, self).__init__()

    def forward(self, scores, labels, reduction='mean', weights=None):
        eps = 1e-10
        
        # Softmax normalization
        scores = scores.exp()
        scores = scores / (scores.sum(1, keepdim=True) + eps)
        
        # Create Boolean mask
        label_mask = labels.bool()
        
        # [Optimized] Vectorized calculation of min_prob for positive classes
        # Set scores of non-target classes to infinity so they don't affect min()
        masked_scores = scores.clone()
        masked_scores[~label_mask] = float('inf')
        
        # Find minimum score among positive labels for each sample
        # values: [N], indices: [N]
        label_probs, _ = masked_scores.min(dim=1)
        
        # Handle cases where a sample might have no positive labels (unlikely but safe)
        # If min is inf, replace with 0 or handle strictly
        label_probs[label_probs == float('inf')] = 0.0
        
        # Reshape for broadcasting: [N, 1]
        label_probs = label_probs.unsqueeze(1)
        
        # Compute mask for negative classes (score < min_positive_score)
        mask = (scores < label_probs).float()
        
        # Compute Complementary Cross Entropy
        loss_per_sample = -((1. - scores + eps).log() * mask).sum(dim=1)
        
        # Apply RAGM Reliability Weights
        if weights is not None:
            if weights.device != scores.device:
                weights = weights.to(scores.device)
            loss_per_sample = loss_per_sample * weights
        
        if reduction == 'mean':
            return loss_per_sample.mean()
        elif reduction == 'sum':
            return loss_per_sample.sum()
        
        return loss_per_sample
    
class DiffusionLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.MSELoss()

    def forward(self, pred_noise, real_noise):
        return self.criterion(pred_noise, real_noise)

