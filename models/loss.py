"""
Optimized Loss Functions for Cross-Modal Person Re-ID
=====================================================
Components:
1. TripletLoss_WRT - Original weighted regularized triplet loss
2. Weak_loss - Original weak supervision loss
3. LabelSmoothingCrossEntropy - NEW: Label smoothing for better generalization
4. InfoNCELoss - NEW: Contrastive learning for cross-modal alignment
5. AdaptiveLossWeighting - NEW: Automatic loss balancing

All original functions and classes are preserved for backward compatibility.
"""

import torch 
import torch.nn as nn
import torch.nn.functional as F


# ============= Original Helper Functions (Keep Unchanged) =============
def normalize(x, axis=-1):
    """Normalize features along specified axis"""
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x


def pdist_torch(emb1, emb2):
    """Compute pairwise Euclidean distance matrix"""
    m, n = emb1.shape[0], emb2.shape[0]
    emb1_pow = torch.pow(emb1, 2).sum(dim=1, keepdim=True).expand(m, n)
    emb2_pow = torch.pow(emb2, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_mtx = emb1_pow + emb2_pow
    dist_mtx = dist_mtx.addmm_(1, -2, emb1, emb2.t())
    dist_mtx = dist_mtx.clamp(min=1e-12).sqrt()
    return dist_mtx


def softmax_weights(dist, mask):
    """Compute softmax weights for distance matrix"""
    max_v = torch.max(dist * mask, dim=1, keepdim=True)[0]
    diff = dist - max_v
    Z = torch.sum(torch.exp(diff) * mask, dim=1, keepdim=True) + 1e-6  # avoid division by zero
    W = torch.exp(diff) * mask / Z
    return W


# ============= Original Loss Classes (Keep Unchanged) =============
class TripletLoss_WRT(nn.Module):
    """
    Weighted Regularized Triplet Loss
    
    Original implementation - preserved for backward compatibility
    """
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
    Weak Supervision Loss
    
    Original implementation - preserved for backward compatibility
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
        scores = scores / ((scores.sum(1, keepdim=True)) + eps)
        
        label_mask = labels.bool()
        label_probs = torch.zeros((n, 1), device=scores.device)
        for i in range(n):
            min_prob = scores[i][label_mask[i]].min()
            label_probs[i] = min_prob
        mask = (scores < label_probs).int()    
        criterion = lambda x: -((1. - x + eps).log() * mask).sum(1).mean()
        return criterion(scores)


# ============= NEW Loss Classes =============
class LabelSmoothingCrossEntropy(nn.Module):
    """
    Label Smoothing Cross Entropy Loss
    
    Prevents overconfidence and improves generalization by smoothing
    hard labels with uniform distribution.
    
    Args:
        epsilon: Smoothing factor (default: 0.1)
        reduction: 'mean', 'sum', or 'none'
    
    Example:
        >>> criterion = LabelSmoothingCrossEntropy(epsilon=0.1)
        >>> loss = criterion(predictions, targets)
    """
    def __init__(self, epsilon=0.1, reduction='mean'):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction
    
    def forward(self, preds, target):
        """
        Args:
            preds: [N, C] prediction logits
            target: [N] ground truth labels (long tensor)
        
        Returns:
            loss: Scalar loss value
        """
        n_classes = preds.size(1)
        log_preds = F.log_softmax(preds, dim=1)
        
        # Create smoothed labels
        with torch.no_grad():
            true_dist = torch.zeros_like(log_preds)
            true_dist.fill_(self.epsilon / (n_classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.epsilon)
        
        loss = torch.sum(-true_dist * log_preds, dim=1)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class InfoNCELoss(nn.Module):
    """
    InfoNCE Contrastive Loss for Cross-Modal Learning
    
    Pulls positive pairs together and pushes negative pairs apart
    in the embedding space using temperature-scaled cosine similarity.
    
    Args:
        temperature: Temperature scaling factor (default: 0.07)
    
    Example:
        >>> criterion = InfoNCELoss(temperature=0.07)
        >>> loss = criterion(features_rgb, features_ir, labels_rgb, labels_ir)
    """
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, features_a, features_b, labels_a, labels_b):
        """
        Args:
            features_a: Features from modality A [N, D]
            features_b: Features from modality B [M, D]
            labels_a: Labels for modality A [N]
            labels_b: Labels for modality B [M]
        
        Returns:
            loss: Scalar contrastive loss
        """
        # Normalize features
        features_a = F.normalize(features_a, dim=1)
        features_b = F.normalize(features_b, dim=1)
        
        # Compute similarity matrix
        similarity = torch.matmul(features_a, features_b.t()) / self.temperature
        
        # Create positive mask
        labels_a = labels_a.unsqueeze(1)  # [N, 1]
        labels_b = labels_b.unsqueeze(0)  # [1, M]
        positive_mask = (labels_a == labels_b).float()
        
        # Avoid numerical instability
        similarity = torch.clamp(similarity, min=-50, max=50)
        
        # Compute loss
        exp_sim = torch.exp(similarity)
        
        # For each sample in A, sum over positive pairs in B
        pos_sum = (exp_sim * positive_mask).sum(dim=1, keepdim=True)
        neg_sum = exp_sim.sum(dim=1, keepdim=True)
        
        # Avoid division by zero
        pos_sum = torch.clamp(pos_sum, min=1e-8)
        
        loss = -torch.log(pos_sum / neg_sum).mean()
        
        return loss


class AdaptiveLossWeighting:
    """
    Adaptive Loss Weighting based on loss magnitude
    
    Automatically balances multiple loss terms using exponential
    moving average (EMA) of loss magnitudes.
    
    Args:
        num_losses: Number of loss terms to balance
        init_weights: Initial weights (optional)
        alpha: EMA coefficient (default: 0.9)
    
    Example:
        >>> weighter = AdaptiveLossWeighting(num_losses=5)
        >>> weighter.update([loss1, loss2, loss3, loss4, loss5])
        >>> weights = weighter.get_weights()
    """
    def __init__(self, num_losses, init_weights=None, alpha=0.9):
        self.num_losses = num_losses
        self.alpha = alpha
        
        if init_weights is None:
            self.weights = [1.0] * num_losses
        else:
            self.weights = init_weights
        
        self.loss_history = [[] for _ in range(num_losses)]
        self.avg_losses = [1.0] * num_losses
    
    def update(self, losses):
        """
        Update weights based on loss magnitudes
        
        Args:
            losses: List of loss values (can contain None for inactive losses)
        """
        for i, loss in enumerate(losses):
            if loss is None or loss == 0:
                continue
                
            self.loss_history[i].append(loss)
            
            # Compute EMA of loss
            if len(self.loss_history[i]) == 1:
                self.avg_losses[i] = loss
            else:
                self.avg_losses[i] = self.alpha * self.avg_losses[i] + (1 - self.alpha) * loss
        
        # Normalize weights inversely proportional to loss magnitude
        total = sum([1.0 / (avg_loss + 1e-8) for avg_loss in self.avg_losses])
        self.weights = [1.0 / (avg_loss + 1e-8) / total * self.num_losses 
                       for avg_loss in self.avg_losses]
    
    def get_weights(self):
        """Get current weights"""
        return self.weights
    
    def get_avg_losses(self):
        """Get average losses"""
        return self.avg_losses
