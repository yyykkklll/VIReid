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

class CrossEntropyLabelSmooth(nn.Module):
    """
    [Strong Baseline] 标签平滑交叉熵损失
    相比标准 CE，它能防止模型对训练数据过拟合，提升泛化能力。
    """
    def __init__(self, num_classes, epsilon=0.1, use_gpu=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (batch_size)
        """
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
        if self.use_gpu: targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).mean(0).sum()
        return loss

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
    Compute contrastive loss for weak supervision
    """
    def __init__(self, tau=0.05, q=0.5, ratio=0):
        super(Weak_loss, self).__init__()
        self.tau = tau
        self.q = q
        self.ratio = ratio

    def forward(self, scores, labels):
        eps = 1e-10
        n, c = scores.shape
        scores = scores.exp()
        scores = scores/((scores.sum(1,keepdim=True))+eps)
        
        # [修改] 关键修复：处理索引标签，转换为 One-hot 掩码
        if labels.dim() == 1:
            # 如果输入是 (Batch,) 的索引，转为 (Batch, Num_Classes) 的 bool 掩码
            label_mask = torch.zeros((n, c), device=scores.device, dtype=torch.bool)
            label_mask.scatter_(1, labels.unsqueeze(1), 1)
        else:
            # 如果已经是掩码形式，直接转 bool
            label_mask = labels.bool()

        label_probs = torch.zeros((n,1), device=scores.device)
        for i in range(n):
            # 此时 label_mask[i] 是一个长度为 C 的向量，其中只有一个 True
            # scores[i][label_mask[i]] 会正确选出那个 True 对应的概率值
            min_prob = scores[i][label_mask[i]].min()
            label_probs[i] = min_prob
            
        mask = (scores < label_probs).int()    
        criterion = lambda x: -((1. - x + eps).log() * mask).sum(1).mean()
        return criterion(scores)