import torch
import torch.nn as nn
import torch.nn.functional as F

class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(InfoNCELoss, self).__init__()
        self.T = temperature

    def forward(self, features, targets=None):
        """
        features: [Batch_Size, Dim]
        简单版：假设 Batch 内前半部分是 View1，后半部分是 View2，互为正样本
        """
        # 归一化
        features = F.normalize(features, dim=1)
        
        # 计算相似度矩阵
        similarity_matrix = torch.matmul(features, features.T)
        
        # 排除自身 (对角线设为非常小)
        mask = torch.eye(features.shape[0], dtype=torch.bool).to(features.device)
        similarity_matrix.masked_fill_(mask, -9e15)
        
        # 找出正样本对 (假设 Batch 结构: [RGB, RGB_aug, IR, IR_aug] 或 [RGB, IR])
        # 这里实现一个通用的 SimCLR 风格 Loss
        # 假设输入已经是 [N, Dim]，其中 N=2B，0..B-1 和 B..2B-1 是正样本
        batch_size = features.shape[0] // 2
        
        # 正样本：对角线偏移 B 的位置
        pos_sim = torch.cat([
            torch.diag(similarity_matrix, batch_size),
            torch.diag(similarity_matrix, -batch_size)
        ], dim=0)
        
        pos_sim = pos_sim / self.T
        
        # 所有样本作为负样本
        # LogSumExp
        neg_sim = similarity_matrix / self.T
        loss = -pos_sim + torch.logsumexp(neg_sim, dim=1)
        
        return loss.mean()

class SinkhornLoss(nn.Module):
    def __init__(self, epsilon=0.05, iterations=3):
        super(SinkhornLoss, self).__init__()
        self.epsilon = epsilon
        self.iterations = iterations

    def forward(self, feat_rgb, feat_ir):
        """
        计算 RGB 分布和 IR 分布之间的最优传输距离 (Sinkhorn Distance)
        """
        # 归一化
        feat_rgb = F.normalize(feat_rgb, dim=1)
        feat_ir = F.normalize(feat_ir, dim=1)
        
        # Cost Matrix (1 - Cosine Similarity)
        C = 1 - torch.mm(feat_rgb, feat_ir.t())
        
        # Sinkhorn 迭代
        # 初始 Log 势能
        mu = torch.zeros(feat_rgb.shape[0], device=feat_rgb.device)
        nu = torch.zeros(feat_ir.shape[0], device=feat_ir.device)
        
        for _ in range(self.iterations):
            # 行归一化
            M = -C / self.epsilon + nu.unsqueeze(0)
            mu = -torch.logsumexp(M, dim=1)
            # 列归一化
            M = -C / self.epsilon + mu.unsqueeze(1)
            nu = -torch.logsumexp(M, dim=0)
            
        # 计算传输计划 P (Log domain)
        U = mu.unsqueeze(1) + nu.unsqueeze(0)
        P = torch.exp(U - C / self.epsilon)
        
        # Loss: 最优传输代价
        loss = torch.sum(P * C)
        return loss

class AdversarialLoss(nn.Module):
    def __init__(self):
        super(AdversarialLoss, self).__init__()
        self.criterion = nn.BCELoss()

    def forward(self, domain_preds, domain_labels):
        return self.criterion(domain_preds, domain_labels)