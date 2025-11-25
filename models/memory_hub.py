import torch
import torch.nn as nn
import torch.nn.functional as F


class MemoryHub(nn.Module):
    def __init__(self, num_parts=6, num_classes=500, feat_dim=256, momentum=0.2, temp=0.05, topk=5):
        super(MemoryHub, self).__init__()
        self.num_parts = num_parts
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.momentum = momentum
        self.temp = temp
        self.topk = topk

        # 记忆库: [K, N, D]
        self.register_buffer('memory', torch.randn(num_parts, num_classes, feat_dim))
        # 初始化归一化
        self.memory = F.normalize(self.memory, p=2, dim=2)

    def forward(self, id_feats):
        """
        图传播：计算 Soft Labels
        id_feats: List of [B, D], len=K
        Return: List of [B, N] (soft labels)
        """
        soft_targets = []
        for k in range(self.num_parts):
            # [B, D] x [N, D]^T -> [B, N]
            # self.memory[k]: [N, D]
            feat = id_feats[k]  # [B, D]
            mem = self.memory[k].clone().detach()  # [N, D]

            sim_matrix = torch.matmul(feat, mem.t())  # [B, N]

            # Top-K 过滤 (Sparse Softmax)
            # 只保留最大的K个值，其余设为负无穷
            topk_val, topk_idx = torch.topk(sim_matrix, self.topk, dim=1)
            mask = torch.zeros_like(sim_matrix).scatter_(1, topk_idx, 1.0)
            sim_matrix_sparse = sim_matrix.masked_fill(mask == 0, -1e9)

            soft_target = F.softmax(sim_matrix_sparse / self.temp, dim=1)
            soft_targets.append(soft_target)

        return soft_targets

    @torch.no_grad()
    def update(self, id_feats, labels):
        """
        动量更新
        id_feats: List of [B, D]
        labels: [B]
        """
        for k in range(self.num_parts):
            # 获取当前 batch 中每个 label 对应的特征中心
            # 简单起见，我们对 batch 内同一个 label 取平均（虽然 ReID batch 通常 label 唯一）
            unique_labels = torch.unique(labels)
            for label in unique_labels:
                # 选出属于该 label 的样本特征
                idx = (labels == label)
                feat_mean = id_feats[k][idx].mean(dim=0)  # [D]
                feat_mean = F.normalize(feat_mean, p=2, dim=0)

                # 动量更新
                # M[k, label] = m * M + (1-m) * new
                self.memory[k, label] = self.momentum * self.memory[k, label] + \
                                        (1 - self.momentum) * feat_mean
                # 再次归一化
                self.memory[k, label] = F.normalize(self.memory[k, label], p=2, dim=0)