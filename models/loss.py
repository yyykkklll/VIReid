"""
models/loss.py - 包含对抗损失的完整版
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiPartIDLoss(nn.Module):
    def __init__(self, label_smoothing=0.1):
        super(MultiPartIDLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    def forward(self, logits_list, labels):
        loss = 0
        for logits in logits_list:
            loss += self.criterion(logits, labels)
        return loss / len(logits_list)

class TripletLoss(nn.Module):
    def __init__(self, margin=0.3):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)
    def forward(self, features, labels):
        dist_mat = torch.cdist(features, features)
        N = dist_mat.size(0)
        is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())
        is_neg = ~is_pos
        mask_diag = torch.eye(N, dtype=torch.bool, device=features.device)
        is_pos = is_pos & ~mask_diag
        dist_ap = []
        dist_an = []
        for i in range(N):
            pos_dists = dist_mat[i][is_pos[i]]
            neg_dists = dist_mat[i][is_neg[i]]
            if pos_dists.numel() > 0: dist_ap.append(pos_dists.max())
            else: dist_ap.append(torch.tensor(0.0, device=features.device))
            if neg_dists.numel() > 0: dist_an.append(neg_dists.min())
            else: dist_an.append(torch.tensor(1e6, device=features.device))
        dist_ap = torch.stack(dist_ap)
        dist_an = torch.stack(dist_an)
        y = torch.ones_like(dist_an)
        return self.ranking_loss(dist_an, dist_ap, y)

class GraphDistillationLoss(nn.Module):
    def __init__(self, temperature=3.0):
        super(GraphDistillationLoss, self).__init__()
        self.temperature = temperature
        self.kl_div = nn.KLDivLoss(reduction='batchmean')
    def forward(self, logits_list, soft_labels_list, entropy_weights=None):
        loss = 0
        T = self.temperature
        for k, (logits, soft_labels) in enumerate(zip(logits_list, soft_labels_list)):
            log_probs = F.log_softmax(logits / T, dim=1)
            kl_loss = self.kl_div(log_probs, soft_labels)
            if entropy_weights is not None:
                kl_loss = (kl_loss * entropy_weights[k]).mean()
            loss += kl_loss * (T * T)
        return loss / len(logits_list)

class TotalLoss(nn.Module):
    def __init__(self, num_parts=6, lambda_graph=0.1, lambda_triplet=1.0, 
                 lambda_adv=0.1, # 新增对抗损失权重
                 label_smoothing=0.1, start_epoch=20):
        super(TotalLoss, self).__init__()
        self.id_loss = MultiPartIDLoss(label_smoothing=label_smoothing)
        self.tri_loss = TripletLoss(margin=0.3)
        self.graph_loss = GraphDistillationLoss(temperature=3.0)
        self.adv_loss = nn.CrossEntropyLoss() # 对抗分类损失
        
        self.lambda_graph = lambda_graph
        self.lambda_triplet = lambda_triplet
        self.lambda_adv = lambda_adv
        self.start_epoch = start_epoch
    
    def forward(self, outputs, labels, current_epoch=0, modality_labels=None, **kwargs):
        # 1. ID & Triplet
        loss_id = self.id_loss(outputs['id_logits'], labels)
        full_feat = torch.cat(outputs['id_features'], dim=1)
        full_feat = F.normalize(full_feat, p=2, dim=1)
        loss_tri = self.tri_loss(full_feat, labels)
        
        # 2. Graph Loss
        loss_graph = torch.tensor(0.0, device=labels.device)
        if current_epoch >= self.start_epoch and outputs['soft_labels'] is not None:
            logits_for_graph = outputs.get('graph_logits', outputs['id_logits'])
            loss_graph = self.graph_loss(logits_for_graph, outputs['soft_labels'], outputs['entropy_weights'])
            
        # 3. [策略三] Adversarial Loss
        loss_adv = torch.tensor(0.0, device=labels.device)
        if outputs['adv_logits'] is not None and modality_labels is not None:
            # 计算所有部件判别器的损失平均
            adv_loss_sum = 0
            for logit in outputs['adv_logits']:
                adv_loss_sum += self.adv_loss(logit, modality_labels)
            loss_adv = adv_loss_sum / len(outputs['adv_logits'])

        total = loss_id + self.lambda_triplet * loss_tri + self.lambda_graph * loss_graph + self.lambda_adv * loss_adv
        
        return total, {
            'loss_total': total.item(),
            'loss_id': loss_id.item(),
            'loss_triplet': loss_tri.item(),
            'loss_graph': loss_graph.item(),
            'loss_adv': loss_adv.item()
        }