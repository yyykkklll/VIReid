"""
models/loss.py - Unified Loss with Stability Fix
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
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)
    
    def forward(self, feat, labels, target_feat=None):
        if target_feat is None: target_feat = feat
        dist = torch.cdist(feat, target_feat)
        N = dist.size(0)
        
        is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())
        is_neg = ~is_pos
        
        if target_feat is feat:
            mask_diag = torch.eye(N, dtype=torch.bool, device=feat.device)
            is_pos = is_pos & ~mask_diag
            
        dist_ap, dist_an = [], []
        for i in range(N):
            dist_ap.append(dist[i][is_pos[i]].max() if is_pos[i].any() else torch.tensor(0., device=feat.device))
            dist_an.append(dist[i][is_neg[i]].min() if is_neg[i].any() else torch.tensor(1e6, device=feat.device))
            
        dist_ap = torch.stack(dist_ap)
        dist_an = torch.stack(dist_an)
        y = torch.ones_like(dist_an)
        return self.ranking_loss(dist_an, dist_ap, y)

class GraphDistillationLoss(nn.Module):
    def __init__(self, temperature=3.0):
        super(GraphDistillationLoss, self).__init__()
        # [关键] 使用 reduction='none' 以便后续进行 clamp 和加权
        self.kl = nn.KLDivLoss(reduction='none')
        self.T = temperature

    def forward(self, logits_list, soft_labels_list, weights=None):
        loss = 0
        for k, (logit, label) in enumerate(zip(logits_list, soft_labels_list)):
            log_prob = F.log_softmax(logit / self.T, dim=1)
            
            # 计算逐样本 KL 散度 [B]
            l = self.kl(log_prob, label).sum(dim=1)
            
            # [关键急救] 截断 Loss，防止脏数据或极端分布导致梯度爆炸
            l = torch.clamp(l, max=5.0)
            
            if weights is not None: 
                l = (l * weights[k]).mean()
            else:
                l = l.mean()
                
            loss += l * (self.T**2)
        return loss / len(logits_list)

class TotalLoss(nn.Module):
    def __init__(self, num_parts=6, lambda_graph=0.1, lambda_triplet=1.0, lambda_adv=0.1,
                 label_smoothing=0.1, start_epoch=20):
        super(TotalLoss, self).__init__()
        self.id_loss = MultiPartIDLoss(label_smoothing)
        self.tri_loss = TripletLoss(margin=0.3)
        self.graph_loss = GraphDistillationLoss(temperature=3.0)
        self.adv_loss = nn.CrossEntropyLoss()
        
        self.lambdas = {'graph': lambda_graph, 'triplet': lambda_triplet, 'adv': lambda_adv}
        self.start_epoch = start_epoch

    def forward(self, outputs, labels, current_epoch=0, modality_labels=None, **kwargs):
        # 1. ID Loss
        L_id = self.id_loss(outputs['id_logits'], labels)
        
        # 2. MTRL Triplet Loss
        feat = F.normalize(torch.cat(outputs['id_features'], dim=1), p=2, dim=1)
        L_tri = self.tri_loss(feat, labels)
        
        L_trans = torch.tensor(0., device=labels.device)
        if outputs['gray_features']:
            feat_gray = F.normalize(torch.cat(outputs['gray_features'], dim=1), p=2, dim=1)
            L_trans = self.tri_loss(feat, labels, feat_gray)
            L_tri += 0.5 * L_trans
            
        # 3. Graph Loss (Safe Mode)
        L_graph = torch.tensor(0., device=labels.device)
        if current_epoch >= self.start_epoch and outputs['soft_labels']:
            # 使用 id_logits 而非 graph_logits (如果模型没有独立 graph head)
            logits = outputs.get('id_logits', [])
            L_graph = self.graph_loss(logits, outputs['soft_labels'], outputs['entropy_weights'])
            
        # 4. Adv Loss
        L_adv = torch.tensor(0., device=labels.device)
        if outputs['adv_logits'] and modality_labels is not None:
            for l in outputs['adv_logits']: L_adv += self.adv_loss(l, modality_labels)
            L_adv /= len(outputs['adv_logits'])

        total = L_id + \
                self.lambdas['triplet'] * L_tri + \
                self.lambdas['graph'] * L_graph + \
                self.lambdas['adv'] * L_adv
                
        return total, {
            'loss_total': total.item(),
            'loss_id': L_id.item(),
            'loss_triplet': L_tri.item(),
            'loss_graph': L_graph.item(),
            'loss_adv': L_adv.item()
        }