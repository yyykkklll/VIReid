"""
models/loss.py (Fixed)
[修改] 将 Graph Loss 介入时间由 5 推迟到 20
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
    def forward(self, features, labels):
        dist_mat = torch.cdist(features, features)
        N = dist_mat.size(0)
        is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())
        dist_ap, _ = torch.max(dist_mat * is_pos.float(), dim=1)
        dist_an, _ = torch.min(dist_mat + is_pos.float() * 1e6, dim=1)
        y = torch.ones_like(dist_an)
        return self.ranking_loss(dist_an, dist_ap, y)

class GraphDistillationLoss(nn.Module):
    def __init__(self, temperature=3.0):
        super(GraphDistillationLoss, self).__init__()
        self.temperature = temperature
        self.kl_div = nn.KLDivLoss(reduction='batchmean')
    
    def forward(self, logits_list, soft_labels_list):
        loss = 0
        T = self.temperature
        for logits, soft_labels in zip(logits_list, soft_labels_list):
            log_probs = F.log_softmax(logits / T, dim=1)
            loss += self.kl_div(log_probs, soft_labels) * (T * T)
        return loss / len(logits_list)

class TotalLoss(nn.Module):
    def __init__(self, lambda_graph=0.1, start_epoch=20, **kwargs):
        super(TotalLoss, self).__init__()
        self.id_loss = MultiPartIDLoss()
        self.tri_loss = TripletLoss(margin=0.3)
        self.graph_loss = GraphDistillationLoss()
        self.lambda_graph = lambda_graph
        self.start_epoch = start_epoch
    
    def forward(self, outputs, labels, current_epoch=0, **kwargs):
        # 1. ID Loss (使用带 Dropout 的 Logits)
        loss_id = self.id_loss(outputs['id_logits'], labels)
        
        # 2. Triplet Loss
        full_feat = torch.cat(outputs['id_features'], dim=1)
        full_feat = F.normalize(full_feat, p=2, dim=1)
        loss_tri = self.tri_loss(full_feat, labels)
        
        # 3. Graph Loss
        loss_graph = torch.tensor(0.0, device=labels.device)
        if current_epoch >= self.start_epoch and 'soft_labels' in outputs:
            # [关键修改] 使用 Clean Logits 计算 Graph Loss
            # 如果模型输出了 graph_logits 则使用，否则回退到 id_logits
            logits_for_graph = outputs.get('graph_logits', outputs['id_logits'])
            loss_graph = self.graph_loss(logits_for_graph, outputs['soft_labels'])
            
        total = loss_id + loss_tri + self.lambda_graph * loss_graph
        
        return total, {
            'loss_total': total.item(),
            'loss_id': loss_id.item(),
            'loss_triplet': loss_tri.item(),
            'loss_graph': loss_graph.item()
        }