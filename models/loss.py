"""
Loss Functions for PF-MGCD
实现四大损失函数:
1. Multi-Part ID Loss (多粒度身份分类损失)
2. Graph Distillation Loss (图蒸馏损失)
3. Orthogonal Loss (正交解耦损失)
4. Modality Loss (模态判别损失)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiPartIDLoss(nn.Module):
    """
    多粒度身份分类损失
    对K个部件分别计算交叉熵损失
    支持Label Smoothing
    """
    
    def __init__(self, num_parts=6, label_smoothing=0.1):
        """
        Args:
            num_parts: 部件数量 K
            label_smoothing: 标签平滑系数
        """
        super(MultiPartIDLoss, self).__init__()
        self.num_parts = num_parts
        self.label_smoothing = label_smoothing
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    
    def forward(self, id_logits_list, labels):
        """
        Args:
            id_logits_list: List of K个logits [B, N]
            labels: 真实标签 [B]
        Returns:
            loss: 标量损失
        """
        total_loss = 0.0
        
        for k in range(self.num_parts):
            logits = id_logits_list[k]  # [B, N]
            loss = self.criterion(logits, labels)
            total_loss += loss
        
        # 平均
        return total_loss / self.num_parts


class GraphDistillationLoss(nn.Module):
    """
    [修复版] 图蒸馏损失 (核心弱监督Loss)
    让学生网络的预测分布拟合图传播的软标签
    使用 Soft Cross Entropy 替代 KLDiv 以避免 NaN
    支持自适应熵加权
    """
    
    def __init__(self, num_parts=6, use_adaptive_weight=True):
        """
        Args:
            num_parts: 部件数量 K
            use_adaptive_weight: 是否使用熵自适应权重
        """
        super(GraphDistillationLoss, self).__init__()
        self.num_parts = num_parts
        self.use_adaptive_weight = use_adaptive_weight
    
    def forward(self, id_logits_list, soft_labels_list, entropy_weights_list=None):
        """
        Args:
            id_logits_list: List of K个学生网络logits [B, N]
            soft_labels_list: List of K个图传播软标签 [B, N]
            entropy_weights_list: List of K个熵权重 [B] (可选)
        Returns:
            loss: 标量损失
        """
        total_loss = 0.0
        
        for k in range(self.num_parts):
            logits = id_logits_list[k]  # [B, N]
            soft_labels = soft_labels_list[k]  # [B, N]
            
            # Log Softmax
            log_probs = F.log_softmax(logits, dim=1)
            
            # [关键修复] 使用软标签交叉熵: -sum(target * log(pred))
            # 避免 KLDiv 在 target=0 时计算 0*log(0) 导致的 NaN
            # 注意：soft_labels 应该是 detached 的 (不需要梯度)
            ce_loss = -torch.sum(soft_labels * log_probs, dim=1)  # [B]
            
            # 应用自适应权重
            if self.use_adaptive_weight and entropy_weights_list is not None:
                weights = entropy_weights_list[k]  # [B]
                ce_loss = ce_loss * weights
            
            total_loss += ce_loss.mean()
        
        # 平均
        return total_loss / self.num_parts


class OrthogonalLoss(nn.Module):
    """
    正交解耦损失
    强制身份特征 f_id 和模态特征 f_mod 在特征空间中正交
    """
    
    def __init__(self, num_parts=6):
        """
        Args:
            num_parts: 部件数量 K
        """
        super(OrthogonalLoss, self).__init__()
        self.num_parts = num_parts
    
    def forward(self, id_features_list, mod_features_list):
        """
        Args:
            id_features_list: List of K个身份特征 [B, D]
            mod_features_list: List of K个模态特征 [B, D]
        Returns:
            loss: 标量损失
        """
        total_loss = 0.0
        
        for k in range(self.num_parts):
            f_id = id_features_list[k]  # [B, D]
            f_mod = mod_features_list[k]  # [B, D]
            
            # 归一化
            f_id_norm = F.normalize(f_id, dim=1)
            f_mod_norm = F.normalize(f_mod, dim=1)
            
            # 计算余弦相似度 (应该接近0)
            cos_sim = torch.sum(f_id_norm * f_mod_norm, dim=1)  # [B]
            
            # 平方损失 (让cos_sim接近0)
            orth_loss = (cos_sim ** 2).mean()
            
            total_loss += orth_loss
        
        # 平均
        return total_loss / self.num_parts


class ModalityLoss(nn.Module):
    """
    模态判别损失
    强制模态特征 f_mod 包含模态信息 (可见光/红外)
    """
    
    def __init__(self, num_parts=6):
        """
        Args:
            num_parts: 部件数量 K
        """
        super(ModalityLoss, self).__init__()
        self.num_parts = num_parts
        self.criterion = nn.CrossEntropyLoss()
    
    def forward(self, mod_logits_list, modality_labels):
        """
        Args:
            mod_logits_list: List of K个模态logits [B, 2]
            modality_labels: 模态标签 [B] (0=可见光, 1=红外)
        Returns:
            loss: 标量损失
        """
        total_loss = 0.0
        
        for k in range(self.num_parts):
            logits = mod_logits_list[k]  # [B, 2]
            loss = self.criterion(logits, modality_labels)
            total_loss += loss
        
        # 平均
        return total_loss / self.num_parts


class TotalLoss(nn.Module):
    """
    总损失函数
    L_total = L_id + λ1*L_graph + λ2*L_orth + λ3*L_mod
    """
    
    def __init__(self, 
                 num_parts=6,
                 lambda_graph=1.0,
                 lambda_orth=0.1,
                 lambda_mod=0.5,
                 label_smoothing=0.1,
                 use_adaptive_weight=True):
        """
        Args:
            num_parts: 部件数量 K
            lambda_graph: 图蒸馏损失权重
            lambda_orth: 正交损失权重
            lambda_mod: 模态损失权重
            label_smoothing: 标签平滑系数
            use_adaptive_weight: 是否使用熵自适应权重
        """
        super(TotalLoss, self).__init__()
        
        self.lambda_graph = lambda_graph
        self.lambda_orth = lambda_orth
        self.lambda_mod = lambda_mod
        
        # 四个损失函数
        self.id_loss_fn = MultiPartIDLoss(num_parts, label_smoothing)
        self.graph_loss_fn = GraphDistillationLoss(num_parts, use_adaptive_weight)
        self.orth_loss_fn = OrthogonalLoss(num_parts)
        self.mod_loss_fn = ModalityLoss(num_parts)
    
    def forward(self, outputs, labels, modality_labels, current_epoch=0, warmup_epochs=10):
        """
        Args:
            outputs: 模型输出字典
            labels: 身份标签 [B]
            modality_labels: 模态标签 [B]
            current_epoch: 当前epoch (用于warmup)
            warmup_epochs: warmup的epoch数
        Returns:
            total_loss: 总损失
            loss_dict: 各项损失的字典
        """
        # 提取需要的输出
        id_logits = outputs['id_logits']
        mod_logits = outputs['mod_logits']
        id_features = outputs['id_features']
        mod_features = outputs['mod_features']
        soft_labels = outputs['soft_labels']
        entropy_weights = outputs['entropy_weights']
        
        # 1. 身份分类损失
        loss_id = self.id_loss_fn(id_logits, labels)
        
        # 2. 图蒸馏损失 (warmup后才启用)
        if current_epoch >= warmup_epochs:
            loss_graph = self.graph_loss_fn(id_logits, soft_labels, entropy_weights)
            graph_weight = self.lambda_graph
        else:
            loss_graph = torch.tensor(0.0, device=labels.device)
            graph_weight = 0.0
        
        # 3. 正交损失
        loss_orth = self.orth_loss_fn(id_features, mod_features)
        
        # 4. 模态损失
        loss_mod = self.mod_loss_fn(mod_logits, modality_labels)
        
        # 总损失
        total_loss = (
            loss_id +
            graph_weight * loss_graph +
            self.lambda_orth * loss_orth +
            self.lambda_mod * loss_mod
        )
        
        # 返回损失字典 (用于日志记录)
        loss_dict = {
            'loss_total': total_loss.item(),
            'loss_id': loss_id.item(),
            'loss_graph': loss_graph.item() if isinstance(loss_graph, torch.Tensor) else 0.0,
            'loss_orth': loss_orth.item(),
            'loss_mod': loss_mod.item()
        }
        
        return total_loss, loss_dict