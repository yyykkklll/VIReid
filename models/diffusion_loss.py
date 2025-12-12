"""
Loss Functions for Feature Diffusion Bridge
包含：噪声预测 MSE、置信度引导、特征分布对齐
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiffusionLoss(nn.Module):
    def __init__(self, mse_weight=1.0, conf_weight=0.1, align_weight=0.05):
        """
        Args:
            mse_weight: 噪声预测 MSE 损失权重
            conf_weight: 红外分类器置信度引导权重
            align_weight: 特征分布对齐损失权重
        """
        super().__init__()
        self.mse_weight = mse_weight
        self.conf_weight = conf_weight
        self.align_weight = align_weight
    
    def forward(self, pred_noise, target_noise, z_0, f_v, W_r=None):
        """
        计算总损失
        Args:
            pred_noise: 预测的噪声 [batch_size, feat_dim]
            target_noise: 真实噪声 [batch_size, feat_dim]
            z_0: 采样的初始点
            f_v: 可见光特征
            W_r: 红外分类器权重 [num_classes, feat_dim]
        Returns:
            loss_dict: 各项损失
        """
        loss_dict = {}
        
        # 1. 噪声预测 MSE Loss（主损失）
        mse_loss = F.mse_loss(pred_noise, target_noise, reduction='mean')
        loss_dict['mse_loss'] = mse_loss
        
        # 2. 置信度引导 Loss（可选）
        conf_loss = torch.tensor(0.0).to(pred_noise.device)
        if W_r is not None and self.conf_weight > 0:
            conf_loss = self.confidence_guidance_loss(z_0, W_r)
            loss_dict['conf_loss'] = conf_loss
        
        # 3. 特征分布对齐 Loss（可选）
        align_loss = torch.tensor(0.0).to(pred_noise.device)
        if self.align_weight > 0:
            align_loss = self.distribution_alignment_loss(z_0, f_v)
            loss_dict['align_loss'] = align_loss
        
        # 总损失
        total_loss = (self.mse_weight * mse_loss + 
                     self.conf_weight * conf_loss + 
                     self.align_weight * align_loss)
        loss_dict['total_loss'] = total_loss
        
        return loss_dict
    
    def confidence_guidance_loss(self, z_0, W_r):
        """
        置信度引导：生成的特征应该在红外分类器上有较高置信度
        策略：最大化 z_0 与最近类别中心的余弦相似度
        Args:
            z_0: 生成的特征 [batch_size, feat_dim]
            W_r: 红外分类器权重 [num_classes, feat_dim]
        Returns:
            conf_loss: 置信度损失（越小越好）
        """
        # L2 归一化
        z_0_norm = F.normalize(z_0, p=2, dim=1)
        W_r_norm = F.normalize(W_r, p=2, dim=1)
        
        # 计算与所有类别的相似度
        sim = torch.mm(z_0_norm, W_r_norm.t())  # [batch_size, num_classes]
        
        # 最大相似度（代表置信度）
        max_sim = sim.max(dim=1)[0]  # [batch_size]
        
        # 损失：最大化置信度 → 最小化负相似度
        conf_loss = -max_sim.mean()
        
        return conf_loss
    
    def distribution_alignment_loss(self, z_0, f_v):
        """
        特征分布对齐：z_0 的统计特性应与 f_v 接近
        策略：匹配均值和标准差
        Args:
            z_0: 生成的特征
            f_v: 可见光特征
        Returns:
            align_loss: 分布对齐损失
        """
        # 计算均值和标准差
        z_mean = z_0.mean(dim=0)
        z_std = z_0.std(dim=0)
        
        f_mean = f_v.mean(dim=0)
        f_std = f_v.std(dim=0)
        
        # L2 距离
        mean_loss = F.mse_loss(z_mean, f_mean)
        std_loss = F.mse_loss(z_std, f_std)
        
        align_loss = mean_loss + std_loss
        
        return align_loss


class AdversarialLoss(nn.Module):
    """
    （可选）对抗损失：使用红外分类器作为判别器
    目标：让生成的 f_r_fake 无法被分类器区分
    """
    def __init__(self, num_classes):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()
    
    def forward(self, f_fake, W_r, pseudo_labels):
        """
        Args:
            f_fake: 生成的特征
            W_r: 红外分类器
            pseudo_labels: 伪标签（从可见光分类结果得到）
        Returns:
            adv_loss: 对抗损失
        """
        # 计算分类 logits
        logits = torch.mm(F.normalize(f_fake, p=2, dim=1), 
                         F.normalize(W_r, p=2, dim=1).t())
        
        # 交叉熵损失（让生成特征能被正确分类）
        adv_loss = self.criterion(logits, pseudo_labels)
        
        return adv_loss
