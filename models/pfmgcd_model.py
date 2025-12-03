"""
models/pfmgcd_model.py - Unified MTRL-Gated Framework
兼容: RegDB (Small), SYSU/LLCM (Large)
核心机制:
1. MTRL: 内部动态生成灰度图，构建 RGB-Gray-IR 三元组约束。
2. Gated GCN: 带有零初始化门控残差的图推理，防止初期噪声破坏特征。
3. Stable Adv: 数值稳定的对抗训练模块。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.autograd import Function

from .pcb_backbone import PCBBackbone, GeMPooling
from .isg_dm import MultiPartISG_DM
from .memory_bank import MultiPartMemoryBank
from .graph_propagation import AdaptiveGraphPropagation

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        nn.init.normal_(m.weight, 1.0, 0.01)
        nn.init.constant_(m.bias, 0.0)

# ==================== 基础组件 ====================

class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None

class GatedGraphReasoning(nn.Module):
    """
    [策略四升级版] 门控图推理 (Gated GCN)
    核心: 引入 alpha 参数，初始化为0。
    out = x + alpha * GCN(x)
    效果: 初始阶段等同于无GCN，随着训练自动寻找最佳推理强度。
    """
    def __init__(self, feature_dim, top_k=5):
        super(GatedGraphReasoning, self).__init__()
        self.top_k = top_k
        # 简单的 GCN 变换
        self.gcn_fc = nn.Linear(feature_dim, feature_dim, bias=False)
        self.relu = nn.ReLU(inplace=True)
        # 零初始化门控因子
        self.alpha = nn.Parameter(torch.zeros(1))
        
        self.gcn_fc.apply(weights_init_kaiming)

    def forward(self, x, memory):
        """
        x: [B, D]
        memory: [N, D]
        """
        if memory is None or memory.size(0) == 0:
            return x
            
        B, D = x.size()
        # 1. 相似度计算 (AMP Safe)
        with torch.cuda.amp.autocast(enabled=False):
            x_fp32 = F.normalize(x.float(), p=2, dim=1)
            mem_fp32 = F.normalize(memory.float(), p=2, dim=1)
            sim = torch.mm(x_fp32, mem_fp32.t()) # [B, N]
        
        # 2. Top-K 检索
        # 限制 K 不能超过 memory 大小
        k = min(self.top_k, sim.size(1))
        topk_val, topk_idx = torch.topk(sim, k=k, dim=1)
        
        # 3. 聚合邻居
        neighbor_feats = F.embedding(topk_idx, memory) # [B, K, D]
        
        # Attention score
        attn = F.softmax(topk_val * 10, dim=1).unsqueeze(2) # [B, K, 1]
        
        # Weighted Sum
        context = (neighbor_feats * attn).sum(dim=1) # [B, D]
        
        # 4. GCN Transform
        out = self.relu(self.gcn_fc(context))
        
        # 5. Gated Residual (关键!)
        # 初期 alpha=0，相当于 identity mapping，不干扰训练
        return x + self.alpha * out

class ModalityDiscriminator(nn.Module):
    """
    [策略三] 模态判别器
    """
    def __init__(self, input_dim):
        super(ModalityDiscriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim // 4),
            nn.BatchNorm1d(input_dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(input_dim // 4, 2) # RGB vs IR
        )
        self.net.apply(weights_init_kaiming)

    def forward(self, x, lambda_adv=0.1):
        x = GradientReversalFunction.apply(x, lambda_adv)
        return self.net(x)

# ==================== 主模型 ====================

class PF_MGCD(nn.Module):
    def __init__(self, num_parts=6, num_identities=395, feature_dim=512,
                 memory_momentum=0.9, temperature=3.0, top_k=5, 
                 pretrained=True, backbone='resnet50', use_ibn=True,
                 use_adversarial=True, use_graph_reasoning=True):
        super(PF_MGCD, self).__init__()
        self.num_parts = num_parts
        self.feature_dim = feature_dim
        self.use_adversarial = use_adversarial
        self.use_graph_reasoning = use_graph_reasoning
        
        # 1. Backbone (IBN)
        self.backbone = PCBBackbone(num_parts, pretrained, backbone, use_ibn)
        
        # 2. ISG-DM
        self.isg_dm = MultiPartISG_DM(num_parts, 2048, feature_dim, feature_dim)
        
        # 3. MTRL Gray Transform
        self.grayscale = T.Grayscale(num_output_channels=3)
        
        # 4. Modules
        self.bottlenecks = nn.ModuleList()
        self.classifiers = nn.ModuleList()
        
        if self.use_graph_reasoning:
            self.gcns = nn.ModuleList()
            
        if self.use_adversarial:
            self.discriminators = nn.ModuleList()
            
        for _ in range(num_parts):
            self.bottlenecks.append(nn.BatchNorm1d(feature_dim))
            self.classifiers.append(nn.Linear(feature_dim, num_identities, bias=False))
            
            if self.use_graph_reasoning:
                self.gcns.append(GatedGraphReasoning(feature_dim, top_k=top_k))
                
            if self.use_adversarial:
                self.discriminators.append(ModalityDiscriminator(feature_dim))
                
        self.bottlenecks.apply(weights_init_kaiming)
        self.classifiers.apply(weights_init_kaiming)
        self.dropout = nn.Dropout(p=0.5)
        
        # 5. Memory Bank & Graph Propagation (Loss)
        self.memory_bank = MultiPartMemoryBank(num_parts, num_identities, feature_dim, memory_momentum)
        self.graph_loss_module = AdaptiveGraphPropagation(temperature, top_k, True, 30.0)

    def forward(self, x, labels=None, current_epoch=0, **kwargs):
        # MTRL 逻辑：训练时生成灰度图
        if self.training:
            x_gray = self.grayscale(x)
            x_all = torch.cat([x, x_gray], dim=0) # [2B, 3, H, W]
            
            # Backbone Forward
            out_all = self.backbone(x_all)
            part_feats_all = out_all[0] if isinstance(out_all, tuple) else out_all
            
            # Split back
            batch_size = x.size(0)
            part_feats_orig = [f[:batch_size] for f in part_feats_all]
            part_feats_gray = [f[batch_size:] for f in part_feats_all]
        else:
            out = self.backbone(x)
            part_feats_orig = out[0] if isinstance(out, tuple) else out
            part_feats_gray = None

        # ISG-DM
        id_feats_orig, _ = self.isg_dm(part_feats_orig)
        if part_feats_gray:
            id_feats_gray, _ = self.isg_dm(part_feats_gray)
        
        # Main Flow
        final_logits = []
        final_feats = []     # For Triplet
        gray_feats_out = []  # For MTRL Triplet
        adv_logits = []
        
        for k in range(self.num_parts):
            feat = id_feats_orig[k]
            
            # [策略四] Gated GCN
            # 只在 memory 初始化后启用，并且不仅依赖 epoch，还依赖 memory 状态
            if self.use_graph_reasoning and self.memory_bank.initialized.sum() > 0:
                mem_k = self.memory_bank.get_part_memory(k).detach()
                feat = self.gcns[k](feat, mem_k)
            
            final_feats.append(feat) # BN前特征用于Triplet
            
            # [策略三] Adversarial
            if self.training and self.use_adversarial:
                # 动态 lambda: 前20 epoch 权重极小，之后增加
                lambda_adv = 0.0 if current_epoch < 5 else 0.1
                adv_logits.append(self.discriminators[k](feat, lambda_adv))
            
            # BNNeck + Classifier
            feat_bn = self.bottlenecks[k](feat)
            
            if self.training:
                final_logits.append(self.classifiers[k](self.dropout(feat_bn)))
                
                # Gray Stream processing (Backbone -> ISG -> (No GCN) -> BN -> Output)
                # Gray流不走GCN，保持作为"纯净"的中间模态锚点
                if part_feats_gray:
                    gray_f = id_feats_gray[k]
                    gray_feats_out.append(gray_f)
            else:
                final_logits.append(self.classifiers[k](feat_bn))

        # Graph Soft Labels
        soft_labels, entropy_weights = None, None
        if self.training and self.memory_bank.initialized.sum() > 0:
            soft_labels, _, entropy_weights = self.graph_loss_module(final_feats, self.memory_bank)

        return {
            'id_logits': final_logits,
            'id_features': final_feats,
            'gray_features': gray_feats_out if self.training else None,
            'adv_logits': adv_logits if self.training and self.use_adversarial else None,
            'soft_labels': soft_labels,
            'entropy_weights': entropy_weights
        }

    def extract_features(self, x, pool_parts=True):
        with torch.no_grad():
            out = self.backbone(x)
            part_feats = out[0] if isinstance(out, tuple) else out
            id_feats, _ = self.isg_dm(part_feats)
            
            bn_feats = []
            for k in range(self.num_parts):
                # 测试时启用GCN吗？
                # 科学的做法：如果训练时GCN学得好，测试时应该用。
                # 但需要加载Memory Bank。为简单稳健，通常ReID测试只用Backbone特征。
                # 如果要极致性能，可以开启，但需确保 Memory Bank 是最新的。
                # 这里我们保持 Strong Baseline 逻辑：只用特征。
                bn_feats.append(self.bottlenecks[k](id_feats[k]))
            
            norm_feats = [F.normalize(f, p=2, dim=1) for f in bn_feats]
            if pool_parts: return torch.cat(norm_feats, dim=1)
            else: return torch.stack(norm_feats, dim=1).mean(dim=1)

    def initialize_memory(self, dataloader, device, teacher_model=None):
        self.eval()
        print("🔄 正在初始化记忆库...")
        all_feats = [[] for _ in range(self.num_parts)]
        all_pids = []
        with torch.no_grad():
            for batch in dataloader:
                imgs, pids = batch[0], batch[1]
                imgs = imgs.to(device)
                
                out = self.backbone(imgs)
                part_feats = out[0] if isinstance(out, tuple) else out
                id_feats, _ = self.isg_dm(part_feats)
                
                for k in range(self.num_parts):
                    all_feats[k].append(id_feats[k].cpu())
                all_pids.append(pids)
        
        for k in range(self.num_parts):
            all_feats[k] = torch.cat(all_feats[k], dim=0).to(device)
        all_pids = torch.cat(all_pids, dim=0).to(device)
        self.memory_bank.initialize_memory(all_feats, all_pids)
        self.train()