"""
跨模态匹配聚合模块 (Cross-Modal Match Aggregation)
=================================================
功能：
1. 提取 RGB 和 IR 模态的特征
2. 使用 Sinkhorn 算法进行全局最优匹配
3. 支持 CLIP 语义特征增强匹配
4. 管理特征记忆库用于一致性约束
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import Counter, OrderedDict


class CMA(nn.Module):
    """
    跨模态匹配聚合器
    """
    def __init__(self, args):
        super(CMA, self).__init__()
        self.device = torch.device(args.device)
        self.num_classes = args.num_classes
        self.T = args.temperature           # Softmax 温度参数
        self.sigma = args.sigma             # 记忆库动量更新系数
        self.args = args
        
        # 特征记忆库：存储每个类别的平均特征 (用于一致性约束)
        self.register_buffer('vis_memory', torch.zeros(self.num_classes, 2048))
        self.register_buffer('ir_memory', torch. zeros(self.num_classes, 2048))
        
        # CLIP 特征缓存 (用于语义匹配)
        self.vis_clip_feats = None
        self.ir_clip_feats = None
        
        # 内部状态
        self.not_saved = True
        self.mode = None
        

    # ==================== 核心接口 ====================
    
    def extract_and_match(self, model, dataset, clip_model=None):
        """
        统一接口：提取特征并执行匹配
        
        Args:
            model: 训练模型
            dataset: 数据集对象
            clip_model:  CLIP 模型 (可选)
            
        Returns:
            v2i_dict: RGB -> IR 匹配字典
            i2v_dict: IR -> RGB 匹配字典
        """
        # 1. 提取特征
        self._extract_features(model, dataset, clip_model)
        
        # 2. 执行匹配
        if hasattr(self. args, 'use_sinkhorn') and self.args.use_sinkhorn:
            print("🔗 使用 Sinkhorn 算法进行全局最优匹配...")
            return self._match_sinkhorn()
        else:
            print("🔗 使用贪婪算法进行快速匹配...")
            return self._match_greedy()
    

    # ==================== 特征提取 ====================
    
    @torch.no_grad()
    def _extract_features(self, model, dataset, clip_model=None):
        """
        提取 RGB 和 IR 模态的特征
        """
        model.set_eval()
        rgb_loader, ir_loader = dataset.get_normal_loader()
        
        print("📊 提取 RGB 特征...")
        rgb_feats, rgb_labels, rgb_cls, rgb_clip = self._extract_single_modal(
            model, rgb_loader, 'rgb', clip_model)
        
        print("📊 提取 IR 特征...")
        ir_feats, ir_labels, ir_cls, ir_clip = self._extract_single_modal(
            model, ir_loader, 'ir', clip_model)
        
        # 保存到内部状态
        self._save_features(
            rgb_cls, ir_cls, rgb_labels, ir_labels, 
            rgb_feats, ir_feats, rgb_clip, ir_clip
        )
    
    
    @torch.no_grad()
    def _extract_single_modal(self, model, loader, modal, clip_model=None):
        """
        提取单个模态的特征
        
        Returns:
            features: BN 特征 [N, 2048]
            labels: 伪标签 [N]
            cls_scores: 分类分数 [N, num_classes]
            clip_feats: CLIP 特征 [N, 1024] (如果启用)
        """
        all_features = []
        all_labels = []
        all_cls_scores = []
        all_clip_feats = []
        
        for imgs_list, infos in loader:
            labels = infos[: , 1]. to(self.device)
            
            # 处理数据增强的情况
            if isinstance(imgs_list, (list, tuple)):
                imgs = imgs_list[0]. to(self.device)  # 只用原始图像
            else: 
                imgs = imgs_list.to(self.device)
            
            # 提取任务模型特征
            _, bn_features = model. model(imgs)
            
            # 根据模态选择分类器 (交叉分类策略)
            if modal == 'rgb':
                cls_scores, _ = model.classifier2(bn_features)  # RGB -> IR 分类器
            else: 
                cls_scores, _ = model.classifier1(bn_features)  # IR -> RGB 分类器
            
            all_features.append(bn_features. cpu())
            all_labels. append(labels. cpu())
            all_cls_scores.append(cls_scores. cpu())
            
            # 提取 CLIP 特征 (如果启用)
            if clip_model is not None: 
                clip_feats = self._extract_clip_features(clip_model, imgs)
                all_clip_feats.append(clip_feats)
        
        # 合并所有 batch
        features = torch.cat(all_features, dim=0)
        labels = torch.cat(all_labels, dim=0)
        cls_scores = torch.cat(all_cls_scores, dim=0)
        clip_feats = torch.cat(all_clip_feats, dim=0) if all_clip_feats else None
        
        return features, labels, cls_scores, clip_feats
    
    
    @torch.no_grad()
    def _extract_clip_features(self, clip_model, imgs):
        """
        提取 CLIP 语义特征 (修复版本)
        
        关键修复：
        1. 直接调用 attnpool，不加索引
        2. 确保输入是标准归一化的图像
        """
        # CLIP 编码图像
        feat_map = clip_model.encode_image(imgs)  # [Batch, 2048, H, W]
        
        # 通过 Attention Pooling 得到全局特征
        if hasattr(clip_model. visual, 'attnpool'):
            # ResNet50 分支：attnpool 直接返回 [Batch, 1024]
            clip_emb = clip_model.visual.attnpool(feat_map)  # ✅ 修复：去掉 [0]
        else:
            # ViT 分支：使用 CLS token
            if isinstance(feat_map, tuple):
                clip_emb = feat_map[-1]  # 取最后一个输出 (通常是投影后的特征)
            else:
                clip_emb = feat_map. mean(dim=[-2, -1])  # 全局平均池化
        
        return clip_emb. detach().cpu()
    
    
    @torch.no_grad()
    def _save_features(self, rgb_cls, ir_cls, rgb_labels, ir_labels, 
                       rgb_features, ir_features, clip_rgb, clip_ir):
        """
        保存特征到内部状态并更新记忆库
        """
        self.mode = 'scores'
        self.not_saved = False
        
        # 保存分类分数 (用于匹配)
        self.vis = F.softmax(self.T * rgb_cls, dim=1).cpu().numpy()
        self.ir = F.softmax(self.T * ir_cls, dim=1).cpu().numpy()
        self.rgb_ids = rgb_labels.cpu()
        self.ir_ids = ir_labels.cpu()
        
        # 更新特征记忆库
        self._update_memory(rgb_features. to(self.device), ir_features.to(self.device),
                            rgb_labels.to(self. device), ir_labels.to(self.device))
        
        # 保存 CLIP 特征
        if clip_rgb is not None and clip_ir is not None: 
            self.vis_clip_feats = clip_rgb
            self.ir_clip_feats = clip_ir
            print(f"✅ CLIP 特征已保存: RGB {clip_rgb.shape}, IR {clip_ir.shape}")
        else:
            self.vis_clip_feats = None
            self.ir_clip_feats = None
    
    
    @torch.no_grad()
    def _update_memory(self, rgb_feats, ir_feats, rgb_labels, ir_labels):
        """
        使用 EMA 更新特征记忆库
        """
        self.vis_memory = self.vis_memory. to(self.device)
        self.ir_memory = self.ir_memory.to(self.device)
        
        # 更新 RGB 记忆
        for label in torch.unique(rgb_labels):
            mask = (rgb_labels == label)
            if mask.any():
                new_feat = rgb_feats[mask]. mean(dim=0)
                self.vis_memory[label] = (1 - self.sigma) * self.vis_memory[label] + \
                                         self.sigma * new_feat
        
        # 更新 IR 记忆
        for label in torch.unique(ir_labels):
            mask = (ir_labels == label)
            if mask.any():
                new_feat = ir_feats[mask]. mean(dim=0)
                self.ir_memory[label] = (1 - self.sigma) * self.ir_memory[label] + \
                                        self.sigma * new_feat
    

    # ==================== Sinkhorn 匹配 ====================
    
    def _match_sinkhorn(self):
        """
        使用 Sinkhorn 算法进行全局最优匹配
        
        核心思想：
        1. 构建相似度矩阵 (专家分数 + CLIP 语义)
        2. Sinkhorn 迭代求解最优传输
        3. 基于传输矩阵生成匹配字典
        """
        # 1. 计算专家相似度
        score_rgb = torch.from_numpy(self.vis).to(self.device)
        score_ir = torch.from_numpy(self.ir).to(self.device)
        score_rgb = F.normalize(score_rgb, dim=1)
        score_ir = F.normalize(score_ir, dim=1)
        sim_expert = torch.matmul(score_rgb, score_ir.T)  # [N_rgb, N_ir]
        
        # 2. 融合 CLIP 语义相似度 (如果启用)
        if self.vis_clip_feats is not None and self.ir_clip_feats is not None:
            clip_rgb = F.normalize(self.vis_clip_feats. to(self.device), dim=1)
            clip_ir = F.normalize(self.ir_clip_feats.to(self.device), dim=1)
            sim_clip = torch.matmul(clip_rgb, clip_ir.T)
            
            w_clip = getattr(self.args, 'w_clip', 0.3)
            sim_final = (1 - w_clip) * sim_expert + w_clip * sim_clip
            print(f"🎯 CLIP 权重:  {w_clip:.2f}")
        else:
            sim_final = sim_expert
        
        # 3. Log-domain Sinkhorn (数值稳定版本)
        epsilon = getattr(self.args, 'sinkhorn_reg', 0.05)
        log_Q = sim_final / epsilon
        
        # 迭代求解
        max_iters = 100
        tolerance = 1e-4
        for iteration in range(max_iters):
            log_Q_prev = log_Q.clone()
            
            # 行归一化 (log-domain)
            log_Q = log_Q - torch.logsumexp(log_Q, dim=1, keepdim=True)
            # 列归一化 (log-domain)
            log_Q = log_Q - torch.logsumexp(log_Q, dim=0, keepdim=True)
            
            # 检查收敛
            if torch.abs(log_Q - log_Q_prev).max() < tolerance:
                print(f"✅ Sinkhorn 在第 {iteration} 轮收敛")
                break
        
        Q = torch.exp(log_Q).cpu().numpy()
        
        # 4. 生成匹配字典 (置信度阈值策略)
        confidence_threshold = 0.5
        v2i, i2v = self._generate_matches_from_Q(Q, confidence_threshold)
        
        print(f"📊 匹配结果: RGB->IR {len(v2i)}/{len(self.rgb_ids)}, "
              f"IR->RGB {len(i2v)}/{len(self.ir_ids)}")
        
        return v2i, i2v
    
    
    def _generate_matches_from_Q(self, Q, threshold):
        """
        从传输矩阵生成匹配字典
        
        策略：基于置信度阈值的软匹配 (相比严格双向验证更宽松)
        """
        v2i = OrderedDict()
        i2v = OrderedDict()
        
        # RGB -> IR 匹配
        max_j = np.argmax(Q, axis=1)
        for i, j in enumerate(max_j):
            if Q[i, j] > threshold:   # 置信度足够高
                rgb_id = self.rgb_ids[i]. item()
                ir_id = self.ir_ids[j]. item()
                if rgb_id not in v2i:  # 避免重复
                    v2i[rgb_id] = ir_id
        
        # IR -> RGB 匹配
        max_i = np.argmax(Q, axis=0)
        for j, i in enumerate(max_i):
            if Q[i, j] > threshold: 
                rgb_id = self.rgb_ids[i].item()
                ir_id = self.ir_ids[j].item()
                if ir_id not in i2v: 
                    i2v[ir_id] = rgb_id
        
        return v2i, i2v
    

    # ==================== 贪婪匹配 (备用) ====================
    
    def _match_greedy(self):
        """
        贪婪匹配算法 (用于快速实验或消融研究)
        """
        # 计算相似度矩阵
        dists = np.matmul(self.vis, self.ir.T)
        
        # 排序并贪婪选择
        sorted_indices = np.argsort(-dists, axis=None)
        sorted_2d = np.unravel_index(sorted_indices, dists.shape)
        idx_rgb, idx_ir = sorted_2d[0], sorted_2d[1]
        
        # 统计匹配频率
        pairs = [(self.rgb_ids[i]. item(), self.ir_ids[j].item()) 
                 for i, j in zip(idx_rgb, idx_ir)]
        pair_counts = Counter(pairs)
        
        # 生成唯一匹配
        v2i, i2v = OrderedDict(), OrderedDict()
        matched_rgb, matched_ir = set(), set()
        
        for (rgb_id, ir_id), count in pair_counts.most_common():
            if rgb_id not in matched_rgb and ir_id not in matched_ir:
                v2i[rgb_id] = ir_id
                i2v[ir_id] = rgb_id
                matched_rgb.add(rgb_id)
                matched_ir.add(ir_id)
        
        print(f"📊 贪婪匹配结果: {len(v2i)} 对")
        return v2i, i2v
    

    # ==================== 工具函数 ====================
    
    def get_memory_features(self):
        """
        获取记忆库特征 (用于一致性约束)
        """
        return self.vis_memory, self.ir_memory
    
    
    def reset(self):
        """
        重置内部状态
        """
        self.not_saved = True
        self.vis_clip_feats = None
        self.ir_clip_feats = None