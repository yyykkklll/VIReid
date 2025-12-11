"""
跨模态行人重识别训练模块
===========================
功能：
1. Phase1: 模态内学习（RGB 和 IR 独立训练）
2. Phase2: 跨模态匹配学习（基于伪标签）
3. 支持多种损失函数：ID Loss, Triplet Loss, CMO Loss, Weak Loss
4. 集成 CLIP 语义增强和 Sinkhorn 全局匹配

作者:  修复优化版本
日期: 2025-01-20
"""

import torch
import torch.nn. functional as F
from models import Model
from datasets import SYSU
from wsl import CMA
from utils import MultiItemAverageMeter, infoEntropy
from tqdm import tqdm


def train(args, model:  Model, dataset, epoch, cma:  CMA, logger, enable_phase1):
    """
    主训练函数
    
    Args:
        args: 训练配置参数
        model: 训练模型
        dataset: 数据集对象
        epoch: 当前轮次
        cma: 跨模态匹配聚合器
        logger: 日志记录器
        enable_phase1: 是否为 Phase1 阶段
        
    Returns:
        loss_dict: 损失字典
        loss_str: 损失字符串
    """
    
    # ==================== 阶段判断与匹配 ====================
    
    if not enable_phase1:
        # Phase2:  执行跨模态匹配
        logger(f"🔄 Epoch {epoch}: 开始跨模态特征提取与匹配...")
        
        r2i_pair_dict, i2r_pair_dict = cma.extract_and_match(
            model=model,
            dataset=dataset,
            clip_model=model.clip_model if hasattr(model, 'clip_model') else None
        )
        
        # 构建匹配关系矩阵
        match_info = _build_matching_matrices(
            args, r2i_pair_dict, i2r_pair_dict, model. device
        )
        
        logger(f"✅ 匹配完成:  Common={len(match_info['common_dict'])}, "
               f"Specific={len(match_info['specific_dict'])}, "
               f"Remain={len(match_info['remain_dict'])}")
        
        # 启用第三个分类器（跨模态分类器）
        if not model.enable_cls3:
            model.enable_cls3 = True
    else:
        match_info = None
    
    
    # ==================== 训练循环 ====================
    
    model.set_train()
    meter = MultiItemAverageMeter()
    
    rgb_loader, ir_loader = dataset.get_train_loader()
    batch_size = args.batch_pidnum * args.pid_numsample
    
    iters = min(len(rgb_loader), len(ir_loader))
    pbar = tqdm(
        zip(rgb_loader, ir_loader), 
        total=iters, 
        desc=f"Epoch {epoch} ({'Phase1' if enable_phase1 else 'Phase2'})",
        leave=False
    )
    
    for (rgb_imgs, ca_imgs, rgb_info), (ir_imgs, aug_imgs, ir_info) in pbar:
        
        # 准备数据
        rgb_imgs = rgb_imgs.to(model.device)
        ca_imgs = ca_imgs.to(model.device)
        ir_imgs = ir_imgs.to(model.device)
        
        if args.dataset == 'regdb':
            aug_imgs = aug_imgs.to(model.device)
            ir_imgs_full = torch.cat([ir_imgs, aug_imgs], dim=0)
        else:
            ir_imgs_full = ir_imgs
        
        color_imgs = torch.cat([rgb_imgs, ca_imgs], dim=0)
        
        rgb_ids = torch.cat([rgb_info[: , 1], rgb_info[:, 1]]).to(model.device)
        ir_ids = ir_info[:, 1].to(model.device)
        if args.dataset == 'regdb':
            ir_ids = torch.cat([ir_ids, ir_ids]).to(model.device)
        
        
        # ==================== 前向传播 ====================
        
        gap_features, bn_features = model.model(color_imgs, ir_imgs_full)
        
        # 模态特定分类
        rgbcls_out, _ = model.classifier1(bn_features)  # RGB 分类器
        ircls_out, _ = model.classifier2(bn_features)   # IR 分类器
        
        # 分离特征和预测
        rgb_features = gap_features[:2 * batch_size]
        ir_features = gap_features[2 * batch_size:]
        
        r2r_cls = rgbcls_out[: 2 * batch_size]   # RGB 图像 -> RGB 分类器
        i2i_cls = ircls_out[2 * batch_size:]    # IR 图像 -> IR 分类器
        r2i_cls = ircls_out[:2 * batch_size]    # RGB 图像 -> IR 分类器
        i2r_cls = rgbcls_out[2 * batch_size:]   # IR 图像 -> RGB 分类器
        
        
        # ==================== 损失计算 ====================
        
        if enable_phase1:
            # Phase1: 模态内学习
            total_loss, losses = _compute_phase1_loss(
                model, args, 
                r2r_cls, i2i_cls, 
                rgb_features, ir_features,
                rgb_ids, ir_ids
            )
        else:
            # Phase2: 跨模态学习
            total_loss, losses = _compute_phase2_loss(
                model, args, epoch,
                bn_features, gap_features,
                r2r_cls, i2i_cls, r2i_cls, i2r_cls,
                rgb_features, ir_features,
                rgb_ids, ir_ids,
                match_info, cma, logger
            )
        
        meter.update(losses)
        
        
        # ==================== 反向传播 ====================
        
        if enable_phase1:
            model.optimizer_phase1.zero_grad()
        else:
            model.optimizer_phase2.zero_grad()
        
        total_loss.backward()
        
        # 梯度裁剪（防止梯度爆炸）
        torch.nn. utils.clip_grad_norm_(
            model.model.parameters(), max_norm=5.0
        )
        
        if enable_phase1:
            model.optimizer_phase1.step()
        else:
            model. optimizer_phase2.step()
        
        # 更新进度条
        pbar.set_postfix({'loss': f'{total_loss.item():.4f}'})
    
    return meter.get_val(), meter.get_str()


# ==================== 辅助函数 ====================

def _build_matching_matrices(args, r2i_dict, i2r_dict, device):
    """
    构建跨模态匹配关系矩阵
    
    匹配类型：
    - Common: 双向一致匹配 (最可靠)
    - Specific: 单向匹配 (中等可靠)
    - Remain: 冲突匹配 (低可靠)
    
    Returns:
        match_info: 包含各类匹配信息的字典
    """
    
    # 分类匹配对
    common_dict = {}      # 双向一致
    specific_dict = {}    # 单向唯一
    remain_dict = {}      # 冲突/剩余
    
    for rgb_id, ir_id in r2i_dict.items():
        if ir_id in i2r_dict and i2r_dict[ir_id] == rgb_id: 
            # 双向一致：RGB->IR 和 IR->RGB 互相指向
            common_dict[rgb_id] = ir_id
        elif rgb_id not in i2r_dict. values() and ir_id not in i2r_dict.keys():
            # 单向唯一：IR 侧没有该匹配
            specific_dict[rgb_id] = ir_id
        else:
            # 冲突：存在其他匹配关系
            remain_dict[rgb_id] = ir_id
    
    for ir_id, rgb_id in i2r_dict.items():
        if (rgb_id, ir_id) not in common_dict. items():
            if rgb_id not in r2i_dict.keys() and ir_id not in r2i_dict.values():
                specific_dict[rgb_id] = ir_id
            else: 
                remain_dict[rgb_id] = ir_id
    
    # 构建匹配矩阵（用于索引）
    num_classes = args.num_classes
    common_rm = torch.zeros(num_classes, num_classes).to(device)
    specific_rm = torch.zeros(num_classes, num_classes).to(device)
    remain_rm = torch.zeros(num_classes, num_classes).to(device)
    
    for rgb_id, ir_id in common_dict.items():
        common_rm[rgb_id, ir_id] = 1.0
    
    for rgb_id, ir_id in specific_dict.items():
        specific_rm[rgb_id, ir_id] = 1.0
    
    for rgb_id, ir_id in remain_dict. items():
        remain_rm[rgb_id, ir_id] = 1.0
    
    # Specific 包含 Common（高置信度匹配）
    specific_rm = specific_rm + common_rm
    
    # 转换为 Tensor 列表（用于快速索引）
    common_matched_rgb = torch.tensor(list(common_dict.keys())).to(device)
    common_matched_ir = torch.tensor(list(common_dict.values())).to(device)
    specific_matched_rgb = torch. tensor(list(specific_dict. keys())).to(device)
    remain_matched_rgb = torch.tensor(list(remain_dict.keys())).to(device)
    remain_matched_ir = torch.tensor(list(remain_dict.values())).to(device)
    
    return {
        'common_dict': common_dict,
        'specific_dict':  specific_dict,
        'remain_dict': remain_dict,
        'common_rm': common_rm,
        'specific_rm': specific_rm,
        'remain_rm': remain_rm,
        'common_matched_rgb': common_matched_rgb,
        'common_matched_ir': common_matched_ir,
        'specific_matched_rgb': specific_matched_rgb,
        'remain_matched_rgb': remain_matched_rgb,
        'remain_matched_ir': remain_matched_ir
    }


def _compute_phase1_loss(model, args, r2r_cls, i2i_cls, 
                         rgb_features, ir_features, rgb_ids, ir_ids):
    """
    Phase1 损失：模态内学习
    
    损失组成：
    - ID Loss: 分类损失
    - Triplet Loss: 三元组损失
    """
    
    # ID 损失（交叉熵）
    r2r_id_loss = model.pid_criterion(r2r_cls, rgb_ids)
    i2i_id_loss = model.pid_criterion(i2i_cls, ir_ids)
    
    # Triplet 损失（度量学习）
    r2r_tri_loss = args.tri_weight * model.tri_criterion(rgb_features, rgb_ids)
    i2i_tri_loss = args.tri_weight * model.tri_criterion(ir_features, ir_ids)
    
    total_loss = r2r_id_loss + i2i_id_loss + r2r_tri_loss + i2i_tri_loss
    
    losses = {
        'r2r_id_loss': r2r_id_loss. item(),
        'i2i_id_loss': i2i_id_loss.item(),
        'r2r_tri_loss': r2r_tri_loss.item(),
        'i2i_tri_loss': i2i_tri_loss.item()
    }
    
    return total_loss, losses


def _compute_phase2_loss(model, args, epoch,
                         bn_features, gap_features,
                         r2r_cls, i2i_cls, r2i_cls, i2r_cls,
                         rgb_features, ir_features,
                         rgb_ids, ir_ids,
                         match_info, cma, logger):
    """
    Phase2 损失：跨模态学习
    
    损失组成：
    - 基础 ID Loss (detached backbone)
    - Triplet Loss (跨模态)
    - CMO Loss (Cross-Modal Consistency)
    - Cross-Modal Classification Loss
    - Weak Supervision Loss (Remain 分支)
    """
    
    batch_size = rgb_ids.size(0)
    
    # ==================== 1. 基础损失（Detached Backbone）====================
    
    # Detach 特征，防止影响 backbone
    dtd_features = bn_features.detach()
    dtd_rgbcls_out, _ = model.classifier1(dtd_features)
    dtd_ircls_out, _ = model.classifier2(dtd_features)
    
    dtd_r2r_cls = dtd_rgbcls_out[:batch_size]
    dtd_i2i_cls = dtd_ircls_out[batch_size:]
    
    r2r_id_loss = model.pid_criterion(dtd_r2r_cls, rgb_ids)
    i2i_id_loss = model. pid_criterion(dtd_i2i_cls, ir_ids)
    
    total_loss = r2r_id_loss + i2i_id_loss
    losses = {
        'r2r_id_loss': r2r_id_loss.item(),
        'i2i_id_loss': i2i_id_loss.item()
    }
    
    
    # ==================== 2. 跨模态 Triplet Loss ====================
    
    # 找到 common 匹配的样本
    common_rgb_indices = torch.isin(rgb_ids, match_info['common_matched_rgb'])
    common_ir_indices = torch.isin(ir_ids, match_info['common_matched_ir'])
    
    if common_rgb_indices.any() and common_ir_indices. any():
        # 提取匹配的特征和标签
        selected_rgb_ids = rgb_ids[common_rgb_indices]
        selected_ir_ids = ir_ids[common_ir_indices]
        
        # 转换标签到对方模态
        translated_rgb_label = torch.nonzero(
            match_info['common_rm'][selected_rgb_ids], as_tuple=False
        )[: , -1]
        translated_ir_label = torch.nonzero(
            match_info['common_rm']. T[selected_ir_ids], as_tuple=False
        )[:, -1]
        
        # 构建跨模态特征集合
        selected_rgb_features = rgb_features[common_rgb_indices]
        selected_ir_features = ir_features[common_ir_indices]
        
        # RGB 侧：自己 + 所有 IR
        matched_rgb_features = torch.cat([selected_rgb_features, ir_features], dim=0)
        matched_rgb_labels = torch.cat([translated_rgb_label, ir_ids], dim=0)
        
        # IR 侧：所有 RGB + 自己
        matched_ir_features = torch.cat([rgb_features, selected_ir_features], dim=0)
        matched_ir_labels = torch.cat([rgb_ids, translated_ir_label], dim=0)
        
        # 计算 Triplet Loss
        tri_loss_rgb = args.tri_weight * model.tri_criterion(
            matched_rgb_features, matched_rgb_labels
        )
        tri_loss_ir = args.tri_weight * model.tri_criterion(
            matched_ir_features, matched_ir_labels
        )
        
        total_loss += tri_loss_rgb + tri_loss_ir
        losses. update({
            'tri_loss_rgb': tri_loss_rgb.item(),
            'tri_loss_ir': tri_loss_ir.item()
        })
        
        
        # ==================== 3. CMO Loss (Cross-Modal Consistency) ====================
        
        # 更新记忆库
        cma._update_memory(
            bn_features[: batch_size], 
            bn_features[batch_size:],
            rgb_ids, 
            ir_ids
        )
        
        # 计算自适应权重（修复版）
        r2i_entropy = infoEntropy(r2i_cls).item()
        i2r_entropy = infoEntropy(i2r_cls).item()
        
        # 熵越小（预测越确定），权重越大
        total_entropy = 2.0 - r2i_entropy - i2r_entropy + 1e-8
        w_r2i = (1.0 - r2i_entropy) / total_entropy
        w_i2r = (1.0 - i2r_entropy) / total_entropy
        
        # 从记忆库获取对应特征
        selected_rgb_memory = cma. vis_memory[translated_ir_label]. detach()
        selected_ir_memory = cma.ir_memory[translated_rgb_label].detach()
        
        # 通过对方分类器预测
        mem_r2i_cls, _ = model.classifier2(selected_rgb_memory)
        mem_i2r_cls, _ = model.classifier1(selected_ir_memory)
        
        # MSE 一致性损失
        cmo_criterion = torch.nn.MSELoss()
        
        if selected_ir_ids.shape[0] > 0:
            r2i_cmo_loss = w_r2i * cmo_criterion(
                dtd_i2i_cls[common_ir_indices], 
                mem_r2i_cls
            )
            if not torch.isnan(r2i_cmo_loss) and not torch.isinf(r2i_cmo_loss):
                total_loss += r2i_cmo_loss
                losses['r2i_cmo_loss'] = r2i_cmo_loss.item()
            else:
                logger(f"⚠️ Epoch {epoch}: r2i_cmo_loss is NaN/Inf, skipped")
        
        if selected_rgb_ids.shape[0] > 0:
            i2r_cmo_loss = w_i2r * cmo_criterion(
                dtd_r2r_cls[common_rgb_indices], 
                mem_i2r_cls
            )
            if not torch.isnan(i2r_cmo_loss) and not torch.isinf(i2r_cmo_loss):
                total_loss += i2r_cmo_loss
                losses['i2r_cmo_loss'] = i2r_cmo_loss. item()
            else:
                logger(f"⚠️ Epoch {epoch}: i2r_cmo_loss is NaN/Inf, skipped")
    
    
    # ==================== 4. Weak Supervision Loss (Remain 分支) ====================
    
    if epoch >= 30 and match_info['remain_dict']: 
        # 获取跨模态分类器的输出
        r2c_cls, _ = model.classifier3(bn_features[: batch_size])
        i2c_cls, _ = model.classifier3(bn_features[batch_size:])
        
        # 找到 remain 匹配的样本
        remain_rgb_indices = torch.isin(rgb_ids, match_info['remain_matched_rgb'])
        remain_ir_indices = torch.isin(ir_ids, match_info['remain_matched_ir'])
        
        if remain_rgb_indices.any():
            remain_rgb_ids = rgb_ids[remain_rgb_indices]
            remain_r2c_cls = r2c_cls[remain_rgb_indices]
            
            # 使用 Weak Loss（软标签）
            weak_r2c_loss = args.weak_weight * model.weak_criterion(
                remain_r2c_cls, 
                match_info['remain_rm'][remain_rgb_ids]
            )
            
            if not torch.isnan(weak_r2c_loss) and not torch.isinf(weak_r2c_loss):
                total_loss += weak_r2c_loss
                losses['weak_r2c_loss'] = weak_r2c_loss.item()
            else:
                logger(f"⚠️ Epoch {epoch}: weak_r2c_loss is NaN/Inf, skipped")
    
    
    # ==================== 5. Cross-Modal Classification Loss ====================
    
    # 获取跨模态分类器输出
    r2c_cls, _ = model.classifier3(bn_features[:batch_size])
    i2c_cls, _ = model.classifier3(bn_features[batch_size:])
    
    # Specific 分支：模态特定伪标签
    specific_rgb_indices = torch.isin(rgb_ids, match_info['specific_matched_rgb'])
    common_rgb_indices = torch.isin(rgb_ids, match_info['common_matched_rgb'])
    rgb_indices = specific_rgb_indices ^ common_rgb_indices  # XOR:  只要 specific
    
    if rgb_indices.any():
        selected_rgb_ids = rgb_ids[rgb_indices]
        selected_r2c_cls = r2c_cls[rgb_indices]
        
        rgb_cross_loss = model.pid_criterion(
            selected_r2c_cls, 
            match_info['specific_rm'][selected_rgb_ids]
        )
        
        if not torch. isnan(rgb_cross_loss) and not torch.isinf(rgb_cross_loss):
            total_loss += rgb_cross_loss
            losses['rgb_cross_loss'] = rgb_cross_loss.item()
    
    # IR 侧：所有样本都参与跨模态分类
    ir_cross_loss = model.pid_criterion(i2c_cls, ir_ids)
    total_loss += ir_cross_loss
    losses['ir_cross_loss'] = ir_cross_loss.item()
    
    return total_loss, losses


# ==================== Baseline 模式（可选）====================

def train_baseline(args, model:  Model, dataset, epoch, logger):
    """
    Baseline 训练模式（有监督学习）
    仅用于对比实验
    """
    model.set_train()
    meter = MultiItemAverageMeter()
    
    rgb_loader, ir_loader = dataset.get_train_loader()
    batch_size = args.batch_pidnum * args.pid_numsample
    
    pbar = tqdm(
        zip(rgb_loader, ir_loader),
        total=min(len(rgb_loader), len(ir_loader)),
        desc=f"Epoch {epoch} (Baseline)",
        leave=False
    )
    
    for (rgb_imgs, ca_imgs, rgb_info), (ir_imgs, aug_imgs, ir_info) in pbar:
        model.optimizer_phase2. zero_grad()
        
        # 准备数据
        rgb_imgs = torch.cat([rgb_imgs, ca_imgs], dim=0).to(model.device)
        ir_imgs = ir_imgs.to(model.device)
        if args.dataset == 'regdb':
            ir_imgs = torch.cat([ir_imgs, aug_imgs. to(model.device)], dim=0)
        
        # 使用真实标签（Ground Truth）
        rgb_gts = torch.cat([rgb_info[: , -1], rgb_info[:, -1]]).to(model.device)
        ir_gts = rgb_info[:, -1].to(model.device)
        if args.dataset == 'regdb':
            ir_gts = torch.cat([ir_gts, ir_gts]).to(model.device)
        
        gts = torch.cat([rgb_gts, ir_gts])
        
        # 前向传播
        gap_features, _ = model.model(rgb_imgs, ir_imgs)
        rgbcls_out, _ = model.classifier1(gap_features)
        
        # 有监督损失
        id_loss = model.pid_criterion(rgbcls_out, gts)
        tri_loss = args.tri_weight * model.tri_criterion(gap_features, gts)
        
        total_loss = id_loss + tri_loss
        total_loss. backward()
        model.optimizer_phase2.step()
        
        meter.update({
            'id_loss': id_loss.item(),
            'tri_loss': tri_loss. item()
        })
        
        pbar.set_postfix({'loss': f'{total_loss.item():.4f}'})
    
    return meter.get_val(), meter.get_str()