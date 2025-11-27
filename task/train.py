"""
PF-MGCD 训练脚本
功能：
1. 支持混合精度训练 (AMP)，兼容 PyTorch 新旧版本 (2.0+ 至 2.5+)
2. 实现 Mean Teacher 策略的 EMA (指数移动平均) 更新
3. 自动管理日志记录与模型保存
"""

import os
import time
import logging
import torch
import torch.nn as nn
from tqdm import tqdm
from models.loss import TotalLoss
from task.test import test

# =============================================================================
# PyTorch 版本兼容性处理 (AMP 模块)
# PyTorch 2.4+ 将 GradScaler 移至 torch.amp
# 旧版本 (如 2.0.1) GradScaler 位于 torch.cuda.amp
# =============================================================================
try:
    # 尝试导入新版本 PyTorch (>=2.4) 的路径
    from torch.amp import autocast, GradScaler
except ImportError:
    # 回退到旧版本 PyTorch (<2.4) 的路径
    from torch.cuda.amp import GradScaler
    from torch import autocast  # torch.autocast 支持 device_type 参数


def setup_logger(log_dir, log_file='train.log'):
    """
    配置并初始化日志记录器
    
    Args:
        log_dir (str): 日志保存目录
        log_file (str): 日志文件名
        
    Returns:
        logger: 配置好的 logging 对象
    """
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_file)
    
    # 如果日志文件已存在，清空内容以便重新记录
    if os.path.exists(log_path):
        open(log_path, 'w').close()
    
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    
    # 清理已存在的处理器，防止重复打印
    if logger.hasHandlers():
        logger.handlers.clear()
        
    # 1. 文件处理器：记录 INFO 及以上级别的所有详细日志
    fh = logging.FileHandler(log_path, mode='a')
    fh.setLevel(logging.INFO)
    fh_formatter = logging.Formatter(
        '[%(asctime)s] - %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    fh.setFormatter(fh_formatter)
    logger.addHandler(fh)
    
    # 2. 控制台处理器：仅输出 WARNING 及以上级别的关键信息
    # 避免在控制台打印冗余的训练状态信息，保持进度条整洁
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING) 
    ch_formatter = logging.Formatter('%(message)s') 
    ch.setFormatter(ch_formatter)
    logger.addHandler(ch)
    
    return logger


def update_ema_variables(model, ema_model, alpha, global_step):
    """
    更新 Mean Teacher 的 EMA 模型参数
    
    Args:
        model (nn.Module): 当前训练的学生模型 (Student)
        ema_model (nn.Module): 教师模型 (Teacher)
        alpha (float): 动量系数 (通常为 0.999)
        global_step (int): 当前全局步数
    """
    # 更新可训练参数 (Weights/Bias)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)
    
    # 更新缓冲区 (Buffers, 如 BatchNorm 的 running_mean/var)
    for ema_buffer, buffer in zip(ema_model.buffers(), model.buffers()):
        if buffer.dtype.is_floating_point:
            ema_buffer.data.mul_(alpha).add_(buffer.data, alpha=1 - alpha)
        else:
            # 对于非浮点类型的 buffer (如 num_batches_tracked)，直接复制
            ema_buffer.data.copy_(buffer.data)


def train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, epoch, logger, args, teacher_model=None):
    """
    执行单个 Epoch 的训练循环
    """
    model.train()
    if teacher_model is not None:
        teacher_model.train()
    
    # 初始化统计指标
    total_loss = 0
    loss_items = {
        'loss_id': 0, 'loss_triplet': 0, 'loss_graph': 0, 'loss_orth': 0, 'loss_mod': 0
    }
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.total_epoch}')
    
    for batch_idx, batch_data in enumerate(pbar):
        # 1. 数据解包与迁移
        if len(batch_data) == 3:
            # 常规格式: images, labels, camera_ids
            images, labels, cams = batch_data
            images, labels, cams = images.to(device), labels.to(device), cams.to(device)
            modality_labels = (cams >= 3).long()
        elif len(batch_data) == 6:
            # 拼接格式: rgb_imgs, ir_imgs, ...
            rgb_imgs, ir_imgs, rgb_labels, ir_labels, rgb_cams, ir_cams = batch_data
            images = torch.cat([rgb_imgs, ir_imgs], dim=0).to(device)
            labels = torch.cat([rgb_labels, ir_labels], dim=0).to(device)
            modality_labels = torch.cat([
                torch.zeros(len(rgb_imgs), dtype=torch.long),
                torch.ones(len(ir_imgs), dtype=torch.long)
            ]).to(device)
        else:
            raise ValueError(f"Unknown batch format: {len(batch_data)}")
        
        optimizer.zero_grad()
        
        # 2. 前向传播与反向传播 (支持 AMP)
        if args.amp and scaler is not None:
            # 开启混合精度上下文
            with autocast(device_type='cuda'):
                outputs = model(images, labels=labels, modality_labels=modality_labels)
                loss, loss_dict = criterion(outputs, labels, modality_labels, current_epoch=epoch)
            
            # 缩放梯度并反向传播
            scaler.scale(loss).backward()
            
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            
            scaler.step(optimizer)
            scaler.update()
        else:
            # 标准精度训练
            outputs = model(images, labels=labels, modality_labels=modality_labels)
            loss, loss_dict = criterion(outputs, labels, modality_labels, current_epoch=epoch)
            loss.backward()
            
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
        
        # 3. Mean Teacher EMA 更新
        if teacher_model is not None:
            update_ema_variables(model, teacher_model, alpha=0.999, global_step=epoch*len(train_loader)+batch_idx)
        
        # 4. 记录损失
        total_loss += loss.item()
        for key in loss_items.keys():
            if key in loss_dict:
                loss_items[key] += loss_dict[key]
        
        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'ID': f'{loss_dict["loss_id"]:.4f}',
            'Graph': f'{loss_dict["loss_graph"]:.4f}'
        })
        
        # 5. 更新记忆库 (Warmup 之后)
        if epoch >= args.warmup_epochs:
            with torch.no_grad():
                # 如果有 Teacher 模型，优先使用 Teacher 的特征更新记忆库 (更加稳定)
                if teacher_model is not None:
                    teacher_outputs = teacher_model(images)
                    model.memory_bank.update_memory(teacher_outputs['id_features'], labels)
                else:
                    model.memory_bank.update_memory(outputs['id_features'], labels)
    
    # 计算并返回 Epoch 平均损失
    num_batches = len(train_loader)
    avg_loss_dict = {k: v / num_batches for k, v in loss_items.items()}
    avg_loss_dict['loss_total'] = total_loss / num_batches
    
    return avg_loss_dict


def train(model, train_loader, dataset_obj, optimizer, scheduler, args, device, teacher_model=None):
    """
    主训练流程入口
    包含初始化、训练循环、验证和模型保存
    """
    
    # 目录准备
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    logger = setup_logger(args.log_dir)
    logger.info(f"Save Dir: {args.save_dir}")
    
    # -------------------------------------------------------------------------
    # 1. 记忆库初始化
    # -------------------------------------------------------------------------
    print("Initializing Memory Bank...") 
    logger.info("\n" + "="*50)
    logger.info("Initializing Memory Bank...")
    logger.info("="*50)
    
    if args.init_memory:
        normal_rgb_loader, _ = dataset_obj.get_normal_loader()
        # 初始化阶段强制使用 Student 模型 (此时 Teacher 尚未完全就绪)
        model.initialize_memory(normal_rgb_loader, device, teacher_model=None)
        
        if teacher_model is not None:
            logger.info("Memory initialized. Mean Teacher is ready for EMA updates.")
    
    # -------------------------------------------------------------------------
    # 2. 训练循环
    # -------------------------------------------------------------------------
    print("Start Training Loop...")
    logger.info("\n" + "="*50)
    logger.info("Start Training Loop")
    logger.info("="*50)
    
    # 初始化损失函数
    criterion = TotalLoss(
        num_parts=args.num_parts,
        label_smoothing=args.label_smoothing,
        lambda_id=1.0,
        lambda_triplet=args.lambda_triplet,
        lambda_graph=args.lambda_graph,
        lambda_orth=args.lambda_orth,
        lambda_mod=args.lambda_mod,
        use_adaptive_weight=False 
    ).to(device)
    
    # 初始化 GradScaler (仅在开启 AMP 时)
    scaler = GradScaler(device='cuda') if args.amp else None
    
    best_rank1 = 0.0
    best_map = 0.0
    
    for epoch in range(args.total_epoch):
        start_time = time.time()
        
        # 执行一个 Epoch 的训练
        avg_losses = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler,
            device, epoch, logger, args, teacher_model
        )
        
        # 更新学习率
        if scheduler is not None:
            scheduler.step()
            
        # 记录日志到文件
        epoch_time = time.time() - start_time
        curr_lr = optimizer.param_groups[0]['lr']
        logger.info(f"Epoch {epoch+1}/{args.total_epoch} [Time: {epoch_time:.1f}s, LR: {curr_lr:.6f}]")
        logger.info(f"  Loss: {avg_losses['loss_total']:.4f} (ID: {avg_losses['loss_id']:.4f}, Graph: {avg_losses['loss_graph']:.4f})")
        
        # 保存常规 Checkpoint
        if (epoch + 1) % args.save_epoch == 0:
            save_path = os.path.join(args.save_dir, f'epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch + 1,
                'model': model.state_dict(),
                'optim': optimizer.state_dict(),
                'args': vars(args)
            }, save_path)
            
        # 执行验证
        if (epoch + 1) % args.eval_epoch == 0:
            print(f"Evaluating at Epoch {epoch+1}...") 
            logger.info("Evaluating (Student Model)...")
            rank1, mAP, mINP = test(
                model, 
                dataset_obj.query_loader, 
                dataset_obj.gallery_loaders, 
                args, device
            )
            
            logger.info(f"Validation: Rank-1 {rank1:.2f}%, mAP {mAP:.2f}%")
            
            # 保存最佳模型
            if rank1 > best_rank1:
                best_rank1 = rank1
                best_map = mAP
                torch.save(model.state_dict(), os.path.join(args.save_dir, 'best_model.pth'))
                if teacher_model is not None:
                    torch.save(teacher_model.state_dict(), os.path.join(args.save_dir, 'best_teacher_model.pth'))
                logger.info(f"New Best Model Saved! (Rank-1: {best_rank1:.2f}%)")
            
            # 验证结束后切回训练模式
            model.train() 

    logger.info("Training Finished.")
    print("Training Finished.")