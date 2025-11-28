"""
PF-MGCD 训练脚本
功能：
1. [兼容性] 支持混合精度训练 (AMP)，自动适配 PyTorch 2.0+ 至 2.5+
2. 实现 Mean Teacher 策略的 EMA (指数移动平均) 更新
3. 自动管理日志记录与模型保存 (控制台仅显示进度条)
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
    配置日志记录器
    """
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_file)
    
    # 覆盖旧日志
    if os.path.exists(log_path):
        open(log_path, 'w').close()
    
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    
    if logger.hasHandlers():
        logger.handlers.clear()
        
    # 1. 文件处理器：记录详细日志 (INFO)
    fh = logging.FileHandler(log_path, mode='a')
    fh.setLevel(logging.INFO)
    fh_formatter = logging.Formatter(
        '[%(asctime)s] - %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    fh.setFormatter(fh_formatter)
    logger.addHandler(fh)
    
    # 2. 控制台处理器：仅记录警告 (WARNING)
    # [用户要求] 保持控制台整洁，只显示 tqdm 进度条，不显示 INFO 日志
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING) 
    ch_formatter = logging.Formatter('%(message)s') 
    ch.setFormatter(ch_formatter)
    logger.addHandler(ch)
    
    return logger


def update_ema_variables(model, ema_model, alpha, global_step):
    """
    Mean Teacher EMA 更新
    """
    # 更新参数
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)
    
    # 更新 Buffer
    for ema_buffer, buffer in zip(ema_model.buffers(), model.buffers()):
        if buffer.dtype.is_floating_point:
            ema_buffer.data.mul_(alpha).add_(buffer.data, alpha=1 - alpha)
        else:
            ema_buffer.data.copy_(buffer.data)


def train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, epoch, logger, args, teacher_model=None):
    """
    训练单个 Epoch
    """
    model.train()
    if teacher_model is not None:
        teacher_model.train()
    
    total_loss = 0
    loss_items = {
        'loss_id': 0, 'loss_triplet': 0, 'loss_graph': 0, 'loss_orth': 0, 'loss_mod': 0
    }
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.total_epoch}')
    
    for batch_idx, batch_data in enumerate(pbar):
        # 1. 解包数据 (适配 UnifiedDataset 返回的格式)
        if len(batch_data) == 3:
            # 旧格式或测试格式: (images, labels, cams)
            images, labels, cams = batch_data
            images, labels, cams = images.to(device), labels.to(device), cams.to(device)
            modality_labels = (cams >= 3).long()
            
        elif len(batch_data) == 4:
            # [修复适配] 新格式: (images, aug_images, labels, cams)
            images, aug_images, labels, cams = batch_data
            images = images.to(device)
            # aug_images 目前暂未使用，如果算法需要可在此处处理
            labels = labels.to(device)
            cams = cams.to(device)
            modality_labels = (cams >= 3).long()
            
        elif len(batch_data) == 6:
            # 拼接格式 (如 LLCM/RegDB 原生 loader): rgb_imgs, ir_imgs, ...
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
        
        # 2. 前向传播 (混合精度)
        if args.amp and scaler is not None:
            with autocast(device_type='cuda'):
                outputs = model(images, labels=labels, modality_labels=modality_labels)
                loss, loss_dict = criterion(outputs, labels, modality_labels, current_epoch=epoch)
            
            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            # 标准精度
            outputs = model(images, labels=labels, modality_labels=modality_labels)
            loss, loss_dict = criterion(outputs, labels, modality_labels, current_epoch=epoch)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
        
        # 3. EMA 更新
        if teacher_model is not None:
            update_ema_variables(model, teacher_model, alpha=0.999, global_step=epoch*len(train_loader)+batch_idx)
        
        # 4. 统计
        total_loss += loss.item()
        for key in loss_items.keys():
            if key in loss_dict:
                loss_items[key] += loss_dict[key]
        
        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'ID': f'{loss_dict["loss_id"]:.4f}',
            'Graph': f'{loss_dict["loss_graph"]:.4f}'
        })
        
        # 5. 记忆库更新
        if epoch >= args.warmup_epochs:
            with torch.no_grad():
                if teacher_model is not None:
                    teacher_outputs = teacher_model(images)
                    model.memory_bank.update_memory(teacher_outputs['id_features'], labels)
                else:
                    model.memory_bank.update_memory(outputs['id_features'], labels)
    
    num_batches = len(train_loader)
    avg_loss_dict = {k: v / num_batches for k, v in loss_items.items()}
    avg_loss_dict['loss_total'] = total_loss / num_batches
    
    return avg_loss_dict


def train(model, train_loader, dataset_obj, optimizer, scheduler, args, device, teacher_model=None):
    """
    完整训练流程
    """
    # 目录准备
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    logger = setup_logger(args.log_dir)
    logger.info(f"Save Dir: {args.save_dir}")
    
    # 1. 初始化记忆库
    print("Initializing Memory Bank...") 
    logger.info("\n" + "="*50)
    logger.info("Initializing Memory Bank...")
    logger.info("="*50)
    
    if args.init_memory:
        normal_rgb_loader, _ = dataset_obj.get_normal_loader()
        # 初始化阶段强制使用 student
        model.initialize_memory(normal_rgb_loader, device, teacher_model=None)
        
        if teacher_model is not None:
            logger.info("Memory initialized. Mean Teacher is ready for EMA updates.")
    
    # 2. 开始训练
    print("Start Training Loop...")
    logger.info("\n" + "="*50)
    logger.info("Start Training Loop")
    logger.info("="*50)
    
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
    
    scaler = GradScaler(device='cuda') if args.amp else None
    
    best_rank1 = 0.0
    best_map = 0.0
    
    for epoch in range(args.total_epoch):
        start_time = time.time()
        
        # 训练
        avg_losses = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler,
            device, epoch, logger, args, teacher_model
        )
        
        if scheduler is not None:
            scheduler.step()
            
        # 记录日志 (仅写入文件，控制台不显示 INFO)
        epoch_time = time.time() - start_time
        curr_lr = optimizer.param_groups[0]['lr']
        logger.info(f"Epoch {epoch+1}/{args.total_epoch} [Time: {epoch_time:.1f}s, LR: {curr_lr:.6f}]")
        logger.info(f"  Loss: {avg_losses['loss_total']:.4f} (ID: {avg_losses['loss_id']:.4f}, Graph: {avg_losses['loss_graph']:.4f})")
        
        # 保存
        if (epoch + 1) % args.save_epoch == 0:
            save_path = os.path.join(args.save_dir, f'epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch + 1,
                'model': model.state_dict(),
                'optim': optimizer.state_dict(),
                'args': vars(args)
            }, save_path)
            
        # 验证
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
            
            if rank1 > best_rank1:
                best_rank1 = rank1
                best_map = mAP
                torch.save(model.state_dict(), os.path.join(args.save_dir, 'best_model.pth'))
                if teacher_model is not None:
                    torch.save(teacher_model.state_dict(), os.path.join(args.save_dir, 'best_teacher_model.pth'))
                logger.info(f"New Best Model Saved! (Rank-1: {best_rank1:.2f}%)")
            
            model.train() 

    logger.info("Training Finished.")
    print("Training Finished.")