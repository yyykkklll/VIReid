"""
Training Script for PF-MGCD
支持混合精度训练 + Mean Teacher EMA 更新 + 自动目录管理
[修复] 解决 EMA 更新类型错误 + 关闭控制台 INFO 日志打印
"""

import os
import time
import logging
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from tqdm import tqdm
from models.loss import TotalLoss
from task.test import test


def setup_logger(log_dir, log_file='train.log'):
    """配置日志记录器"""
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_file)
    
    # 覆盖旧日志
    if os.path.exists(log_path):
        open(log_path, 'w').close()
    
    # 创建 logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.propagate = False  # 防止重复打印
    
    # 清除已有的 handlers
    if logger.hasHandlers():
        logger.handlers.clear()
        
    # 1. 文件处理器：记录所有 INFO 及以上日志 (保留详细日志在文件中)
    fh = logging.FileHandler(log_path, mode='a')
    fh.setLevel(logging.INFO)
    fh_formatter = logging.Formatter(
        '[%(asctime)s] - %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    fh.setFormatter(fh_formatter)
    logger.addHandler(fh)
    
    # 2. 控制台处理器：[关键修改] 只输出 WARNING 及以上日志
    # 这样控制台就不会打印 INFO 信息 (如时间和状态分割线)，只显示进度条
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING) 
    ch_formatter = logging.Formatter('%(message)s') 
    ch.setFormatter(ch_formatter)
    logger.addHandler(ch)
    
    return logger


def update_ema_variables(model, ema_model, alpha, global_step):
    """
    Mean Teacher 核心：指数移动平均 (EMA) 更新
    """
    # 1. 更新参数 (Parameters)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)
    
    # 2. 更新缓冲区 (Buffers) - [已修复类型错误]
    for ema_buffer, buffer in zip(ema_model.buffers(), model.buffers()):
        if buffer.dtype.is_floating_point:
            ema_buffer.data.mul_(alpha).add_(buffer.data, alpha=1 - alpha)
        else:
            ema_buffer.data.copy_(buffer.data)


def train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, epoch, logger, args, teacher_model=None):
    """训练单个Epoch"""
    model.train()
    if teacher_model is not None:
        teacher_model.train()
    
    total_loss = 0
    loss_items = {
        'loss_id': 0, 'loss_triplet': 0, 'loss_graph': 0, 'loss_orth': 0, 'loss_mod': 0
    }
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.total_epoch}')
    
    for batch_idx, batch_data in enumerate(pbar):
        # 1. 解包数据
        if len(batch_data) == 3:
            images, labels, cams = batch_data
            images, labels, cams = images.to(device), labels.to(device), cams.to(device)
            modality_labels = (cams >= 3).long()
        elif len(batch_data) == 6:
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
        
        # 2. 前向传播
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
    
    # 返回平均损失
    num_batches = len(train_loader)
    avg_loss_dict = {k: v / num_batches for k, v in loss_items.items()}
    avg_loss_dict['loss_total'] = total_loss / num_batches
    
    return avg_loss_dict


def train(model, train_loader, dataset_obj, optimizer, scheduler, args, device, teacher_model=None):
    """完整训练流程"""
    
    # 准备目录和日志
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    logger = setup_logger(args.log_dir)
    # 这条只会写到文件，不会打印到屏幕
    logger.info(f"Save Dir: {args.save_dir}")
    
    # 1. 初始化记忆库
    print("Initializing Memory Bank...") # 使用print在控制台提示进度
    logger.info("\n" + "="*50)
    logger.info("Initializing Memory Bank...")
    logger.info("="*50)
    
    if args.init_memory:
        normal_rgb_loader, _ = dataset_obj.get_normal_loader()
        # 初始化时强制使用 student 本身 (此时 teacher==student)
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
            
        # 记录日志 (仅写入文件)
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
            print(f"Evaluating at Epoch {epoch+1}...") # 控制台提示
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