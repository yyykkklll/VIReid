"""
PF-MGCD 训练脚本 (Advanced Version)
[恢复] Memory Bank 更新逻辑
"""

import os
import time
import logging
import torch
import torch.nn as nn
from tqdm import tqdm
from models.loss import TotalLoss
from task.test import test

try:
    from torch.amp import autocast, GradScaler
except ImportError:
    from torch.cuda.amp import GradScaler
    from torch import autocast

def setup_logger(log_dir, log_file='train.log'):
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_file)
    if os.path.exists(log_path): open(log_path, 'w').close()
    
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if logger.hasHandlers(): logger.handlers.clear()
        
    fh = logging.FileHandler(log_path, mode='a')
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter('[%(asctime)s] - %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(fh)
    
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING) 
    ch.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(ch)
    return logger

def update_ema_variables(model, ema_model, alpha, global_step):
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)
    for ema_buffer, buffer in zip(ema_model.buffers(), model.buffers()):
        if buffer.dtype.is_floating_point:
            ema_buffer.data.mul_(alpha).add_(buffer.data, alpha=1 - alpha)
        else:
            ema_buffer.data.copy_(buffer.data)

def train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, epoch, logger, args, teacher_model=None):
    model.train()
    if teacher_model: teacher_model.train()
    
    total_loss = 0
    loss_items = {'loss_id': 0, 'loss_triplet': 0, 'loss_graph': 0}
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.total_epoch}')
    
    for batch_idx, batch_data in enumerate(pbar):
        # 数据解包
        if len(batch_data) == 3:
            images, labels, cams = batch_data
            images, labels, cams = images.to(device), labels.to(device), cams.to(device)
            modality_labels = (cams >= 3).long()
        elif len(batch_data) == 6: # Adapter 可能会返回 6 个元素如果没用 collate_fn 整理，但在 Adapter 中我们已经封装好了 __getitem__ 返回 3 个
             # 如果 Adapter 返回的是 (img, label, cam)，则这里无需特殊处理
             # 但如果是原始 RegDB/SYSU 的 native loader，可能会拼成 6 个
             # 我们的新 Adapter 统一返回 3 个，所以主要走上面的逻辑
             pass 
        
        # 兼容性处理：如果 UnifiedDataset.__getitem__ 返回 3 个元素
        if isinstance(batch_data, list) and len(batch_data) == 3:
             pass # 上面已经处理
        
        optimizer.zero_grad()
        
        # 前向传播
        if args.amp and scaler is not None:
            with autocast(device_type='cuda'):
                outputs = model(images, labels=labels)
                loss, loss_dict = criterion(outputs, labels, current_epoch=epoch)
            
            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images, labels=labels)
            loss, loss_dict = criterion(outputs, labels, current_epoch=epoch)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
        
        # EMA 更新
        if teacher_model:
            update_ema_variables(model, teacher_model, alpha=0.999, global_step=epoch*len(train_loader)+batch_idx)
        
        # 统计
        total_loss += loss.item()
        for key in loss_items.keys():
            if key in loss_dict: loss_items[key] += loss_dict[key]
        
        pbar.set_postfix({'Loss': f'{loss.item():.4f}', 'ID': f'{loss_dict["loss_id"]:.4f}', 'G': f'{loss_dict["loss_graph"]:.4f}'})
        
        # [核心] 更新记忆库 (Warmup 后)
        if epoch >= args.warmup_epochs:
            with torch.no_grad():
                # 使用当前 Batch 的特征更新记忆库
                # 这里使用 outputs['id_features'] (Pre-BN)
                model.memory_bank.update_memory(outputs['id_features'], labels)
    
    num_batches = len(train_loader)
    avg_loss_dict = {k: v / num_batches for k, v in loss_items.items()}
    avg_loss_dict['loss_total'] = total_loss / num_batches
    return avg_loss_dict

def train(model, train_loader, dataset_obj, optimizer, scheduler, args, device, teacher_model=None):
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    logger = setup_logger(args.log_dir)
    logger.info(f"Save Dir: {args.save_dir}")
    
    # 1. 初始化记忆库
    print("Initializing Memory Bank...") 
    if args.init_memory:
        normal_rgb_loader, _ = dataset_obj.get_normal_loader()
        # 初始化阶段强制使用 student
        model.initialize_memory(normal_rgb_loader, device, teacher_model=None)
        logger.info("Memory initialized.")
    
    # 2. 损失函数
    print("Start Training Loop (Advanced)...")
    criterion = TotalLoss(
        num_parts=args.num_parts,
        lambda_graph=args.lambda_graph, # 从 args 传入权重
        label_smoothing=args.label_smoothing,
        start_epoch=20
    ).to(device)
    
    scaler = GradScaler(device='cuda') if args.amp else None
    best_rank1 = 0.0
    
    for epoch in range(args.total_epoch):
        start_time = time.time()
        
        avg_losses = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler,
            device, epoch, logger, args, teacher_model
        )
        
        if scheduler: scheduler.step()
        
        epoch_time = time.time() - start_time
        curr_lr = optimizer.param_groups[0]['lr']
        logger.info(f"Epoch {epoch+1}/{args.total_epoch} [Time: {epoch_time:.1f}s, LR: {curr_lr:.6f}]")
        logger.info(f"  Loss: {avg_losses['loss_total']:.4f} (ID: {avg_losses['loss_id']:.4f}, Graph: {avg_losses['loss_graph']:.4f})")
        
        if (epoch + 1) % args.save_epoch == 0:
            torch.save({
                'epoch': epoch + 1,
                'model': model.state_dict(),
                'optim': optimizer.state_dict(),
                'args': vars(args)
            }, os.path.join(args.save_dir, f'epoch_{epoch+1}.pth'))
            
        if (epoch + 1) % args.eval_epoch == 0:
            print(f"Evaluating at Epoch {epoch+1}...") 
            logger.info("Evaluating...")
            rank1, mAP, mINP = test(model, dataset_obj.query_loader, dataset_obj.gallery_loaders, args, device)
            logger.info(f"Validation: Rank-1 {rank1:.2f}%, mAP {mAP:.2f}%")
            
            if rank1 > best_rank1:
                best_rank1 = rank1
                torch.save(model.state_dict(), os.path.join(args.save_dir, 'best_model.pth'))
                if teacher_model:
                    torch.save(teacher_model.state_dict(), os.path.join(args.save_dir, 'best_teacher_model.pth'))
                logger.info(f"New Best Model Saved! (Rank-1: {best_rank1:.2f}%)")
            
            model.train() 

    logger.info("Training Finished.")
    print("Training Finished.")