"""
Training Script for PF-MGCD
支持混合精度训练 + 改进的loss计算 + 自动目录管理
兼容 PyTorch 1.6+ (包括旧版和新版)
"""

import os
import time
import logging
import torch
import torch.nn as nn
from tqdm import tqdm
from models.loss import TotalLoss
from task.test import test

# [修复1] 兼容不同PyTorch版本的AMP导入
try:
    # PyTorch >= 1.12 (新版API)
    from torch.amp import autocast, GradScaler
except ImportError:
    # PyTorch < 1.12 (旧版API)
    from torch.cuda.amp import autocast, GradScaler


def setup_logger(log_dir, log_file='train.log'):
    """设置日志 - 自动创建目录"""
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_file)
    
    # [改进] 如果日志文件存在，清空它（覆盖模式）
    if os.path.exists(log_path):
        open(log_path, 'w').close()
    
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] - %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(log_path, mode='a'),  # append模式
            logging.StreamHandler()
        ],
        force=True  # 强制重新配置logging
    )
    return logging.getLogger(__name__)


def train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, epoch, logger, args):
    """训练一个epoch"""
    model.train()
    
    total_loss = 0
    loss_items = {
        'loss_id': 0,
        'loss_triplet': 0,
        'loss_graph': 0,
        'loss_orth': 0,
        'loss_mod': 0
    }
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.total_epoch}')
    
    for batch_idx, batch_data in enumerate(pbar):
        # 解包batch数据
        if len(batch_data) == 3:
            # 标准格式: (images, labels, cams)
            images, labels, cams = batch_data
            images = images.to(device)
            labels = labels.to(device)
            cams = cams.to(device)
            
            # 推断模态标签（基于camera ID）
            modality_labels = (cams >= 3).long()
        
        elif len(batch_data) == 6:
            # 分离RGB和IR的格式: (rgb_imgs, ir_imgs, rgb_labels, ir_labels, rgb_cams, ir_cams)
            rgb_imgs, ir_imgs, rgb_labels, ir_labels, rgb_cams, ir_cams = batch_data
            
            # 合并RGB和IR数据
            images = torch.cat([rgb_imgs, ir_imgs], dim=0).to(device)
            labels = torch.cat([rgb_labels, ir_labels], dim=0).to(device)
            cams = torch.cat([rgb_cams, ir_cams], dim=0).to(device)
            
            # 模态标签 (0=RGB, 1=IR)
            modality_labels = torch.cat([
                torch.zeros(len(rgb_imgs), dtype=torch.long),
                torch.ones(len(ir_imgs), dtype=torch.long)
            ]).to(device)
        
        else:
            raise ValueError(f"Unexpected batch format with {len(batch_data)} elements")
        
        optimizer.zero_grad()
        
        # 混合精度训练
        if args.amp and scaler is not None:
            # [修复2] 兼容旧版autocast (不支持device_type参数)
            try:
                amp_context = autocast(device_type='cuda')
            except TypeError:
                # 旧版PyTorch不支持device_type参数
                amp_context = autocast()
            
            with amp_context:
                # 前向传播
                outputs = model(images, labels=labels, modality_labels=modality_labels)
                
                # 计算损失
                loss, loss_dict = criterion(
                    outputs, 
                    labels, 
                    modality_labels, 
                    current_epoch=epoch
                )
            
            # 反向传播
            scaler.scale(loss).backward()
            
            # 梯度裁剪
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            
            scaler.step(optimizer)
            scaler.update()
        else:
            # 标准训练
            outputs = model(images, labels=labels, modality_labels=modality_labels)
            loss, loss_dict = criterion(
                outputs, 
                labels, 
                modality_labels, 
                current_epoch=epoch
            )
            
            loss.backward()
            
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            
            optimizer.step()
        
        # 统计损失
        total_loss += loss.item()
        for key in loss_items.keys():
            if key in loss_dict:
                loss_items[key] += loss_dict[key]
        
        # 更新进度条
        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'ID': f'{loss_dict["loss_id"]:.4f}',
            'Trip': f'{loss_dict["loss_triplet"]:.4f}',
            'Graph': f'{loss_dict["loss_graph"]:.4f}'
        })
        
        # 更新记忆库（warmup后）
        if epoch >= args.warmup_epochs:
            with torch.no_grad():
                model.memory_bank.update_memory(
                    outputs['id_features'],
                    labels
                )
    
    # 计算平均损失
    num_batches = len(train_loader)
    avg_loss_dict = {
        'loss_total': total_loss / num_batches,
        'loss_id': loss_items['loss_id'] / num_batches,
        'loss_triplet': loss_items['loss_triplet'] / num_batches,
        'loss_graph': loss_items['loss_graph'] / num_batches,
        'loss_orth': loss_items['loss_orth'] / num_batches,
        'loss_mod': loss_items['loss_mod'] / num_batches
    }
    
    return avg_loss_dict


def train(model, train_loader, dataset_obj, optimizer, scheduler, args, device):
    """
    完整训练流程
    自动根据save_dir和log_dir创建/覆盖目录
    """
    
    # ===== [关键改进] 自动创建目录 =====
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"Directory Setup:")
    print(f"  Save Dir: {args.save_dir}")
    print(f"  Log Dir:  {args.log_dir}")
    print(f"{'='*70}\n")
    
    # 设置日志
    logger = setup_logger(args.log_dir)
    
    logger.info("\n" + "="*50)
    logger.info("Initializing Memory Bank...")
    logger.info("="*50)
    
    # 初始化记忆库
    if args.init_memory:
        normal_rgb_loader, normal_ir_loader = dataset_obj.get_normal_loader()
        model.initialize_memory(normal_rgb_loader, device)
    
    logger.info("\n" + "="*50)
    logger.info("Start Training")
    logger.info("="*50)
    logger.info(f"Total Epochs: {args.total_epoch}")
    logger.info(f"Warmup Epochs: {args.warmup_epochs}")
    logger.info(f"Learning Rate: {args.lr}")
    logger.info(f"Lambda Graph: {args.lambda_graph}")
    logger.info(f"Lambda Orth: {args.lambda_orth}")
    logger.info(f"Lambda Mod: {args.lambda_mod}")
    logger.info(f"Lambda Triplet: {args.lambda_triplet}")
    logger.info("="*50 + "\n")
    
    # 创建损失函数
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
    
    # [修复3] 兼容不同版本的GradScaler初始化
    scaler = None
    if args.amp:
        try:
            # 新版API (支持device参数)
            scaler = GradScaler(device='cuda')
        except TypeError:
            # 旧版API (不支持device参数)
            scaler = GradScaler()
    
    # 最佳模型记录
    best_rank1 = 0.0
    best_map = 0.0
    best_epoch = 0
    
    # 开始训练
    for epoch in range(args.total_epoch):
        epoch_start_time = time.time()
        
        # 训练一个epoch
        avg_loss_dict = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler,
            device, epoch, logger, args
        )
        
        # 学习率调度
        if scheduler is not None:
            scheduler.step()
        
        # 记录训练信息
        epoch_time = time.time() - epoch_start_time
        current_lr = optimizer.param_groups[0]['lr']
        
        logger.info(f"Epoch {epoch+1}/{args.total_epoch} - Time: {epoch_time:.2f}s - LR: {current_lr:.6f}")
        logger.info(f"  Total Loss: {avg_loss_dict['loss_total']:.4f}")
        logger.info(f"  ID Loss:    {avg_loss_dict['loss_id']:.4f}")
        logger.info(f"  Triplet:    {avg_loss_dict['loss_triplet']:.4f}")
        logger.info(f"  Graph Loss: {avg_loss_dict['loss_graph']:.4f}")
        logger.info(f"  Orth Loss:  {avg_loss_dict['loss_orth']:.4f}")
        logger.info(f"  Mod Loss:   {avg_loss_dict['loss_mod']:.4f}")
        
        # ===== [改进] 简化保存路径，直接使用save_dir =====
        if (epoch + 1) % args.save_epoch == 0:
            save_path = os.path.join(args.save_dir, f'epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_rank1': best_rank1,
                'best_map': best_map,
                'args': vars(args)  # 保存训练配置
            }, save_path)
            logger.info(f"  Model saved to {save_path}")
        
        # 验证
        if (epoch + 1) % args.eval_epoch == 0:
            logger.info("-" * 50)
            logger.info("Running Validation...")
            logger.info("-" * 50)
            
            # 测试
            rank1, mAP, mINP = test(
                model, 
                dataset_obj.query_loader, 
                dataset_obj.gallery_loaders, 
                args, 
                device
            )
            
            logger.info(f"Validation Results: Rank-1 {rank1:.2f}, mAP {mAP:.2f}, mINP {mINP:.2f}")
            logger.info("-" * 50 + "\n")
            
            # 更新最佳模型
            if rank1 > best_rank1:
                best_rank1 = rank1
                best_map = mAP
                best_epoch = epoch + 1
                
                # ===== [改进] 简化保存路径 =====
                best_model_path = os.path.join(args.save_dir, 'best_model.pth')
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'rank1': rank1,
                    'mAP': mAP,
                    'mINP': mINP,
                    'args': vars(args)
                }, best_model_path)
                logger.info(f"✓ Best model updated! Rank-1 {rank1:.2f}, mAP {mAP:.2f}")
                logger.info(f"  Saved to: {best_model_path}")
            
            # 切回训练模式
            model.train()
    
    # 训练结束
    logger.info("\n" + "="*50)
    logger.info("Training Completed!")
    logger.info("="*50)
    logger.info(f"Best Epoch: {best_epoch}")
    logger.info(f"Best Rank-1: {best_rank1:.2f}%")
    logger.info(f"Best mAP: {best_map:.2f}%")
    logger.info(f"Model saved in: {args.save_dir}")
    logger.info("="*50)
