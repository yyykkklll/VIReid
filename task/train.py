"""
task/train.py - Enhanced Training with Better LR Schedule
"""
import os
import time
import logging
import torch
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
    logger = logging.getLogger('train')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    
    fh = logging.FileHandler(os.path.join(log_dir, log_file), mode='a')
    fh.setFormatter(logging.Formatter('[%(asctime)s] - %(levelname)s: %(message)s',
                                       datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(fh)
    return logger


def get_lr(epoch, args):
    """WarmupMultiStep 学习率策略"""
    if epoch < args.warmup_epochs:
        # 线性 warmup
        return args.lr * (epoch + 1) / args.warmup_epochs
    else:
        # MultiStep 衰减
        lr = args.lr
        for milestone in [30, 50]:  # 在 epoch 30 和 50 衰减
            if epoch >= milestone:
                lr *= 0.1
        return lr


def train_one_epoch(model, train_loader, criterion, optimizer, scaler,
                    device, epoch, args):
    model.train()
    total_loss = 0
    loss_items = {'loss_id': 0, 'loss_triplet': 0, 'loss_xmodal': 0}
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.total_epoch}')
    
    for batch_data in pbar:
        if len(batch_data) == 3:
            images, labels, cam_ids = batch_data
            cam_ids = cam_ids.to(device)
        else:
            images, labels = batch_data[:2]
            cam_ids = None
        
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        if args.amp and scaler is not None:
            with autocast(device_type='cuda'):
                outputs = model(images, labels=labels)
                loss, loss_dict = criterion(outputs, labels, cam_ids=cam_ids)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images, labels=labels)
            loss, loss_dict = criterion(outputs, labels, cam_ids=cam_ids)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
        
        total_loss += loss.item()
        for k in loss_items:
            if k in loss_dict:
                loss_items[k] += loss_dict[k]
        
        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'ID': f'{loss_dict.get("loss_id", 0):.4f}',
            'Tri': f'{loss_dict.get("loss_triplet", 0):.4f}',
            'XM': f'{loss_dict.get("loss_xmodal", 0):.4f}'
        })
    
    n = len(train_loader)
    return {k: v / n for k, v in loss_items.items()} | {'loss_total': total_loss / n}


def train(model, train_loader, dataset_obj, optimizer, scheduler, args, device,
          teacher_model=None, start_epoch=0):
    os.makedirs(args.save_dir, exist_ok=True)
    logger = setup_logger(args.log_dir)
    logger.info(f"📁 Save Dir: {args.save_dir}")
    
    # 增强版损失函数
    criterion = TotalLoss(
        num_parts=args.num_parts,
        num_classes=args.num_classes,
        feature_dim=args.feature_dim,
        lambda_triplet=getattr(args, 'lambda_triplet', 1.0),
        lambda_xmodal=getattr(args, 'lambda_xmodal', 0.5),
        label_smoothing=args.label_smoothing
    ).to(device)
    
    scaler = GradScaler() if args.amp else None
    best_rank1 = 0.0
    best_mAP = 0.0
    no_improve_count = 0
    
    for epoch in range(start_epoch, args.total_epoch):
        start_time = time.time()
        
        # 动态学习率
        lr = get_lr(epoch, args)
        for pg in optimizer.param_groups:
            pg['lr'] = lr
        
        avg_losses = train_one_epoch(model, train_loader, criterion, optimizer,
                                     scaler, device, epoch, args)
        
        # 日志
        logger.info(f"Epoch {epoch+1}/{args.total_epoch} "
                    f"[⏱️ {time.time()-start_time:.1f}s, 📉 LR: {lr:.6f}]")
        logger.info(f"  Loss: {avg_losses['loss_total']:.4f} "
                    f"(ID: {avg_losses['loss_id']:.4f}, "
                    f"Tri: {avg_losses['loss_triplet']:.4f}, "
                    f"XM: {avg_losses['loss_xmodal']:.4f})")
        
        # 验证
        if (epoch + 1) % args.eval_epoch == 0:
            rank1, mAP, mINP = test(model, dataset_obj.query_loader,
                                    dataset_obj.gallery_loaders, args, device)
            logger.info(f"📈 Validation: Rank-1={rank1:.2f}%, mAP={mAP:.2f}%, mINP={mINP:.2f}%")
            
            # 综合评分 (Rank-1 权重更高)
            score = 0.6 * rank1 + 0.4 * mAP
            best_score = 0.6 * best_rank1 + 0.4 * best_mAP
            
            if score > best_score:
                best_rank1 = rank1
                best_mAP = mAP
                no_improve_count = 0
                torch.save({
                    'model': model.state_dict(),
                    'rank1': rank1,
                    'mAP': mAP,
                    'epoch': epoch + 1
                }, os.path.join(args.save_dir, 'best_model.pth'))
                logger.info(f"🏆 New Best!  (Rank-1: {rank1:.2f}%, mAP: {mAP:.2f}%)")
            else:
                no_improve_count += 1
            
            model.train()
        
        # 保存 checkpoint
        if (epoch + 1) % args.save_epoch == 0:
            torch.save({
                'epoch': epoch + 1,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, os.path.join(args.save_dir, f'epoch_{epoch+1}.pth'))
    
    logger.info(f"🎉 Complete! Best Rank-1: {best_rank1:.2f}%, Best mAP: {best_mAP:.2f}%")