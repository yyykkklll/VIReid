"""
task/train.py - 适配模态标签生成
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

@torch.no_grad()
def update_ema_variables(model, ema_model, alpha, global_step):
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)
    for ema_buffer, buffer in zip(ema_model.buffers(), model.buffers()):
        if buffer.dtype.is_floating_point:
            ema_buffer.data.mul_(alpha).add_(buffer.data, alpha=1 - alpha)
        else:
            ema_buffer.data.copy_(buffer.data)

def train_one_epoch(model, train_loader, criterion, optimizer, scaler, 
                    device, epoch, logger, args, teacher_model=None):
    model.train()
    if teacher_model: teacher_model.train()
    total_loss = 0
    loss_items = {'loss_id': 0, 'loss_triplet': 0, 'loss_graph': 0, 'loss_adv': 0}
    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.total_epoch}')
    
    for batch_idx, batch_data in enumerate(pbar):
        if len(batch_data) == 3:
            images, labels, cams = batch_data
            images = images.to(device)
            labels = labels.to(device)
            cams = cams.to(device)
            # 生成模态标签: 0 for RGB, 1 for IR
            # 假设 cams >= 3 是 IR (SYSU: 3,6; RegDB: 3; LLCM: adapter logic)
            modality_labels = (cams >= 3).long()
        else:
            # 兼容旧接口，如果没有 cam 信息，默认全 RGB (不应发生)
            images, labels = batch_data[:2]
            images = images.to(device)
            labels = labels.to(device)
            modality_labels = torch.zeros(labels.size(0), dtype=torch.long, device=device)

        optimizer.zero_grad()
        if args.amp and scaler is not None:
            with autocast(device_type='cuda'):
                # 传入 modality_labels 用于对抗训练
                outputs = model(images, labels=labels)
                loss, loss_dict = criterion(outputs, labels, current_epoch=epoch, modality_labels=modality_labels)
            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images, labels=labels)
            loss, loss_dict = criterion(outputs, labels, current_epoch=epoch, modality_labels=modality_labels)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
        
        if teacher_model:
            global_step = epoch * len(train_loader) + batch_idx
            update_ema_variables(model, teacher_model, alpha=0.999, global_step=global_step)
        
        total_loss += loss.item()
        for key in loss_items.keys():
            if key in loss_dict: loss_items[key] += loss_dict[key]
        
        postfix = {
            'Loss': f'{loss.item():.4f}',
            'ID': f'{loss_dict.get("loss_id", 0):.4f}'
        }
        if args.lambda_graph > 0: postfix['Graph'] = f'{loss_dict.get("loss_graph", 0):.4f}'
        if args.lambda_adv > 0: postfix['Adv'] = f'{loss_dict.get("loss_adv", 0):.4f}'
        pbar.set_postfix(postfix)
        
        if epoch >= args.warmup_epochs:
            if hasattr(model, 'memory_bank') and model.memory_bank is not None:
                with torch.no_grad():
                    if 'id_features' in outputs:
                        detached_features = [feat.detach() for feat in outputs['id_features']]
                        model.memory_bank.update_memory(detached_features, labels)

    avg_loss_dict = {k: v / len(train_loader) for k, v in loss_items.items()}
    avg_loss_dict['loss_total'] = total_loss / len(train_loader)
    return avg_loss_dict

def train(model, train_loader, dataset_obj, optimizer, scheduler, args, device,
          teacher_model=None, start_epoch=0):
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    logger = setup_logger(args.log_dir)
    logger.info(f"📁 Save Dir: {args.save_dir}")
    
    if start_epoch == 0 and args.init_memory:
        if hasattr(model, 'memory_bank') and model.memory_bank is not None:
            print("🔄 正在初始化记忆库...")
            normal_rgb_loader, _ = dataset_obj.get_normal_loader()
            model.initialize_memory(normal_rgb_loader, device, teacher_model=None)
            logger.info("✅ 记忆库初始化完成")
    
    criterion = TotalLoss(
        num_parts=args.num_parts,
        lambda_graph=args.lambda_graph,
        lambda_triplet=getattr(args, 'lambda_triplet', 1.0),
        lambda_adv=getattr(args, 'lambda_adv', 0.1), # 传入对抗权重
        label_smoothing=args.label_smoothing,
        start_epoch=20
    ).to(device)
    
    scaler = GradScaler(device='cuda') if args.amp else None
    best_rank1 = 0.0
    
    for epoch in range(start_epoch, args.total_epoch):
        start_time = time.time()
        avg_losses = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler,
            device, epoch, logger, args, teacher_model
        )
        if scheduler: scheduler.step()
        
        epoch_time = time.time() - start_time
        curr_lr = optimizer.param_groups[0]['lr']
        logger.info(f"Epoch {epoch+1}/{args.total_epoch} [⏱️ {epoch_time:.1f}s, 📉 LR: {curr_lr:.6f}]")
        log_msg = f"  Loss: {avg_losses['loss_total']:.4f} (ID: {avg_losses['loss_id']:.4f}, Triplet: {avg_losses['loss_triplet']:.4f}"
        if args.lambda_graph > 0: log_msg += f", Graph: {avg_losses['loss_graph']:.4f}"
        if args.lambda_adv > 0: log_msg += f", Adv: {avg_losses['loss_adv']:.4f}"
        log_msg += ")"
        logger.info(log_msg)
        
        if (epoch + 1) % args.save_epoch == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'model': model.state_dict(),
                'best_rank1': best_rank1
            }
            torch.save(checkpoint, os.path.join(args.save_dir, f'epoch_{epoch+1}.pth'))
        
        if (epoch + 1) % args.eval_epoch == 0:
            rank1, mAP, mINP = test(model, dataset_obj.query_loader, dataset_obj.gallery_loaders, args, device)
            logger.info(f"📈 Validation: Rank-1={rank1:.2f}%, mAP={mAP:.2f}%")
            if rank1 > best_rank1:
                best_rank1 = rank1
                torch.save({'model': model.state_dict(), 'rank1': rank1}, os.path.join(args.save_dir, 'best_model.pth'))
                logger.info(f"🏆 New Best Model! (Rank-1: {best_rank1:.2f}%)")
            model.train()