"""
task/train.py - PF-MGCD训练脚本 (完整修复版)

修复日志:
1. [P0] 支持断点恢复start_epoch参数
2. [P0] 保存Teacher模型状态到checkpoint
3. [P2] 添加@torch.no_grad装饰器防止内存泄漏
4. 添加详细的中文注释

功能:
- 训练循环管理
- EMA更新Teacher模型
- 记忆库动态更新
- 定期评估和保存
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
    """
    创建日志记录器
    
    Args:
        log_dir: 日志目录
        log_file: 日志文件名
    Returns:
        logger: logging.Logger对象
    """
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_file)
    
    # 清空旧日志
    if os.path.exists(log_path):
        open(log_path, 'w').close()
    
    # 创建logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    
    # 清除旧handlers
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # 文件handler
    fh = logging.FileHandler(log_path, mode='a')
    fh.setLevel(logging.INFO)
    fh.setFormatter(
        logging.Formatter(
            '[%(asctime)s] - %(levelname)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    )
    logger.addHandler(fh)
    
    # 控制台handler（只输出WARNING及以上）
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)
    ch.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(ch)
    
    return logger


@torch.no_grad()  # [修复] 添加装饰器，确保不构建计算图
def update_ema_variables(model, ema_model, alpha, global_step):
    """
    EMA (Exponential Moving Average) 更新Teacher模型权重
    
    Teacher参数更新公式:
    θ_teacher ← α * θ_teacher + (1-α) * θ_student
    
    Args:
        model: Student模型
        ema_model: Teacher模型（EMA）
        alpha: 动量系数，通常取0.999
        global_step: 全局训练步数（用于动态调整alpha，可选）
    """
    # 更新可训练参数
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)
    
    # 更新buffer（如BN的running_mean/running_var）
    for ema_buffer, buffer in zip(ema_model.buffers(), model.buffers()):
        if buffer.dtype.is_floating_point:
            # 浮点型buffer使用EMA更新
            ema_buffer.data.mul_(alpha).add_(buffer.data, alpha=1 - alpha)
        else:
            # 整型buffer直接复制（如记忆库的initialized标记）
            ema_buffer.data.copy_(buffer.data)


def train_one_epoch(model, train_loader, criterion, optimizer, scaler, 
                    device, epoch, logger, args, teacher_model=None):
    """
    单个epoch的训练
    
    Args:
        model: Student模型
        train_loader: 训练数据加载器
        criterion: 损失函数
        optimizer: 优化器
        scaler: AMP的GradScaler（混合精度训练）
        device: 计算设备
        epoch: 当前epoch索引
        logger: 日志记录器
        args: 超参数配置
        teacher_model: Teacher模型（可选）
    
    Returns:
        avg_loss_dict: Dict，平均损失字典
    """
    model.train()
    if teacher_model:
        teacher_model.train()  # BN层需要train模式
    
    # 损失统计
    total_loss = 0
    loss_items = {'loss_id': 0, 'loss_triplet': 0, 'loss_graph': 0}
    
    # 进度条
    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.total_epoch}')
    
    for batch_idx, batch_data in enumerate(pbar):
        # 1. 数据解包（兼容多种格式）
        if len(batch_data) == 3:
            images, labels, cams = batch_data
            images = images.to(device)
            labels = labels.to(device)
            cams = cams.to(device)
            modality_labels = (cams >= 3).long()  # 假设cam>=3为红外
        else:
            # 其他格式
            images, labels = batch_data[:2]
            images = images.to(device)
            labels = labels.to(device)
        
        # 2. 清空梯度
        optimizer.zero_grad()
        
        # 3. 前向传播 + 反向传播
        if args.amp and scaler is not None:
            # [混合精度训练] 使用FP16加速
            with autocast(device_type='cuda'):
                outputs = model(images, labels=labels)
                loss, loss_dict = criterion(outputs, labels, current_epoch=epoch)
            
            # 缩放损失并反向传播
            scaler.scale(loss).backward()
            
            # 梯度裁剪（防止梯度爆炸）
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)  # 先unscale才能裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            
            # 更新参数
            scaler.step(optimizer)
            scaler.update()
        else:
            # [FP32训练] 标准流程
            outputs = model(images, labels=labels)
            loss, loss_dict = criterion(outputs, labels, current_epoch=epoch)
            loss.backward()
            
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            
            optimizer.step()
        
        # 4. EMA更新Teacher模型
        if teacher_model:
            global_step = epoch * len(train_loader) + batch_idx
            update_ema_variables(model, teacher_model, alpha=0.999, global_step=global_step)
        
        # 5. 统计损失
        total_loss += loss.item()
        for key in loss_items.keys():
            if key in loss_dict:
                loss_items[key] += loss_dict[key]
        
        # 更新进度条
        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'ID': f'{loss_dict.get("loss_id", 0):.4f}',
            'Graph': f'{loss_dict.get("loss_graph", 0):.4f}'
        })
        
        # 6. [核心] 更新记忆库（Warmup后启用）
        # 修复后 (task/train.py)
        if epoch >= args.warmup_epochs:
            with torch.no_grad():
                if 'id_features' in outputs:
                    # [修复] 显式detach特征，确保完全切断计算图
                    detached_features = [feat.detach() for feat in outputs['id_features']]
                    model.memory_bank.update_memory(detached_features, labels)

    
    # 计算平均损失
    num_batches = len(train_loader)
    avg_loss_dict = {k: v / num_batches for k, v in loss_items.items()}
    avg_loss_dict['loss_total'] = total_loss / num_batches
    
    return avg_loss_dict


def train(model, train_loader, dataset_obj, optimizer, scheduler, args, device,
          teacher_model=None, start_epoch=0):
    """
    训练主函数
    
    Args:
        model: Student模型
        train_loader: 训练数据加载器
        dataset_obj: 数据集对象（用于验证）
        optimizer: 优化器
        scheduler: 学习率调度器
        args: 超参数配置
        device: 计算设备
        teacher_model: Teacher模型（可选）
        start_epoch: [修复] 断点恢复的起始epoch
    """
    # 创建保存目录和日志
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    logger = setup_logger(args.log_dir)
    
    logger.info(f"📁 Save Dir: {args.save_dir}")
    logger.info(f"📊 Training: Epoch {start_epoch+1} ~ {args.total_epoch}")
    
    # 1. 初始化记忆库（仅在从头训练时）
    if start_epoch == 0 and args.init_memory:
        print("🔄 正在初始化记忆库...")
        normal_rgb_loader, _ = dataset_obj.get_normal_loader()
        model.initialize_memory(normal_rgb_loader, device, teacher_model=None)
        logger.info("✅ 记忆库初始化完成")
    
    # 2. 创建损失函数
    print(f"🚀 开始训练 (Epoch {start_epoch+1} ~ {args.total_epoch})...")
    criterion = TotalLoss(
        num_parts=args.num_parts,
        lambda_graph=args.lambda_graph,
        lambda_triplet=getattr(args, 'lambda_triplet', 1.0),
        label_smoothing=args.label_smoothing,
        start_epoch=20  # Graph Loss启动epoch
    ).to(device)
    
    # 混合精度训练
    scaler = GradScaler(device='cuda') if args.amp else None
    
    # 最佳模型跟踪
    best_rank1 = 0.0
    
    # 3. [修复] 从start_epoch开始训练
    for epoch in range(start_epoch, args.total_epoch):
        start_time = time.time()
        
        # 训练一个epoch
        avg_losses = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler,
            device, epoch, logger, args, teacher_model
        )
        
        # 更新学习率
        if scheduler:
            scheduler.step()
        
        # 记录日志
        epoch_time = time.time() - start_time
        curr_lr = optimizer.param_groups[0]['lr']
        logger.info(f"Epoch {epoch+1}/{args.total_epoch} [⏱️ {epoch_time:.1f}s, 📉 LR: {curr_lr:.6f}]")
        logger.info(
            f"  Loss: {avg_losses['loss_total']:.4f} "
            f"(ID: {avg_losses['loss_id']:.4f}, "
            f"Triplet: {avg_losses.get('loss_triplet', 0):.4f}, "
            f"Graph: {avg_losses.get('loss_graph', 0):.4f})"
        )
        
        # 4. [修复] 保存checkpoint（包含Teacher状态）
        if (epoch + 1) % args.save_epoch == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'model': model.state_dict(),
                'optim': optimizer.state_dict(),
                'scheduler': scheduler.state_dict() if scheduler else None,
                'teacher': teacher_model.state_dict() if teacher_model else None,
                'best_rank1': best_rank1,
                'args': vars(args)
            }
            save_path = os.path.join(args.save_dir, f'epoch_{epoch+1}.pth')
            torch.save(checkpoint, save_path)
            logger.info(f"💾 Checkpoint saved: {save_path}")
        
        # 5. 定期评估
        if (epoch + 1) % args.eval_epoch == 0:
            print(f"\n📊 评估模型 (Epoch {epoch+1})...")
            logger.info("🔍 开始评估...")
            
            # 调用测试函数
            rank1, mAP, mINP = test(
                model, 
                dataset_obj.query_loader,
                dataset_obj.gallery_loaders,
                args,
                device
            )
            
            logger.info(f"📈 Validation: Rank-1={rank1:.2f}%, mAP={mAP:.2f}%, mINP={mINP:.2f}%")
            
            # 保存最佳模型
            if rank1 > best_rank1:
                best_rank1 = rank1
                best_checkpoint = {
                    'epoch': epoch + 1,
                    'model': model.state_dict(),
                    'teacher': teacher_model.state_dict() if teacher_model else None,
                    'rank1': rank1,
                    'mAP': mAP,
                    'mINP': mINP
                }
                torch.save(best_checkpoint, os.path.join(args.save_dir, 'best_model.pth'))
                logger.info(f"🏆 New Best Model! (Rank-1: {best_rank1:.2f}%)")
            
            # 恢复训练模式
            model.train()
    
    logger.info("✅ 训练完成!")
    print("\n🎉 训练完成!")
