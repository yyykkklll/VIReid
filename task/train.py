"""
Training Pipeline for PF-MGCD
实现PF-MGCD的完整训练流程
"""

import os
import sys
import time

import torch
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.loss import TotalLoss


def train_one_epoch(model, dataloader, criterion, optimizer, epoch, args, device):
    """
    训练一个epoch
    Args:
        model: PF_MGCD模型
        dataloader: 训练数据加载器
        criterion: 损失函数 (TotalLoss)
        optimizer: 优化器
        epoch: 当前epoch
        args: 参数配置
        device: 设备
    Returns:
        avg_loss_dict: 平均损失字典
    """
    model.train()

    # 损失累加器
    loss_accum = {
        'loss_total': 0.0,
        'loss_id': 0.0,
        'loss_graph': 0.0,
        'loss_orth': 0.0,
        'loss_mod': 0.0
    }

    # 进度条
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}/{args.total_epoch}')

    for batch_idx, batch_data in enumerate(pbar):
        # 解包数据
        if len(batch_data) == 3:
            images, labels, cam_ids = batch_data
            modality_labels = (cam_ids >= 3).long()  # SYSU: cam_id >= 3 为红外
        else:
            images, labels = batch_data[:2]
            modality_labels = torch.zeros(labels.size(0), dtype=torch.long)

        images = images.to(device)
        labels = labels.to(device)
        modality_labels = modality_labels.to(device)

        # ===== 前向传播 (不更新记忆库) =====
        outputs = model(
            images,
            labels=labels,
            modality_labels=modality_labels,
            update_memory=False  # 在反向传播前不更新
        )

        # ===== 计算损失 =====
        total_loss, loss_dict = criterion(
            outputs,
            labels,
            modality_labels,
            current_epoch=epoch,
            warmup_epochs=args.warmup_epochs
        )

        # ===== 反向传播 =====
        optimizer.zero_grad()
        total_loss.backward()

        # 梯度裁剪 (可选)
        if hasattr(args, 'grad_clip') and args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        optimizer.step()

        # ===== 更新记忆库 (在梯度计算完成后) =====
        if epoch >= args.warmup_epochs:
            with torch.no_grad():
                # 提取特征（不需要梯度）
                id_features = outputs['id_features']
                # 确保特征已经detach
                id_features_detached = [f.detach().clone() for f in id_features]
                # 更新记忆库
                model.memory_bank.update_memory(id_features_detached, labels)

        # 累加损失
        for key in loss_accum.keys():
            loss_accum[key] += loss_dict[key]

        # 更新进度条
        pbar.set_postfix({
            'Loss': f"{loss_dict['loss_total']:.4f}",
            'ID': f"{loss_dict['loss_id']:.4f}",
            'Graph': f"{loss_dict['loss_graph']:.4f}",
        })

    # 计算平均损失
    num_batches = len(dataloader)
    avg_loss_dict = {key: value / num_batches for key, value in loss_accum.items()}

    return avg_loss_dict


def train(model, train_loader, val_loader, optimizer, scheduler, args, device):
    """
    完整训练流程
    Args:
        model: PF_MGCD模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器 (可选)
        optimizer: 优化器
        scheduler: 学习率调度器
        args: 参数配置
        device: 设备
    """
    # 创建损失函数
    criterion = TotalLoss(
        num_parts=args.num_parts,
        lambda_graph=args.lambda_graph,
        lambda_orth=args.lambda_orth,
        lambda_mod=args.lambda_mod,
        label_smoothing=args.label_smoothing,
        use_adaptive_weight=True
    ).to(device)

    # 初始化记忆库
    if hasattr(args, 'init_memory') and args.init_memory:
        print("\n" + "=" * 50)
        print("Initializing Memory Bank...")
        print("=" * 50)
        model.initialize_memory(train_loader, device)

    # 最佳模型跟踪
    best_metric = 0.0
    best_epoch = 0

    # 训练日志
    print("\n" + "=" * 50)
    print("Start Training")
    print("=" * 50)
    print(f"Total Epochs: {args.total_epoch}")
    print(f"Warmup Epochs: {args.warmup_epochs}")
    print(f"Learning Rate: {args.lr}")
    print(f"Lambda Graph: {args.lambda_graph}")
    print(f"Lambda Orth: {args.lambda_orth}")
    print(f"Lambda Mod: {args.lambda_mod}")
    print("=" * 50 + "\n")

    # 开始训练
    for epoch in range(1, args.total_epoch + 1):
        epoch_start_time = time.time()

        # 训练一个epoch
        avg_loss_dict = train_one_epoch(
            model, train_loader, criterion, optimizer, epoch, args, device
        )

        # 学习率调度
        if scheduler is not None:
            scheduler.step()

        # 打印日志
        epoch_time = time.time() - epoch_start_time
        current_lr = optimizer.param_groups[0]['lr']

        print(f"\nEpoch {epoch}/{args.total_epoch} - Time: {epoch_time:.2f}s - LR: {current_lr:.6f}")
        print(f"  Total Loss: {avg_loss_dict['loss_total']:.4f}")
        print(f"  ID Loss:    {avg_loss_dict['loss_id']:.4f}")
        print(f"  Graph Loss: {avg_loss_dict['loss_graph']:.4f}")
        print(f"  Orth Loss:  {avg_loss_dict['loss_orth']:.4f}")
        print(f"  Mod Loss:   {avg_loss_dict['loss_mod']:.4f}")

        # 保存检查点
        if epoch % args.save_epoch == 0 or epoch == args.total_epoch:
            save_path = os.path.join(args.save_dir, f'pfmgcd_epoch{epoch}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'loss_dict': avg_loss_dict,
            }, save_path)
            print(f"  Model saved to {save_path}")

        # 验证 (可选)
        if val_loader is not None and epoch % args.eval_epoch == 0:
            print("\n" + "-" * 50)
            print("Running Validation...")
            print("-" * 50)

            # 导入测试函数
            try:
                from task.test import test
                rank1, mAP = test(model, val_loader, args, device)

                print(f"Validation Results: Rank-1: {rank1:.2f}%, mAP: {mAP:.2f}%")
                print("-" * 50 + "\n")

                # 保存最佳模型
                current_metric = rank1 + mAP
                if current_metric > best_metric:
                    best_metric = current_metric
                    best_epoch = epoch
                    best_path = os.path.join(args.save_dir, 'pfmgcd_best.pth')
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'rank1': rank1,
                        'mAP': mAP,
                    }, best_path)
                    print(f"  Best model updated! (Rank-1: {rank1:.2f}%, mAP: {mAP:.2f}%)")
            except ImportError:
                print("  Test module not found, skipping validation")
                print("-" * 50 + "\n")

    # 训练完成
    print("\n" + "=" * 50)
    print("Training Completed!")
    print("=" * 50)
    if val_loader is not None:
        print(f"Best Epoch: {best_epoch}")
        print(f"Best Metric: {best_metric:.2f}")
    print("=" * 50 + "\n")
