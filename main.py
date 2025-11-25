"""
Main Entry Point for PF-MGCD Training and Testing
"""

import os
import sys
import random
import argparse
import numpy as np
import torch

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='PF-MGCD for VI-ReID')
    
    # ===== 基础设置 =====
    parser.add_argument('--dataset', type=str, default='sysu',
                        choices=['sysu', 'regdb', 'llcm'],
                        help='数据集选择')
    parser.add_argument('--data-path', type=str, default='./datasets',
                        help='数据集根路径')
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'test'],
                        help='运行模式')
    parser.add_argument('--resume', type=str, default='',
                        help='恢复训练的检查点路径')
    parser.add_argument('--gpu', type=str, default='0',
                        help='使用的GPU编号')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    
    # ===== PF-MGCD模型参数 =====
    parser.add_argument('--num-parts', type=int, default=6,
                        help='部件数量 K')
    parser.add_argument('--feature-dim', type=int, default=256,
                        help='解耦后的特征维度')
    parser.add_argument('--memory-momentum', type=float, default=0.9,
                        help='记忆库动量系数')
    parser.add_argument('--temperature', type=float, default=3.0,
                        help='Softmax温度参数')
    parser.add_argument('--top-k', type=int, default=5,
                        help='图传播的Top-K邻居数量')
    parser.add_argument('--pretrained', action='store_true',
                        help='使用预训练的ResNet50')
    
    # ===== 原有数据集参数 (兼容原代码) =====
    parser.add_argument('--num-classes', type=int, default=395,
                        help='类别数量')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='数据加载worker数')
    parser.add_argument('--pid-numsample', type=int, default=4,
                        help='每个ID的样本数')
    parser.add_argument('--batch-pidnum', type=int, default=8,
                        help='每batch的ID数')
    parser.add_argument('--test-batch', type=int, default=64,
                        help='测试batch大小')
    parser.add_argument('--img-w', type=int, default=144,
                        help='图像宽度')
    parser.add_argument('--img-h', type=int, default=288,
                        help='图像高度')
    parser.add_argument('--relabel', action='store_true', default=True,
                        help='是否重新标注')
    parser.add_argument('--search-mode', type=str, default='all',
                        choices=['all', 'indoor'],
                        help='搜索模式')
    parser.add_argument('--gall-mode', type=str, default='single',
                        choices=['single', 'multi'],
                        help='检索模式')
    parser.add_argument('--test-mode', type=str, default='v2t',
                        choices=['v2t', 't2v'],
                        help='测试模式(LLCM)')
    parser.add_argument('--trial', type=int, default=1,
                        help='RegDB trial编号')
    
    # ===== 损失函数权重 =====
    parser.add_argument('--lambda-graph', type=float, default=1.0,
                        help='图蒸馏损失权重')
    parser.add_argument('--lambda-orth', type=float, default=0.1,
                        help='正交损失权重')
    parser.add_argument('--lambda-mod', type=float, default=0.5,
                        help='模态损失权重')
    parser.add_argument('--label-smoothing', type=float, default=0.1,
                        help='标签平滑系数')
    
    # ===== 训练参数 =====
    parser.add_argument('--total-epoch', type=int, default=120,
                        help='总训练轮数')
    parser.add_argument('--warmup-epochs', type=int, default=10,
                        help='Warmup轮数')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='批次大小')
    parser.add_argument('--lr', type=float, default=0.0003,
                        help='学习率')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        help='权重衰减')
    parser.add_argument('--grad-clip', type=float, default=5.0,
                        help='梯度裁剪阈值')
    
    # ===== 学习率调度 =====
    parser.add_argument('--lr-scheduler', type=str, default='step',
                        choices=['step', 'cosine', 'plateau'],
                        help='学习率调度器')
    parser.add_argument('--lr-step', type=int, default=40,
                        help='StepLR的步长')
    parser.add_argument('--lr-gamma', type=float, default=0.1,
                        help='StepLR的衰减系数')
    
    # ===== 记忆库初始化 =====
    parser.add_argument('--init-memory', action='store_true',
                        help='训练前初始化记忆库')
    
    # ===== 保存和日志 =====
    parser.add_argument('--save-dir', type=str, default='./checkpoints',
                        help='模型保存目录')
    parser.add_argument('--log-dir', type=str, default='./logs',
                        help='日志保存目录')
    parser.add_argument('--save-epoch', type=int, default=10,
                        help='每隔多少epoch保存模型')
    parser.add_argument('--eval-epoch', type=int, default=10,
                        help='每隔多少epoch评估模型')
    
    # ===== 测试参数 =====
    parser.add_argument('--test-batch-size', type=int, default=64,
                        help='测试批次大小')
    parser.add_argument('--model-path', type=str, default='',
                        help='测试模型路径')
    parser.add_argument('--pool-parts', action='store_true',
                        help='测试时是否合并部件特征')
    parser.add_argument('--distance-metric', type=str, default='euclidean',
                        choices=['euclidean', 'cosine'],
                        help='距离度量方式')
    
    args = parser.parse_args()
    
    # 根据数据集设置num_classes
    if args.dataset == 'sysu':
        args.num_classes = 395
    elif args.dataset == 'regdb':
        args.num_classes = 206
    elif args.dataset == 'llcm':
        args.num_classes = 713
    
    return args


def main():
    """主函数"""
    # 解析参数
    args = parse_args()
    
    # 设置GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # 打印配置
    print("="*70)
    print(" "*20 + "PF-MGCD Configuration")
    print("="*70)
    print(f"{'Dataset':<20}: {args.dataset}")
    print(f"{'Mode':<20}: {args.mode}")
    print(f"{'Device':<20}: {device}")
    print(f"{'Num Parts':<20}: {args.num_parts}")
    print(f"{'Feature Dim':<20}: {args.feature_dim}")
    print(f"{'Total Epochs':<20}: {args.total_epoch}")
    print(f"{'Warmup Epochs':<20}: {args.warmup_epochs}")
    print(f"{'Learning Rate':<20}: {args.lr}")
    print(f"{'Lambda Graph':<20}: {args.lambda_graph}")
    print(f"{'Lambda Orth':<20}: {args.lambda_orth}")
    print(f"{'Lambda Mod':<20}: {args.lambda_mod}")
    print("="*70 + "\n")
    
    # 创建模型
    print("Creating PF-MGCD model...")
    from models.pfmgcd_model import PF_MGCD
    
    model = PF_MGCD(
        num_parts=args.num_parts,
        num_identities=args.num_classes,
        feature_dim=args.feature_dim,
        memory_momentum=args.memory_momentum,
        temperature=args.temperature,
        top_k=args.top_k,
        pretrained=args.pretrained
    ).to(device)
    
    print(f"Model created successfully!")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M\n")
    
    # 训练模式
    if args.mode == 'train':
        # 获取数据加载器
        from datasets.dataloader_adapter import get_dataloader
        train_loader, val_loader = get_dataloader(args)
        
        # 创建优化器
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
        
        # 创建学习率调度器
        if args.lr_scheduler == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, 
                step_size=args.lr_step, 
                gamma=args.lr_gamma
            )
        elif args.lr_scheduler == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=args.total_epoch
            )
        else:
            scheduler = None
        
        # 开始训练
        from task.train import train
        train(model, train_loader, val_loader, optimizer, scheduler, args, device)
    
    # 测试模式
    elif args.mode == 'test':
        if args.model_path:
            print(f"Loading model from: {args.model_path}")
            checkpoint = torch.load(args.model_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            print("Model loaded successfully!\n")
        
        from datasets.dataloader_adapter import get_dataloader
        _, test_loader = get_dataloader(args)
        
        from task.test import test
        test(model, test_loader, args, device)


if __name__ == '__main__':
    main()