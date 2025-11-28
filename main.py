"""
Main Entry Point for PF-MGCD Training and Testing
[优化版] 集成 Mean Teacher 策略：教师网络作为学生网络的 EMA 镜像
[修复] 修复 lr-step 参数解析，支持 "20,40" 格式的 MultiStepLR
"""

import os
import sys
import random
import argparse
import numpy as np
import torch

# 添加项目路径到系统路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def set_seed(seed=42):
    """设置随机种子以确保实验可复现"""
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
                        choices=['sysu', 'regdb', 'llcm'], help='数据集选择')
    parser.add_argument('--data-path', type=str, default='./datasets', help='数据集根路径')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'], help='运行模式')
    parser.add_argument('--resume', type=str, default='', help='恢复训练的检查点路径')
    parser.add_argument('--gpu', type=str, default='0', help='使用的GPU编号')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    
    # ===== PF-MGCD模型参数 =====
    parser.add_argument('--num-parts', type=int, default=6, help='部件数量 K')
    parser.add_argument('--feature-dim', type=int, default=256, help='解耦后的特征维度')
    parser.add_argument('--memory-momentum', type=float, default=0.9, help='记忆库动量系数')
    parser.add_argument('--temperature', type=float, default=3.0, help='Softmax温度参数')
    parser.add_argument('--top-k', type=int, default=5, help='图传播的Top-K邻居数量')
    parser.add_argument('--pretrained', action='store_true', help='使用预训练的ResNet')
    
    # 骨干网络与精度
    parser.add_argument('--backbone', type=str, default='resnet50',
                        choices=['resnet50', 'resnet101', 'resnet152'], help='ResNet骨干网络类型')
    parser.add_argument('--amp', action='store_true', help='启用自动混合精度训练(A100推荐)')
    
    # ===== 数据集参数 =====
    parser.add_argument('--num-classes', type=int, default=395, help='类别数量')
    parser.add_argument('--num-workers', type=int, default=4, help='数据加载worker数')
    parser.add_argument('--pid-numsample', type=int, default=4, help='每个ID的样本数')
    parser.add_argument('--batch-pidnum', type=int, default=8, help='每batch的ID数')
    parser.add_argument('--test-batch', type=int, default=64, help='测试batch大小')
    parser.add_argument('--img-w', type=int, default=144, help='图像宽度')
    parser.add_argument('--img-h', type=int, default=288, help='图像高度')
    parser.add_argument('--relabel', action='store_true', default=True, help='是否重新标注')
    parser.add_argument('--search-mode', type=str, default='all', choices=['all', 'indoor'], help='搜索模式')
    parser.add_argument('--gall-mode', type=str, default='single', choices=['single', 'multi'], help='检索模式')
    parser.add_argument('--test-mode', type=str, default='v2t', choices=['v2t', 't2v'], help='测试模式(LLCM)')
    parser.add_argument('--trial', type=int, default=1, help='RegDB trial编号')
    
    # ===== 损失函数权重 =====
    parser.add_argument('--lambda-graph', type=float, default=1.0, help='图蒸馏损失权重')
    parser.add_argument('--lambda-orth', type=float, default=0.01, help='正交损失权重')
    parser.add_argument('--lambda-mod', type=float, default=0.5, help='模态损失权重')
    parser.add_argument('--lambda-triplet', type=float, default=0.5, help='Triplet损失权重')
    parser.add_argument('--label-smoothing', type=float, default=0.1, help='标签平滑系数')
    
    # ===== 训练参数 =====
    parser.add_argument('--total-epoch', type=int, default=120, help='总训练轮数')
    parser.add_argument('--warmup-epochs', type=int, default=10, help='Warmup轮数')
    parser.add_argument('--batch-size', type=int, default=32, help='批次大小')
    parser.add_argument('--lr', type=float, default=0.0003, help='学习率')
    parser.add_argument('--weight-decay', type=float, default=5e-4, help='权重衰减')
    parser.add_argument('--grad-clip', type=float, default=5.0, help='梯度裁剪阈值')
    
    # ===== 学习率调度 =====
    parser.add_argument('--lr-scheduler', type=str, default='step', choices=['step', 'cosine', 'plateau'])
    # [修复] 改为 str 类型，以支持 "20,40" 这样的列表输入
    parser.add_argument('--lr-step', type=str, default='40', help='StepLR的步长 或 MultiStepLR的里程碑(逗号分隔)')
    parser.add_argument('--lr-gamma', type=float, default=0.1, help='StepLR的衰减系数')
    
    # ===== 记忆库初始化 =====
    parser.add_argument('--init-memory', action='store_true', help='训练前初始化记忆库')
    
    # ===== 保存和日志 =====
    parser.add_argument('--save-dir', type=str, default='./checkpoints', help='模型保存目录')
    parser.add_argument('--log-dir', type=str, default='./logs', help='日志保存目录')
    parser.add_argument('--save-epoch', type=int, default=10, help='每隔多少epoch保存模型')
    parser.add_argument('--eval-epoch', type=int, default=10, help='每隔多少epoch评估模型')
    
    # ===== 测试参数 =====
    parser.add_argument('--test-batch-size', type=int, default=64, help='测试批次大小')
    parser.add_argument('--model-path', type=str, default='', help='测试模型路径')
    parser.add_argument('--pool-parts', action='store_true', help='测试时是否合并部件特征')
    parser.add_argument('--distance-metric', type=str, default='euclidean', choices=['euclidean', 'cosine'])
    
    args = parser.parse_args()
    
    # 数据集类别数配置
    if args.dataset == 'sysu':
        args.num_classes = 395
    elif args.dataset == 'regdb':
        args.num_classes = 206
    elif args.dataset == 'llcm':
        args.num_classes = 713
    
    return args


def main():
    """主函数"""
    args = parse_args()
    
    # 环境设置
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(args.seed)
    
    # 目录准备
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # 打印配置
    print("="*70)
    print(" "*20 + "PF-MGCD Configuration (Mean Teacher)")
    print("="*70)
    print(f"{'Dataset':<20}: {args.dataset}")
    print(f"{'Mode':<20}: {args.mode}")
    print(f"{'Backbone':<20}: {args.backbone.upper()}")
    print(f"{'Mixed Precision':<20}: {'Enabled' if args.amp else 'Disabled'}")
    print(f"{'Num Parts':<20}: {args.num_parts}")
    print(f"{'LR Schedule':<20}: {args.lr_scheduler} (Step: {args.lr_step})")
    print("="*70 + "\n")
    
    # 1. 创建学生模型 (PF-MGCD Student)
    print("Creating PF-MGCD Student Model...")
    from models.pfmgcd_model import PF_MGCD
    
    model = PF_MGCD(
        num_parts=args.num_parts,
        num_identities=args.num_classes,
        feature_dim=args.feature_dim,
        memory_momentum=args.memory_momentum,
        temperature=args.temperature,
        top_k=args.top_k,
        pretrained=args.pretrained,
        backbone=args.backbone
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Student Model Params: {total_params:.2f}M\n")
    
    # 2. 训练流程
    if args.mode == 'train':
        # 数据加载
        from datasets.dataloader_adapter import get_dataloader
        train_loader, val_loader = get_dataloader(args)
        
        # 创建 Mean Teacher 模型
        print("Creating Mean Teacher Network (EMA Clone)...")
        
        # Teacher 结构与 Student 完全一致，但 pretrained=False 以避免重复加载权重
        teacher_model = PF_MGCD(
            num_parts=args.num_parts,
            num_identities=args.num_classes,
            feature_dim=args.feature_dim,
            memory_momentum=args.memory_momentum,
            temperature=args.temperature,
            top_k=args.top_k,
            pretrained=False, # 稍后会加载Student权重
            backbone=args.backbone
        ).to(device)
        
        # 初始化 Teacher 权重为 Student 的副本
        teacher_model.load_state_dict(model.state_dict())
        
        # 冻结 Teacher 的所有参数 (不参与梯度下降，只通过 EMA 更新)
        for param in teacher_model.parameters():
            param.requires_grad = False
            
        print("Mean Teacher initialized (Weights copied from Student, Grads disabled).\n")
        
        # 优化器
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
        
        # 学习率调度 [修复逻辑]
        if args.lr_scheduler == 'step':
            # 判断是否为多个 milestones (例如 "20,40")
            if ',' in args.lr_step:
                milestones = [int(x) for x in args.lr_step.split(',')]
                print(f"Using MultiStepLR with milestones: {milestones}")
                scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=args.lr_gamma)
            else:
                step_size = int(args.lr_step)
                print(f"Using StepLR with step_size: {step_size}")
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=args.lr_gamma)
        
        elif args.lr_scheduler == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.total_epoch)
        else:
            scheduler = None
        
        # 数据集对象 (用于验证和初始化)
        if args.dataset == 'sysu':
            from datasets.sysu import SYSU
            dataset_obj = SYSU(args)
        elif args.dataset == 'regdb':
            from datasets.regdb import RegDB
            dataset_obj = RegDB(args)
        elif args.dataset == 'llcm':
            from datasets.llcm import LLCM
            dataset_obj = LLCM(args)
        
        # 进入训练循环 (传入 teacher_model)
        from task.train import train
        train(model, train_loader, dataset_obj, optimizer, scheduler, args, device, teacher_model)
    
    # 3. 测试流程
    elif args.mode == 'test':
        if args.model_path:
            print(f"Loading model from: {args.model_path}")
            checkpoint = torch.load(args.model_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            print("Model loaded successfully!\n")
        
        # 加载测试数据
        if args.dataset == 'sysu':
            from datasets.sysu import SYSU
            dataset_obj = SYSU(args)
        elif args.dataset == 'regdb':
            from datasets.regdb import RegDB
            dataset_obj = RegDB(args)
        elif args.dataset == 'llcm':
            from datasets.llcm import LLCM
            dataset_obj = LLCM(args)
        
        from task.test import test
        test(model, dataset_obj.query_loader, dataset_obj.gallery_loaders, args, device)


if __name__ == '__main__':
    main()