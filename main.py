"""
main.py - PF-MGCD 主入口程序 (Modular Ultimate Version)

功能:
- 支持基础 Strong Baseline (IBN+GeM)
- 支持高级特性开关 (Adversarial, Graph Reasoning)
- 自动适配不同数据集的训练策略
"""

import os
import sys
import argparse
import torch

# 添加项目路径到系统路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import set_seed

def parse_args():
    parser = argparse.ArgumentParser(description='PF-MGCD for Visible-Infrared Person Re-Identification')
    
    # ==================== 基础设置 ====================
    parser.add_argument('--dataset', type=str, default='sysu', choices=['sysu', 'regdb', 'llcm'])
    parser.add_argument('--data-path', type=str, default='./datasets')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--resume', type=str, default='')
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--seed', type=int, default=42)
    
    # ==================== 模型参数 ====================
    parser.add_argument('--num-parts', type=int, default=6)
    parser.add_argument('--feature-dim', type=int, default=512)
    parser.add_argument('--memory-momentum', type=float, default=0.9)
    parser.add_argument('--temperature', type=float, default=3.0)
    parser.add_argument('--top-k', type=int, default=5)
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--backbone', type=str, default='resnet50')
    parser.add_argument('--amp', action='store_true', help='启用混合精度训练')
    
    # [核心策略开关]
    parser.add_argument('--use-ibn', action='store_true', default=True, help='开启 IBN-Net (默认开启)')
    parser.add_argument('--use-adversarial', action='store_true', help='开启模态对抗训练 (策略三)')
    parser.add_argument('--use-graph-reasoning', action='store_true', help='开启图推理 GCN (策略四)')
    
    # ==================== 数据集参数 ====================
    parser.add_argument('--num-classes', type=int, default=395)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--pid-numsample', type=int, default=8)
    parser.add_argument('--batch-pidnum', type=int, default=8)
    parser.add_argument('--test-batch', type=int, default=128)
    parser.add_argument('--img-w', type=int, default=144)
    parser.add_argument('--img-h', type=int, default=288)
    parser.add_argument('--relabel', action='store_true', default=True)
    parser.add_argument('--search-mode', type=str, default='all', choices=['all', 'indoor'])
    parser.add_argument('--gall-mode', type=str, default='single', choices=['single', 'multi'])
    parser.add_argument('--test-mode', type=str, default='v2t', choices=['v2t', 't2v'])
    parser.add_argument('--trial', type=int, default=1)
    
    # ==================== 损失函数权重 ====================
    parser.add_argument('--lambda-graph', type=float, default=0.3, help='图蒸馏损失权重')
    parser. add_argument('--lambda-triplet', type=float, default=1.5, help='三元组损失权重')
    parser.add_argument('--lambda-adv', type=float, default=0.1, help='对抗损失权重')
    parser.add_argument('--lambda-center', type=float, default=0.005, help='Center Loss权重')  # 新增
    parser.add_argument('--graph-start-epoch', type=int, default=10, help='Graph Loss启用epoch')  # 新增
    parser.add_argument('--label-smoothing', type=float, default=0.1)
    parser.add_argument('--lambda-xmodal', type=float, default=0.5, help='跨模态损失权重')
    #保留旧参数接口防止报错 (实际已移除逻辑)
    parser.add_argument('--lambda-orth', type=float, default=0.0)
    parser.add_argument('--lambda-mod', type=float, default=0.0)
    
    # ==================== 训练参数 ====================
    parser.add_argument('--total-epoch', type=int, default=120)
    parser.add_argument('--warmup-epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.00035)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--grad-clip', type=float, default=5.0)
    parser.add_argument('--lr-scheduler', type=str, default='cosine')
    parser.add_argument('--lr-step', type=str, default='40,70')
    parser.add_argument('--lr-gamma', type=float, default=0.1)
    parser.add_argument('--init-memory', action='store_true')
    
    # ==================== 保存和日志 ====================
    parser.add_argument('--save-dir', type=str, default='./checkpoints')
    parser.add_argument('--log-dir', type=str, default='./logs')
    parser.add_argument('--save-epoch', type=int, default=10)
    parser.add_argument('--eval-epoch', type=int, default=5)
    
    # ==================== 测试参数 ====================
    parser.add_argument('--model-path', type=str, default='')
    parser.add_argument('--pool-parts', action='store_true')
    parser.add_argument('--distance-metric', type=str, default='euclidean')
    
    args = parser.parse_args()
    
    # 自动设置类别数
    if args.dataset == 'sysu': args.num_classes = 395
    elif args.dataset == 'regdb': args.num_classes = 206
    elif args.dataset == 'llcm': args.num_classes = 713
    
    return args

def main():
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(args.seed)
    
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    print("="*70)
    print(f"Dataset: {args.dataset.upper()} | Mode: {args.mode.upper()}")
    print(f"Strategies: IBN={args.use_ibn}, Adv={args.use_adversarial}, GCN={args.use_graph_reasoning}")
    print("="*70)
    
    # 创建模型
    print("🔧 Creating Student Model...")
    from models.pfmgcd_model import PF_MGCD
    
    model = PF_MGCD(
        num_parts=args.num_parts,
        num_identities=args.num_classes,
        feature_dim=args.feature_dim,
        memory_momentum=args.memory_momentum,
        temperature=args.temperature,
        top_k=args.top_k,
        pretrained=args.pretrained,
        backbone=args.backbone,
        use_ibn=args.use_ibn,
        use_adversarial=args.use_adversarial,      # [新增]
        use_graph_reasoning=args.use_graph_reasoning # [新增]
    ).to(device)
    
    if args.mode == 'train':
        from datasets.dataloader_adapter import get_dataloader
        train_loader, _ = get_dataloader(args)
        
        print("🔧 Creating Teacher Model...")
        teacher_model = PF_MGCD(
            num_parts=args.num_parts,
            num_identities=args.num_classes,
            feature_dim=args.feature_dim,
            memory_momentum=args.memory_momentum,
            temperature=args.temperature,
            top_k=args.top_k,
            pretrained=False,
            backbone=args.backbone,
            use_ibn=args.use_ibn,
            use_adversarial=args.use_adversarial,
            use_graph_reasoning=args.use_graph_reasoning
        ).to(device)
        
        teacher_model.load_state_dict(model.state_dict())
        for param in teacher_model.parameters(): param.requires_grad = False
        
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        
        # 简单断点恢复逻辑 (仅权重)
        start_epoch = 0
        if args.resume and os.path.isfile(args.resume):
            print(f"📂 Loading checkpoint: {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device)
            start_epoch = checkpoint.get('epoch', 0)
            model.load_state_dict(checkpoint['model'])
            if 'teacher' in checkpoint and checkpoint['teacher']:
                teacher_model.load_state_dict(checkpoint['teacher'])
            if 'optim' in checkpoint: optimizer.load_state_dict(checkpoint['optim'])
            print(f"✅ Resuming from epoch {start_epoch+1}")

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.total_epoch, last_epoch=-1)
        # 如果需要恢复 scheduler 状态可在此添加
        
        if args.dataset == 'sysu': from datasets.sysu import SYSU; dataset_obj = SYSU(args)
        elif args.dataset == 'regdb': from datasets.regdb import RegDB; dataset_obj = RegDB(args)
        elif args.dataset == 'llcm': from datasets.llcm import LLCM; dataset_obj = LLCM(args)
        
        from task.train import train
        train(model, train_loader, dataset_obj, optimizer, scheduler, args, device, teacher_model, start_epoch)
        
    elif args.mode == 'test':
        print(f"📂 Loading test model: {args.model_path}")
        checkpoint = torch.load(args.model_path, map_location=device)
        if 'model' in checkpoint: model.load_state_dict(checkpoint['model'])
        elif 'model_state_dict' in checkpoint: model.load_state_dict(checkpoint['model_state_dict'])
        
        if args.dataset == 'sysu': from datasets.sysu import SYSU; dataset_obj = SYSU(args)
        elif args.dataset == 'regdb': from datasets.regdb import RegDB; dataset_obj = RegDB(args)
        elif args.dataset == 'llcm': from datasets.llcm import LLCM; dataset_obj = LLCM(args)
        
        from task.test import test
        test(model, dataset_obj.query_loader, dataset_obj.gallery_loaders, args, device)

if __name__ == '__main__':
    main()