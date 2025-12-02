"""
main.py - PF-MGCD 主入口程序 (完整修复版)

修复日志:
1. [P0] 完善断点恢复逻辑 (Teacher + Scheduler + Memory Bank)
2. [P0] 修复测试模式checkpoint key兼容性问题
3. [P1] 添加记忆库状态检查和恢复
4. 删除重复的set_seed函数
5. 添加详细的中文注释

功能:
- 参数解析和配置管理
- 模型创建和初始化
- 训练/测试流程控制
- 断点恢复和保存

作者: PF-MGCD Team
日期: 2025-12-02
"""

import os
import sys
import argparse
import torch

# 添加项目路径到系统路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# [修复] 使用utils.py的set_seed，删除本地重复定义
from utils import set_seed


def parse_args():
    """
    解析命令行参数
    
    Returns:
        args: Namespace对象，包含所有超参数
    """
    parser = argparse.ArgumentParser(
        description='PF-MGCD for Visible-Infrared Person Re-Identification'
    )
    
    # ==================== 基础设置 ====================
    parser.add_argument('--dataset', type=str, default='sysu',
                        choices=['sysu', 'regdb', 'llcm'],
                        help='数据集选择: sysu(SYSU-MM01), regdb(RegDB), llcm(LLCM)')
    parser.add_argument('--data-path', type=str, default='./datasets',
                        help='数据集根路径')
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'test'],
                        help='运行模式: train(训练), test(测试)')
    parser.add_argument('--resume', type=str, default='',
                        help='恢复训练的检查点路径 (例如: checkpoints/sysu/epoch_50.pth)')
    parser.add_argument('--gpu', type=str, default='0',
                        help='使用的GPU编号 (例如: 0 或 0,1)')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子 (用于可复现性)')
    
    # ==================== PF-MGCD模型参数 ====================
    parser.add_argument('--num-parts', type=int, default=6,
                        help='人体部件数量K (默认6: 头-上身-中身-下身-腿部-脚部)')
    parser.add_argument('--feature-dim', type=int, default=512,
                        help='解耦后的特征维度D (Clean Baseline默认512)')
    parser.add_argument('--memory-momentum', type=float, default=0.9,
                        help='记忆库动量更新系数m (范围0~1)')
    parser.add_argument('--temperature', type=float, default=3.0,
                        help='图传播Softmax温度T (越大分布越平滑)')
    parser.add_argument('--top-k', type=int, default=5,
                        help='图传播Top-K邻居数量 (保留接口兼容)')
    parser.add_argument('--pretrained', action='store_true',
                        help='是否使用ImageNet预训练的ResNet权重')
    
    # 骨干网络与精度
    parser.add_argument('--backbone', type=str, default='resnet50',
                        choices=['resnet50', 'resnet101', 'resnet152'],
                        help='ResNet骨干网络类型')
    parser.add_argument('--amp', action='store_true',
                        help='启用自动混合精度训练AMP (A100推荐, 可加速2x)')
    
    # ==================== 数据集参数 ====================
    parser.add_argument('--num-classes', type=int, default=395,
                        help='类别数量N (自动根据数据集设置)')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='数据加载worker进程数')
    parser.add_argument('--pid-numsample', type=int, default=8,
                        help='每个ID的样本数 (用于PK采样)')
    parser.add_argument('--batch-pidnum', type=int, default=8,
                        help='每batch的ID数 (batch_size = pid_numsample * batch_pidnum)')
    parser.add_argument('--test-batch', type=int, default=128,
                        help='测试时的batch大小')
    parser.add_argument('--img-w', type=int, default=144,
                        help='输入图像宽度')
    parser.add_argument('--img-h', type=int, default=288,
                        help='输入图像高度')
    parser.add_argument('--relabel', action='store_true', default=True,
                        help='是否重新标注ID (弱监督场景)')
    parser.add_argument('--search-mode', type=str, default='all',
                        choices=['all', 'indoor'],
                        help='SYSU检索模式: all(全部摄像头), indoor(仅室内)')
    parser.add_argument('--gall-mode', type=str, default='single',
                        choices=['single', 'multi'],
                        help='SYSU Gallery模式: single(单次), multi(多次)')
    parser.add_argument('--test-mode', type=str, default='v2t',
                        choices=['v2t', 't2v'],
                        help='LLCM测试模式: v2t(可见光→红外), t2v(红外→可见光)')
    parser.add_argument('--trial', type=int, default=1,
                        help='RegDB trial编号 (1~10)')
    
    # ==================== 损失函数权重 ====================
    parser.add_argument('--lambda-graph', type=float, default=0.1,
                        help='图蒸馏损失权重 λ_graph')
    parser.add_argument('--lambda-orth', type=float, default=0.1,
                        help='正交损失权重 λ_orth (保留接口)')
    parser.add_argument('--lambda-mod', type=float, default=0.5,
                        help='模态判别损失权重 λ_mod (保留接口)')
    parser.add_argument('--lambda-triplet', type=float, default=0.5,
                        help='三元组损失权重 λ_triplet')
    parser.add_argument('--label-smoothing', type=float, default=0.1,
                        help='标签平滑系数 (范围0~1)')
    
    # ==================== 训练参数 ====================
    parser.add_argument('--total-epoch', type=int, default=120,
                        help='总训练轮数')
    parser.add_argument('--warmup-epochs', type=int, default=10,
                        help='Warmup轮数 (前期不启用Graph Loss)')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='训练批次大小')
    parser.add_argument('--lr', type=float, default=0.00035,
                        help='初始学习率')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        help='权重衰减系数 (L2正则)')
    parser.add_argument('--grad-clip', type=float, default=5.0,
                        help='梯度裁剪阈值 (防止梯度爆炸)')
    
    # ==================== 学习率调度 ====================
    parser.add_argument('--lr-scheduler', type=str, default='cosine',
                        choices=['step', 'cosine', 'plateau'],
                        help='学习率调度策略')
    parser.add_argument('--lr-step', type=str, default='40,70',
                        help='StepLR的步长 或 MultiStepLR的里程碑 (逗号分隔)')
    parser.add_argument('--lr-gamma', type=float, default=0.1,
                        help='StepLR的学习率衰减系数')
    
    # ==================== 记忆库初始化 ====================
    parser.add_argument('--init-memory', action='store_true',
                        help='训练前初始化记忆库 (推荐开启)')
    
    # ==================== 保存和日志 ====================
    parser.add_argument('--save-dir', type=str, default='./checkpoints',
                        help='模型保存目录')
    parser.add_argument('--log-dir', type=str, default='./logs',
                        help='日志保存目录')
    parser.add_argument('--save-epoch', type=int, default=10,
                        help='每隔多少epoch保存模型')
    parser.add_argument('--eval-epoch', type=int, default=5,
                        help='每隔多少epoch评估模型')
    
    # ==================== 测试参数 ====================
    parser.add_argument('--model-path', type=str, default='',
                        help='测试模型路径 (例如: checkpoints/best_model.pth)')
    parser.add_argument('--pool-parts', action='store_true',
                        help='测试时是否拼接所有部件特征 (True: K*D, False: D)')
    parser.add_argument('--distance-metric', type=str, default='euclidean',
                        choices=['euclidean', 'cosine'],
                        help='距离度量方式')
    
    args = parser.parse_args()
    
    # 根据数据集自动设置类别数
    if args.dataset == 'sysu':
        args.num_classes = 395
    elif args.dataset == 'regdb':
        args.num_classes = 206
    elif args.dataset == 'llcm':
        args.num_classes = 713
    
    return args


def main():
    """
    主函数
    """
    # ==================== 1. 解析参数和环境设置 ====================
    args = parse_args()
    
    # 设置GPU环境
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 设置随机种子（确保可复现性）
    set_seed(args.seed)
    
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # 打印配置信息
    print("="*70)
    print(" "*18 + "PF-MGCD Configuration (Bug Fixed)")
    print("="*70)
    print(f"{'Dataset':<20}: {args.dataset.upper()}")
    print(f"{'Mode':<20}: {args.mode.upper()}")
    print(f"{'Backbone':<20}: {args.backbone.upper()}")
    print(f"{'Mixed Precision':<20}: {'✅ Enabled' if args.amp else '❌ Disabled'}")
    print(f"{'Num Parts':<20}: {args.num_parts}")
    print(f"{'Feature Dim':<20}: {args.feature_dim}")
    print(f"{'LR Schedule':<20}: {args.lr_scheduler}")
    print(f"{'Total Epochs':<20}: {args.total_epoch}")
    if args.resume:
        print(f"{'Resume':<20}: {args.resume}")
    print("="*70 + "\n")
    
    # ==================== 2. 创建模型 ====================
    print("🔧 创建PF-MGCD Student模型...")
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
    print(f"✅ Student模型参数量: {total_params:.2f}M\n")
    
    # ==================== 3. 训练模式 ====================
    if args.mode == 'train':
        # 3.1 加载训练数据
        from datasets.dataloader_adapter import get_dataloader
        train_loader, _ = get_dataloader(args)
        print(f"📊 训练数据: {len(train_loader)} batches\n")
        
        # 3.2 创建Teacher模型 (Mean Teacher架构)
        print("🔧 创建Mean Teacher模型...")
        teacher_model = PF_MGCD(
            num_parts=args.num_parts,
            num_identities=args.num_classes,
            feature_dim=args.feature_dim,
            memory_momentum=args.memory_momentum,
            temperature=args.temperature,
            top_k=args.top_k,
            pretrained=False,  # Teacher从Student复制权重
            backbone=args.backbone
        ).to(device)
        
        # 初始化Teacher权重为Student的副本
        teacher_model.load_state_dict(model.state_dict())
        
        # 冻结Teacher参数（不参与梯度下降，仅通过EMA更新）
        for param in teacher_model.parameters():
            param.requires_grad = False
        
        print("✅ Teacher模型初始化完成 (权重已从Student复制)\n")
        
        # 3.3 创建优化器
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
        
        # ==================== [修复] 断点恢复逻辑 ====================
        start_epoch = 0
        scheduler_state_loaded = False
        
        if args.resume and os.path.isfile(args.resume):
            print(f"📂 加载checkpoint: {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device)
            
            # 恢复epoch
            start_epoch = checkpoint.get('epoch', 0)
            print(f"   └─ Epoch: {start_epoch}")
            
            # 恢复Student模型
            if 'model' in checkpoint:
                model.load_state_dict(checkpoint['model'])
                print(f"   └─ ✅ Student模型权重已恢复")
            else:
                raise KeyError("❌ Checkpoint中缺少'model'键!")
            
            # [修复] 恢复Teacher模型
            if 'teacher' in checkpoint and checkpoint['teacher'] is not None:
                teacher_model.load_state_dict(checkpoint['teacher'])
                print(f"   └─ ✅ Teacher模型权重已恢复")
            else:
                # 降级方案：从Student复制
                teacher_model.load_state_dict(model.state_dict())
                print(f"   └─ ⚠️  Checkpoint中无Teacher权重，已从Student复制")
            
            # 恢复优化器
            if 'optim' in checkpoint:
                optimizer.load_state_dict(checkpoint['optim'])
                print(f"   └─ ✅ 优化器状态已恢复")
            
            # 标记scheduler状态
            if 'scheduler' in checkpoint and checkpoint['scheduler'] is not None:
                scheduler_state_loaded = True
                print(f"   └─ ✅ Scheduler状态已标记为待恢复")
            
            # [修复] 检查记忆库状态
            if hasattr(model, 'memory_bank'):
                num_initialized = model.memory_bank.initialized.sum().item()
                total_ids = model.memory_bank.num_identities
                print(f"   └─ 📊 记忆库状态: {num_initialized}/{total_ids} IDs已初始化")
                
                if num_initialized == 0:
                    print(f"   └─ ⚠️  警告: 记忆库未初始化，将在训练开始前重新初始化")
            
            print(f"✅ 断点恢复完成! 将从Epoch {start_epoch+1}继续训练\n")
        
        elif args.resume:
            print(f"❌ 未找到checkpoint: {args.resume}\n")
        
        # 3.4 创建学习率调度器
        scheduler = None
        if args.lr_scheduler == 'step':
            if ',' in args.lr_step:
                # MultiStepLR
                milestones = [int(x) for x in args.lr_step.split(',')]
                print(f"📉 使用MultiStepLR，里程碑: {milestones}")
                scheduler = torch.optim.lr_scheduler.MultiStepLR(
                    optimizer,
                    milestones=milestones,
                    gamma=args.lr_gamma,
                    last_epoch=-1  # 重置last_epoch
                )
            else:
                # StepLR
                step_size = int(args.lr_step)
                print(f"📉 使用StepLR，步长: {step_size}")
                scheduler = torch.optim.lr_scheduler.StepLR(
                    optimizer,
                    step_size=step_size,
                    gamma=args.lr_gamma,
                    last_epoch=-1
                )
        elif args.lr_scheduler == 'cosine':
            print(f"📉 使用CosineAnnealingLR，T_max={args.total_epoch}")
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=args.total_epoch,
                last_epoch=-1
            )
        
        # [修复] 恢复scheduler状态
        if scheduler is not None and scheduler_state_loaded and 'scheduler' in checkpoint:
            try:
                scheduler.load_state_dict(checkpoint['scheduler'])
                print(f"✅ Scheduler状态已恢复\n")
            except Exception as e:
                print(f"⚠️  Scheduler恢复失败: {e}，使用默认状态\n")
        
        # 3.5 加载数据集对象（用于验证和记忆库初始化）
        if args.dataset == 'sysu':
            from datasets.sysu import SYSU
            dataset_obj = SYSU(args)
        elif args.dataset == 'regdb':
            from datasets.regdb import RegDB
            dataset_obj = RegDB(args)
        elif args.dataset == 'llcm':
            from datasets.llcm import LLCM
            dataset_obj = LLCM(args)
        
        # 3.6 进入训练循环
        from task.train import train
        train(
            model=model,
            train_loader=train_loader,
            dataset_obj=dataset_obj,
            optimizer=optimizer,
            scheduler=scheduler,
            args=args,
            device=device,
            teacher_model=teacher_model,
            start_epoch=start_epoch  # [修复] 传递start_epoch
        )
    
    # ==================== 4. 测试模式 ====================
    elif args.mode == 'test':
        if not args.model_path:
            raise ValueError("❌ 测试模式需要指定 --model-path 参数!")
        
        print(f"📂 加载测试模型: {args.model_path}")
        checkpoint = torch.load(args.model_path, map_location=device)
        
        # [修复] 兼容多种checkpoint key格式
        if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
            print("✅ 模型加载成功 (key='model')")
        elif 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print("✅ 模型加载成功 (key='model_state_dict')")
        else:
            raise KeyError(
                "❌ Checkpoint中未找到模型权重!\n"
                "   尝试的key: 'model', 'model_state_dict'"
            )
        
        # 打印模型信息（如果有）
        if 'epoch' in checkpoint:
            print(f"   └─ Epoch: {checkpoint['epoch']}")
        if 'rank1' in checkpoint:
            print(f"   └─ Rank-1: {checkpoint['rank1']:.2f}%")
        if 'mAP' in checkpoint:
            print(f"   └─ mAP: {checkpoint['mAP']:.2f}%")
        print()
        
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
        
        # 运行测试
        from task.test import test
        test(
            model=model,
            query_loader=dataset_obj.query_loader,
            gallery_loaders=dataset_obj.gallery_loaders,
            args=args,
            device=device
        )


if __name__ == '__main__':
    main()
