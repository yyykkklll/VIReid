"""
Utility Functions for PF-MGCD
工具函数集合
"""

import numpy as np
import os
import random
import torch


def set_seed(seed=42):
    """
    设置随机种子以确保可复现性
    Args:
        seed: 随机种子
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_logger(log_dir, log_name='train.log'):
    """
    创建日志记录器
    Args:
        log_dir: 日志目录
        log_name: 日志文件名
    Returns:
        logger: 日志记录器
    """
    import logging
    
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_name)
    
    # 创建logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # 创建文件handler
    fh = logging.FileHandler(log_path, mode='a')
    fh.setLevel(logging.INFO)
    
    # 创建控制台handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # 定义格式
    formatter = logging.Formatter(
        '[%(asctime)s] - %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    # 添加handler
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger


def save_checkpoint(state, save_dir, filename='checkpoint.pth'):
    """
    保存检查点
    Args:
        state: 状态字典
        save_dir: 保存目录
        filename: 文件名
    """
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, filename)
    torch.save(state, filepath)
    print(f"Checkpoint saved to {filepath}")


def load_checkpoint(filepath, model, optimizer=None, scheduler=None):
    """
    加载检查点
    Args:
        filepath: 检查点路径
        model: 模型
        optimizer: 优化器（可选）
        scheduler: 学习率调度器（可选）
    Returns:
        start_epoch: 起始epoch
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint not found: {filepath}")
    
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    start_epoch = checkpoint.get('epoch', 0) + 1
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    print(f"Checkpoint loaded from {filepath}")
    print(f"Resuming from epoch {start_epoch}")
    
    return start_epoch


def count_parameters(model):
    """
    计算模型参数量
    Args:
        model: 模型
    Returns:
        total: 总参数量
        trainable: 可训练参数量
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def get_lr(optimizer):
    """
    获取当前学习率
    Args:
        optimizer: 优化器
    Returns:
        lr: 学习率
    """
    for param_group in optimizer.param_groups:
        return param_group['lr']


class AverageMeter:
    """
    计算并存储平均值和当前值
    """
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class WarmupScheduler:
    """
    Warmup学习率调度器
    """
    def __init__(self, optimizer, warmup_epochs, base_lr, after_scheduler=None):
        """
        Args:
            optimizer: 优化器
            warmup_epochs: warmup的epoch数
            base_lr: 基础学习率
            after_scheduler: warmup后使用的调度器
        """
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.base_lr = base_lr
        self.after_scheduler = after_scheduler
        self.current_epoch = 0
    
    def step(self):
        """
        更新学习率
        """
        if self.current_epoch < self.warmup_epochs:
            # Warmup阶段: 线性增长
            lr = self.base_lr * (self.current_epoch + 1) / self.warmup_epochs
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        else:
            # Warmup后使用after_scheduler
            if self.after_scheduler is not None:
                self.after_scheduler.step()
        
        self.current_epoch += 1


def visualize_parts(images, num_parts=6):
    """
    可视化部件切分
    Args:
        images: 图像张量 [B, 3, H, W]
        num_parts: 部件数量
    Returns:
        visualization: 可视化结果
    """
    import matplotlib.pyplot as plt
    
    B, C, H, W = images.shape
    part_height = H // num_parts
    
    fig, axes = plt.subplots(1, num_parts + 1, figsize=(15, 3))
    
    # 显示原图
    img = images[0].permute(1, 2, 0).cpu().numpy()
    img = (img - img.min()) / (img.max() - img.min())
    axes[0].imshow(img)
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    # 显示各个部件
    for i in range(num_parts):
        start = i * part_height
        end = (i + 1) * part_height
        part = images[0, :, start:end, :].permute(1, 2, 0).cpu().numpy()
        part = (part - part.min()) / (part.max() - part.min())
        axes[i + 1].imshow(part)
        axes[i + 1].set_title(f'Part {i+1}')
        axes[i + 1].axis('off')
    
    plt.tight_layout()
    return fig


def print_config(args):
    """
    打印配置信息
    Args:
        args: 参数配置
    """
    print("="*70)
    print(" "*25 + "Configuration")
    print("="*70)
    
    # 按类别组织参数
    categories = {
        'Basic': ['dataset', 'mode', 'gpu', 'seed'],
        'Model': ['num_parts', 'feature_dim', 'memory_momentum', 'temperature', 'top_k'],
        'Loss': ['lambda_graph', 'lambda_orth', 'lambda_mod', 'label_smoothing'],
        'Training': ['total_epoch', 'warmup_epochs', 'batch_size', 'lr', 'weight_decay'],
        'Paths': ['data_path', 'save_dir', 'log_dir']
    }
    
    for category, keys in categories.items():
        print(f"\n{category}:")
        for key in keys:
            if hasattr(args, key):
                value = getattr(args, key)
                print(f"  {key:20s}: {value}")
    
    print("="*70 + "\n")