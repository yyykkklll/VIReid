"""
模型管理模块
============
功能：
1. 统一管理 Backbone、Classifier、CLIP 等模型组件
2. 优化器和学习率调度器配置
3. 模型保存和加载
4. 训练/评估模式切换

作者:  修复优化版本
日期: 2025-01-20
"""

import torch
import torch.nn as nn
import os

from .classifier import Image_Classifier
from .agw import AGW
from .clip_model import CLIP, load_clip_to_cpu
from .optim import WarmupMultiStepLR
from .loss import TripletLoss_WRT, Weak_loss
from utils import os_walk


# ==================== 模型注册表 ====================

_models = {
    "resnet":  AGW,      # AGW (ResNet50 backbone)
    "clip": CLIP,       # CLIP-based model (不推荐用于主干网络)
}


def create(args):
    """
    模型工厂函数
    
    Args:
        args: 配置参数
        
    Returns:
        Model 实例
    """
    if args.arch not in _models:
        raise KeyError(f"Unknown backbone: {args.arch}. Available: {list(_models.keys())}")
    
    print(f"🏗️  Creating model with backbone: {args.arch}")
    return Model(args)


# ==================== 主模型类 ====================

class Model: 
    """
    跨模态行人重识别模型管理器
    
    组件：
    - Backbone:  特征提取网络 (AGW/ResNet50)
    - Classifier1: RGB 分类器
    - Classifier2: IR 分类器
    - Classifier3: 跨模态分类器
    - CLIP (可选): 语义增强模块
    """
    
    def __init__(self, args):
        self.args = args
        self. mode = args.mode
        self. device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        self.save_path = os.path.join(args.save_path, "models/")
        
        # 训练超参数
        self.lr = args.lr
        self. weight_decay = args.weight_decay
        self.milestones = args.milestones
        self.resume = args.resume
        
        # ==================== 构建模型组件 ====================
        
        print(f"📦 Building backbone: {args.arch}")
        self.model = _models[args.arch](args).to(self.device)
        
        print(f"📦 Building classifiers (num_classes={args.num_classes})")
        self.classifier1 = Image_Classifier(args).to(self.device)  # RGB 分类器
        self.classifier2 = Image_Classifier(args).to(self.device)  # IR 分类器
        self. classifier3 = Image_Classifier(args).to(self.device)  # 跨模态分类器
        self.enable_cls3 = False  # Phase1 不使用 classifier3
        
        # ==================== CLIP 语义模块（可选）====================
        
        self.clip_model = None
        if hasattr(args, 'use_clip') and args.use_clip:
            print("🎨 Loading CLIP as Semantic Referee...")
            self.clip_model = self._build_clip_model(args)
            print(f"✅ CLIP loaded successfully")
        
        # ==================== 优化器和损失函数 ====================
        
        self._init_optimizer()
        self._init_criterion()
        
        print(f"✅ Model initialized on {self.device}")
    
    
    def _build_clip_model(self, args):
        """
        构建 CLIP 模型（修复版）
        
        关键修复：
        1. 正确计算特征图分辨率
        2. 适配不同的输入图像尺寸
        """
        # 计算 CLIP 的特征图分辨率
        # ResNet50: stride=32, 所以输出分辨率 = (H-32)/32 + 1
        # 例如:  288x144 -> (288-32)/32+1 = 9, (144-32)/32+1 = 4
        h_resolution = (args.img_h - 32) // 32 + 1
        w_resolution = (args.img_w - 32) // 32 + 1
        
        print(f"   CLIP feature map resolution: {h_resolution}x{w_resolution}")
        print(f"   Input image resolution: {args.img_h}x{args.img_w}")
        
        clip_model = load_clip_to_cpu(
            backbone_name='RN50',
            h_resolution=h_resolution,
            w_resolution=w_resolution,
            vision_stride_size=32  # ResNet50 的总 stride
        )
        
        clip_model.to(self.device)
        clip_model.eval()
        
        # 冻结所有参数（CLIP 仅用于特征提取）
        for param in clip_model.parameters():
            param.requires_grad = False
        
        return clip_model
    
    
    def _init_optimizer(self):
        """
        初始化优化器（双阶段）
        
        Phase1: Backbone + Classifier1 + Classifier2
        Phase2: Phase1 + Classifier3
        """
        # Phase1 参数组
        params_phase1 = []
        for module in [self.model, self.classifier1, self.classifier2]:
            for name, param in module.named_parameters():
                if not param.requires_grad:
                    continue
                
                # 分类器层使用 2x 学习率
                if 'classifier' in name:
                    params_phase1.append({
                        'params': [param],
                        'lr': 2.0 * self.lr,
                        'weight_decay': self.weight_decay
                    })
                else:
                    params_phase1.append({
                        'params': [param],
                        'lr': self.lr,
                        'weight_decay': self.weight_decay
                    })
        
        # Phase2 参数组（包含 Phase1 + Classifier3）
        params_phase2 = params_phase1.copy()
        for name, param in self.classifier3.named_parameters():
            if param.requires_grad:
                params_phase2.append({
                    'params': [param],
                    'lr': 2.0 * self.lr,
                    'weight_decay': self.weight_decay
                })
        
        # 创建优化器
        self. optimizer_phase1 = torch.optim.Adam(params_phase1)
        self.optimizer_phase2 = torch.optim.Adam(params_phase2)
        
        # 学习率调度器
        self.scheduler_phase1 = WarmupMultiStepLR(
            self.optimizer_phase1,
            milestones=self. milestones,
            gamma=0.1,
            warmup_factor=0.01,
            warmup_iters=10,
            mode='cls'
        )
        self.scheduler_phase2 = WarmupMultiStepLR(
            self.optimizer_phase2,
            milestones=self.milestones,
            gamma=0.1,
            warmup_factor=0.01,
            warmup_iters=10,
            mode='cls'
        )
    
    
    def _init_criterion(self):
        """
        初始化损失函数
        """
        self.pid_criterion = nn.CrossEntropyLoss()
        self.tri_criterion = TripletLoss_WRT()
        self.weak_criterion = Weak_loss()
    
    
    # ==================== 模式切换 ====================
    
    def set_train(self):
        """设置为训练模式"""
        self.model.train()
        self.classifier1.train()
        self.classifier2.train()
        if self.enable_cls3:
            self.classifier3.train()
    
    
    def set_eval(self):
        """设置为评估模式"""
        self.model.eval()
        self.classifier1.eval()
        self.classifier2.eval()
        self.classifier3.eval()
    
    
    # ==================== 模型保存与加载 ====================
    
    def save_model(self, epoch, is_best=False):
        """
        保存模型检查点
        
        Args: 
            epoch: 当前轮次
            is_best: 是否为最佳模型
        """
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        
        # 构建状态字典
        state_dict = {
            'epoch':  epoch,
            'backbone':  self.model.state_dict(),
            'classifier1': self. classifier1.state_dict(),
            'classifier2': self.classifier2.state_dict(),
            'classifier3': self.classifier3.state_dict(),
            'optimizer_phase1': self.optimizer_phase1.state_dict(),
            'optimizer_phase2': self. optimizer_phase2.state_dict(),
        }
        
        if is_best:
            # 保存为最佳模型
            model_path = os.path.join(self.save_path, 'model_best.pth')
            torch.save(state_dict, model_path)
            print(f"💾 Best model saved to {model_path}")
            
            # 删除旧的最佳模型（可选）
            self._cleanup_old_models(keep_best=True)
        else:
            # 定期保存
            model_path = os.path.join(self.save_path, f'model_{epoch}.pth')
            torch.save(state_dict, model_path)
            print(f"💾 Checkpoint saved to {model_path}")
    
    
    def resume_model(self, model_path=None):
        """
        加载模型检查点
        
        Args:
            model_path: 指定模型路径，None 则自动加载最新模型
        """
        if model_path is None:
            # 自动查找最新模型
            model_path = self._find_latest_model()
        
        if model_path is None or not os.path.exists(model_path):
            print("⚠️  No checkpoint found, starting from scratch")
            return
        
        print(f"📂 Loading checkpoint from {model_path}")
        
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # 加载模型权重
            self.model.load_state_dict(checkpoint['backbone'], strict=False)
            self.classifier1.load_state_dict(checkpoint['classifier1'], strict=False)
            self.classifier2.load_state_dict(checkpoint['classifier2'], strict=False)
            
            # 尝试加载 classifier3（可能不存在）
            if 'classifier3' in checkpoint:
                self.classifier3.load_state_dict(checkpoint['classifier3'], strict=False)
            
            # 加载优化器状态（可选）
            if self.mode == 'train' and 'optimizer_phase2' in checkpoint:
                try:
                    self.optimizer_phase2.load_state_dict(checkpoint['optimizer_phase2'])
                except:
                    print("⚠️  Failed to load optimizer state, using fresh optimizer")
            
            print(f"✅ Model loaded successfully from epoch {checkpoint. get('epoch', 'unknown')}")
            
        except Exception as e:
            print(f"❌ Failed to load checkpoint: {e}")
            print("   Starting from scratch...")
    
    
    def _find_latest_model(self):
        """
        查找最新的模型检查点
        
        Returns:
            模型路径或 None
        """
        if not os.path.exists(self.save_path):
            return None
        
        root, _, files = os_walk(self.save_path)
        
        # 过滤 .pth 文件
        pth_files = [f for f in files if f.endswith('. pth')]
        
        if not pth_files:
            return None
        
        # 优先加载 best 模型
        if 'model_best.pth' in pth_files:
            return os.path.join(root, 'model_best.pth')
        
        # 否则加载最新的 epoch 模型
        epochs = []
        for f in pth_files:
            try:
                epoch = int(f.replace('.pth', '').split('_')[-1])
                epochs.append((epoch, f))
            except: 
                continue
        
        if epochs:
            latest_file = max(epochs, key=lambda x: x[0])[1]
            return os. path.join(root, latest_file)
        
        return None
    
    
    def _cleanup_old_models(self, keep_best=True, keep_recent=3):
        """
        清理旧的模型检查点
        
        Args: 
            keep_best: 是否保留 best 模型
            keep_recent: 保留最近 N 个模型
        """
        if not os.path.exists(self.save_path):
            return
        
        root, _, files = os_walk(self.save_path)
        pth_files = [f for f in files if f.endswith('. pth')]
        
        # 提取 epoch 信息
        epoch_files = []
        for f in pth_files:
            if f == 'model_best.pth' and keep_best:
                continue  # 保留 best 模型
            
            try:
                epoch = int(f.replace('.pth', '').split('_')[-1])
                epoch_files.append((epoch, f))
            except:
                continue
        
        # 按 epoch 排序
        epoch_files.sort(key=lambda x: x[0], reverse=True)
        
        # 删除旧模型
        for epoch, filename in epoch_files[keep_recent:]:
            file_path = os.path.join(root, filename)
            try:
                os.remove(file_path)
                print(f"🗑️  Removed old checkpoint: {filename}")
            except:
                pass
    
    
    # ==================== 工具方法 ====================
    
    def count_parameters(self):
        """
        统计模型参数量
        
        Returns:
            total:  总参数量
            trainable: 可训练参数量
        """
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p. numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"📊 Model parameters:")
        print(f"   Total: {total / 1e6:.2f}M")
        print(f"   Trainable: {trainable / 1e6:.2f}M")
        
        return total, trainable
    
    
    def get_learning_rate(self):
        """
        获取当前学习率
        """
        return self.optimizer_phase2.param_groups[0]['lr']