"""
模型初始化模块 - 集成特征扩散桥 + 余弦退火调度器
"""
import torch
import os
import re

from .classifier import Image_Classifier
from utils import os_walk
from .agw import AGW
from .clip_model import CLIP
from .optim import WarmupMultiStepLR
from .loss import TripletLoss_WRT, Weak_loss
from .feature_diffusion import FeatureDiffusionBridge

_models = {
    "resnet": AGW,
    "clip-resnet": CLIP,
    "vit": 0,
}


def create(args):
    """创建模型实例"""
    if args.arch not in _models:
        raise KeyError("Unknown backbone:", args.arch)
    print('Loading {} architecture for {} dataset...'.format(args.arch, args.dataset))
    return Model(args)


class Model:
    def __init__(self, args):
        self.mode = args.mode
        self.device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        self.save_path = os.path.join(args.save_path, "models/")
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.milestones = args.milestones
        self.resume = args.resume
        self.args = args

        # 主干网络
        self.model = _models[args.arch](args).to(self.device)
        
        # 三个分类器
        self.classifier1 = Image_Classifier(args).to(self.device)  # RGB classifier
        self.classifier2 = Image_Classifier(args).to(self.device)  # IR classifier
        self.classifier3 = Image_Classifier(args).to(self.device)  # Common classifier
        self.enable_cls3 = False

        # ========== 特征扩散桥 ==========
        self.use_diffusion = getattr(args, 'use_diffusion', False)
        if self.use_diffusion:
            self.diffusion_bridge = FeatureDiffusionBridge(
                feat_dim=2048,
                hidden_dim=getattr(args, 'diffusion_hidden', 1024),
                num_steps=getattr(args, 'diffusion_steps', 10),
                beta_schedule='cosine',
                device=self.device
            ).to(self.device)
            print(f"✓ Feature Diffusion Bridge initialized (T={args.diffusion_steps}, hidden={args.diffusion_hidden})")
        # ===============================

        self._init_optimizer()
        self._init_criterion()

    def _init_optimizer(self):
        """初始化优化器和学习率调度器（使用余弦退火）"""
        params_phase1 = []
        params_phase2 = []
        
        # ========== Phase 1 参数 ==========
        for part in (self.model, self.classifier1, self.classifier2):
            for key, value in part.named_parameters():
                if not value.requires_grad:
                    continue
                lr = 2 * self.lr if "classifier" in key else self.lr
                params_phase1.append({
                    "params": [value], 
                    "lr": lr, 
                    "weight_decay": self.weight_decay
                })
        
        params_phase2.extend(params_phase1)
        
        # ========== Phase 2 额外参数 ==========
        # Classifier3
        for key, value in self.classifier3.named_parameters():
            if value.requires_grad:
                params_phase2.append({
                    "params": [value], 
                    "lr": 2 * self.lr, 
                    "weight_decay": self.weight_decay
                })
        
        # 扩散模块参数
        if self.use_diffusion:
            diffusion_lr = getattr(self.args, 'diffusion_lr', self.lr)
            for key, value in self.diffusion_bridge.named_parameters():
                if value.requires_grad:
                    params_phase2.append({
                        "params": [value], 
                        "lr": diffusion_lr, 
                        "weight_decay": self.weight_decay
                    })
            print(f"✓ Diffusion parameters added to optimizer (lr={diffusion_lr})")
        
        # ========== 优化器 ==========
        self.optimizer_phase1 = torch.optim.Adam(params_phase1)
        self.optimizer_phase2 = torch.optim.Adam(params_phase2)
        
        # ========== 学习率调度器（余弦退火） ==========
        use_cosine_annealing = getattr(self.args, 'use_cosine_annealing', True)
        
        if use_cosine_annealing:
            # Phase 1: 余弦退火调度器
            self.scheduler_phase1 = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer_phase1,
                T_max=self.args.stage1_epoch,
                eta_min=1e-7  # 最小学习率
            )
            
            # Phase 2: 余弦退火调度器
            self.scheduler_phase2 = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer_phase2,
                T_max=self.args.stage2_epoch - self.args.stage1_epoch,
                eta_min=1e-7
            )
            
            print("✓ Using CosineAnnealingLR scheduler")
        else:
            # 原始的 WarmupMultiStepLR（保留兼容性）
            self.scheduler_phase1 = WarmupMultiStepLR(
                self.optimizer_phase1, 
                self.milestones,
                gamma=0.1, 
                warmup_factor=0.01, 
                warmup_iters=10, 
                mode='cls'
            )
            self.scheduler_phase2 = WarmupMultiStepLR(
                self.optimizer_phase2, 
                self.milestones,
                gamma=0.1, 
                warmup_factor=0.01, 
                warmup_iters=10, 
                mode='cls'
            )
            print("✓ Using WarmupMultiStepLR scheduler")
    
    def _init_criterion(self):
        """初始化损失函数"""
        self.pid_criterion = torch.nn.CrossEntropyLoss()
        self.tri_criterion = TripletLoss_WRT()
        self.weak_criterion = Weak_loss()

    def set_train(self):
        """设置训练模式"""
        self.model.train()
        self.classifier1.train()
        self.classifier2.train()
        self.classifier3.train()
        if self.use_diffusion:
            self.diffusion_bridge.train()

    def set_eval(self):
        """设置评估模式"""
        self.model.eval()
        self.classifier1.eval()
        self.classifier2.eval()
        self.classifier3.eval()
        if self.use_diffusion:
            self.diffusion_bridge.eval()

    def save_model(self, save_epoch, is_best):
        """保存模型（包含扩散模块）"""
        if is_best:
            model_file_path = os.path.join(self.save_path, f'model_{save_epoch}.pth')
            
            # 清理旧模型
            root, _, files = os_walk(self.save_path)
            for file in files:
                if '.pth' in file:
                    try:
                        file_iters = int(file.replace('.pth', '').split('_')[1])
                        if file_iters < save_epoch:
                            old_path = os.path.join(root, file)
                            if os.path.exists(old_path):
                                os.remove(old_path)
                    except:
                        continue

            # 保存状态字典
            all_state_dict = {
                'backbone': self.model.state_dict(),
                'classifier1': self.classifier1.state_dict(),
                'classifier2': self.classifier2.state_dict(),
                'classifier3': self.classifier3.state_dict(),
                'epoch': save_epoch
            }
            
            # 保存扩散模块
            if self.use_diffusion:
                all_state_dict['diffusion_bridge'] = self.diffusion_bridge.state_dict()
            
            torch.save(all_state_dict, model_file_path)
            print(f'✓ Model saved: {model_file_path}')
        
    def resume_model(self, specified_model=None):
        """恢复模型（包含扩散模块）"""
        if specified_model is None:
            root, _, files = os_walk(self.save_path)
            self.resume_epoch = 0
            
            if len(files) > 0:
                indexes = []
                for f in files:
                    if '.pth' in f:
                        try:
                            idx = int(f.replace('.pth', '').split('_')[-1])
                            indexes.append(idx)
                        except:
                            continue
                
                if indexes:
                    indexes = sorted(list(set(indexes)))
                    model_path = os.path.join(self.save_path, f'model_{indexes[-1]}.pth')
                    
                    if self.resume or self.mode == 'test':
                        self._load_checkpoint(model_path)
                        self.resume_epoch = indexes[-1]
                        print(f'✓ Loaded checkpoint: {model_path}')
                        print(f'  Resuming from epoch {self.resume_epoch}')
                    else:
                        os.remove(model_path)
                        print('✗ Existing model removed (training from scratch)')
                else:
                    print('✗ No valid checkpoint found')
            else:
                print('✗ No model files in save directory')
            
            print(f'Starting from epoch {self.resume_epoch}')
        else:
            self._load_checkpoint(specified_model)
            self.resume_epoch = 0
            print(f'✓ Loaded specified model: {specified_model}')
            print(f'Starting from epoch {self.resume_epoch}')
    
    def _load_checkpoint(self, model_path):
        """加载检查点（内部方法）"""
        loaded_dict = torch.load(model_path, map_location=self.device)
        
        self.model.load_state_dict(loaded_dict['backbone'], strict=False)
        self.classifier1.load_state_dict(loaded_dict['classifier1'], strict=False)
        self.classifier2.load_state_dict(loaded_dict['classifier2'], strict=False)
        
        if 'classifier3' in loaded_dict:
            self.classifier3.load_state_dict(loaded_dict['classifier3'], strict=False)
        
        # 加载扩散模块
        if self.use_diffusion and 'diffusion_bridge' in loaded_dict:
            self.diffusion_bridge.load_state_dict(loaded_dict['diffusion_bridge'], strict=False)
            print("✓ Diffusion bridge loaded from checkpoint")
        elif self.use_diffusion:
            print("⚠️ Warning: Diffusion bridge not found in checkpoint, using random initialization")
