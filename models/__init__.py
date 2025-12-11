"""
Model Management Module
=======================
Components: Backbone, Classifiers, CLIP, Optimizers, Schedulers
Author: Fixed Version
Date: 2025-12-11
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


_models = {
    "resnet": AGW,
    "clip": CLIP,
}


def create(args):
    """Model factory function"""
    if args.arch not in _models:
        raise KeyError(f"Unknown backbone: {args.arch}. Available: {list(_models.keys())}")
    return Model(args)


class Model:
    """Cross-modal Person Re-ID Model Manager"""
    
    def __init__(self, args):
        self.args = args
        self.mode = args.mode
        self.device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else "cpu")
        self.save_path = os.path.join(args.save_path, "models/")
        
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.milestones = args.milestones
        self.resume_epoch = 0
        
        print(f"Building backbone: {args.arch}")
        self.model = _models[args.arch](args).to(self.device)
        
        print(f"Building classifiers (num_classes={args.num_classes})")
        self.classifier1 = Image_Classifier(args).to(self.device)
        self.classifier2 = Image_Classifier(args).to(self.device)
        self.classifier3 = Image_Classifier(args).to(self.device)
        self.enable_cls3 = False
        
        self.clip_model = None
        if hasattr(args, 'use_clip') and args.use_clip:
            print("Loading CLIP as Semantic Referee...")
            self.clip_model = self._build_clip_model(args)
        
        self._init_optimizer()
        self._init_criterion()
        print(f"Model initialized on {self.device}")
    
    def _build_clip_model(self, args):
        """Build CLIP model with correct feature map resolution"""
        h_resolution = (args.img_h - 32) // 32 + 1
        w_resolution = (args.img_w - 32) // 32 + 1
        
        print(f"  CLIP feature map: {h_resolution}x{w_resolution}")
        print(f"  Input image: {args.img_h}x{args.img_w}")
        
        clip_model = load_clip_to_cpu(
            backbone_name='RN50',
            h_resolution=h_resolution,
            w_resolution=w_resolution,
            vision_stride_size=32
        )
        
        clip_model.to(self.device)
        clip_model.eval()
        
        for param in clip_model.parameters():
            param.requires_grad = False
        
        return clip_model
    
    def _init_optimizer(self):
        """Initialize optimizers for two-stage training"""
        params_phase1 = []
        for module in [self.model, self.classifier1, self.classifier2]:
            for name, param in module.named_parameters():
                if not param.requires_grad:
                    continue
                lr = 2.0 * self.lr if 'classifier' in name else self.lr
                params_phase1.append({
                    'params': [param],
                    'lr': lr,
                    'weight_decay': self.weight_decay
                })
        
        params_phase2 = params_phase1.copy()
        for name, param in self.classifier3.named_parameters():
            if param.requires_grad:
                params_phase2.append({
                    'params': [param],
                    'lr': 2.0 * self.lr,
                    'weight_decay': self.weight_decay
                })
        
        self.optimizer_phase1 = torch.optim.Adam(params_phase1)
        self.optimizer_phase2 = torch.optim.Adam(params_phase2)
        
        self.scheduler_phase1 = WarmupMultiStepLR(
            self.optimizer_phase1, milestones=self.milestones,
            gamma=0.1, warmup_factor=0.1, warmup_iters=5, mode='cls'
        )
        self.scheduler_phase2 = WarmupMultiStepLR(
            self.optimizer_phase2, milestones=self.milestones,
            gamma=0.1, warmup_factor=0.1, warmup_iters=5, mode='cls'
        )
    
    def _init_criterion(self):
        """Initialize loss functions"""
        self.pid_criterion = nn.CrossEntropyLoss()
        self.tri_criterion = TripletLoss_WRT()
        self.weak_criterion = Weak_loss()
    
    def set_train(self):
        """Set model to training mode"""
        self.model.train()
        self.classifier1.train()
        self.classifier2.train()
        if self.enable_cls3:
            self.classifier3.train()
    
    def set_eval(self):
        """Set model to evaluation mode"""
        self.model.eval()
        self.classifier1.eval()
        self.classifier2.eval()
        self.classifier3.eval()
    
    def save_model(self, epoch, is_best=False):
        """Save model checkpoint with error handling"""
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        
        state_dict = {
            'epoch': epoch,
            'backbone': self.model.state_dict(),
            'classifier1': self.classifier1.state_dict(),
            'classifier2': self.classifier2.state_dict(),
            'classifier3': self.classifier3.state_dict(),
            'optimizer_phase1': self.optimizer_phase1.state_dict(),
            'optimizer_phase2': self.optimizer_phase2.state_dict(),
        }
        
        if is_best:
            model_path = os.path.join(self.save_path, 'model_best.pth')
        else:
            model_path = os.path.join(self.save_path, f'model_{epoch}.pth')
        
        try:
            temp_path = model_path + '.tmp'
            torch.save(state_dict, temp_path)
            
            if os.path.exists(model_path):
                os.remove(model_path)
            os.rename(temp_path, model_path)
            
            print(f"Model saved to {model_path}")
            
            if not is_best:
                self._cleanup_old_models(keep_best=True, keep_recent=3)
                
        except Exception as e:
            print(f"Failed to save model: {e}")
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except:
                    pass
    
    def resume_model(self, model_path=None):
        """Load model checkpoint"""
        if model_path is None:
            model_path = self._find_latest_model()
        
        if model_path is None or not os.path.exists(model_path):
            print("No checkpoint found, starting from scratch")
            self.resume_epoch = 0
            return
        
        print(f"Loading checkpoint from {model_path}")
        
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            
            self.model.load_state_dict(checkpoint['backbone'], strict=False)
            self.classifier1.load_state_dict(checkpoint['classifier1'], strict=False)
            self.classifier2.load_state_dict(checkpoint['classifier2'], strict=False)
            
            if 'classifier3' in checkpoint:
                self.classifier3.load_state_dict(checkpoint['classifier3'], strict=False)
            
            self.resume_epoch = checkpoint.get('epoch', 0)
            
            if self.mode == 'train' and 'optimizer_phase2' in checkpoint:
                try:
                    self.optimizer_phase2.load_state_dict(checkpoint['optimizer_phase2'])
                except:
                    print("Failed to load optimizer state, using fresh optimizer")
            
            print(f"Model loaded successfully from epoch {self.resume_epoch}")
            
        except Exception as e:
            print(f"Failed to load checkpoint: {e}")
            self.resume_epoch = 0
    
    def _find_latest_model(self):
        """Find the latest checkpoint"""
        if not os.path.exists(self.save_path):
            return None
        
        root, _, files = os_walk(self.save_path)
        pth_files = [f for f in files if f.endswith('.pth') and not f.endswith('.tmp')]
        
        if not pth_files:
            return None
        
        if 'model_best.pth' in pth_files:
            return os.path.join(root, 'model_best.pth')
        
        epochs = []
        for f in pth_files:
            try:
                epoch = int(f.replace('.pth', '').split('_')[-1])
                epochs.append((epoch, f))
            except:
                continue
        
        if epochs:
            latest_file = max(epochs, key=lambda x: x[0])[1]
            return os.path.join(root, latest_file)
        
        return None
    
    def _cleanup_old_models(self, keep_best=True, keep_recent=3):
        """Clean up old model checkpoints"""
        if not os.path.exists(self.save_path):
            return
        
        root, _, files = os_walk(self.save_path)
        pth_files = [f for f in files if f.endswith('.pth') and not f.endswith('.tmp')]
        
        epoch_files = []
        for f in pth_files:
            if f == 'model_best.pth' and keep_best:
                continue
            
            try:
                epoch = int(f.replace('.pth', '').split('_')[-1])
                epoch_files.append((epoch, f))
            except:
                continue
        
        epoch_files.sort(key=lambda x: x[0], reverse=True)
        
        for epoch, filename in epoch_files[keep_recent:]:
            file_path = os.path.join(root, filename)
            try:
                os.remove(file_path)
                print(f"Removed old checkpoint: {filename}")
            except:
                pass
    
    def count_parameters(self):
        """Count model parameters"""
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"Model parameters:")
        print(f"  Total: {total / 1e6:.2f}M")
        print(f"  Trainable: {trainable / 1e6:.2f}M")
        
        return total, trainable
    
    def get_learning_rate(self):
        """Get current learning rate"""
        return self.optimizer_phase2.param_groups[0]['lr']
