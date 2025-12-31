import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from .agw import AGW
from .classifier import Image_Classifier


def embed_net(class_num=None, arch='resnet50', drop=0.0, part=0):
    """Factory function to create AGW backbone"""
    class Args: pass
    args = Args()
    return AGW(args)


def create(args):
    """Factory function to create model"""
    return Model(args)


class Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        
        self.args = args
        self.device = torch.device(f'cuda:{args.device}')
        self.num_classes = args.num_classes
        
        # Training state
        self.resume_epoch = 0
        self.enable_cls3 = False
        
        # Diffusion configuration
        self.use_diffusion = getattr(args, 'use_diffusion', False)
        self.diffusion_weight = getattr(args, 'diffusion_weight', 0.05)
        
        # Build components
        self._build_backbone(args)
        self._build_classifiers(args)
        self._build_diffusion_bridge(args)
        self._build_ragm(args)
        self._build_loss_functions(args)
        
        # Initialize optimizers
        self.configure_optimizer_phase1(args)
        
        self.to(self.device)

    def _build_backbone(self, args):
        self.model = embed_net(class_num=args.num_classes, arch=args.arch)
    
    def _build_classifiers(self, args):
        class ClassifierArgs: num_classes = args.num_classes
        clf_args = ClassifierArgs()
        
        self.classifier1 = Image_Classifier(clf_args)
        self.classifier2 = Image_Classifier(clf_args)
        self.classifier3 = None  # Initialized in Phase 2
    
    def _build_diffusion_bridge(self, args):
        if not self.use_diffusion:
            self.diffusion = None
            self.diffusion_bridge = None
            return
        
        from .feature_diffusion import HierarchicalDiffusionBridge
        self.diffusion = HierarchicalDiffusionBridge(
            feat_dim=2048,
            hidden_dim=getattr(args, 'diffusion_hidden', 1024),
            feature_steps=getattr(args, 'feature_diffusion_steps', 5),
            semantic_steps=getattr(args, 'semantic_diffusion_steps', 10),
            num_heads=getattr(args, 'cross_attn_heads', 4),
            dropout=getattr(args, 'cross_attn_dropout', 0.1),
            device=self.device,
            use_memory=getattr(args, 'use_memory_bank', True),
            memory_slots=getattr(args, 'memory_size_per_class', 5),
            num_classes=args.num_classes
        )
        self.diffusion_bridge = self.diffusion
        print(f"✓ Diffusion Bridge Initialized")
    
    def _build_ragm(self, args):
        from .reliability_gating import ReliabilityGating
        self.ragm = ReliabilityGating(
            temperature=0.1, 
            momentum=0.9
        )

    def _build_loss_functions(self, args):
        from .loss import TripletLoss_WRT, Weak_loss
        self.pid_criterion = nn.CrossEntropyLoss()
        self.tri_criterion = TripletLoss_WRT()
        self.weak_criterion = Weak_loss()
    
    def forward(self, imgs_rgb=None, imgs_ir=None):
        if imgs_rgb is not None:
            _, feats = self.model(x1=imgs_rgb)
        elif imgs_ir is not None:
            _, feats = self.model(x2=imgs_ir)
        else:
            raise ValueError("Input needed")
        return F.normalize(feats, dim=1)
    
    def configure_optimizer_phase1(self, args):
        params = [
            {'params': self.model.parameters(), 'lr': args.lr},
            {'params': self.classifier1.parameters(), 'lr': args.lr * 2},
            {'params': self.classifier2.parameters(), 'lr': args.lr * 2}
        ]
        self.optimizer_phase1 = torch.optim.Adam(params, weight_decay=args.weight_decay)
        self.scheduler_phase1 = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer_phase1, milestones=args.milestones, gamma=0.1
        )
    
    def transition_to_phase2(self, args, logger=None):
        """Switch model to Phase 2 with proper initialization"""
        
        # Clean up Phase 1 optimizer
        if hasattr(self, 'optimizer_phase1'):
            del self.optimizer_phase1
        if hasattr(self, 'scheduler_phase1'):
            del self.scheduler_phase1
        
        # Initialize Classifier3 from Classifier1 weights (Transfer Learning)
        if self.classifier3 is None:
            class ClassifierArgs: num_classes = args.num_classes
            self.classifier3 = Image_Classifier(ClassifierArgs()).to(self.device)
            # Load weights from trained classifier1
            self.classifier3.load_state_dict(self.classifier1.state_dict())
            self.enable_cls3 = True
            if logger: logger('✓ Classifier3 initialized from Classifier1 weights')

        # Configure Phase 2 optimizer with Differential LR
        self.configure_optimizer_phase2(args)
        
        if logger: logger('✓ Phase 2 Optimizer configured with differential learning rates')

    def configure_optimizer_phase2(self, args):
        """
        Phase 2 Optimizer:
        - Backbone & Old Classifiers: 0.1x LR (Prevent forgetting)
        - Diffusion & New Classifier: 1.0x LR (Fast learning)
        """
        
        # 🟢 关键修改：定义差异化学习率
        # Phase 1 结束时 LR 已经衰减了 10 倍，所以这里 Backbone 应该保持低 LR
        reduced_lr = args.lr * 0.1 
        full_lr = args.lr 
        
        param_set = set()
        params = []
        
        # 1. Backbone -> Low LR (0.1x)
        for p in self.model.parameters():
            if id(p) not in param_set:
                param_set.add(id(p))
                params.append({'params': [p], 'lr': reduced_lr})
        
        # 2. Classifier 1 & 2 -> Low LR (Scaled 2x relative to backbone base)
        # Typically classifiers use 2x backbone LR. So here: 0.1 * 2 = 0.2x
        for p in self.classifier1.parameters():
            if id(p) not in param_set:
                param_set.add(id(p))
                params.append({'params': [p], 'lr': reduced_lr * 2})
                
        for p in self.classifier2.parameters():
            if id(p) not in param_set:
                param_set.add(id(p))
                params.append({'params': [p], 'lr': reduced_lr * 2})
        
        # 3. Classifier 3 (New) -> High LR (Full speed)
        if self.classifier3 is not None:
            for p in self.classifier3.parameters():
                if id(p) not in param_set:
                    param_set.add(id(p))
                    params.append({'params': [p], 'lr': full_lr * 2})
        
        # 4. Diffusion (New) -> High LR (Specific config)
        if self.use_diffusion:
            diff_lr = getattr(args, 'diffusion_lr', full_lr)
            for p in self.diffusion.parameters():
                if id(p) not in param_set:
                    param_set.add(id(p))
                    # 🟢 Explicitly set weight_decay=0.0 for Diffusion stability
                    params.append({'params': [p], 'lr': diff_lr, 'weight_decay': 0.0})
        
        # 5. RAGM -> Low LR
        for p in self.ragm.parameters():
            if id(p) not in param_set:
                param_set.add(id(p))
                params.append({'params': [p], 'lr': reduced_lr})
        
        # Optimizer with Global Weight Decay
        self.optimizer_phase2 = torch.optim.Adam(params, weight_decay=args.weight_decay)
        
        # Scheduler (Cosine Annealing)
        if getattr(args, 'use_cosine_annealing', True):
            self.scheduler_phase2 = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer_phase2, T_max=args.stage2_epoch, eta_min=1e-6
            )
        else:
            self.scheduler_phase2 = torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer_phase2, milestones=args.milestones, gamma=0.1
            )

    def save_model(self, epoch, is_best=False):
        save_path = self.args.save_path
        os.makedirs(os.path.join(save_path, 'models'), exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model': self.model.state_dict(),
            'classifier1': self.classifier1.state_dict(),
            'classifier2': self.classifier2.state_dict(),
            'ragm': self.ragm.state_dict(),
        }
        
        if self.enable_cls3 and self.classifier3 is not None:
            checkpoint['classifier3'] = self.classifier3.state_dict()
        
        if self.use_diffusion:
            temp_cma = getattr(self.diffusion, 'cma', None)
            if temp_cma: self.diffusion.cma = None
            checkpoint['diffusion'] = self.diffusion.state_dict()
            if temp_cma: self.diffusion.cma = temp_cma

        if hasattr(self, 'optimizer_phase2'):
            checkpoint['optimizer_phase2'] = self.optimizer_phase2.state_dict()
            checkpoint['scheduler_phase2'] = self.scheduler_phase2.state_dict()
        
        torch.save(checkpoint, os.path.join(save_path, 'models', 'checkpoint.pth'))
        if is_best:
            torch.save(checkpoint, os.path.join(save_path, 'models', 'checkpoint_best.pth'))
            print(f"✓ Saved Best Epoch {epoch}")

    def resume_model(self, model_path=None):
        if model_path is None or model_path == 'default':
            model_path = os.path.join(self.args.save_path, 'models', 'checkpoint.pth')
        
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Load core components
        self.model.load_state_dict(checkpoint['model'])
        self.classifier1.load_state_dict(checkpoint['classifier1'])
        self.classifier2.load_state_dict(checkpoint['classifier2'])
        
        # Load Classifier3 (If resuming a Phase 2 model)
        if 'classifier3' in checkpoint:
            if self.classifier3 is None:
                class CArgs: num_classes = self.num_classes
                self.classifier3 = Image_Classifier(CArgs()).to(self.device)
            self.classifier3.load_state_dict(checkpoint['classifier3'])
            self.enable_cls3 = True
        
        # Load Diffusion
        if self.use_diffusion and 'diffusion' in checkpoint:
            self.diffusion.load_state_dict(checkpoint['diffusion'], strict=False)
        
        # Load Optimizer (Phase 2)
        if 'optimizer_phase2' in checkpoint:
            try:
                self.configure_optimizer_phase2(self.args)
                self.optimizer_phase2.load_state_dict(checkpoint['optimizer_phase2'])
                self.scheduler_phase2.load_state_dict(checkpoint['scheduler_phase2'])
            except:
                print("⚠️ Optimizer param groups mismatch, skipping optimizer load.")
        
        self.resume_epoch = checkpoint.get('epoch', 0)
        print(f"✓ Resumed from Epoch {self.resume_epoch}")

    def set_train(self):
        self.model.train()
        self.classifier1.train()
        self.classifier2.train()
        if self.enable_cls3 and self.classifier3 is not None:
            self.classifier3.train()
        self.ragm.train()
    
    def set_eval(self):
        self.model.eval()
        self.classifier1.eval()
        self.classifier2.eval()
        if self.enable_cls3 and self.classifier3 is not None:
            self.classifier3.eval()
        self.ragm.eval()