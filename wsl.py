import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict

# 导入 PRUD
from models.prud import PRUD

class CMA(nn.Module):
    def __init__(self, args):
        super().__init__()
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_classes = args.num_classes
        self.T = args.temperature
        self.sigma = args.sigma
        self.proto_momentum = getattr(args, 'prototype_momentum', 0.9)
        self.use_ccpa = getattr(args, 'use_cycle_consistency', False)
        
        # State flags
        self.not_saved = True
        self.current_epoch = 0
        self.ragm_module = None
        
        # Memory Banks
        self.register_buffer('vis_memory', torch.zeros(self.num_classes, 2048))
        self.register_buffer('ir_memory', torch.zeros(self.num_classes, 2048))
        
        # Auxiliary Prototypes
        self.register_buffer('prototypes_v', torch.randn(self.num_classes, 2048))
        self.register_buffer('prototypes_r', torch.randn(self.num_classes, 2048))
        self.prototypes_initialized = False
        
        # PRUD Initialization
        if self.use_ccpa:
            print("=> [CMA] Initializing PRUD (Prototype-Rectified Unidirectional Distillation)...")
            self.prud = PRUD(
                num_classes=self.num_classes,
                rectification_threshold=0.3,
                temp=0.1,
                momentum=0.9
            )
        else:
            self.prud = None

    def set_epoch(self, epoch):
        self.current_epoch = epoch
        if self.prud is not None:
            self.prud.set_epoch(epoch)
            
    def set_ragm_module(self, ragm):
        self.ragm_module = ragm

    def _recursive_clean(self, module):
        """Helper to recursively remove weakref proxies from all submodules"""
        import weakref
        
        # 1. Clean current module's _modules
        if hasattr(module, '_modules'):
            keys_to_remove = []
            for name, child in list(module._modules.items()):
                if isinstance(child, weakref.ProxyTypes):
                    print(f"⚠️ [CMA] Deep Clean: Removing weakref proxy '{name}' from {type(module).__name__}")
                    keys_to_remove.append(name)
            
            for k in keys_to_remove:
                del module._modules[k]
        
        # 2. Recurse into children
        if hasattr(module, 'named_children'):
            for name, child in module.named_children():
                self._recursive_clean(child)

    def extract(self, args, model, dataset):
        """
        Initialize memory banks with features extracted from the entire dataset.
        Called once at the beginning of Phase 2.
        """
        print("=> [CMA] Extracting features to initialize memory banks...")
        
        # ==================== [DEEP CLEAN FIX] ====================
        print("=> [CMA] Running Deep Clean on model...")
        self._recursive_clean(model)
        if isinstance(model, nn.DataParallel):
             self._recursive_clean(model.module)
        # ==========================================================

        model.eval()
        rgb_loader, ir_loader = dataset.get_train_loader()
        
        # Extract RGB features
        rgb_features = OrderedDict()
        
        with torch.no_grad():
            # ✅ FIX: data loader yields (img, path, info), where info has shape [B, 4]
            # 'pids' variable name changed to 'info' to reflect reality
            for imgs, _, info in rgb_loader:
                imgs = imgs.to(self.device)
                
                # Extract actual PIDs from index 1 (Consistent with train.py)
                pids = info[:, 1]
                
                # ✅ FIX: AGW.forward(x1, x2) - RGB goes to x1
                if isinstance(model, nn.DataParallel):
                    output = model.module.model(x1=imgs, x2=None)
                else:
                    output = model.model(x1=imgs, x2=None)
                
                features = output[1] 
                
                for f, pid in zip(features, pids):
                    pid = pid.item() # Now pid is a scalar, .item() works
                    if pid not in rgb_features:
                        rgb_features[pid] = []
                    rgb_features[pid].append(f)
        
        # Extract IR features
        ir_features = OrderedDict()
        with torch.no_grad():
            # ✅ FIX: same for IR loader
            for imgs, _, info in ir_loader:
                imgs = imgs.to(self.device)
                pids = info[:, 1] # Extract PIDs
                
                # ✅ FIX: AGW.forward(x1, x2) - IR goes to x2
                if isinstance(model, nn.DataParallel):
                    output = model.module.model(x1=None, x2=imgs)
                else:
                    output = model.model(x1=None, x2=imgs)
                
                features = output[1]
                
                for f, pid in zip(features, pids):
                    pid = pid.item()
                    if pid not in ir_features:
                        ir_features[pid] = []
                    ir_features[pid].append(f)

        # Initialize Memory Banks
        print("=> [CMA] Averaging features...")
        for pid in range(self.num_classes):
            if pid in rgb_features:
                stack_v = torch.stack(rgb_features[pid])
                self.vis_memory[pid] = F.normalize(stack_v.mean(0), dim=0)
                if hasattr(self, 'prototypes_v'):
                    self.prototypes_v[pid] = self.vis_memory[pid]
            
            if pid in ir_features:
                stack_r = torch.stack(ir_features[pid])
                self.ir_memory[pid] = F.normalize(stack_r.mean(0), dim=0)
                if hasattr(self, 'prototypes_r'):
                    self.prototypes_r[pid] = self.ir_memory[pid]

        self.prototypes_initialized = True
        print("=> [CMA] Memory initialization done.")

    def update_memory(self, features_v, features_r, ids_v, ids_r):
        features_v = F.normalize(features_v, dim=1)
        features_r = F.normalize(features_r, dim=1)
        
        for pid in ids_v.unique():
            mask = (ids_v == pid)
            if mask.sum() > 0:
                feat_mean = features_v[mask].mean(0)
                feat_mean = F.normalize(feat_mean, dim=0)
                self.vis_memory[pid] = self.proto_momentum * self.vis_memory[pid] + \
                                     (1 - self.proto_momentum) * feat_mean
                self.vis_memory[pid] = F.normalize(self.vis_memory[pid], dim=0)

        for pid in ids_r.unique():
            mask = (ids_r == pid)
            if mask.sum() > 0:
                feat_mean = features_r[mask].mean(0)
                feat_mean = F.normalize(feat_mean, dim=0)
                self.ir_memory[pid] = self.proto_momentum * self.ir_memory[pid] + \
                                    (1 - self.proto_momentum) * feat_mean
                self.ir_memory[pid] = F.normalize(self.ir_memory[pid], dim=0)

    def update_prototypes(self, features, ids, modality):
        target_memory = self.prototypes_v if modality == 'v' else self.prototypes_r
        features = F.normalize(features, dim=1)
        
        for pid in ids.unique():
            mask = (ids == pid)
            if mask.sum() > 0:
                feat_mean = features[mask].mean(0)
                self.proto_momentum = 0.9
                target_memory[pid] = self.proto_momentum * target_memory[pid] + \
                                   (1 - self.proto_momentum) * feat_mean
                target_memory[pid] = F.normalize(target_memory[pid], dim=0)

    def prepare_rectified_prototypes(self, diffusion_bridge):
        if self.prud is None:
            return
        self.prud.prepare_rectified_prototypes(
            prototypes_r=self.ir_memory, 
            prototypes_v=self.vis_memory, 
            diffusion_bridge=diffusion_bridge
        )
    
    def get_distillation_weights(self, rgb_ids, ir_ids):
        if self.prud is None:
            return torch.ones_like(rgb_ids, dtype=torch.float), torch.ones_like(ir_ids, dtype=torch.float)
        return self.prud.get_distillation_weights(rgb_ids, ir_ids)
    
    def get_prototypes(self, modality):
        return self.vis_memory if modality == 'v' else self.ir_memory

    def get_label(self):
        sim = torch.mm(self.vis_memory, self.ir_memory.t())
        cost = 1 - sim
        P = self._sinkhorn(cost, epsilon=0.05, max_iter=50)
        
        r2i_pair = {}
        i2r_pair = {}
        assign_thresh = 1.0 / self.num_classes
        
        vals, idxs = P.max(dim=1)
        for i, (val, idx) in enumerate(zip(vals, idxs)):
            if val > assign_thresh:
                r2i_pair[i] = idx.item()
                
        vals, idxs = P.max(dim=0)
        for i, (val, idx) in enumerate(zip(vals, idxs)):
            if val > assign_thresh:
                i2r_pair[i] = idx.item()
                
        return r2i_pair, i2r_pair

    def _sinkhorn(self, C, epsilon=0.05, max_iter=50):
        n = C.shape[0]
        m = C.shape[1]
        K = torch.exp(-C / epsilon)
        u = torch.ones(n, device=self.device) / n
        v = torch.ones(m, device=self.device) / m
        
        for _ in range(max_iter):
            u = 1.0 / (torch.matmul(K, v) + 1e-8)
            v = 1.0 / (torch.matmul(K.t(), u) + 1e-8)
            
        P = torch.diag(u) @ K @ torch.diag(v)
        return P