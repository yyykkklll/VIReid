import torch
import os

from .classifier import Image_Classifier
from utils import os_walk

# 引入 Backbone (仅保留 VMamba)
# [Clean Up] 移除了 CLIP 和 ViT 的引用，确保架构纯净
from .vmamba import build_vmamba 

# 引入优化器和损失函数
from .optim import WarmupMultiStepLR
from .loss import TripletLoss_WRT, Weak_loss, CrossEntropyLabelSmooth, ConsistencyLoss

# 注册支持的模型架构
# [Note] CLIP 和 ViT 已被弃用，不再支持调用
_models = {
    "vmamba": build_vmamba, 
}

def create(args):
    if args.arch not in _models:
        # 如果用户试图使用 vit 或 clip，直接抛出更明确的错误
        raise KeyError(f"Unknown backbone: {args.arch}. Currently only 'vmamba' is supported.")
    print('正在加载模型架构: {} ...'.format(args.arch))
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
        
        self.resume_epoch = 0 

        # 1. 构建骨干网络 (Backbone)
        print(f"Building Backbone: {args.arch}")
        self.model = _models[args.arch](args).to(self.device)
        
        # 2. 确定特征维度
        # VMamba-Tiny/Small/Base 的最终输出维度均为 768
        # 不再依赖 args.feat_dim 的手动输入，直接锁定为 768 以匹配预训练权重
        if args.arch == 'vmamba':
             self.in_dim = 768 
             print(f">>> Auto-detected VMamba dimension: {self.in_dim}")
        else:
             # 保留此分支以防未来扩展其他架构
             self.in_dim = getattr(args, 'feat_dim', 768)

        # 3. 同步参数供 Classifier 使用
        args.feat_dim = self.in_dim 
        
        # 4. 构建分类器 (Classifiers)
        self.classifier1 = Image_Classifier(args).to(self.device) # RGB Expert
        self.classifier2 = Image_Classifier(args).to(self.device) # IR Expert
        self.classifier3 = Image_Classifier(args).to(self.device) # Shared Expert
        self.enable_cls3 = False 

        # 5. 初始化优化器和损失函数
        self._init_optimizer()
        self._init_criterion()

    def _init_optimizer(self):
        """
        使用 AdamW 优化器
        分类器部分的学习率设置为 Backbone 的 2 倍
        """
        params_phase1 = []
        params_phase2 = []
        
        # Phase 1 参数
        for part in (self.model, self.classifier1, self.classifier2):
            for key, value in part.named_parameters():
                if value.requires_grad:
                    if "classifier" in key:
                        params_phase1 += [{"params": [value], 
                                           "lr": 2 * self.lr,
                                           "weight_decay": self.weight_decay}]
                    else:
                        params_phase1 += [{"params": [value], 
                                           "lr": self.lr,
                                           "weight_decay": self.weight_decay}]
        
        # Phase 2 参数
        params_phase2.extend(params_phase1)
        
        for key, value in self.classifier3.named_parameters():
            if value.requires_grad:
                params_phase2 += [{"params": [value], 
                                   "lr": 2 * self.lr,
                                   "weight_decay": self.weight_decay}]
        
        self.optimizer_phase1 = torch.optim.AdamW(params_phase1)
        self.optimizer_phase2 = torch.optim.AdamW(params_phase2)
        
        self.scheduler_phase1 = WarmupMultiStepLR(self.optimizer_phase1, self.milestones,
                                           gamma=0.1, warmup_factor=0.01, warmup_iters=10, mode='cls')
        self.scheduler_phase2 = WarmupMultiStepLR(self.optimizer_phase2, self.milestones,
                                           gamma=0.1, warmup_factor=0.01, warmup_iters=10, mode='cls')

    def _init_criterion(self):
        self.pid_criterion = CrossEntropyLabelSmooth(num_classes=self.args.num_classes, epsilon=0.1, use_gpu=True)
        self.tri_criterion = TripletLoss_WRT()
        self.weak_criterion = Weak_loss()
        self.cons_criterion = ConsistencyLoss() 

    def set_train(self):
        self.model.train()
        self.classifier1.train()
        self.classifier2.train()
        self.classifier3.train()

    def set_eval(self):
        self.model.eval()
        self.classifier1.eval()
        self.classifier2.eval()
        self.classifier3.eval()

    def save_model(self, save_epoch, is_best):
        if is_best:
            model_file_path = os.path.join(self.save_path, 'model_{}.pth'.format(save_epoch))
            root, _, files = os_walk(self.save_path)
            for file in files:
                if '.pth' not in file: continue
                try:
                    file_iters = int(file.replace('.pth', '').split('_')[1])
                    if file_iters <= save_epoch: 
                        remove_file_path = os.path.join(root, 'model_{}.pth'.format(file_iters))
                        os.remove(remove_file_path)
                except:
                    pass

            all_state_dict = {
                'backbone': self.model.state_dict(),
                'classifier1': self.classifier1.state_dict(),
                'classifier2': self.classifier2.state_dict(),
                'classifier3': self.classifier3.state_dict()
            }
            torch.save(all_state_dict, model_file_path)
            print(f"Model saved to {model_file_path}")
        
    def resume_model(self, specified_model=None):
        model_path = None
        if specified_model is None or specified_model == 'default':
            root, _, files = os_walk(self.save_path)
            if len(files) > 0:
                indexes = []
                for file in files:
                    if 'model_' in file and '.pth' in file:
                        try:
                            indexes.append(int(file.replace('.pth', '').split('_')[-1]))
                        except: pass
                if indexes:
                    indexes = sorted(list(set(indexes)), reverse=False)
                    model_path = os.path.join(self.save_path, 'model_{}.pth'.format(indexes[-1]))
        else:
            model_path = specified_model

        if model_path and os.path.exists(model_path):
            print(f'Loading checkpoint from {model_path}')
            loaded_dict = torch.load(model_path, map_location=self.device)
            
            self.model.load_state_dict(loaded_dict['backbone'], strict=False)
            self.classifier1.load_state_dict(loaded_dict['classifier1'], strict=False)
            self.classifier2.load_state_dict(loaded_dict['classifier2'], strict=False)
            if 'classifier3' in loaded_dict:
                self.classifier3.load_state_dict(loaded_dict['classifier3'], strict=False)
            
            try:
                self.resume_epoch = int(os.path.basename(model_path).split('_')[1].split('.')[0])
            except:
                self.resume_epoch = 0
            
            print(f'Resumed from epoch {self.resume_epoch}')
        else:
            print('No valid checkpoint found. Starting from scratch.')
            self.resume_epoch = 0