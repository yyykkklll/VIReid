import torch
import os
import re

from .classifier import Image_Classifier

from utils import os_walk
from .agw import AGW
from .clip_model import CLIP
from .optim import WarmupMultiStepLR
from .loss import TripletLoss_WRT, Weak_loss, WeightedContrastiveLoss, LabelSmoothingCrossEntropy
_models = {
    "resnet": AGW,  # visual encoder AGW, no text encoder
    "clip-resnet": CLIP,  # resnet50 + transformer
    "vit": 0,  # visual encoder vit, no text encoder
    #"clip-vit": CLIP,  # both vit-b/16 + transformer
}



def create(args):
    """
    Create a dataset instance.
    """
    if args.arch not in _models:
        raise KeyError("Unknown backbone:", args.arch)
    print('loading {} dataset ...'.format(args.dataset))
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
        self.resume_epoch = 0  # Initialize resume_epoch to 0

        self.model = _models[args.arch](args).to(self.device)
        self.classifier1 = Image_Classifier(args).to(self.device) # RGB classifier
        self.classifier2 = Image_Classifier(args).to(self.device) # IR classifier
        self.classifier3 = Image_Classifier(args).to(self.device) # Common classifier
        self.enable_cls3 = False

        self._init_optimizer()
        self._init_criterion()

    def _init_optimizer(self):
        ###########################
        ###专家学习率适配###
        params_phase1 = []
        params_phase2 = []
        for part in (self.model, self.classifier1, self.classifier2):
            for key, value in part.named_parameters():
                if value.requires_grad and key.find("classifier") != -1:
                    params_phase1 += [{"params": [value], 
                                "lr": 2 * self.lr,
                                "weight_decay": self.weight_decay}]
                elif value.requires_grad:
                    params_phase1 += [{"params": [value], 
                                "lr": self.lr,
                                "weight_decay": self.weight_decay}]
        params_phase2.extend(params_phase1)
        for key, value in self.classifier3.named_parameters():
            if value.requires_grad:
                params_phase2 += [{"params": [value], 
                            "lr": 2 * self.lr,
                            "weight_decay": self.weight_decay}]
        self.optimizer_phase1 = torch.optim.Adam(params_phase1)
        self.optimizer_phase2 = torch.optim.Adam(params_phase2)
        self.scheduler_phase1 = WarmupMultiStepLR(self.optimizer_phase1, self.milestones,
                                           gamma=0.1, warmup_factor=0.01, warmup_iters=10, mode='cls')
        self.scheduler_phase2 = WarmupMultiStepLR(self.optimizer_phase2, self.milestones,
                                           gamma=0.1, warmup_factor=0.01, warmup_iters=10, mode='cls')
    def _init_criterion(self):
        self.pid_criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
        self.tri_criterion = TripletLoss_WRT()
        self.weak_criterion = Weak_loss()

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

    def save_model(self, save_epoch, filename):
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
            
        model_file_path = os.path.join(self.save_path, filename)
        
        all_state_dict = {'backbone': self.model.state_dict(),
                            'classifier1': self.classifier1.state_dict(),
                            'classifier2': self.classifier2.state_dict(),
                            'classifier3': self.classifier3.state_dict(),
                            'epoch': save_epoch}
        
        torch.save(all_state_dict, model_file_path)
        print(f'Saved model to {model_file_path}')

    def resume_model(self, specified_model=None):
        '''
        # load the weights from existed file
        model_epoch: the epoch of the model to load(optional)
        '''
        if specified_model is None:
            # Check for best_phase2.pth then best_phase1.pth
            phase2_path = os.path.join(self.save_path, 'best_phase2.pth')
            phase1_path = os.path.join(self.save_path, 'best_phase1.pth')
            
            model_path = None
            if os.path.exists(phase2_path):
                model_path = phase2_path
            elif os.path.exists(phase1_path):
                model_path = phase1_path
            
            self.resume_epoch = 0
            if model_path is not None:
                if self.resume or self.mode == 'test':
                    loaded_dict = torch.load(model_path, map_location=self.device)
                    self.model.load_state_dict(loaded_dict['backbone'], strict=False)
                    self.classifier1.load_state_dict(loaded_dict['classifier1'], strict=False)
                    self.classifier2.load_state_dict(loaded_dict['classifier2'], strict=False)
                    try:
                        self.classifier3.load_state_dict(loaded_dict['classifier3'], strict=False)
                    except:
                        pass
                    
                    if 'epoch' in loaded_dict:
                        self.resume_epoch = loaded_dict['epoch']
                    else:
                        # Fallback if epoch not in dict (unlikely with new save)
                        print("Warning: 'epoch' not found in checkpoint. Starting from 0.")
                        self.resume_epoch = 0

                    print(f'load {model_path}')
                else:
                    # If not resume/test, we don't delete files anymore as they are best checkpoints
                    print('Not resuming, starting from scratch.')
            else:
                print('valid model file not existed!')
            print(f'from {self.resume_epoch} epoch training')

        else:
            model_path = specified_model
            loaded_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(loaded_dict['backbone'], strict=False)
            self.classifier1.load_state_dict(loaded_dict['classifier1'], strict=False)
            self.classifier2.load_state_dict(loaded_dict['classifier2'], strict=False)
            try:
                self.classifier3.load_state_dict(loaded_dict['classifier3'], strict=False)
            except:
                pass
            
            if 'epoch' in loaded_dict:
                 self.resume_epoch = loaded_dict['epoch']
            else:
                 self.resume_epoch = 0
            
            print(f'load {model_path}')
            print(f'from {self.resume_epoch} epoch training')
