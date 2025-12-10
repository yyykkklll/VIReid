import torch
import os
from .resnet import build_resnet 
from .projector import Projector
from .discriminator import DomainDiscriminator
from .loss import InfoNCELoss, SinkhornLoss, AdversarialLoss
from utils import os_walk

# 仅保留 ResNet
_models = {
    "resnet50": build_resnet,
}

def create(args):
    return UnsupervisedModel(args)

class UnsupervisedModel:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        self.save_path = os.path.join(args.save_path, "models/")
        
        print(f"Building Backbone: {args.arch}")
        self.backbone = _models[args.arch](args).to(self.device)
        self.feat_dim = 2048 # ResNet50
        
        print("Building Unsupervised Components...")
        self.projector = Projector(self.feat_dim, 256).to(self.device)
        self.discriminator = DomainDiscriminator(self.feat_dim).to(self.device)
        
        self._init_optimizer()
        self._init_criterion()

    def _init_optimizer(self):
        # Backbone 和 Projector 是一伙的
        self.opt_backbone = torch.optim.AdamW([
            {'params': self.backbone.parameters()},
            {'params': self.projector.parameters(), 'lr': self.args.lr * 2}
        ], lr=self.args.lr, weight_decay=self.args.weight_decay)
        
        # Discriminator 是对手
        self.opt_disc = torch.optim.Adam(
            self.discriminator.parameters(), lr=self.args.lr, betas=(0.5, 0.999)
        )

    def _init_criterion(self):
        self.criterion_nce = InfoNCELoss(temperature=0.07).to(self.device)
        self.criterion_ot = SinkhornLoss(epsilon=0.05).to(self.device)
        self.criterion_adv = AdversarialLoss().to(self.device)

    def set_train(self):
        self.backbone.train()
        self.projector.train()
        self.discriminator.train()

    def set_eval(self):
        self.backbone.eval()
        self.projector.eval()
        self.discriminator.eval()
        
    def save_model(self, epoch, is_best):
        if is_best:
            path = os.path.join(self.save_path, f"model_best.pth")
        else:
            path = os.path.join(self.save_path, f"model_epoch_{epoch}.pth")
            
        state = {
            'backbone': self.backbone.state_dict(),
            'projector': self.projector.state_dict(),
            'discriminator': self.discriminator.state_dict(),
            'epoch': epoch
        }
        torch.save(state, path)
        print(f"Saved model to {path}")

    def load_model(self, path):
        if not os.path.exists(path): return
        state = torch.load(path, map_location=self.device)
        self.backbone.load_state_dict(state['backbone'], strict=False)
        print(f"Loaded backbone from {path}")