import torch
import torch.nn as nn
import timm

class ResNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        print("Loading ResNet50 with Online Pretrained Weights...")
        self.model = timm.create_model('resnet50', pretrained=True, num_classes=0, global_pool='')
        
        # Last Stride = 1 (ReID Standard Trick)
        self.model.layer4[0].conv2.stride = (1, 1)
        self.model.layer4[0].downsample[0].stride = (1, 1)
        
        self.in_planes = 2048
        
        self.bn = nn.BatchNorm1d(self.in_planes)
        self.bn.bias.requires_grad_(False)
        nn.init.constant_(self.bn.weight, 1.0)
        nn.init.constant_(self.bn.bias, 0.0)
        
        self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(self, x1=None, x2=None):
        if x1 is None: x = x2
        elif x2 is None: x = x1
        else: x = torch.cat([x1, x2], dim=0)
        
        x = self.model(x)
        feat = self.gap(x).flatten(1)
        bn_feat = self.bn(feat)
        return feat, bn_feat

def build_resnet(args):
    return ResNet(args)