# models/pcb_resnet.py
import torch.nn as nn
import torchvision.models as models


class PCB_ResNet(nn.Module):
    def __init__(self, pretrained=True):
        super(PCB_ResNet, self).__init__()
        resnet50 = models.resnet50(pretrained=pretrained)

        # 按照 AGW 和方案书设定，去掉 layer4 的下采样 stride，保留更大的特征图
        # Output size will be [B, 2048, 24, 8] for input [B, 3, 384, 128] or similar scaling
        resnet50.layer4[0].conv2.stride = (1, 1)
        resnet50.layer4[0].downsample[0].stride = (1, 1)

        self.backbone = nn.Sequential(
            resnet50.conv1,
            resnet50.bn1,
            resnet50.relu,
            resnet50.maxpool,
            resnet50.layer1,
            resnet50.layer2,
            resnet50.layer3,
            resnet50.layer4
        )

    def forward(self, x):
        # 直接返回特征图 [B, 2048, H, W]
        return self.backbone(x)