import torch.nn as nn
import torch

def weights_init_kaiming(m):
    """
    Kaiming 初始化，用于初始化卷积层、全连接层和BN层
    """
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def weights_init(m):
    """
    正态分布初始化，主要用于分类器的全连接层
    """
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)

class Normalize(nn.Module):
    """
    L2 特征归一化层
    """
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, input):
        norm = input.pow(self.power).sum(dim=1, keepdim=True).pow(1. / self.power)
        output = input / norm
        return output

class GeneralizedMeanPooling(nn.Module):
    """
    广义平均池化 (GeM Pooling)，可调节关注度
    """
    def __init__(self, norm, output_size=1, eps=1e-6):
        super(GeneralizedMeanPooling, self).__init__()
        assert norm > 0
        self.p = float(norm)
        self.output_size = output_size
        self.eps = eps

    def forward(self, x):
        x = x.clamp(min=self.eps).pow(self.p)
        return torch.nn.functional.adaptive_avg_pool2d(x, self.output_size).pow(1. / self.p)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + str(self.p) + ', ' \
               + 'output_size=' + str(self.output_size) + ')'

class GeneralizedMeanPoolingP(GeneralizedMeanPooling):
    """
    可学习参数 p 的 GeM Pooling
    """
    def __init__(self, norm=3, output_size=1, eps=1e-6):
        super(GeneralizedMeanPoolingP, self).__init__(norm, output_size, eps)
        self.p = nn.Parameter(torch.ones(1) * norm)

class Text_Classifier(nn.Module):
    """
    文本模态分类器（用于 CLIP 相关模块）
    """
    def __init__(self, args):
        super(Text_Classifier, self).__init__()
        self.num_classes = args.num_classes
        # CLIP text encoder 输出通常是 1024 或 512，这里假设配合原代码逻辑为 1024
        self.BN = nn.BatchNorm1d(1024)
        self.BN.apply(weights_init_kaiming)

        self.classifier = nn.Linear(1024, self.num_classes, bias=False)
        self.classifier.apply(weights_init)

        self.l2_norm = Normalize(2)

    def forward(self, features):
        bn_features = self.BN(features.squeeze())
        cls_score = self.classifier(bn_features)
        if self.training:
            return cls_score
        else:
            return self.l2_norm(bn_features)

class Image_Classifier(nn.Module):
    """
    图像模态分类器
    输入: BN 后的特征 (Bottleneck Feature)
    输出: (分类分数, 归一化特征)
    """
    def __init__(self, args):
        super(Image_Classifier, self).__init__()
        self.num_classes = args.num_classes
        
        # [修改逻辑] 动态获取特征维度，兼容 ViT (768) 和 ResNet (2048)
        # 如果 args 中没有定义 feat_dim，则默认回退到 2048 以兼容旧代码
        self.in_dim = getattr(args, 'feat_dim', 2048)
        
        self.classifier = nn.Linear(self.in_dim, self.num_classes, bias=False)
        self.classifier.apply(weights_init)

        self.l2_norm = Normalize(2)

    def forward(self, x_bn):
        # x_bn: 经过 Bottleneck (BN层) 后的特征
        x_score = self.classifier(x_bn)
        return x_score, self.l2_norm(x_bn)