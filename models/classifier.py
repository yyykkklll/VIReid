import torch
import torch.nn as nn
import torch.nn.functional as F

def weights_init_kaiming(m):
    """Kaiming initialization for linear and conv layers."""
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode="fan_out")
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode="fan_in")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def weights_init(m):
    """Standard normal initialization."""
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=1e-3)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)

class Normalize(nn.Module):
    """L2 Normalization layer (Optimized with F.normalize)."""
    def __init__(self, power=2):
        super().__init__()
        self.power = power

    def forward(self, x):
        return F.normalize(x, p=self.power, dim=1)

class GeneralizedMeanPooling(nn.Module):
    """Generalized Mean Pooling (GeM)."""
    def __init__(self, norm, output_size=1, eps=1e-6):
        super().__init__()
        assert norm > 0
        self.p = float(norm)
        self.output_size = output_size
        self.eps = eps

    def forward(self, x):
        x = x.clamp(min=self.eps).pow(self.p)
        return F.adaptive_avg_pool2d(x, self.output_size).pow(1.0 / self.p)

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p}, output_size={self.output_size})"

class GeneralizedMeanPoolingP(GeneralizedMeanPooling):
    """GeM with trainable power parameter."""
    def __init__(self, norm=3, output_size=1, eps=1e-6):
        super().__init__(norm, output_size, eps)
        self.p = nn.Parameter(torch.ones(1) * norm)

class Image_Classifier(nn.Module):
    """
    Standard Linear Classifier with L2 Normalized Output.
    Output: (logits, normalized_features)
    """
    def __init__(self, args):
        super().__init__()
        self.num_classes = args.num_classes
        
        # Classifier layer (bias=False is standard for ReID)
        self.classifier = nn.Linear(2048, self.num_classes, bias=False)
        self.classifier.apply(weights_init)

        self.l2_norm = Normalize(2)

    def forward(self, x_bn):
        # x_bn: [B, 2048]
        x_score = self.classifier(x_bn)
        return x_score, self.l2_norm(x_bn)

# Alias for compatibility
Classifier = Image_Classifier