from __future__ import absolute_import
from PIL import Image
import numpy as np
import random
import math
import torchvision.transforms as transforms
import torch
import torch.fft

normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def frequency_mix(rgb_imgs, ir_imgs, ratio=0.1):
    """
    [新增] 频域混合增强
    将 RGB 图像的低频（风格）与 IR 图像的高频（内容）混合，反之亦然
    
    Args: 
        rgb_imgs:  RGB 图像 tensor [B, C, H, W]
        ir_imgs: IR 图像 tensor [B, C, H, W]
        ratio: 低频区域的比例 (0~1)
    
    Returns:
        mix_rgb: 混合后的伪 RGB (IR 内容 + RGB 风格)
        mix_ir:  混合后的伪 IR (RGB 内容 + IR 风格)
    """
    B, C, H, W = rgb_imgs.shape
    
    # 创建低频掩码 (中心区域)
    mask = torch.zeros(H, W, device=rgb_imgs. device)
    center_h, center_w = H // 2, W // 2
    r_h, r_w = int(H * ratio), int(W * ratio)
    
    # 低频区域设为 1
    mask[center_h - r_h:center_h + r_h, center_w - r_w:center_w + r_w] = 1.0
    mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    
    # FFT 变换
    rgb_fft = torch. fft.fft2(rgb_imgs)
    ir_fft = torch. fft.fft2(ir_imgs)
    
    # 将零频移到中心
    rgb_fft_shift = torch.fft.fftshift(rgb_fft)
    ir_fft_shift = torch.fft.fftshift(ir_fft)
    
    # 频域混合
    # mix_rgb:  RGB 的低频 (风格) + IR 的高频 (内容)
    mix_rgb_fft = rgb_fft_shift * mask + ir_fft_shift * (1 - mask)
    # mix_ir: IR 的低频 (风格) + RGB 的高频 (内容)
    mix_ir_fft = ir_fft_shift * mask + rgb_fft_shift * (1 - mask)
    
    # 逆变换
    mix_rgb_fft = torch.fft.ifftshift(mix_rgb_fft)
    mix_ir_fft = torch.fft.ifftshift(mix_ir_fft)
    
    mix_rgb = torch.fft.ifft2(mix_rgb_fft).real
    mix_ir = torch.fft.ifft2(mix_ir_fft).real
    
    # Clamp 到有效范围
    mix_rgb = torch.clamp(mix_rgb, min=-3, max=3)  # 已归一化的图像范围
    mix_ir = torch.clamp(mix_ir, min=-3, max=3)
    
    return mix_rgb, mix_ir


class StyleAug:
    def __init__(self):
        pass
    def __call__(self, img):
        c = img.shape[0]
        if c == 1:
            factor = random.uniform(0.5, 1.5)
            img = img * factor
        elif c == 3:
            factors = [random.uniform(0.5, 1.5) for _ in range(3)]
            for i in range(3):
                img[i] *= factors[i]
            img = img[torch.randperm(3)]
        else:
            pass
        return img. clamp(0, 1)


class ChannelRandomErasing:
    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=[0.4914, 0.4822, 0.4465]):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self. r1 = r1
        
    def __call__(self, img):
        if random. uniform(0, 1) > self.probability:
            return img
        for attempt in range(100):
            area = img.size()[1] * img.size()[2]
            target_area = random. uniform(self.sl, self.sh) * area
            aspect_ratio = random. uniform(self.r1, 1 / self.r1)
            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))
            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1 + h, y1:y1 + w] = self. mean[0]
                    img[1, x1:x1 + h, y1:y1 + w] = self. mean[1]
                    img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
                else: 
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                return img
        return img


class ChannelExchange(object):
    def __init__(self, gray=2):
        self.gray = gray
        
    def __call__(self, img):
        idx = random.randint(0, self.gray)
        if idx == 0:
            img[1, :, :] = img[0, : , :]
            img[2, :, :] = img[0, :, :]
        elif idx == 1:
            img[0, :, : ] = img[1, :, :]
            img[2, :, :] = img[1, :, :]
        elif idx == 2:
            img[0, :, :] = img[2, :, :]
            img[1, :, :] = img[2, : , :]
        else:
            tmp_img = 0.2989 * img[0, :, :] + 0.5870 * img[1, :, :] + 0.1140 * img[2, :, :]
            img[0, :, : ] = tmp_img
            img[1, :, :] = tmp_img
            img[2, :, : ] = tmp_img
        return img


class ChannelAdapGray(object):
    def __init__(self, probability=0.5):
        self.probability = probability
        
    def __call__(self, img):
        idx = random.randint(0, 3)
        if idx == 0:
            img[1, :, :] = img[0, :, :]
            img[2, :, :] = img[0, : , :]
        elif idx == 1:
            img[0, :, : ] = img[1, :, :]
            img[2, :, :] = img[1, :, :]
        elif idx == 2:
            img[0, :, :] = img[2, :, :]
            img[1, :, : ] = img[2, :, :]
        else: 
            if random.uniform(0, 1) > self.probability:
                img = img
            else:
                tmp_img = 0.2989 * img[0, : , :] + 0.5870 * img[1, :, :] + 0.1140 * img[2, :, :]
                img[0, :, :] = tmp_img
                img[1, :, :] = tmp_img
                img[2, : , :] = tmp_img
        return img


def get_train_transforms(h, w, modal='rgb'):
    if modal == 'rgb':
        transform_normal = transforms.Compose([
            transforms. Pad(10),
            transforms.RandomCrop((h, w)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
            ChannelRandomErasing(probability=0.5)
        ])
        
        transform_ca = transforms.Compose([
            transforms. Pad(10),
            transforms.RandomCrop((h, w)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
            ChannelRandomErasing(probability=0.5),
            ChannelExchange(gray=2)
        ])
        return transform_normal, transform_ca
        
    elif modal == 'ir': 
        transform_normal = transforms.Compose([
            transforms. Pad(10),
            transforms.RandomCrop((h, w)),
            transforms. RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
            ChannelRandomErasing(probability=0.5),
            ChannelAdapGray(probability=0.5)
        ])
        
        transform_aug = transforms.Compose([
            transforms. Pad(10),
            transforms.RandomCrop((h, w)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            StyleAug(),
            normalize,
            ChannelRandomErasing(probability=0.5),
            ChannelAdapGray(probability=0.5)
        ])
        return transform_normal, transform_aug


def get_test_transformer(h, w):
    transform_test = transforms.Compose([
        transforms. Resize((h, w)),
        transforms.ToTensor(),
        normalize
    ])
    return transform_test