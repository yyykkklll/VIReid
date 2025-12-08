from __future__ import absolute_import
from PIL import Image
import numpy as np
import random
import math
import torchvision.transforms as transforms
import torch

normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

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
        return img.clamp(0, 1)

class ChannelRandomErasing:
    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=[0.4914, 0.4822, 0.4465]):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1
    def __call__(self, img):
        if random.uniform(0, 1) > self.probability:
            return img
        for attempt in range(100):
            area = img.size()[1] * img.size()[2]
            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)
            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))
            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                    img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
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
            img[1, :, :] = img[0, :, :]
            img[2, :, :] = img[0, :, :]
        elif idx == 1:
            img[0, :, :] = img[1, :, :]
            img[2, :, :] = img[1, :, :]
        elif idx == 2:
            img[0, :, :] = img[2, :, :]
            img[1, :, :] = img[2, :, :]
        else:
            tmp_img = 0.2989 * img[0, :, :] + 0.5870 * img[1, :, :] + 0.1140 * img[2, :, :]
            img[0, :, :] = tmp_img
            img[1, :, :] = tmp_img
            img[2, :, :] = tmp_img
        return img

class ChannelAdapGray(object):
    def __init__(self, probability=0.5):
        self.probability = probability
    def __call__(self, img):
        idx = random.randint(0, 3)
        if idx == 0:
            img[1, :, :] = img[0, :, :]
            img[2, :, :] = img[0, :, :]
        elif idx == 1:
            img[0, :, :] = img[1, :, :]
            img[2, :, :] = img[1, :, :]
        elif idx == 2:
            img[0, :, :] = img[2, :, :]
            img[1, :, :] = img[2, :, :]
        else:
            if random.uniform(0, 1) > self.probability:
                img = img
            else:
                tmp_img = 0.2989 * img[0, :, :] + 0.5870 * img[1, :, :] + 0.1140 * img[2, :, :]
                img[0, :, :] = tmp_img
                img[1, :, :] = tmp_img
                img[2, :, :] = tmp_img
        return img

def get_train_transforms(h, w, modal='rgb'):
    """
    [修改] 移除了 transforms.ToPILImage()
    因为传入的已经是 PIL Image
    """
    if modal == 'rgb':
        # RGB Transform
        transform_normal = transforms.Compose([
            # transforms.ToPILImage(), # 移除
            transforms.Pad(10),
            transforms.RandomCrop((h, w)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
            ChannelRandomErasing(probability=0.5)
        ])
        
        transform_ca = transforms.Compose([
            # transforms.ToPILImage(), # 移除
            transforms.Pad(10),
            transforms.RandomCrop((h, w)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
            ChannelRandomErasing(probability=0.5),
            ChannelExchange(gray=2)
        ])
        return transform_normal, transform_ca
        
    elif modal == 'ir':
        # IR Transform
        transform_normal = transforms.Compose([
            # transforms.ToPILImage(), # 移除
            transforms.Pad(10),
            transforms.RandomCrop((h, w)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
            ChannelRandomErasing(probability=0.5),
            ChannelAdapGray(probability=0.5)
        ])
        
        transform_aug = transforms.Compose([
            # transforms.ToPILImage(), # 移除
            transforms.Pad(10),
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
        # transforms.ToPILImage(), # 移除
        transforms.Resize((h, w)),
        transforms.ToTensor(),
        normalize
    ])
    return transform_test