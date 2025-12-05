from __future__ import absolute_import
import random
import math
import torch
import torchvision.transforms as transforms

# ImageNet 均值和方差
normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

class ChannelRandomErasing:
    """ 随机通道擦除 (Random Erasing) """
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
    """ 随机通道交换，模拟模态差异 [关键增强] """
    def __init__(self, gray=2):
        self.gray = gray

    def __call__(self, img):
        idx = random.randint(0, self.gray)
        if idx == 0:
            # R <- G, B <- G
            img[0, :, :] = img[1, :, :]
            img[2, :, :] = img[1, :, :]
        elif idx == 1:
            # R <- B, G <- B
            img[0, :, :] = img[2, :, :]
            img[1, :, :] = img[2, :, :]
        # idx == 2: 原图不变
        return img

class ChannelAdapGray(object):
    """ 自适应灰度增强 """
    def __init__(self, probability=0.5):
        self.probability = probability

    def __call__(self, img):
        if random.uniform(0, 1) > self.probability:
            return img
        # 转灰度并复制到3通道
        tmp_img = 0.2989 * img[0, :, :] + 0.5870 * img[1, :, :] + 0.1140 * img[2, :, :]
        img[0, :, :] = tmp_img
        img[1, :, :] = tmp_img
        img[2, :, :] = tmp_img
        return img

# ================= 优化后的变换策略 =================

# 1. 可见光训练变换
transform_color_normal = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((288, 144)),
            transforms.Pad(10),
            transforms.RandomCrop((288, 144)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
            # [关键] 随机通道交换
            transforms.RandomApply([ChannelExchange(gray=2)], p=0.5), 
            ChannelRandomErasing(probability=0.5)
])

# 兼容接口别名
transform_color_ca = transform_color_normal
transform_color_sa = transform_color_normal

# 2. 红外光训练变换
transform_infrared_normal = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((288, 144)),
            transforms.Pad(10),
            transforms.RandomCrop((288, 144)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
            # [关键] 自适应灰度
            ChannelAdapGray(probability=0.5),
            ChannelRandomErasing(probability=0.5)
])

# 兼容接口别名
transform_infrared_sa = transform_infrared_normal

# 3. 测试变换
transform_test = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((288, 144)),
            transforms.ToTensor(),
            normalize])

# [修复 Import Error] 补全 transform_test_sa
# 在强基线模式下，测试时不需要额外的 Style Augmentation，直接复用标准测试变换即可
transform_test_sa = transform_test