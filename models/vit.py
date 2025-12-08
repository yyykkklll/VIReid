import torch
import torch.nn as nn
import timm
import logging
import os
import torch.nn.functional as F
import math

def weights_init_kaiming(m):
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

def resize_pos_embed(posemb, posemb_new, hight, width):
    """
    Rescale the grid of position embeddings when loading from state_dict.
    Adapted from timm.
    """
    # posemb: [1, 197, 768] (197 = 14*14 + 1)
    # posemb_new: [1, N, 768]
    
    ntok_new = posemb_new.shape[1]
    if posemb.shape[1] == ntok_new:
        return posemb

    print(f'Resizing position embedding from {posemb.shape} to {posemb_new.shape} for {hight}x{width}')
    
    posemb_token, posemb_grid = posemb[:, :1], posemb[:, 1:]
    ntok_new -= 1
    
    gs_old = int(math.sqrt(len(posemb_grid[0]))) # 14
    
    # [1, 196, 768] -> [1, 14, 14, 768] -> [1, 768, 14, 14]
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    
    # Interpolation
    posemb_grid = F.interpolate(posemb_grid, size=(hight, width), mode='bilinear', align_corners=False)
    
    # [1, 768, H, W] -> [1, 768, H*W] -> [1, H*W, 768]
    posemb_grid = posemb_grid.flatten(2).transpose(1, 2)
    
    posemb = torch.cat([posemb_token, posemb_grid], dim=1)
    return posemb

class VisionTransformer(nn.Module):
    def __init__(self, args):
        super(VisionTransformer, self).__init__()
        
        # 1. 确定输入尺寸
        img_h = args.img_h
        img_w = args.img_w
        
        # 2. 创建模型 (传入 img_size)
        # 注意：patch_size=16，所以输入尺寸必须能被16整除
        print(f"Creating ViT model with input size ({img_h}, {img_w})...")
        self.model = timm.create_model(
            'vit_base_patch16_224', 
            pretrained=False, 
            img_size=(img_h, img_w), 
            drop_path_rate=0.1
        )
        
        # 3. 智能加载权重 (带插值)
        pretrained_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'pretrained', 'vit_base_patch16_224.bin')
        
        if os.path.exists(pretrained_path):
            print(f"Loading local pretrained weights from: {pretrained_path}")
            try:
                checkpoint = torch.load(pretrained_path, map_location='cpu')
                if 'model' in checkpoint:
                    checkpoint = checkpoint['model']
                
                # 获取当前模型的 state_dict
                model_dict = self.model.state_dict()
                
                # 处理位置编码不匹配问题
                if 'pos_embed' in checkpoint:
                    pos_embed_checkpoint = checkpoint['pos_embed']
                    pos_embed_model = model_dict['pos_embed']
                    
                    # 计算 grid size (e.g., 256/16=16, 128/16=8)
                    grid_h = img_h // 16
                    grid_w = img_w // 16
                    
                    # 执行插值
                    new_pos_embed = resize_pos_embed(pos_embed_checkpoint, pos_embed_model, grid_h, grid_w)
                    checkpoint['pos_embed'] = new_pos_embed
                
                # 加载权重
                msg = self.model.load_state_dict(checkpoint, strict=False)
                print(f"Weights loaded. Missing keys: {msg.missing_keys}")
                
            except Exception as e:
                print(f"Error loading weights: {e}")
        else:
            print(f"WARNING: Local weights not found at {pretrained_path}")

        # 4. 修改结构
        self.model.head = nn.Identity()
        self.in_planes = 768
        self.bn = nn.BatchNorm1d(self.in_planes)
        self.bn.bias.requires_grad_(False)
        self.bn.apply(weights_init_kaiming)

    def forward(self, x1=None, x2=None):
        if x1 is not None and x2 is not None:
            x = torch.cat((x1, x2), 0)
        elif x1 is not None:
            x = x1
        elif x2 is not None:
            x = x2
        else:
            raise ValueError("x1 and x2 cannot be both None")

        features = self.model.forward_features(x)
        
        # 兼容不同版本的 timm 输出
        if isinstance(features, tuple):
            features = features[0]
        if len(features.shape) == 3:
            features = features[:, 0]

        bn_features = self.bn(features)

        return features, bn_features

def build_vit(args):
    return VisionTransformer(args)