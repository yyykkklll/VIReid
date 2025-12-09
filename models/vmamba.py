import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math
import os

# 检查 mamba 环境
try:
    from mamba_ssm import Mamba
except ImportError:
    Mamba = None
    print("【警告】未检测到 mamba_ssm，将无法正常构建网络！")

class CSSF_Fusion(nn.Module):
    """
    [创新点] 跨扫描状态融合 (Cross-Scan State Fusion)
    """
    def __init__(self, dim):
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(1, 1, dim)) 
        self.beta = nn.Parameter(torch.zeros(1, 1, dim))
        self.act = nn.Tanh() # 增加非线性，使融合更平滑

    def forward(self, x_rgb, x_ir):
        # x: [B, L, D]
        # 门控系数
        gate_rgb = 0.5 * self.act(self.alpha) + 0.5 # 0~1
        gate_ir = 0.5 * self.act(self.beta) + 0.5
        
        # 状态交换
        out_rgb = (1 - gate_rgb) * x_rgb + gate_rgb * x_ir
        out_ir = (1 - gate_ir) * x_ir + gate_ir * x_rgb
        return out_rgb, out_ir

class PatchEmbed(nn.Module):
    def __init__(self, img_size=256, patch_size=4, in_chans=3, embed_dim=96):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        x = self.norm(x)
        return x

class VSSBlock(nn.Module):
    def __init__(self, hidden_dim=0, drop_path=0., norm_layer=nn.LayerNorm, d_state=16):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        # 核心 Mamba 算子
        self.self_attention = Mamba(
            d_model=hidden_dim,
            d_state=d_state,
            d_conv=4,
            expand=2
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, input):
        x = input + self.drop_path(self.self_attention(self.ln_1(input)))
        return x

class PatchMerging(nn.Module):
    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

class VMamba_Stream(nn.Module):
    """
    单流 VMamba Backbone (匹配官方 VMamba-Tiny 结构)
    """
    def __init__(self, img_size=(256, 128), 
                 patch_size=4, in_chans=3, 
                 embed_dim=96, depths=[2, 2, 9, 2], 
                 dims=[96, 192, 384, 768], 
                 drop_path_rate=0.2):
        super().__init__()
        
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, 
                                      in_chans=in_chans, embed_dim=embed_dim)
        
        self.layers = nn.ModuleList()
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        
        # Build Layers (4 Stages)
        for i_layer in range(len(depths)):
            # 1. Layers (Blocks)
            blocks = nn.ModuleList([
                VSSBlock(
                    hidden_dim=dims[i_layer],
                    drop_path=dpr[sum(depths[:i_layer]) + i],
                    norm_layer=nn.LayerNorm
                )
                for i in range(depths[i_layer])
            ])
            
            # 2. Downsample (PatchMerging) - except last layer
            if i_layer < len(depths) - 1:
                # Calculate resolution for current stage
                curr_h = img_size[0] // (4 * (2**i_layer))
                curr_w = img_size[1] // (4 * (2**i_layer))
                downsample = PatchMerging((curr_h, curr_w), dims[i_layer])
            else:
                downsample = None
                
            self.layers.append(nn.ModuleDict({
                'blocks': blocks,
                'downsample': downsample
            }))

        self.norm = nn.LayerNorm(dims[-1])
        
    def forward_features(self, x):
        # 这个函数仅用于单流测试，实际训练在 DualVMamba 中手动控制
        pass

class DualVMamba(nn.Module):
    def __init__(self, args):
        super().__init__()
        # VMamba-Tiny Config
        self.img_h = args.img_h
        self.img_w = args.img_w
        
        # 定义双流
        # 注意: 这里的 config 必须严格匹配预训练权重
        cfg = dict(
            img_size=(args.img_h, args.img_w),
            embed_dim=96,
            depths=[2, 2, 9, 2],
            dims=[96, 192, 384, 768]
        )
        
        self.rgb_stream = VMamba_Stream(**cfg)
        self.ir_stream = VMamba_Stream(**cfg)
        
        # 融合模块 (插入在每个 Stage 结束前)
        self.fusions = nn.ModuleList([
            CSSF_Fusion(96),  # Stage 1
            CSSF_Fusion(192), # Stage 2
            CSSF_Fusion(384), # Stage 3
            CSSF_Fusion(768)  # Stage 4
        ])
        
        # Final Norm & Head
        self.bn = nn.BatchNorm1d(768)
        nn.init.constant_(self.bn.weight, 1.0)
        nn.init.constant_(self.bn.bias, 0.0)

    def load_pretrained(self, path):
        if not os.path.exists(path):
            print(f"【错误】未找到预训练权重: {path}")
            return
            
        print(f">>> Loading VMamba-Tiny weights from {path}")
        state_dict = torch.load(path, map_location='cpu')
        if 'model' in state_dict:
            state_dict = state_dict['model']
            
        # 构造新的 dict 适配双流
        new_dict = {}
        for k, v in state_dict.items():
            # 官方权重 key 类似于: layers.0.blocks.0...
            # 我们需要复制两份: rgb_stream.layers.0... 和 ir_stream.layers.0...
            
            # 1. 复制给 RGB
            new_key_rgb = "rgb_stream." + k
            new_dict[new_key_rgb] = v
            
            # 2. 复制给 IR
            new_key_ir = "ir_stream." + k
            new_dict[new_key_ir] = v
            
        # 加载
        msg = self.load_state_dict(new_dict, strict=False)
        # missing keys应该是 fusion 模块和 bn，这是正常的
        print(f">>> Weights loaded. Missing keys (expected for fusions/head): {len(msg.missing_keys)}")

    def forward(self, x1=None, x2=None):
        # 统一处理输入，支持单模态测试
        if x1 is None: x1 = x2 # 占位，实际上不会用到
        if x2 is None: x2 = x1
        
        # Patch Embed
        x_rgb = self.rgb_stream.patch_embed(x1)
        x_ir = self.ir_stream.patch_embed(x2)
        
        # 逐 Stage 交互
        # VMamba-Tiny 有 4 个 stages
        for stage_idx in range(4):
            layer_group_rgb = self.rgb_stream.layers[stage_idx]
            layer_group_ir = self.ir_stream.layers[stage_idx]
            
            # 1. Run Blocks
            for blk in layer_group_rgb['blocks']:
                x_rgb = blk(x_rgb)
            for blk in layer_group_ir['blocks']:
                x_ir = blk(x_ir)
                
            # 2. [Scheme A] CSSF Fusion
            # 在下采样之前进行状态交互
            x_rgb, x_ir = self.fusions[stage_idx](x_rgb, x_ir)
            
            # 3. Downsample (Patch Merging)
            if layer_group_rgb['downsample'] is not None:
                x_rgb = layer_group_rgb['downsample'](x_rgb)
                x_ir = layer_group_ir['downsample'](x_ir)
                
        # Final Norm
        x_rgb = self.rgb_stream.norm(x_rgb) # B, L, C
        x_ir = self.ir_stream.norm(x_ir)
        
        # Global Average Pooling
        feat_rgb = x_rgb.mean(dim=1)
        feat_ir = x_ir.mean(dim=1)
        
        # 根据输入情况返回
        # 训练时 (x1!=None, x2!=None) -> cat
        # 测试时 (单模态) -> return single
        
        # 这里的逻辑主要适配 task/train.py 和 task/test.py
        # 训练时我们传入了 cat 后的数据，这里需要拆分吗？
        # 不，train.py 传入的是 full_rgb 和 full_ir，都是完整的 batch
        
        # 简单处理：如果是训练模式，总是返回 cat
        if self.training:
            feat = torch.cat([feat_rgb, feat_ir], dim=0)
            return feat, self.bn(feat)
        else:
            # 测试模式，判断哪个是真的输入
            # 由于 DualVMamba 结构特殊，测试时我们建议只调用 model.forward_features 单流
            # 但为了兼容 main.py 接口，我们做个判断
            # main.py 测试时通常是一次传一个模态
            # 如果两个输入完全一样（上面做的占位），说明是单模态
            if torch.equal(x1, x2): 
                # 假设是测试 RGB 或 IR，由于参数共享或独立，这里有点歧义
                # 但我们的 DualVMamba 是双流独立的。
                # 测试脚本通常会分别测。这里为了简便，返回 RGB 流的结果
                # *注意*：实际部署时需根据 mode 选择 stream。
                # 鉴于 ReID 测试时通常只跑 model(x1=input)，我们在 forward 入口处改进
                return feat_rgb, self.bn(feat_rgb)
            else:
                 feat = torch.cat([feat_rgb, feat_ir], dim=0)
                 return feat, self.bn(feat)

    # 覆盖 forward 处理单模态测试的特殊情况
    def forward(self, x1=None, x2=None):
        if x1 is not None and x2 is None:
            # Test RGB
            x = self.rgb_stream.patch_embed(x1)
            for stage_idx in range(4):
                layer = self.rgb_stream.layers[stage_idx]
                for blk in layer['blocks']: x = blk(x)
                if layer['downsample']: x = layer['downsample'](x)
            x = self.rgb_stream.norm(x).mean(1)
            return x, self.bn(x)
            
        if x1 is None and x2 is not None:
            # Test IR
            x = self.ir_stream.patch_embed(x2)
            for stage_idx in range(4):
                layer = self.ir_stream.layers[stage_idx]
                for blk in layer['blocks']: x = blk(x)
                if layer['downsample']: x = layer['downsample'](x)
            x = self.ir_stream.norm(x).mean(1)
            return x, self.bn(x)
            
        # Training (Dual)
        x_rgb = self.rgb_stream.patch_embed(x1)
        x_ir = self.ir_stream.patch_embed(x2)
        
        for stage_idx in range(4):
            l_rgb = self.rgb_stream.layers[stage_idx]
            l_ir = self.ir_stream.layers[stage_idx]
            
            for blk in l_rgb['blocks']: x_rgb = blk(x_rgb)
            for blk in l_ir['blocks']: x_ir = blk(x_ir)
            
            # [Fusion]
            x_rgb, x_ir = self.fusions[stage_idx](x_rgb, x_ir)
            
            if l_rgb['downsample']:
                x_rgb = l_rgb['downsample'](x_rgb)
                x_ir = l_ir['downsample'](x_ir)
                
        feat_rgb = self.rgb_stream.norm(x_rgb).mean(1)
        feat_ir = self.ir_stream.norm(x_ir).mean(1)
        
        feat = torch.cat([feat_rgb, feat_ir], dim=0)
        return feat, self.bn(feat)

def build_vmamba(args):
    model = DualVMamba(args)
    # 自动加载
    path = './pretrained/vmamba_tiny.pth'
    model.load_pretrained(path)
    return model