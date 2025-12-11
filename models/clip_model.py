"""
CLIP 模型加载模块（超简化版本）
================================
策略：直接使用官方 CLIP 包，避免所有自定义代码的问题
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def load_clip_to_cpu(backbone_name, h_resolution, w_resolution, vision_stride_size):
    """
    加载官方 CLIP 并调整位置编码
    
    Args:
        backbone_name: 'RN50', 'RN101', 'ViT-B/16'
        h_resolution: 目标特征图高度
        w_resolution: 目标特征图宽度
        vision_stride_size:  保留用于兼容（未使用）
    
    Returns:
        CLIP 模型
        
    注意：
        官方 CLIP 的 encode_image() 已经返回最终池化后的特征
        ResNet:  [batch, 1024]
        ViT:  [batch, 512/768]
    """
    print(f"   Loading official CLIP:  {backbone_name}")
    print(f"   Target feature map:  {h_resolution}x{w_resolution}")
    
    try:
        # 尝试导入官方 CLIP
        import clip as official_clip
    except ImportError:
        print("   ❌ Official CLIP not found. Installing...")
        import subprocess
        import sys
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "git+https://github.com/openai/CLIP.git"
        ])
        import clip as official_clip
    
    # 加载模型
    device = "cpu"
    model, preprocess = official_clip.load(backbone_name, device=device)
    
    print(f"   ✅ Model loaded")
    
    # 调整位置编码
    if hasattr(model.visual, 'attnpool'):
        # ResNet 分支
        old_pos_embed = model.visual. attnpool.positional_embedding.data
        print(f"   Original position embedding: {old_pos_embed.shape}")
        
        cls_pos = old_pos_embed[0:1]
        spatial_pos = old_pos_embed[1:]
        
        old_tokens = spatial_pos.shape[0]
        old_size = int(old_tokens ** 0.5)
        
        if old_size * old_size != old_tokens: 
            raise ValueError(f"Invalid position embedding size: {old_tokens}")
        
        new_tokens = h_resolution * w_resolution
        
        if new_tokens != old_tokens:
            print(f"   Resizing:  {old_size}x{old_size} -> {h_resolution}x{w_resolution}")
            
            dim = spatial_pos.shape[-1]
            spatial_pos = spatial_pos.reshape(1, old_size, old_size, dim).permute(0, 3, 1, 2)
            spatial_pos = F.interpolate(
                spatial_pos,
                size=(h_resolution, w_resolution),
                mode='bilinear',
                align_corners=False
            )
            spatial_pos = spatial_pos.permute(0, 2, 3, 1).reshape(-1, dim)
            
            new_pos_embed = torch. cat([cls_pos, spatial_pos], dim=0)
            model.visual.attnpool.positional_embedding = nn.Parameter(new_pos_embed)
            
            print(f"   New position embedding: {new_pos_embed.shape}")
        else:
            print(f"   Position embedding size matches")
    
    elif hasattr(model.visual, 'positional_embedding'):
        # ViT 分支
        old_pos_embed = model.visual.positional_embedding.data
        cls_pos = old_pos_embed[0:1]
        spatial_pos = old_pos_embed[1:]
        
        old_tokens = spatial_pos.shape[0]
        old_size = int(old_tokens ** 0.5)
        new_tokens = h_resolution * w_resolution
        
        if new_tokens != old_tokens: 
            print(f"   Resizing ViT: {old_size}x{old_size} -> {h_resolution}x{w_resolution}")
            
            dim = spatial_pos.shape[-1]
            spatial_pos = spatial_pos.reshape(1, old_size, old_size, dim).permute(0, 3, 1, 2)
            spatial_pos = F.interpolate(spatial_pos, size=(h_resolution, w_resolution), mode='bilinear', align_corners=False)
            spatial_pos = spatial_pos.permute(0, 2, 3, 1).reshape(-1, dim)
            
            new_pos_embed = torch.cat([cls_pos, spatial_pos], dim=0)
            model.visual.positional_embedding = nn.Parameter(new_pos_embed)
    
    print(f"   ✅ CLIP loaded and configured successfully")
    
    model.eval()
    return model


# ==================== 占位类 ====================

class CLIP(nn.Module):
    """占位类（已弃用）"""
    def __init__(self, args):
        super(CLIP, self).__init__()
        print("⚠️ CLIP class is deprecated")
        self.dummy = nn.Linear(1, 1)
    
    def forward(self, x1=None, x2=None):
        raise NotImplementedError("Use load_clip_to_cpu() instead")


# ==================== 测试函数 ====================

def test_clip_loading():
    """测试 CLIP 加载"""
    configs = [(288, 144), (256, 128), (384, 192)]
    
    for h, w in configs:
        print(f"\n{'='*60}")
        print(f"Testing resolution: {h}x{w}")
        print(f"{'='*60}")
        
        h_res = (h - 32) // 32 + 1
        w_res = (w - 32) // 32 + 1
        
        try:
            clip_model = load_clip_to_cpu('RN50', h_res, w_res, 32)
            
            # 测试前向传播
            dummy_input = torch.randn(2, 3, h, w)
            
            with torch.no_grad():
                # ✅ 官方 CLIP 的 encode_image 已经返回最终特征
                clip_feat = clip_model.encode_image(dummy_input)
            
            print(f"✅ Success!")
            print(f"   Input: {dummy_input.shape}")
            print(f"   CLIP features: {clip_feat.shape}")
            
            # 验证特征维度（RN50 输出 1024 维）
            expected_dim = 1024
            assert clip_feat.shape == (2, expected_dim), f"Expected (2, {expected_dim}), got {clip_feat.shape}"
            
            print(f"   ✅ All tests passed!")
            
        except Exception as e:
            print(f"❌ Failed: {e}")
            import traceback
            traceback.print_exc()


if __name__ == '__main__':
    test_clip_loading()