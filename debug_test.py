import torch
from datasets.regdb import RegDB
from models import PF_MGCD
import argparse

args = argparse.Namespace(
    dataset='regdb', data_path='./datasets', trial=1,
    num_workers=4, pid_numsample=4, batch_pidnum=8,
    test_batch=64, img_w=144, img_h=288,
    relabel=False, search_mode='all', gall_mode='single',
    num_classes=206, num_parts=6, feature_dim=256,
    pool_parts=True, distance_metric='cosine'
)

# 加载数据
dataset = RegDB(args)

# 加载模型
device = torch.device('cuda')
model = PF_MGCD(num_parts=6, num_identities=206, feature_dim=256).to(device)
checkpoint = torch.load('./checkpoints/regdb/pfmgcd_epoch10.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 测试一个batch
query_batch = next(iter(dataset.query_loader))
imgs, labels, cams = query_batch
imgs = imgs.to(device)

with torch.no_grad():
    features = model.extract_features(imgs, pool_parts=True)
    
print(f"Feature shape: {features.shape}")
print(f"Feature stats:")
print(f"  Min: {features.min():.4f}")
print(f"  Max: {features.max():.4f}")
print(f"  Mean: {features.mean():.4f}")
print(f"  Std: {features.std():.4f}")
print(f"  Norm (should be ~1.0): {features.norm(dim=1).mean():.4f}")
