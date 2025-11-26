import torch
import torch.nn.functional as F
from datasets.regdb import RegDB
from models import PF_MGCD
import argparse
import numpy as np

args = argparse.Namespace(
    dataset='regdb', data_path='./datasets', trial=1,
    num_workers=4, pid_numsample=4, batch_pidnum=8,
    test_batch=64, img_w=144, img_h=288,
    relabel=False, search_mode='all', gall_mode='single',
    num_classes=206, num_parts=6, feature_dim=256,
    pool_parts=True, distance_metric='cosine'
)

# 加载数据和模型
dataset = RegDB(args)
device = torch.device('cuda')
model = PF_MGCD(num_parts=6, num_identities=206, feature_dim=256).to(device)
checkpoint = torch.load('./checkpoints/regdb/pfmgcd_epoch10.pth', weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print("Extracting features...")

# 提取query特征 (只取前100个样本加速)
query_features = []
query_labels = []
with torch.no_grad():
    for i, (imgs, labels, cams) in enumerate(dataset.query_loader):
        if i >= 2:  # 只取2个batch
            break
        imgs = imgs.to(device)
        feats = model.extract_features(imgs, pool_parts=True)
        query_features.append(feats.cpu())
        query_labels.append(labels)

query_features = torch.cat(query_features, dim=0)
query_labels = torch.cat(query_labels, dim=0)

# 提取gallery特征
gallery_features = []
gallery_labels = []
with torch.no_grad():
    for i, (imgs, labels, cams) in enumerate(dataset.gallery_loader):
        if i >= 2:  # 只取2个batch
            break
        imgs = imgs.to(device)
        feats = model.extract_features(imgs, pool_parts=True)
        gallery_features.append(feats.cpu())
        gallery_labels.append(labels)

gallery_features = torch.cat(gallery_features, dim=0)
gallery_labels = torch.cat(gallery_labels, dim=0)

print(f"Query: {query_features.shape}, Gallery: {gallery_features.shape}")

# 计算余弦距离
query_features_norm = F.normalize(query_features, p=2, dim=1)
gallery_features_norm = F.normalize(gallery_features, p=2, dim=1)
cosine_sim = torch.mm(query_features_norm, gallery_features_norm.t())
cosine_dist = 1 - cosine_sim

print(f"\nCosine Distance Matrix:")
print(f"  Shape: {cosine_dist.shape}")
print(f"  Min: {cosine_dist.min():.4f}")
print(f"  Max: {cosine_dist.max():.4f}")
print(f"  Mean: {cosine_dist.mean():.4f}")
print(f"  Std: {cosine_dist.std():.4f}")

# 分析前5个query
print(f"\nAnalyzing first 5 queries:")
for i in range(min(5, len(query_labels))):
    q_label = query_labels[i].item()
    dists = cosine_dist[i].numpy()
    
    # 找到相同ID的gallery样本
    same_id_mask = (gallery_labels == q_label).numpy()
    same_id_dists = dists[same_id_mask]
    diff_id_dists = dists[~same_id_mask]
    
    # 最近的5个
    nearest_indices = np.argsort(dists)[:5]
    nearest_labels = gallery_labels[nearest_indices].numpy()
    nearest_dists = dists[nearest_indices]
    
    print(f"\nQuery {i} (ID={q_label}):")
    print(f"  Nearest 5 labels: {nearest_labels}")
    print(f"  Nearest 5 dists:  {nearest_dists}")
    print(f"  Same ID in gallery: {same_id_mask.sum()} samples")
    if same_id_dists.size > 0:
        print(f"  Same ID dist: min={same_id_dists.min():.4f}, mean={same_id_dists.mean():.4f}")
    if diff_id_dists.size > 0:
        print(f"  Diff ID dist: min={diff_id_dists.min():.4f}, mean={diff_id_dists.mean():.4f}")
    print(f"  Correct? {q_label in nearest_labels}")
