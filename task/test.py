"""
Testing Pipeline for PF-MGCD
实现PF-MGCD的测试和评估流程
"""

import os
import time
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


def extract_features(model, dataloader, device, pool_parts=True):
    model.eval()
    
    features_list = []
    labels_list = []
    cam_ids_list = []
    
    print("Extracting features...")
    with torch.no_grad():
        for batch_data in tqdm(dataloader):
            if len(batch_data) == 3:
                images, labels, cam_ids = batch_data
            else:
                images, labels = batch_data[:2]
                cam_ids = torch.zeros(labels.size(0), dtype=torch.long)
            
            images = images.to(device)
            
            # 提取特征
            batch_features = model.extract_features(images, pool_parts=pool_parts)
            
            # [兼容性修复] 如果返回的是列表，在特征维度拼接
            if isinstance(batch_features, (list, tuple)):
                batch_features = torch.cat(batch_features, dim=1)
            
            features_list.append(batch_features.cpu())
            labels_list.append(labels)
            cam_ids_list.append(cam_ids)
    
    # 合并所有特征
    features = torch.cat(features_list, dim=0)
    labels = torch.cat(labels_list, dim=0)
    cam_ids = torch.cat(cam_ids_list, dim=0)
    
    return features, labels, cam_ids


def compute_distance_matrix(query_features, gallery_features, metric='euclidean'):
    if metric == 'euclidean':
        dist_matrix = torch.cdist(query_features, gallery_features, p=2)
    elif metric == 'cosine':
        query_features = F.normalize(query_features, p=2, dim=1)
        gallery_features = F.normalize(gallery_features, p=2, dim=1)
        dist_matrix = 1 - torch.mm(query_features, gallery_features.t())
    else:
        raise ValueError(f"Unknown metric: {metric}")
    return dist_matrix


def evaluate_rank(dist_matrix, query_labels, gallery_labels, 
                  query_cam_ids, gallery_cam_ids, max_rank=20):
    num_query = dist_matrix.shape[0]
    
    # 转换为numpy
    dist_matrix = dist_matrix.numpy()
    query_labels = query_labels.numpy()
    gallery_labels = gallery_labels.numpy()
    query_cam_ids = query_cam_ids.numpy()
    gallery_cam_ids = gallery_cam_ids.numpy()
    
    all_cmc = []
    all_AP = []
    all_INP = []
    
    for i in range(num_query):
        q_label = query_labels[i]
        q_cam = query_cam_ids[i]
        
        matches = (gallery_labels == q_label)
        junk_mask = (gallery_labels == q_label) & (gallery_cam_ids == q_cam)
        matches = matches & ~junk_mask
        
        if matches.sum() == 0:
            continue
        
        indices = np.argsort(dist_matrix[i])
        matches = matches[indices]
        
        # CMC
        cmc = matches.cumsum()
        cmc[cmc > 1] = 1
        all_cmc.append(cmc[:max_rank])
        
        # AP
        num_rel = matches.sum()
        tmp_cmc = matches.cumsum()
        tmp_cmc = [x / (i + 1.0) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * matches
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)
        
        # INP
        pos_indices = np.where(matches)[0]
        if len(pos_indices) > 0:
            INP = pos_indices[0] / (2 * len(pos_indices))
            all_INP.append(INP)
    
    # [关键修复] 检查是否为空
    if len(all_cmc) == 0:
        print("Warning: No valid matches found during evaluation.")
        return np.zeros(max_rank), 0.0, 0.0
    
    all_cmc = np.asarray(all_cmc).astype(np.float32)
    cmc = all_cmc.mean(axis=0)
    mAP = np.mean(all_AP)
    mINP = np.mean(all_INP) if len(all_INP) > 0 else 0.0
    
    return cmc, mAP, mINP


def test(model, test_loader, args, device):
    print("\n" + "="*50)
    print("Starting Evaluation")
    print("="*50)
    
    pool_parts = args.pool_parts if hasattr(args, 'pool_parts') else True
    features, labels, cam_ids = extract_features(model, test_loader, device, pool_parts)
    
    print(f"\nExtracted features shape: {features.shape}")
    print(f"Number of identities: {len(labels.unique())}")
    
    # [关键修复] 打乱数据，确保 Query 和 Gallery 有重叠的 ID
    num_samples = features.shape[0]
    # 固定随机种子以保证验证结果的一致性
    g_cpu = torch.Generator()
    g_cpu.manual_seed(args.seed if hasattr(args, 'seed') else 42)
    perm = torch.randperm(num_samples, generator=g_cpu)
    
    features = features[perm]
    labels = labels[perm]
    cam_ids = cam_ids[perm]
    
    # 分割
    num_query = num_samples // 2
    query_features = features[:num_query]
    query_labels = labels[:num_query]
    query_cam_ids = cam_ids[:num_query]
    
    gallery_features = features[num_query:]
    gallery_labels = labels[num_query:]
    gallery_cam_ids = cam_ids[num_query:]
    
    print(f"\nQuery samples: {num_query}")
    print(f"Gallery samples: {num_samples - num_query}")
    
    print("\nComputing distance matrix...")
    metric = args.distance_metric if hasattr(args, 'distance_metric') else 'euclidean'
    dist_matrix = compute_distance_matrix(query_features, gallery_features, metric)
    
    print("\nEvaluating...")
    cmc, mAP, mINP = evaluate_rank(
        dist_matrix, query_labels, gallery_labels,
        query_cam_ids, gallery_cam_ids, max_rank=20
    )
    
    print("\n" + "="*50)
    print("Evaluation Results")
    print("="*50)
    print(f"mAP: {mAP*100:.2f}%")
    print(f"mINP: {mINP*100:.2f}%")
    print(f"CMC Scores:")
    for r in [1, 5, 10, 20]:
        if r <= len(cmc):
            print(f"  Rank-{r:2d}: {cmc[r-1]*100:.2f}%")
        else:
            print(f"  Rank-{r:2d}: 0.00%")
    print("="*50 + "\n")
    
    rank1 = cmc[0] * 100 if len(cmc) > 0 else 0.0
    return rank1, mAP * 100


def test_cross_modality(model, query_loader, gallery_loader, args, device):
    # 跨模态测试逻辑（通常不需要打乱，因为Loader已经分好了）
    # ... (保持原样或按需使用上面的 helper 函数)
    pass