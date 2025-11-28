"""
PF-MGCD 测试与评估脚本
功能：
1. 提取 Query 和 Gallery 特征
2. 计算距离矩阵 (Cosine/Euclidean)
3. 评估 CMC, mAP 和 mINP 指标
"""

import os
import time
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


def extract_features(model, dataloader, device, pool_parts=True):
    """
    提取特征
    """
    model.eval()
    
    features_list = []
    labels_list = []
    cam_ids_list = []
    
    with torch.no_grad():
        for batch_data in tqdm(dataloader, desc="Extracting features"):
            if len(batch_data) == 3:
                images, labels, cam_ids = batch_data
            else:
                images, labels = batch_data[:2]
                cam_ids = torch.zeros(labels.size(0), dtype=torch.long)
            
            images = images.to(device)
            
            # 提取特征
            batch_features = model.extract_features(images, pool_parts=pool_parts)
            
            # 如果返回的是列表 (多部件特征)，拼接它们
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
    """
    计算距离矩阵
    """
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
    """
    评估检索性能 (CMC, mAP, mINP)
    """
    num_query = dist_matrix.shape[0]
    
    # 转换为 numpy
    dist_matrix = dist_matrix.numpy()
    query_labels = query_labels.numpy()
    gallery_labels = gallery_labels.numpy()
    query_cam_ids = query_cam_ids.numpy()
    gallery_cam_ids = gallery_cam_ids.numpy()
    
    all_cmc = []
    all_AP = []
    all_INP = []
    
    num_valid_queries = 0
    
    for i in range(num_query):
        q_label = query_labels[i]
        q_cam = query_cam_ids[i]
        
        # 1. 找到匹配样本
        matches = (gallery_labels == q_label)
        
        # 2. 过滤掉同相机下的同ID样本 (Junk)
        junk_mask = (gallery_labels == q_label) & (gallery_cam_ids == q_cam)
        valid_matches = matches & ~junk_mask
        
        # 如果没有有效匹配，跳过
        if valid_matches.sum() == 0:
            if matches.sum() > 0:
                valid_matches = matches
            else:
                continue
        
        num_valid_queries += 1
        
        # 3. 按距离排序
        indices = np.argsort(dist_matrix[i])
        valid_matches = valid_matches[indices]
        
        # --- 计算 CMC ---
        cmc = valid_matches.cumsum()
        cmc[cmc > 1] = 1
        all_cmc.append(cmc[:max_rank])
        
        # --- 计算 mAP ---
        num_rel = valid_matches.sum()
        tmp_cmc = valid_matches.cumsum()
        tmp_cmc = [x / (i + 1.0) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * valid_matches
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)
        
        # --- 计算 mINP (修复版) ---
        # [修复] 移除了原有的 2.0 系数
        # 修正公式: 1.0 / first_pos_rank (根据用户要求)
        pos_indices = np.where(valid_matches)[0]
        if len(pos_indices) > 0:
            first_pos_rank = pos_indices[0] + 1  # rank从1开始
            INP = 1.0 / first_pos_rank
            all_INP.append(INP)
    
    if len(all_cmc) == 0:
        print("Warning: No valid matches found during evaluation.")
        return np.zeros(max_rank), 0.0, 0.0
    
    print(f"  Valid queries: {num_valid_queries}/{num_query}")
    
    all_cmc = np.asarray(all_cmc).astype(np.float32)
    cmc = all_cmc.mean(axis=0)
    mAP = np.mean(all_AP)
    mINP = np.mean(all_INP) if len(all_INP) > 0 else 0.0
    
    return cmc, mAP, mINP


def test(model, query_loader, gallery_loaders, args, device):
    """
    跨模态测试主流程
    """
    print("\n" + "="*50)
    print(f"Starting Evaluation on {args.dataset.upper()} Dataset")
    print("="*50)
    
    pool_parts = args.pool_parts if hasattr(args, 'pool_parts') else True
    
    # 1. 提取 Query 特征
    print("\n[1/3] Extracting query features...")
    query_features, query_labels, query_cam_ids = extract_features(
        model, query_loader, device, pool_parts
    )
    print(f"  Query features shape: {query_features.shape}")
    print(f"  Query samples: {len(query_labels)}")
    print(f"  Query unique IDs: {len(query_labels.unique())}")
    
    # 2. 提取 Gallery 特征
    print("\n[2/3] Extracting gallery features...")
    gallery_features_list = []
    gallery_labels_list = []
    gallery_cam_ids_list = []
    
    for idx, gallery_loader in enumerate(gallery_loaders):
        print(f"  Processing gallery {idx+1}/{len(gallery_loaders)}...")
        g_feat, g_labels, g_cam_ids = extract_features(
            model, gallery_loader, device, pool_parts
        )
        gallery_features_list.append(g_feat)
        gallery_labels_list.append(g_labels)
        gallery_cam_ids_list.append(g_cam_ids)
    
    # 合并所有 Gallery
    gallery_features = torch.cat(gallery_features_list, dim=0)
    gallery_labels = torch.cat(gallery_labels_list, dim=0)
    gallery_cam_ids = torch.cat(gallery_cam_ids_list, dim=0)
    
    print(f"  Gallery features shape: {gallery_features.shape}")
    print(f"  Gallery samples: {len(gallery_labels)}")
    
    # 3. 计算距离矩阵并评估
    print("\n[3/3] Computing distance matrix and evaluating...")
    metric = args.distance_metric if hasattr(args, 'distance_metric') else 'euclidean'
    dist_matrix = compute_distance_matrix(query_features, gallery_features, metric)
    
    # 评估指标
    cmc, mAP, mINP = evaluate_rank(
        dist_matrix, query_labels, gallery_labels,
        query_cam_ids, gallery_cam_ids, max_rank=20
    )
    
    # 打印结果
    print("\n" + "="*50)
    print("Evaluation Results")
    print("="*50)
    print(f"mAP:  {mAP*100:.2f}%")
    print(f"mINP: {mINP*100:.2f}%")
    print(f"CMC Scores:")
    for r in [1, 5, 10, 20]:
        if r <= len(cmc):
            print(f"  Rank-{r:2d}: {cmc[r-1]*100:.2f}%")
        else:
            print(f"  Rank-{r:2d}: 0.00%")
    print("="*50 + "\n")
    
    rank1 = cmc[0] * 100 if len(cmc) > 0 else 0.0
    return rank1, mAP * 100, mINP * 100