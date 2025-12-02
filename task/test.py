import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import sys
import os

# 确保能导入 utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import re_ranking

def extract_features(model, dataloader, device, pool_parts=True):
    model.eval()
    features_list, labels_list, cam_ids_list = [], [], []
    
    with torch.no_grad():
        for batch_data in tqdm(dataloader, desc="Extracting"):
            if len(batch_data) == 3:
                images, labels, cam_ids = batch_data
            else:
                images, labels = batch_data[:2]
                cam_ids = torch.zeros(labels.size(0), dtype=torch.long)
            
            images = images.to(device)
            batch_features = model.extract_features(images, pool_parts=pool_parts)
            
            features_list.append(batch_features.cpu())
            labels_list.append(labels)
            cam_ids_list.append(cam_ids)
    
    return torch.cat(features_list), torch.cat(labels_list), torch.cat(cam_ids_list)

def test(model, query_loader, gallery_loaders, args, device):
    print("\n" + "="*50)
    print(f"Starting Evaluation (With Re-Ranking)")
    
    # 1. 提取特征
    q_feat, q_labels, q_cams = extract_features(model, query_loader, device, args.pool_parts)
    
    g_feats, g_labels, g_cams = [], [], []
    for g_loader in gallery_loaders:
        gf, gl, gc = extract_features(model, g_loader, device, args.pool_parts)
        g_feats.append(gf)
        g_labels.append(gl)
        g_cams.append(gc)
    
    g_feat = torch.cat(g_feats)
    g_label = torch.cat(g_labels)
    g_cam = torch.cat(g_cams)
    
    print(f"Query: {q_feat.shape}, Gallery: {g_feat.shape}")
    
    # 2. 计算距离 (强制使用 Re-Ranking)
    print("Applying Re-Ranking (k-reciprocal)...")
    try:
        # Re-ranking 使用欧氏距离，通常输入未归一化的特征效果也不错，
        # 但既然网络输出已归一化，直接用即可。
        dist_mat = re_ranking(q_feat, g_feat, k1=20, k2=6, lambda_value=0.3)
    except Exception as e:
        print(f"Re-ranking failed: {e}. Falling back to Cosine.")
        q_feat = F.normalize(q_feat, p=2, dim=1)
        g_feat = F.normalize(g_feat, p=2, dim=1)
        dist_mat = 1 - torch.mm(q_feat, g_feat.t()).numpy()
    
    # 3. 评估
    return evaluate_rank(dist_mat, q_labels.numpy(), g_label.numpy(), q_cams.numpy(), g_cam.numpy())

def evaluate_rank(dist_matrix, query_labels, gallery_labels, query_cam_ids, gallery_cam_ids, max_rank=20):
    num_query = dist_matrix.shape[0]
    all_cmc, all_AP, all_INP = [], [], []
    
    for i in range(num_query):
        q_label = query_labels[i]
        q_cam = query_cam_ids[i]
        
        matches = (gallery_labels == q_label)
        junk_mask = (gallery_labels == q_label) & (gallery_cam_ids == q_cam)
        valid_matches = matches & ~junk_mask
        
        if valid_matches.sum() == 0: continue
        
        indices = np.argsort(dist_matrix[i])
        valid_matches = valid_matches[indices]
        
        # CMC
        cmc = valid_matches.cumsum()
        cmc[cmc > 1] = 1
        all_cmc.append(cmc[:max_rank])
        
        # AP
        num_rel = valid_matches.sum()
        tmp_cmc = valid_matches.cumsum()
        tmp_cmc = [x / (i + 1.0) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * valid_matches
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)
        
        # INP
        pos_indices = np.where(valid_matches)[0]
        if len(pos_indices) > 0:
            INP = 1.0 / (pos_indices[0] + 1)
            all_INP.append(INP)
            
    if len(all_cmc) == 0: return 0.0, 0.0, 0.0
    
    cmc = np.asarray(all_cmc).astype(np.float32).mean(axis=0)
    mAP = np.mean(all_AP)
    mINP = np.mean(all_INP)
    
    print(f"Rank-1: {cmc[0]*100:.2f}%, mAP: {mAP*100:.2f}%, mINP: {mINP*100:.2f}%")
    return cmc[0]*100, mAP*100, mINP*100