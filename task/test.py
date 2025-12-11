"""
Cross-Modal Person Re-Identification Testing Module
===================================================
Features: Feature extraction, distance computation, evaluation metrics
Author: Fixed Version
Date: 2025-12-11
"""

import torch
import numpy as np
from sklearn.preprocessing import normalize


def extract_ir_features(loader, model, num_samples):
    """
    Extract IR modality features with test-time augmentation
    
    Args:
        loader: Data loader
        model: Trained model
        num_samples: Total number of samples
        
    Returns:
        ir_feat: Normalized features [N, 2048]
    """
    ptr = 0
    ir_feat = np.zeros((num_samples, 2048))
    
    with torch.no_grad():
        for batch_idx, (input, label) in enumerate(loader):
            batch_num = input.size(0)
            input = input.to(model.device)
            
            # ✅ 修复1: 使用位置参数而非关键字参数
            _, feat = model.model(None, input)  # (x1=None, x2=input)
            
            # Test-time augmentation: horizontal flip
            imgs_flip = input.flip(-1)
            _, feat_flip = model.model(None, imgs_flip)
            
            # ✅ 修复2: 在BN特征层面融合，不经过分类器
            feat = (feat + feat_flip) / 2.0
            
            ir_feat[ptr:ptr + batch_num, :] = feat.detach().cpu().numpy()
            ptr = ptr + batch_num
    
    return ir_feat


def extract_rgb_features(loader, model, num_samples):
    """
    Extract RGB modality features with test-time augmentation
    
    Args:
        loader: Data loader
        model: Trained model
        num_samples: Total number of samples
        
    Returns:
        rgb_feat: Normalized features [N, 2048]
    """
    ptr = 0
    rgb_feat = np.zeros((num_samples, 2048))
    
    with torch.no_grad():
        for batch_idx, (input, label) in enumerate(loader):
            batch_num = input.size(0)
            input = input.to(model.device)
            
            # ✅ 修复1: 使用位置参数而非关键字参数
            _, feat = model.model(input, None)  # (x1=input, x2=None)
            
            # Test-time augmentation: horizontal flip
            imgs_flip = input.flip(-1)
            _, feat_flip = model.model(imgs_flip, None)
            
            # ✅ 修复2: 在BN特征层面融合，不经过分类器
            feat = (feat + feat_flip) / 2.0
            
            rgb_feat[ptr:ptr + batch_num, :] = feat.detach().cpu().numpy()
            ptr = ptr + batch_num
    
    return rgb_feat


def test(args, model, dataset, *epoch):
    """
    Main testing function
    
    Args:
        args: Configuration arguments
        model: Trained model
        dataset: Dataset object
        epoch: Optional epoch number for logging
        
    Returns:
        all_cmc: Cumulative Matching Characteristic curve
        all_mAP: Mean Average Precision
        all_mINP: Mean Inverse Negative Penalty
    """
    model.set_eval()
    torch.set_grad_enabled(False)
    
    all_cmc = 0
    all_mAP = 0
    all_mINP = 0
    
    if args.dataset == 'sysu' or args.dataset == 'llcm':
        # SYSU-MM01 or LLCM dataset
        query_loader = dataset.query_loader
        query_num = dataset.n_query
        
        # Extract query features
        if args.dataset == 'sysu' or args.test_mode == "t2v":
            query_feat = extract_ir_features(query_loader, model, query_num)
        elif args.dataset == 'llcm' and args.test_mode == "v2t":
            query_feat = extract_rgb_features(query_loader, model, query_num)
        
        # ✅ 修复3: 归一化特征
        query_feat = normalize(query_feat, axis=1)
        
        query_label = dataset.query.test_label
        query_cam = dataset.query.test_cam
        
        # Evaluate on 10 random gallery splits
        for i in range(10):
            gall_num = dataset.n_gallery
            gall_loader = dataset.gallery_loaders[i]
            
            # Extract gallery features
            if args.dataset == 'sysu' or args.test_mode == "t2v":
                gall_feat = extract_rgb_features(gall_loader, model, gall_num)
            elif args.dataset == 'llcm' and args.test_mode == "v2t":
                gall_feat = extract_ir_features(gall_loader, model, gall_num)
            
            # ✅ 修复3: 归一化特征
            gall_feat = normalize(gall_feat, axis=1)
            
            gall_label = dataset.gall_info[i][0]
            gall_cam = dataset.gall_info[i][1]
            
            # Compute similarity matrix (cosine similarity)
            distmat = np.matmul(query_feat, gall_feat.T)
            
            # Evaluate
            if args.dataset == 'sysu':
                cmc, mAP, mINP = eval_sysu(-distmat, query_label, gall_label, 
                                           query_cam, gall_cam)
            elif args.dataset == 'llcm':
                cmc, mAP, mINP = eval_llcm(-distmat, query_label, gall_label, 
                                           query_cam, gall_cam)
            
            all_cmc += cmc
            all_mAP += mAP
            all_mINP += mINP
        
        # Average over 10 trials
        all_cmc = all_cmc / 10
        all_mAP = all_mAP / 10
        all_mINP = all_mINP / 10

    elif args.dataset == 'regdb':
        # RegDB dataset
        ir_loader = dataset.query_loader
        query_num = dataset.n_query
        ir_feat = extract_ir_features(ir_loader, model, query_num)
        ir_label = dataset.query.test_label
        
        rgb_loader = dataset.gallery_loader
        gall_num = dataset.n_gallery
        rgb_feat = extract_rgb_features(rgb_loader, model, gall_num)
        rgb_label = dataset.gallery.test_label
        
        # ✅ 修复3: 归一化特征
        ir_feat = normalize(ir_feat, axis=1)
        rgb_feat = normalize(rgb_feat, axis=1)
        
        # Compute similarity and evaluate
        if args.test_mode == "t2v":
            distmat = np.matmul(ir_feat, rgb_feat.T)
            cmc, mAP, mINP = eval_regdb(-distmat, ir_label, rgb_label)
        elif args.test_mode == "v2t":
            distmat = np.matmul(rgb_feat, ir_feat.T)
            cmc, mAP, mINP = eval_regdb(-distmat, rgb_label, ir_label)
        
        all_cmc, all_mAP, all_mINP = cmc, mAP, mINP
    
    torch.set_grad_enabled(True)
    return all_cmc, all_mAP, all_mINP


def eval_sysu(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=20):
    """
    Evaluation with SYSU-MM01 protocol
    
    Key: Remove gallery samples from the same camera as query when 
    query is from camera 3 and gallery is from camera 2.
    
    Args:
        distmat: Distance matrix [num_query, num_gallery]
        q_pids: Query person IDs
        g_pids: Gallery person IDs
        q_camids: Query camera IDs
        g_camids: Gallery camera IDs
        max_rank: Maximum rank for CMC computation
        
    Returns:
        new_all_cmc: CMC curve
        mAP: Mean Average Precision
        mINP: Mean Inverse Negative Penalty
    """
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
        print(f"Note: number of gallery samples is quite small, got {num_g}")
    
    indices = np.argsort(distmat, axis=1)
    pred_label = g_pids[indices]
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    new_all_cmc = []
    all_cmc = []
    all_AP = []
    all_INP = []
    num_valid_q = 0.
    
    for q_idx in range(num_q):
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        order = indices[q_idx]
        remove = (q_camid == 3) & (g_camids[order] == 2)
        keep = np.invert(remove)
        
        # Compute single-gallery-shot CMC
        new_cmc = pred_label[q_idx][keep]
        new_index = np.unique(new_cmc, return_index=True)[1]
        new_cmc = [new_cmc[index] for index in sorted(new_index)]
        
        new_match = (new_cmc == q_pid).astype(np.int32)
        new_cmc = new_match.cumsum()
        new_all_cmc.append(new_cmc[:max_rank])
        
        # Compute multi-gallery-shot CMC
        orig_cmc = matches[q_idx][keep]
        if not np.any(orig_cmc):
            continue

        cmc = orig_cmc.cumsum()

        # Compute mINP
        pos_idx = np.where(orig_cmc == 1)
        pos_max_idx = np.max(pos_idx)
        inp = cmc[pos_max_idx] / (pos_max_idx + 1.0)
        all_INP.append(inp)

        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # Compute average precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"
    
    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    
    new_all_cmc = np.asarray(new_all_cmc).astype(np.float32)
    new_all_cmc = new_all_cmc.sum(0) / num_valid_q
    
    mAP = np.mean(all_AP)
    mINP = np.mean(all_INP)
    
    return new_all_cmc, mAP, mINP


def eval_llcm(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=20):
    """
    Evaluation with LLCM protocol
    
    Key: Remove gallery samples that have the same person ID and camera ID 
    as the query (following the original dataset setting).
    
    Args:
        distmat: Distance matrix [num_query, num_gallery]
        q_pids: Query person IDs
        g_pids: Gallery person IDs
        q_camids: Query camera IDs
        g_camids: Gallery camera IDs
        max_rank: Maximum rank for CMC computation
        
    Returns:
        new_all_cmc: CMC curve
        mAP: Mean Average Precision
        mINP: Mean Inverse Negative Penalty
    """
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
        print(f"Note: number of gallery samples is quite small, got {num_g}")
    
    indices = np.argsort(distmat, axis=1)
    pred_label = g_pids[indices]
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)
    
    new_all_cmc = []
    all_cmc = []
    all_AP = []
    all_INP = []
    num_valid_q = 0.
    
    for q_idx in range(num_q):
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        # Compute single-gallery-shot CMC
        new_cmc = pred_label[q_idx][keep]
        new_index = np.unique(new_cmc, return_index=True)[1]
        new_cmc = [new_cmc[index] for index in sorted(new_index)]
        
        new_match = (new_cmc == q_pid).astype(np.int32)
        new_cmc = new_match.cumsum()
        new_all_cmc.append(new_cmc[:max_rank])
        
        # Compute multi-gallery-shot CMC
        orig_cmc = matches[q_idx][keep]
        if not np.any(orig_cmc):
            continue

        cmc = orig_cmc.cumsum()

        # Compute mINP
        pos_idx = np.where(orig_cmc == 1)
        pos_max_idx = np.max(pos_idx)
        inp = cmc[pos_max_idx] / (pos_max_idx + 1.0)
        all_INP.append(inp)

        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # Compute average precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"
    
    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    
    new_all_cmc = np.asarray(new_all_cmc).astype(np.float32)
    new_all_cmc = new_all_cmc.sum(0) / num_valid_q
    
    mAP = np.mean(all_AP)
    mINP = np.mean(all_INP)
    
    return new_all_cmc, mAP, mINP


def eval_regdb(distmat, q_pids, g_pids, max_rank=20):
    """
    Evaluation with RegDB protocol
    
    Key: Remove gallery samples with the same person ID as query.
    
    Args:
        distmat: Distance matrix [num_query, num_gallery]
        q_pids: Query person IDs
        g_pids: Gallery person IDs
        max_rank: Maximum rank for CMC computation
        
    Returns:
        all_cmc: CMC curve
        mAP: Mean Average Precision
        mINP: Mean Inverse Negative Penalty
    """
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
        print(f"Note: number of gallery samples is quite small, got {num_g}")
    
    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    all_cmc = []
    all_AP = []
    all_INP = []
    num_valid_q = 0.

    # RegDB assumes different cameras for query and gallery
    q_camids = np.ones(num_q).astype(np.int32)
    g_camids = 2 * np.ones(num_g).astype(np.int32)
    
    for q_idx in range(num_q):
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        raw_cmc = matches[q_idx][keep]
        if not np.any(raw_cmc):
            continue

        cmc = raw_cmc.cumsum()

        # Compute mINP
        pos_idx = np.where(raw_cmc == 1)
        pos_max_idx = np.max(pos_idx)
        inp = cmc[pos_max_idx] / (pos_max_idx + 1.0)
        all_INP.append(inp)

        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # Compute average precision
        num_rel = raw_cmc.sum()
        tmp_cmc = raw_cmc.cumsum()
        tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * raw_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)
    mINP = np.mean(all_INP)
    
    return all_cmc, mAP, mINP
