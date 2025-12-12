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
from utils import eval_regdb, eval_sysu, AverageMeter


def extract_ir_features(loader, model, num_samples):
    """Extract IR modality features with test-time augmentation"""
    ptr = 0
    ir_feat = np.zeros((num_samples, 2048))
    
    # 使用上下文管理器，确保异常安全
    with torch.no_grad():
        for batch_idx, (input, label) in enumerate(loader):
            batch_num = input.size(0)
            input = input.to(model.device)
            
            _, feat = model.model(None, input)
            
            # Test-time augmentation: horizontal flip
            imgs_flip = input.flip(-1)
            _, feat_flip = model.model(None, imgs_flip)
            
            feat = (feat + feat_flip) / 2.0
            
            ir_feat[ptr:ptr + batch_num, :] = feat.detach().cpu().numpy()
            ptr = ptr + batch_num
    
    return ir_feat


def extract_rgb_features(loader, model, num_samples):
    """Extract RGB modality features with test-time augmentation"""
    ptr = 0
    rgb_feat = np.zeros((num_samples, 2048))
    
    with torch.no_grad():
        for batch_idx, (input, label) in enumerate(loader):
            batch_num = input.size(0)
            input = input.to(model.device)
            
            _, feat = model.model(input, None)
            
            # Test-time augmentation: horizontal flip
            imgs_flip = input.flip(-1)
            _, feat_flip = model.model(imgs_flip, None)
            
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
    """
    model.set_eval()
    
    # 修复核心 BUG：
    # 移除全局 torch.set_grad_enabled(False/True)
    # 使用 with torch.no_grad() 块，确保即使发生异常退出，梯度状态也会自动恢复
    with torch.no_grad():
        all_cmc = 0
        all_mAP = 0
        all_mINP = 0
        
        if args.dataset == 'sysu' or args.dataset == 'llcm':
            # SYSU-MM01 or LLCM dataset
            print("Extracting Query Features...")
            query_loader = dataset.query_loader
            query_num = dataset.n_query
            
            if args.dataset == 'sysu' or args.test_mode == "t2v":
                query_feat = extract_ir_features(query_loader, model, query_num)
            elif args.dataset == 'llcm' and args.test_mode == "v2t":
                query_feat = extract_rgb_features(query_loader, model, query_num)
            
            query_feat = normalize(query_feat, axis=1)
            query_label = dataset.query.test_label
            query_cam = dataset.query.test_cam
            
            # Evaluate on 10 random gallery splits
            for i in range(10):
                print(f"Extracting Gallery Features... Trial {i+1}/10")
                gall_num = dataset.n_gallery
                gall_loader = dataset.gallery_loaders[i]
                
                if args.dataset == 'sysu' or args.test_mode == "t2v":
                    gall_feat = extract_rgb_features(gall_loader, model, gall_num)
                elif args.dataset == 'llcm' and args.test_mode == "v2t":
                    gall_feat = extract_ir_features(gall_loader, model, gall_num)
                
                gall_feat = normalize(gall_feat, axis=1)
                gall_label = dataset.gall_info[i][0]
                gall_cam = dataset.gall_info[i][1]
                
                # Compute similarity (cosine similarity)
                distmat = np.matmul(query_feat, gall_feat.T)
                
                # Evaluate
                if args.dataset == 'sysu':
                    cmc, mAP, mINP = eval_sysu(-distmat, query_label, gall_label, 
                                            query_cam, gall_cam)
                elif args.dataset == 'llcm':
                    cmc, mAP, mINP = eval_sysu(-distmat, query_label, gall_label, 
                                            query_cam, gall_cam)
                
                all_cmc += cmc
                all_mAP += mAP
                all_mINP += mINP
            
            all_cmc = all_cmc / 10
            all_mAP = all_mAP / 10
            all_mINP = all_mINP / 10

        elif args.dataset == 'regdb':
            # RegDB dataset
            print("Extracting Query Features (IR)...")
            ir_loader = dataset.query_loader
            query_num = dataset.n_query
            ir_feat = extract_ir_features(ir_loader, model, query_num)
            ir_label = dataset.query.test_label
            
            print("Extracting Gallery Features (RGB)...")
            rgb_loader = dataset.gallery_loader
            gall_num = dataset.n_gallery
            rgb_feat = extract_rgb_features(rgb_loader, model, gall_num)
            rgb_label = dataset.gallery.test_label
            
            # Normalize features
            ir_feat = normalize(ir_feat, axis=1)
            rgb_feat = normalize(rgb_feat, axis=1)
            
            # Compute similarity and evaluate
            if args.test_mode == "t2v" or args.test_mode == "all":
                print("Computing T2V (IR->RGB) similarity...")
                distmat = np.matmul(ir_feat, rgb_feat.T)
                cmc, mAP, mINP = eval_regdb(-distmat, ir_label, rgb_label)
            elif args.test_mode == "v2t":
                print("Computing V2T (RGB->IR) similarity...")
                distmat = np.matmul(rgb_feat, ir_feat.T)
                cmc, mAP, mINP = eval_regdb(-distmat, rgb_label, ir_label)
            
            all_cmc, all_mAP, all_mINP = cmc, mAP, mINP
    
    # 恢复训练模式（可选，通常在 train loop 中会再次调用 model.set_train()）
    return all_cmc, all_mAP, all_mINP