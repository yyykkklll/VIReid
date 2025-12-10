import torch
import numpy as np
from utils import fliplr

def extract_features(loader, model, modal='rgb'):
    ptr = 0
    feat_dim = 2048 # ResNet50 GAP output
    features = np.zeros((len(loader.dataset), feat_dim))
    
    with torch.no_grad():
        for batch_idx, (input, label) in enumerate(loader):
            batch_num = input.size(0)
            input = input.to(model.device)
            
            # [修正] 去掉 .model，直接调用 backbone 实例
            if modal == 'rgb':
                gap_feat, _ = model.backbone(x1=input)
            else:
                gap_feat, _ = model.backbone(x2=input)
                
            # Flip Augmentation
            input_flip = fliplr(input)
            if modal == 'rgb':
                gap_flip, _ = model.backbone(x1=input_flip)
            else:
                gap_flip, _ = model.backbone(x2=input_flip)
            
            feat = (gap_feat + gap_flip) / 2
            
            # 归一化
            feat = torch.nn.functional.normalize(feat, dim=1)
            
            features[ptr:ptr + batch_num, :] = feat.detach().cpu().numpy()
            ptr = ptr + batch_num
            
    return features

def test(args, model, dataset, current_epoch=0):
    model.set_eval()
    
    # ---------------- RegDB Evaluation ----------------
    # Query: Thermal, Gallery: Visible
    query_loader = dataset.query_loader
    gallery_loader = dataset.gallery_loader
    
    print("Extracting IR Query Features...")
    query_feat = extract_features(query_loader, model, modal='ir')
    print("Extracting RGB Gallery Features...")
    gall_feat = extract_features(gallery_loader, model, modal='rgb')
    
    query_label = dataset.query.test_label
    gall_label = dataset.gallery.test_label
    
    # 计算距离
    distmat = - np.matmul(query_feat, gall_feat.T)
    
    cmc, mAP, mINP = eval_regdb(distmat, query_label, gall_label)
    
    return cmc, mAP, mINP

def eval_regdb(distmat, q_pids, g_pids, max_rank=20):
    num_q, num_g = distmat.shape
    if num_g < max_rank: max_rank = num_g
    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    all_cmc = []
    all_AP = []
    all_INP = []
    num_valid_q = 0.

    for q_idx in range(num_q):
        raw_cmc = matches[q_idx]
        if not np.any(raw_cmc): continue

        cmc = raw_cmc.cumsum()
        pos_idx = np.where(raw_cmc == 1)
        pos_max_idx = np.max(pos_idx)
        inp = cmc[pos_max_idx]/ (pos_max_idx + 1.0)
        all_INP.append(inp)

        cmc[cmc > 1] = 1
        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        num_rel = raw_cmc.sum()
        tmp_cmc = raw_cmc.cumsum()
        tmp_cmc = [x / (i+1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * raw_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)
    mINP = np.mean(all_INP)
    return all_cmc, mAP, mINP