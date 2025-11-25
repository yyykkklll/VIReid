"""
Testing Pipeline for PF-MGCD
实现PF-MGCD的测试和评估流程
"""

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


def extract_features(model, dataloader, device, pool_parts=True):
    """
    提取所有样本的特征
    Args:
        model: PF_MGCD模型
        dataloader: 数据加载器
        device: 设备
        pool_parts: 是否合并部件特征
    Returns:
        features: 特征矩阵 [N, D]
        labels: 标签 [N]
        cam_ids: 相机ID [N]
    """
    model.eval()

    features_list = []
    labels_list = []
    cam_ids_list = []

    print("Extracting features...")
    with torch.no_grad():
        for batch_data in tqdm(dataloader):
            # 解包数据
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
    """
    计算查询集和检索集之间的距离矩阵
    Args:
        query_features: 查询特征 [N_q, D]
        gallery_features: 检索特征 [N_g, D]
        metric: 距离度量 ('euclidean' or 'cosine')
    Returns:
        dist_matrix: 距离矩阵 [N_q, N_g]
    """
    if metric == 'euclidean':
        # 欧氏距离
        dist_matrix = torch.cdist(query_features, gallery_features, p=2)
    elif metric == 'cosine':
        # 余弦距离
        query_features = F.normalize(query_features, p=2, dim=1)
        gallery_features = F.normalize(gallery_features, p=2, dim=1)
        dist_matrix = 1 - torch.mm(query_features, gallery_features.t())
    else:
        raise ValueError(f"Unknown metric: {metric}")

    return dist_matrix


def evaluate_rank(dist_matrix, query_labels, gallery_labels,
                  query_cam_ids, gallery_cam_ids, max_rank=20):
    """
    计算CMC和mAP
    Args:
        dist_matrix: 距离矩阵 [N_q, N_g]
        query_labels: 查询标签 [N_q]
        gallery_labels: 检索标签 [N_g]
        query_cam_ids: 查询相机ID [N_q]
        gallery_cam_ids: 检索相机ID [N_g]
        max_rank: 最大rank
    Returns:
        cmc: CMC曲线 [max_rank]
        mAP: 平均精度均值
        mINP: 平均逆负惩罚
    """
    num_query = dist_matrix.shape[0]
    num_gallery = dist_matrix.shape[1]

    # 转换为numpy
    dist_matrix = dist_matrix.numpy()
    query_labels = query_labels.numpy()
    gallery_labels = gallery_labels.numpy()
    query_cam_ids = query_cam_ids.numpy()
    gallery_cam_ids = gallery_cam_ids.numpy()

    # 初始化结果
    all_cmc = []
    all_AP = []
    all_INP = []

    for i in range(num_query):
        # 获取当前查询
        q_label = query_labels[i]
        q_cam = query_cam_ids[i]

        # 计算匹配
        matches = (gallery_labels == q_label)

        # 移除同一相机下的同一身份（如果需要）
        junk_mask = (gallery_labels == q_label) & (gallery_cam_ids == q_cam)
        matches = matches & ~junk_mask

        if matches.sum() == 0:
            continue

        # 获取排序索引
        indices = np.argsort(dist_matrix[i])
        matches = matches[indices]

        # 计算CMC
        cmc = matches.cumsum()
        cmc[cmc > 1] = 1
        all_cmc.append(cmc[:max_rank])

        # 计算AP
        num_rel = matches.sum()
        tmp_cmc = matches.cumsum()
        tmp_cmc = [x / (i + 1.0) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * matches
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

        # 计算INP
        pos_indices = np.where(matches)[0]
        if len(pos_indices) > 0:
            INP = pos_indices[0] / (2 * len(pos_indices))
            all_INP.append(INP)

    # [关键修复] 检查是否有有效匹配结果
    if len(all_cmc) == 0:
        print("Warning: No valid matches found during evaluation.")
        return np.zeros(max_rank), 0.0, 0.0

    # 计算平均值
    all_cmc = np.asarray(all_cmc).astype(np.float32)
    cmc = all_cmc.mean(axis=0)
    mAP = np.mean(all_AP)
    mINP = np.mean(all_INP) if len(all_INP) > 0 else 0.0

    return cmc, mAP, mINP


def test(model, test_loader, args, device):
    """
    测试模型性能
    Args:
        model: PF_MGCD模型
        test_loader: 测试数据加载器（包含query和gallery）
        args: 参数配置
        device: 设备
    Returns:
        rank1: Rank-1准确率
        mAP: 平均精度均值
    """
    print("\n" + "=" * 50)
    print("Starting Evaluation")
    print("=" * 50)

    # 提取特征
    pool_parts = args.pool_parts if hasattr(args, 'pool_parts') else True
    features, labels, cam_ids = extract_features(model, test_loader, device, pool_parts)

    print(f"\nExtracted features shape: {features.shape}")
    print(f"Number of identities: {len(labels.unique())}")

    # 分割query和gallery
    # 假设前半部分是query，后半部分是gallery (标准SYSU处理方式)
    num_samples = features.shape[0]
    num_query = num_samples // 2

    query_features = features[:num_query]
    query_labels = labels[:num_query]
    query_cam_ids = cam_ids[:num_query]

    gallery_features = features[num_query:]
    gallery_labels = labels[num_query:]
    gallery_cam_ids = cam_ids[num_query:]

    print(f"\nQuery samples: {num_query}")
    print(f"Gallery samples: {num_samples - num_query}")

    # 计算距离矩阵
    print("\nComputing distance matrix...")
    metric = args.distance_metric if hasattr(args, 'distance_metric') else 'euclidean'
    dist_matrix = compute_distance_matrix(query_features, gallery_features, metric)

    # 评估
    print("\nEvaluating...")
    cmc, mAP, mINP = evaluate_rank(
        dist_matrix, query_labels, gallery_labels,
        query_cam_ids, gallery_cam_ids, max_rank=20
    )

    # 打印结果
    print("\n" + "=" * 50)
    print("Evaluation Results")
    print("=" * 50)
    print(f"mAP: {mAP * 100:.2f}%")
    print(f"mINP: {mINP * 100:.2f}%")
    print(f"CMC Scores:")

    # 安全打印 Rank
    for r in [1, 5, 10, 20]:
        if r <= len(cmc):
            print(f"  Rank-{r:2d}: {cmc[r - 1] * 100:.2f}%")
        else:
            print(f"  Rank-{r:2d}: 0.00%")

    print("=" * 50 + "\n")

    # 安全返回
    rank1 = cmc[0] * 100 if len(cmc) > 0 else 0.0
    return rank1, mAP * 100


def test_cross_modality(model, query_loader, gallery_loader, args, device):
    """
    跨模态测试（可见光查询 vs 红外检索，或反之）
    """
    print("\n" + "=" * 50)
    print("Starting Cross-Modality Evaluation")
    print("=" * 50)

    # 提取查询特征
    print("\n--- Extracting Query Features ---")
    pool_parts = args.pool_parts if hasattr(args, 'pool_parts') else True
    query_features, query_labels, query_cam_ids = extract_features(
        model, query_loader, device, pool_parts
    )

    # 提取检索特征
    print("\n--- Extracting Gallery Features ---")
    gallery_features, gallery_labels, gallery_cam_ids = extract_features(
        model, gallery_loader, device, pool_parts
    )

    print(f"\nQuery samples: {query_features.shape[0]}")
    print(f"Gallery samples: {gallery_features.shape[0]}")

    # 计算距离矩阵
    print("\nComputing distance matrix...")
    metric = args.distance_metric if hasattr(args, 'distance_metric') else 'euclidean'
    dist_matrix = compute_distance_matrix(query_features, gallery_features, metric)

    # 评估
    print("\nEvaluating...")
    cmc, mAP, mINP = evaluate_rank(
        dist_matrix, query_labels, gallery_labels,
        query_cam_ids, gallery_cam_ids, max_rank=20
    )

    # 打印结果
    print("\n" + "=" * 50)
    print("Cross-Modality Evaluation Results")
    print("=" * 50)
    print(f"mAP: {mAP * 100:.2f}%")
    print(f"mINP: {mINP * 100:.2f}%")
    print(f"CMC Scores:")
    for r in [1, 5, 10, 20]:
        if r <= len(cmc):
            print(f"  Rank-{r:2d}: {cmc[r - 1] * 100:.2f}%")
    print("=" * 50 + "\n")

    rank1 = cmc[0] * 100 if len(cmc) > 0 else 0.0
    return rank1, mAP * 100
