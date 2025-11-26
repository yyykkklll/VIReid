"""
Adapter for original dataset loaders to work with PF-MGCD
将原有的数据加载器适配到PF-MGCD模型
"""

import torch
from torch.utils.data import DataLoader, Dataset


class UnifiedDataset(Dataset):
    """
    统一的数据集包装器
    将RGB和IR数据合并到一个数据集中
    """
    
    def __init__(self, rgb_dataset, ir_dataset):
        """
        Args:
            rgb_dataset: RGB训练数据集
            ir_dataset: IR训练数据集
        """
        self.rgb_dataset = rgb_dataset
        self.ir_dataset = ir_dataset
        
        # 合并样本
        self.total_len = len(rgb_dataset) + len(ir_dataset)
        
    def __len__(self):
        return self.total_len
    
    def __getitem__(self, idx):
        """
        返回格式: (image, label, cam_id)
        """
        if idx < len(self.rgb_dataset):
            # RGB样本
            data = self.rgb_dataset[idx]
            if isinstance(data, tuple) and len(data) == 3:
                # 训练模式: (img, aug_img, info)
                img, aug_img, info = data
                # info: [index, pid, camid, gt_label]
                label = int(info[1])
                cam_id = int(info[2])
                return img, label, cam_id
            else:
                # 测试模式: (img, info)
                img, info = data
                label = int(info[1])
                cam_id = int(info[2])
                return img, label, cam_id
        else:
            # IR样本
            ir_idx = idx - len(self.rgb_dataset)
            data = self.ir_dataset[ir_idx]
            if isinstance(data, tuple) and len(data) == 3:
                # 训练模式
                img, aug_img, info = data
                label = int(info[1])
                cam_id = int(info[2]) + 3  # IR相机ID偏移
                return img, label, cam_id
            else:
                # 测试模式
                img, info = data
                label = int(info[1])
                cam_id = int(info[2]) + 3
                return img, label, cam_id


class SimpleUnifiedSampler:
    """
    简单的统一采样器
    交替采样RGB和IR数据
    """
    
    def __init__(self, rgb_len, ir_len, batch_size):
        self.rgb_len = rgb_len
        self.ir_len = ir_len
        self.batch_size = batch_size
        self.total_len = rgb_len + ir_len
        
    def __iter__(self):
        # 生成交替索引
        rgb_indices = torch.randperm(self.rgb_len).tolist()
        ir_indices = (torch.randperm(self.ir_len) + self.rgb_len).tolist()
        
        # 交替合并
        indices = []
        max_len = max(len(rgb_indices), len(ir_indices))
        for i in range(max_len):
            if i < len(rgb_indices):
                indices.append(rgb_indices[i])
            if i < len(ir_indices):
                indices.append(ir_indices[i])
        
        return iter(indices)
    
    def __len__(self):
        return self.total_len


def get_train_loader_unified(dataset_obj, args):
    """
    获取统一的训练数据加载器
    Args:
        dataset_obj: SYSU/RegDB/LLCM数据集对象
        args: 参数配置
    Returns:
        train_loader: 统一的训练数据加载器
    """
    # 设置为训练模式
    dataset_obj.train_rgb.load_mode = 'train'
    dataset_obj.train_ir.load_mode = 'train'
    
    # 获取原始采样器（用于设置sampler_idx）
    if hasattr(dataset_obj, 'train_rgb') and hasattr(dataset_obj, 'train_ir'):
        from datasets.sysu import SYSU_Sampler
        from datasets.regdb import RegDB_Sampler
        from datasets.llcm import LLCM_Sampler
        
        # 根据数据集类型选择采样器
        if args.dataset == 'sysu':
            sampler = SYSU_Sampler(args, dataset_obj.train_rgb.label, dataset_obj.train_ir.label)
        elif args.dataset == 'regdb':
            sampler = RegDB_Sampler(args, dataset_obj.train_rgb.label, dataset_obj.train_ir.label)
        elif args.dataset == 'llcm':
            sampler = LLCM_Sampler(args, dataset_obj.train_rgb.label, dataset_obj.train_ir.label)
        else:
            raise ValueError(f"Unknown dataset: {args.dataset}")
        
        # 设置采样索引
        dataset_obj.train_rgb.sampler_idx = sampler.rgb_index
        dataset_obj.train_ir.sampler_idx = sampler.ir_index
    
    # 创建统一数据集
    unified_dataset = UnifiedDataset(dataset_obj.train_rgb, dataset_obj.train_ir)
    
    # 创建数据加载器
    train_loader = DataLoader(
        unified_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    return train_loader


def get_test_loader_unified(dataset_obj, args):
    """
    获取测试数据加载器
    Args:
        dataset_obj: SYSU/RegDB/LLCM数据集对象
        args: 参数配置
    Returns:
        query_loader: 查询集加载器
        gallery_loaders: 检索集加载器列表
    """
    if hasattr(dataset_obj, 'query_loader'):
        return dataset_obj.query_loader, dataset_obj.gallery_loaders
    else:
        return None, None


def get_dataloader(args):
    """
    主数据加载器函数
    Args:
        args: 参数配置
    Returns:
        train_loader: 训练数据加载器
        test_loader: 测试数据加载器（暂时返回None）
    """
    print(f"\nLoading {args.dataset} dataset from {args.data_path}...")
    
    # 根据数据集类型加载
    if args.dataset == 'sysu':
        from datasets.sysu import SYSU
        dataset_obj = SYSU(args)
        
    elif args.dataset == 'regdb':
        from datasets.regdb import RegDB
        dataset_obj = RegDB(args)
        
    elif args.dataset == 'llcm':
        from datasets.llcm import LLCM
        dataset_obj = LLCM(args)
        
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    # 获取统一的训练数据加载器
    train_loader = get_train_loader_unified(dataset_obj, args)
    
    # 获取测试数据加载器
    query_loader, gallery_loaders = get_test_loader_unified(dataset_obj, args)
    
    print(f"DataLoader created successfully!")
    print(f"  Train batches: {len(train_loader)}")
    if query_loader:
        print(f"  Query batches: {len(query_loader)}")
        print(f"  Gallery loaders: {len(gallery_loaders) if gallery_loaders else 0}")
    print()
    
    # 返回训练加载器和查询加载器（作为验证）
    return train_loader, query_loader