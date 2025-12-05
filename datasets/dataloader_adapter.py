"""
Adapter for original dataset loaders to work with PF-MGCD
(Final Fix V2: Sync Virtual Lengths)
"""

import torch
from torch.utils.data import DataLoader, Dataset, Sampler

class UnifiedDataset(Dataset):
    """
    统一的数据集包装器 (修复长度对齐问题)
    """
    def __init__(self, rgb_dataset, ir_dataset):
        self.rgb_dataset = rgb_dataset
        self.ir_dataset = ir_dataset
        
        # [关键修复] 优先使用 sampler_idx 的长度 (虚拟长度)，
        # 只有在没有 sampler_idx 时才使用物理长度。
        # 这确保了 Dataset 的分界线与 UnifiedBatchSampler 的偏移量完全一致。
        if hasattr(rgb_dataset, 'sampler_idx') and rgb_dataset.sampler_idx is not None:
            self.rgb_len = len(rgb_dataset.sampler_idx)
        else:
            self.rgb_len = len(rgb_dataset)

        if hasattr(ir_dataset, 'sampler_idx') and ir_dataset.sampler_idx is not None:
            self.ir_len = len(ir_dataset.sampler_idx)
        else:
            self.ir_len = len(ir_dataset)

        self.total_len = self.rgb_len + self.ir_len
        
    def __len__(self):
        return self.total_len
    
    def __getitem__(self, idx):
        """
        返回格式: (image, label, cam_id, modality_id)
        modality_id: 0 for RGB, 1 for IR
        """
        # 使用虚拟长度判断分界
        if idx < self.rgb_len:
            # RGB样本
            # 注意: 如果 idx 超出了物理长度，dataset 内部通过 sampler_idx[idx] 映射回物理索引，是安全的
            data = self.rgb_dataset[idx]
            if isinstance(data, tuple) and len(data) == 3:
                img, aug_img, info = data
                label = int(info[1])
                cam_id = int(info[2])
                return img, label, cam_id, 0
            else:
                img, info = data
                label = int(info[1])
                cam_id = int(info[2])
                return img, label, cam_id, 0
        else:
            # IR样本 (减去虚拟长度偏移量)
            ir_idx = idx - self.rgb_len
            data = self.ir_dataset[ir_idx]
            if isinstance(data, tuple) and len(data) == 3:
                img, aug_img, info = data
                label = int(info[1])
                cam_id = int(info[2]) + 3 
                return img, label, cam_id, 1
            else:
                img, info = data
                label = int(info[1])
                cam_id = int(info[2]) + 3
                return img, label, cam_id, 1

class UnifiedBatchSampler(Sampler):
    """
    统一交错 Batch 采样器
    """
    def __init__(self, rgb_len, ir_len, batch_size):
        self.rgb_len = rgb_len
        self.ir_len = ir_len
        self.batch_size = batch_size
        self.num_batches = min(rgb_len, ir_len) // batch_size

    def __iter__(self):
        for i in range(self.num_batches):
            batch_indices = []
            
            # RGB 索引段
            start_rgb = i * self.batch_size
            end_rgb = start_rgb + self.batch_size
            batch_indices.extend(range(start_rgb, end_rgb))
            
            # IR 索引段 (使用传入的 rgb_len 作为偏移，必须与 Dataset 中的 rgb_len 一致)
            start_ir = i * self.batch_size + self.rgb_len
            end_ir = start_ir + self.batch_size
            batch_indices.extend(range(start_ir, end_ir))
            
            yield batch_indices

    def __len__(self):
        return self.num_batches

def get_train_loader_unified(dataset_obj, args):
    """
    获取统一的训练数据加载器
    """
    # 1. 设置模式
    dataset_obj.train_rgb.load_mode = 'train'
    dataset_obj.train_ir.load_mode = 'train'
    
    # 2. 初始化底层 Sampler并赋值
    if hasattr(dataset_obj, 'train_rgb') and hasattr(dataset_obj, 'train_ir'):
        from datasets.sysu import SYSU_Sampler
        from datasets.regdb import RegDB_Sampler
        from datasets.llcm import LLCM_Sampler
        
        if args.dataset == 'sysu':
            sampler = SYSU_Sampler(args, dataset_obj.train_rgb.label, dataset_obj.train_ir.label)
        elif args.dataset == 'regdb':
            sampler = RegDB_Sampler(args, dataset_obj.train_rgb.label, dataset_obj.train_ir.label)
        elif args.dataset == 'llcm':
            sampler = LLCM_Sampler(args, dataset_obj.train_rgb.label, dataset_obj.train_ir.label)
        else:
            raise ValueError(f"Unknown dataset: {args.dataset}")
        
        dataset_obj.train_rgb.sampler_idx = sampler.rgb_index
        dataset_obj.train_ir.sampler_idx = sampler.ir_index
    
    # 3. 创建统一数据集 (注意：此时 dataset_obj.train_rgb 已经有了 sampler_idx)
    # UnifiedDataset 初始化时会读取 sampler_idx 的长度
    unified_dataset = UnifiedDataset(dataset_obj.train_rgb, dataset_obj.train_ir)
    
    # 4. 创建 BatchSampler
    # 传入的长度必须是 unified_dataset 内部使用的长度
    batch_sampler = UnifiedBatchSampler(
        unified_dataset.rgb_len, # 直接使用 Dataset 计算出的长度，确保一致
        unified_dataset.ir_len,
        args.batch_size
    )
    
    # 5. 创建 DataLoader
    train_loader = DataLoader(
        unified_dataset,
        batch_sampler=batch_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    
    return train_loader

def get_test_loader_unified(dataset_obj, args):
    if hasattr(dataset_obj, 'query_loader'):
        return dataset_obj.query_loader, dataset_obj.gallery_loaders
    else:
        return None, None

def get_dataloader(args):
    print(f"\nLoading {args.dataset} dataset from {args.data_path}...")
    
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
    
    train_loader = get_train_loader_unified(dataset_obj, args)
    query_loader, gallery_loaders = get_test_loader_unified(dataset_obj, args)
    
    print(f"DataLoader created successfully!")
    # BatchSampler 使得 len(loader) 返回 batch 数量
    print(f"  Train batches: {len(train_loader)} (Batch Size: {args.batch_size}x2)")
    
    if query_loader:
        print(f"  Query batches: {len(query_loader)}")
    print()
    
    return train_loader, query_loader