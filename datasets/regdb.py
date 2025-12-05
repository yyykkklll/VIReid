"""
datasets/regdb.py - RegDB 数据集加载模块 (Strict Version)
"""

import os
import numpy as np
import torch.utils.data as data
from PIL import Image
from .data_process import *

class RegDB:
    """
    RegDB 数据集控制器
    """
    def __init__(self, args):
        self.args = args
        self.trial = args.trial
        self.path = os.path.join(args.data_path, "RegDB/")
        
        # 参数校验
        assert os.path.exists(self.path), f"RegDB path does not exist: {self.path}"
        
        self.num_workers = args.num_workers
        self.pid_numsample = args.pid_numsample
        self.batch_pidnum = args.batch_pidnum
        self.batch_size = self.pid_numsample * self.batch_pidnum
        self.test_batch = args.test_batch

        # 初始化数据集
        print(f"Dataset: Loading RegDB (Trial {self.trial})...")
        self.train_rgb = RegDB_train(args, self.path, self.trial, modal='rgb')
        self.train_ir = RegDB_train(args, self.path, self.trial, modal='ir')

        # 确保 RGB 和 IR 的 ID 数量一致，否则采样器会出错
        assert self.train_rgb.num_classes == self.train_ir.num_classes, \
            f"ID mismatch: RGB({self.train_rgb.num_classes}) vs IR({self.train_ir.num_classes})"

        # 初始化测试集
        self.query = RegDB_test(args, self.path, mode="thermal", trial=self.trial)
        self.gallery = RegDB_test(args, self.path, mode="visible", trial=self.trial)
        
        self.query_loader = data.DataLoader(
            self.query, self.test_batch, shuffle=False, num_workers=self.num_workers
        )
        self.gallery_loader = data.DataLoader(
            self.gallery, self.test_batch, shuffle=False, num_workers=self.num_workers
        )
        self.gallery_loaders = [self.gallery_loader]

    def get_normal_loader(self):
        """获取用于初始化记忆库的普通 DataLoader (仅 RGB)"""
        self.train_rgb.load_mode = 'test'
        loader = data.DataLoader(
            self.train_rgb, batch_size=self.test_batch, 
            num_workers=self.num_workers, shuffle=False
        )
        return loader, None


class RegDB_train(data.Dataset):
    """
    RegDB 训练数据集
    """
    def __init__(self, args, data_path, trial, modal=None):
        self.num_classes = 0 # 将在 _init_data 中更新
        self.relabel = args.relabel
        self.data_path = data_path
        
        # 确定的 Transform 接口
        self.transform_color = transform_color_normal
        self.transform_ir = transform_infrared_normal
        self.transform_test = transform_test
        
        # 输入校验
        if modal not in ['rgb', 'ir']:
            raise ValueError(f"Invalid modal type: {modal}. Must be 'rgb' or 'ir'.")
        self.modal = modal
        self.trial = trial
        self.sampler_idx = None
        self.load_mode = 'train'
        
        self._init_data()

    def _init_data(self):
        # 1. 路径构建
        if self.modal == 'rgb':
            idx_file = os.path.join(self.data_path, f'idx/train_visible_{self.trial}.txt')
        else:
            idx_file = os.path.join(self.data_path, f'idx/train_thermal_{self.trial}.txt')
            
        if not os.path.exists(idx_file):
            raise FileNotFoundError(f"Index file not found: {idx_file}")

        # 2. 数据加载
        img_paths, labels = self._load_data_file(idx_file)
        
        # 3. 预加载图片到内存
        self.train_image = []
        for path in img_paths:
            full_path = os.path.join(self.data_path, path)
            if not os.path.exists(full_path):
                raise FileNotFoundError(f"Image not found: {full_path}")
            
            # 统一转为 RGB 并 Resize，存储为 ndarray
            img = Image.open(full_path).convert('RGB')
            img = img.resize((144, 288), Image.LANCZOS)
            self.train_image.append(np.array(img))

        self.train_image = np.array(self.train_image)
        
        # 4. 构建 Info 矩阵
        length = len(labels)
        train_idx = np.arange(length).reshape(-1, 1)
        train_label = np.array(labels).reshape(-1, 1)
        
        # 显式设置 Camera ID (RGB=0, IR=1)
        cam_val = 0 if self.modal == 'rgb' else 1
        train_cam = np.full((length, 1), cam_val)
        
        train_info = np.concatenate((train_idx, train_label, train_cam), axis=1)
        
        # 5. ID 重映射
        self.train_info, self.pid2label = self._relabel(train_info)
        self.label = self.train_info[:, 1]
        self.num_classes = len(self.pid2label)

    def _load_data_file(self, file_path):
        with open(file_path, 'r') as f:
            lines = f.read().splitlines()
        paths = [x.split(' ')[0] for x in lines]
        labels = [int(x.split(' ')[1]) for x in lines]
        return paths, labels

    def _relabel(self, info):
        """严格的 ID 重映射：确保映射到 [0, N-1]"""
        raw_labels = info[:, 1]
        unique_pids = sorted(list(set(raw_labels)))
        
        pid2label = {pid: i for i, pid in enumerate(unique_pids)}
        
        # 执行重映射
        for i in range(len(info)):
            info[i, 1] = pid2label[raw_labels[i]]
            
        return info, pid2label

    def __len__(self):
        return len(self.train_image)

    def __getitem__(self, index):
        # 索引映射
        if self.load_mode == 'train' and self.sampler_idx is not None:
            real_index = self.sampler_idx[index]
        else:
            real_index = index

        info = self.train_info[real_index]
        img_arr = self.train_image[real_index] # Type: numpy.ndarray (H, W, C)
        
        if self.load_mode == 'train':
            # 应用增强
            # data_process.py 中的 transforms.Compose 第一个操作是 ToPILImage()
            # 它明确支持 numpy.ndarray 输入，无需转换
            if self.modal == 'rgb':
                img = self.transform_color(img_arr)
                aug_img = self.transform_color(img_arr)
            else:
                img = self.transform_ir(img_arr)
                aug_img = self.transform_ir(img_arr)
            
            return img, aug_img, info
        else:
            # 测试模式
            img = self.transform_test(img_arr)
            return img, info


class RegDB_test(data.Dataset):
    """
    RegDB 测试数据集
    """
    def __init__(self, args, data_path, mode, trial):
        self.data_path = data_path
        self.transform = transform_test
        
        if mode == "visible":
            idx_file = os.path.join(data_path, f'idx/test_visible_{trial}.txt')
            cam_val = 0
        else:
            idx_file = os.path.join(data_path, f'idx/test_thermal_{trial}.txt')
            cam_val = 1

        if not os.path.exists(idx_file):
            raise FileNotFoundError(f"Test index file not found: {idx_file}")

        with open(idx_file, 'r') as f:
            lines = f.read().splitlines()
            
        self.img_paths = [os.path.join(data_path, x.split(' ')[0]) for x in lines]
        self.labels = np.array([int(x.split(' ')[1]) for x in lines])
        self.cams = np.full(len(self.labels), cam_val)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        path = self.img_paths[index]
        # 测试集读取必须健壮
        if not os.path.exists(path):
             raise FileNotFoundError(f"Test image not found: {path}")

        img = Image.open(path).convert('RGB')
        img = img.resize((144, 288), Image.LANCZOS)
        img = np.array(img)
        img = self.transform(img)
            
        return img, self.labels[index], self.cams[index]


class RegDB_Sampler(data.Sampler):
    """
    RegDB 专用采样器
    """
    def __init__(self, args, rgb_labels, ir_labels):
        self.rgb_labels = rgb_labels
        self.ir_labels = ir_labels
        self.batch_pidnum = args.batch_pidnum
        self.pid_numsample = args.pid_numsample
        self.length = max(len(rgb_labels), len(ir_labels))
        
        self.rgb_dict = self._build_dict(rgb_labels)
        self.ir_dict = self._build_dict(ir_labels)
        
        self.pids = sorted(list(set(rgb_labels)))
        self.rgb_index = []
        self.ir_index = []
        self._generate_indices()

    def _build_dict(self, labels):
        d = {}
        for idx, label in enumerate(labels):
            if label not in d:
                d[label] = []
            d[label].append(idx)
        return d

    def _generate_indices(self):
        self.rgb_index = []
        self.ir_index = []
        
        num_batches = self.length // (self.batch_pidnum * self.pid_numsample) + 1
        
        for _ in range(num_batches):
            batch_pids = np.random.choice(self.pids, self.batch_pidnum, replace=False)
            
            for pid in batch_pids:
                # RGB Sampling
                rgb_idxs = self.rgb_dict[pid]
                replace_rgb = len(rgb_idxs) < self.pid_numsample
                sel_rgb = np.random.choice(rgb_idxs, self.pid_numsample, replace=replace_rgb)
                self.rgb_index.extend(sel_rgb)
                
                # IR Sampling
                ir_idxs = self.ir_dict[pid]
                replace_ir = len(ir_idxs) < self.pid_numsample
                sel_ir = np.random.choice(ir_idxs, self.pid_numsample, replace=replace_ir)
                self.ir_index.extend(sel_ir)

    def __iter__(self):
        self._generate_indices()
        return iter(range(len(self.rgb_index)))

    def __len__(self):
        return len(self.rgb_index)