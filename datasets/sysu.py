"""
datasets/sysu.py - SYSU-MM01 数据集加载模块 (Strict Version)

包含:
1. SYSU (控制器)
2. SYSU_train (训练集 Dataset，基于预处理的 .npy 文件)
3. SYSU_test (测试集 Dataset)
4. SYSU_Sampler (自定义采样器)
"""

import os
import random
import numpy as np
import torch.utils.data as data
from PIL import Image
from .data_process import *

class SYSU:
    """
    SYSU-MM01 数据集控制器
    注意：SYSU 训练通常依赖预处理好的 .npy 文件 (由 pre_process_sysu.py 生成)
    """
    def __init__(self, args):
        self.args = args
        self.path = os.path.join(args.data_path, "SYSU-MM01/")
        
        # 路径校验
        assert os.path.exists(self.path), f"SYSU dataset path not found: {self.path}"
        
        self.num_workers = args.num_workers
        self.pid_numsample = args.pid_numsample
        self.batch_pidnum = args.batch_pidnum
        self.batch_size = self.pid_numsample * self.batch_pidnum
        self.test_batch = args.test_batch

        print("Dataset: Loading SYSU-MM01...")
        
        # 初始化训练集
        self.train_rgb = SYSU_train(args, self.path, modal='rgb')
        self.train_ir = SYSU_train(args, self.path, modal='ir')

        # 确保 RGB 和 IR ID 一致
        assert self.train_rgb.num_classes == self.train_ir.num_classes, \
            f"SYSU ID mismatch: RGB({self.train_rgb.num_classes}) vs IR({self.train_ir.num_classes})"

        # 初始化 Query (Query 只有 1 个)
        self.query = SYSU_test(args, self.path, mode="query")
        self.query_loader = data.DataLoader(
            self.query, self.test_batch, shuffle=False, num_workers=self.num_workers
        )

        # 初始化 Gallery (SYSU 协议包含 10 次随机划分)
        self.gallery_list = []
        self.gallery_loaders = []
        self.n_gallery = 0
        
        print("Dataset: Building SYSU Gallery (10 trials)...")
        for i in range(10):
            gallery = SYSU_test(args, self.path, mode="gallery", trial=i)
            self.gallery_list.append(gallery)
            
            g_loader = data.DataLoader(
                gallery, self.test_batch, shuffle=False, num_workers=self.num_workers
            )
            self.gallery_loaders.append(g_loader)
            
            if i == 0:
                self.n_gallery = len(gallery)

    def get_normal_loader(self):
        """获取用于初始化记忆库的普通 DataLoader (仅 RGB)"""
        self.train_rgb.load_mode = 'test'
        loader = data.DataLoader(
            self.train_rgb, batch_size=self.test_batch,
            num_workers=self.num_workers, shuffle=False
        )
        return loader, None


class SYSU_train(data.Dataset):
    """
    SYSU 训练数据集
    依赖 pre_process_sysu.py 生成的 .npy 文件
    """
    def __init__(self, args, data_path, modal=None):
        self.num_classes = 0
        self.relabel = args.relabel
        self.data_path = data_path
        
        self.transform_color = transform_color_normal
        self.transform_ir = transform_infrared_normal
        self.transform_test = transform_test
        
        if modal not in ['rgb', 'ir']:
            raise ValueError(f"Invalid modal: {modal}")
        self.modal = modal
        
        self.sampler_idx = None
        self.load_mode = 'train'

        self._init_data()

    def _init_data(self):
        # 1. 加载预处理的 .npy 文件
        if self.modal == 'rgb':
            img_path = os.path.join(self.data_path, "train_rgb_modified_img.npy")
            info_path = os.path.join(self.data_path, "train_rgb_info.npy")
        else:
            img_path = os.path.join(self.data_path, "train_ir_modified_img.npy")
            info_path = os.path.join(self.data_path, "train_ir_info.npy")
            
        if not os.path.exists(img_path) or not os.path.exists(info_path):
            raise FileNotFoundError(
                f"SYSU pre-processed files not found:\n  {img_path}\n  {info_path}\n"
                "Please run 'python pre_process_sysu.py' first!"
            )
            
        # 2. 加载数据到内存
        print(f"  Loading {self.modal.upper()} data from .npy files...")
        self.train_image = np.load(img_path) # [N, H, W, C]
        train_info = np.load(info_path)      # [index, label, cam_id]
        
        # 3. ID 重映射
        self.train_info, self.pid2label = self._relabel(train_info)
        self.label = self.train_info[:, 1]
        self.num_classes = len(self.pid2label)

    def _relabel(self, info):
        """严格的 ID 重映射"""
        raw_labels = info[:, 1]
        unique_pids = sorted(list(set(raw_labels)))
        pid2label = {pid: i for i, pid in enumerate(unique_pids)}
        
        for i in range(len(info)):
            info[i, 1] = pid2label[raw_labels[i]]
            
        return info, pid2label

    def __len__(self):
        return len(self.train_image)

    def __getitem__(self, index):
        if self.load_mode == 'train' and self.sampler_idx is not None:
            real_index = self.sampler_idx[index]
        else:
            real_index = index

        info = self.train_info[real_index]
        img_arr = self.train_image[real_index] # ndarray
        
        if self.load_mode == 'train':
            if self.modal == 'rgb':
                img = self.transform_color(img_arr)
                aug_img = self.transform_color(img_arr)
            else:
                img = self.transform_ir(img_arr)
                aug_img = self.transform_ir(img_arr)
            return img, aug_img, info
        else:
            img = self.transform_test(img_arr)
            return img, info


class SYSU_test(data.Dataset):
    """
    SYSU 测试数据集
    """
    def __init__(self, args, data_path, mode, search_mode="all", gall_mode="single", trial=0):
        self.data_path = data_path
        self.mode = mode
        self.search_mode = args.search_mode
        self.gall_mode = args.gall_mode
        self.transform = transform_test
        
        if mode == "query":
            img_paths, labels, cams = self._process_query_sysu()
        elif mode == "gallery":
            img_paths, labels, cams = self._process_gallery_sysu(trial)
        else:
            raise ValueError(f"Invalid mode: {mode}")

        self.img_paths = img_paths
        self.labels = labels
        self.cams = cams

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        path = self.img_paths[index]
        if not os.path.exists(path):
            # 容错处理：如果是 gallery 多张图模式，这里可能是 list
            if isinstance(path, list):
                # 这里暂不处理 multi-shot gallery 的复杂情况，遵循原代码逻辑
                # 原代码 logic: gall_img.append(random.choice(new_files)) for single
                # gall_img.append(np.random.choice(new_files, 10)) for multi
                # 如果是 multi，下面的 Image.open 会报错，需注意 args.gall_mode
                pass 
            else:
                 raise FileNotFoundError(f"Image not found: {path}")

        try:
            img = Image.open(path).convert('RGB')
            img = img.resize((144, 288), Image.LANCZOS)
            img = np.array(img)
            img = self.transform(img)
        except Exception as e:
            print(f"Error loading {path}: {e}")
            img = torch.zeros(3, 288, 144)

        return img, self.labels[index], self.cams[index]

    def _process_query_sysu(self):
        # 确定 IR 相机
        if self.search_mode == "all":
            ir_cameras = ["cam3", "cam6"]
        elif self.search_mode == "indoor":
            ir_cameras = ["cam3", "cam6"]
        else:
            raise ValueError(f"Unknown search mode: {self.search_mode}")

        file_path = os.path.join(self.data_path, "exp/test_id.txt")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"ID list not found: {file_path}")

        with open(file_path, "r") as file:
            ids = file.read().splitlines()
            ids = [int(y) for y in ids[0].split(",")]
            ids = ["%04d" % x for x in ids]

        files_ir = []
        for id in sorted(ids):
            for cam in ir_cameras:
                img_dir = os.path.join(self.data_path, cam, id)
                if os.path.isdir(img_dir):
                    new_files = sorted([os.path.join(img_dir, i) for i in os.listdir(img_dir)])
                    files_ir.extend(new_files)
        
        query_img = []
        query_id = []
        query_cam = []
        for img_path in files_ir:
            # 解析路径: .../cam3/0001/0001.jpg
            # camid: cam3 -> 3
            # pid: 0001 -> 1
            camid = int(os.path.basename(os.path.dirname(os.path.dirname(img_path)))[-1])
            pid = int(os.path.basename(os.path.dirname(img_path)))
            
            query_img.append(img_path)
            query_id.append(pid)
            query_cam.append(camid)

        return query_img, np.array(query_id), np.array(query_cam)

    def _process_gallery_sysu(self, seed):
        random.seed(seed)
        np.random.seed(seed)

        if self.search_mode == 'all':
            rgb_cameras = ['cam1', 'cam2', 'cam4', 'cam5']
        elif self.search_mode == 'indoor':
            rgb_cameras = ['cam1', 'cam2']
        else:
             raise ValueError(f"Unknown search mode: {self.search_mode}")

        file_path = os.path.join(self.data_path, 'exp/test_id.txt')
        with open(file_path, 'r') as file:
            ids = file.read().splitlines()
            ids = [int(y) for y in ids[0].split(',')]
            ids = ["%04d" % x for x in ids]

        files_rgb = []
        for id in sorted(ids):
            for cam in rgb_cameras:
                img_dir = os.path.join(self.data_path, cam, id)
                if os.path.isdir(img_dir):
                    new_files = sorted([os.path.join(img_dir, i) for i in os.listdir(img_dir)])
                    
                    if self.gall_mode == 'single':
                        files_rgb.append(random.choice(new_files))
                    elif self.gall_mode == 'multi':
                        # 如果图片少于10张，全部选上；否则选10张
                        if len(new_files) < 10:
                            files_rgb.extend(new_files)
                        else:
                            files_rgb.extend(np.random.choice(new_files, 10, replace=False))
        
        gall_img = []
        gall_id = []
        gall_cam = []

        for img_path in files_rgb:
            camid = int(os.path.basename(os.path.dirname(os.path.dirname(img_path)))[-1])
            pid = int(os.path.basename(os.path.dirname(img_path)))
            
            gall_img.append(img_path)
            gall_id.append(pid)
            gall_cam.append(camid)

        return gall_img, np.array(gall_id), np.array(gall_cam)


class SYSU_Sampler(data.Sampler):
    """
    SYSU 专用采样器
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
        self.num_classes = len(self.pids)
        
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
                # RGB
                rgb_idxs = self.rgb_dict[pid]
                replace_rgb = len(rgb_idxs) < self.pid_numsample
                sel_rgb = np.random.choice(rgb_idxs, self.pid_numsample, replace=replace_rgb)
                self.rgb_index.extend(sel_rgb)
                
                # IR
                ir_idxs = self.ir_dict[pid]
                replace_ir = len(ir_idxs) < self.pid_numsample
                sel_ir = np.random.choice(ir_idxs, self.pid_numsample, replace=replace_ir)
                self.ir_index.extend(sel_ir)

    def __iter__(self):
        self._generate_indices()
        return iter(range(len(self.rgb_index)))

    def __len__(self):
        return len(self.rgb_index)