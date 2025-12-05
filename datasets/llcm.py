"""
datasets/llcm.py - LLCM 数据集加载模块 (Strict Version)

包含:
1. LLCM (控制器)
2. LLCM_train (训练集 Dataset)
3. LLCM_test (测试集 Dataset)
4. LLCM_Sampler (自定义采样器)
"""

import os
import re
import random
import numpy as np
import torch.utils.data as data
from PIL import Image
from .data_process import *

class LLCM:
    """
    LLCM 数据集控制器
    """
    def __init__(self, args):
        self.args = args
        self.test_mode = args.test_mode # 'v2t' or 't2v'
        self.path = os.path.join(args.data_path, "LLCM/")
        
        assert os.path.exists(self.path), f"LLCM path not found: {self.path}"

        self.num_workers = args.num_workers
        self.pid_numsample = args.pid_numsample
        self.batch_pidnum = args.batch_pidnum
        self.batch_size = self.pid_numsample * self.batch_pidnum
        self.test_batch = args.test_batch

        print("Dataset: Loading LLCM...")
        
        self.train_rgb = LLCM_train(args, self.path, modal='rgb')
        self.train_ir = LLCM_train(args, self.path, modal='ir')

        assert self.train_rgb.num_classes == self.train_ir.num_classes, \
             f"LLCM ID mismatch: RGB({self.train_rgb.num_classes}) vs IR({self.train_ir.num_classes})"

        self.gallery_list = []
        self.gallery_loaders = []
        
        # 根据测试模式初始化 Query 和 Gallery (10 trials)
        print(f"Dataset: Building LLCM Test Sets (Mode: {self.test_mode})...")
        
        if self.test_mode == 'v2t': # Vis Query, Thermal Gallery
            self.query = LLCM_test(args, self.path, 'rgb', trial=0) # Query 这里的 trial 参数无影响，但保持接口一致
            for i in range(10):
                gallery = LLCM_test(args, self.path, 'ir', trial=i)
                self.gallery_list.append(gallery)
                
        elif self.test_mode == 't2v': # Thermal Query, Vis Gallery
            self.query = LLCM_test(args, self.path, 'ir', trial=0)
            for i in range(10):
                gallery = LLCM_test(args, self.path, 'rgb', trial=i)
                self.gallery_list.append(gallery)
        else:
            raise ValueError(f"Invalid test_mode: {self.test_mode}")
            
        self.n_query = len(self.query)
        self.n_gallery = len(self.gallery_list[0]) # 记录第一个 gallery 的大小作为参考
        
        self.query_loader = data.DataLoader(
            self.query, self.test_batch, shuffle=False, num_workers=self.num_workers
        )
        
        for gal in self.gallery_list:
            self.gallery_loaders.append(data.DataLoader(
                gal, self.test_batch, shuffle=False, num_workers=self.num_workers
            ))

    def get_normal_loader(self):
        """用于初始化的 RGB Loader"""
        self.train_rgb.load_mode = 'test'
        loader = data.DataLoader(
            self.train_rgb, batch_size=self.test_batch,
            num_workers=self.num_workers, shuffle=False
        )
        return loader, None


class LLCM_train(data.Dataset):
    """
    LLCM 训练数据集
    """
    def __init__(self, args, data_path, trial=None, modal=None):
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
        self.trial = trial

        self._init_data()

    def _init_data(self):
        # 1. 读取列表文件
        if self.modal == 'rgb':
            txt_file = os.path.join(self.data_path, 'idx/train_vis.txt')
        else:
            txt_file = os.path.join(self.data_path, 'idx/train_nir.txt')
            
        if not os.path.exists(txt_file):
            raise FileNotFoundError(f"Train list not found: {txt_file}")
            
        # 2. 解析文件
        img_paths, labels, cam_ids = self._load_data(txt_file)
        
        # 3. 加载图片到内存
        self.train_image = []
        for path in img_paths:
            full_path = os.path.join(self.data_path, path)
            if not os.path.exists(full_path):
                raise FileNotFoundError(f"Image not found: {full_path}")
                
            img = Image.open(full_path).convert('RGB')
            img = img.resize((144, 288), Image.LANCZOS)
            self.train_image.append(np.array(img))
            
        self.train_image = np.array(self.train_image)
        
        # 4. 构建 Info
        length = len(labels)
        train_idx = np.arange(length).reshape(-1, 1)
        train_label = np.array(labels).reshape(-1, 1)
        train_cam = np.array(cam_ids).reshape(-1, 1)
        
        train_info = np.concatenate((train_idx, train_label, train_cam), axis=1)
        
        # 5. ID 重映射
        self.train_info, self.pid2label = self._relabel(train_info)
        self.label = self.train_info[:, 1]
        self.num_classes = len(self.pid2label)

    def _load_data(self, input_data_path):
        with open(input_data_path) as f:
            lines = f.read().splitlines()
            
        paths = [s.split(' ')[0] for s in lines]
        labels = [int(s.split(' ')[1]) for s in lines]
        
        # 解析 Camera ID (e.g., "..._c1_...")
        # 正则: 匹配 _c 后面的数字
        pattern = re.compile(r'_c(\d+)')
        cam_ids = []
        for path in paths:
            match = pattern.search(path)
            if match:
                cam_ids.append(int(match.group(1)))
            else:
                # 默认值，或者报错
                print(f"Warning: Could not parse cam ID from {path}, setting to 0")
                cam_ids.append(0)
                
        return paths, labels, cam_ids

    def _relabel(self, info):
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
        img_arr = self.train_image[real_index]
        
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


class LLCM_test(data.Dataset):
    """
    LLCM 测试数据集
    """
    def __init__(self, args, data_path, modal, trial=-1):
        self.data_path = data_path
        self.modal = modal
        self.transform = transform_test
        
        # 确定相机目录
        if self.modal == "rgb":
            cameras = [f'test_vis/cam{i}' for i in range(1, 10)] # cam1~cam9
        elif self.modal == "ir":
            # 注意: LLCM NIR 部分可能没有 cam3 (依据原始代码逻辑)
            # 原始代码: cam1,2,4,5,6,7,8,9
            cameras = ['test_nir/cam1', 'test_nir/cam2', 
                       'test_nir/cam4', 'test_nir/cam5', 
                       'test_nir/cam6', 'test_nir/cam7', 
                       'test_nir/cam8', 'test_nir/cam9']
        else:
            raise ValueError(f"Invalid modal: {modal}")

        # 读取测试 ID 列表
        id_file = os.path.join(self.data_path, 'idx/test_id.txt')
        if not os.path.exists(id_file):
            raise FileNotFoundError(f"Test ID file not found: {id_file}")
            
        with open(id_file, 'r') as file:
            ids = file.read().splitlines()
            ids = [int(y) for y in ids[0].split(',')]
            ids = ["%04d" % x for x in ids]

        # 收集图片文件
        self.img_paths = []
        self.labels = []
        self.cams = []
        
        # 设置随机种子 (用于 Gallery 随机采样)
        if trial == -1:
            random.seed(0) # 默认
        else:
            random.seed(trial)

        for id in sorted(ids):
            for cam in cameras:
                img_dir = os.path.join(self.data_path, cam, id)
                if os.path.isdir(img_dir):
                    new_files = sorted([os.path.join(img_dir, i) for i in os.listdir(img_dir)])
                    
                    # 采样逻辑: Query 全量，Gallery 随机选 1 张
                    # 根据 LLCM 协议: 
                    # 通常 Query 是确定的 (test_mode决定)，Gallery 是随机的
                    # 这里原始代码逻辑是：如果 trial != -1 (即 Gallery 模式)，则随机选一张
                    if trial == -1:
                        selected_files = new_files
                    else:
                        if len(new_files) > 0:
                            selected_files = [random.choice(new_files)]
                        else:
                            selected_files = []
                            
                    for f in selected_files:
                        self.img_paths.append(f)
                        # 解析 ID 和 Cam
                        # path format: .../camX/0001/img.jpg
                        # camid parsing from folder name 'camX'
                        # 原始代码 path string split logic: camid, pid = int(img_path.split('cam')[1][0]), ...
                        # 这是一个比较脆弱的解析，这里优化一下
                        try:
                            # 找到 'cam' 第一次出现的位置，取后面的一位数字
                            # 或者更稳健地: 提取路径中的 camX
                            cam_match = re.search(r'cam(\d+)', f)
                            if cam_match:
                                camid = int(cam_match.group(1))
                            else:
                                camid = 0
                                
                            pid = int(id)
                            
                            self.labels.append(pid)
                            self.cams.append(camid)
                        except Exception as e:
                            print(f"Error parsing path {f}: {e}")

        self.labels = np.array(self.labels)
        self.cams = np.array(self.cams)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        path = self.img_paths[index]
        if not os.path.exists(path):
            raise FileNotFoundError(f"Test image not found: {path}")

        try:
            img = Image.open(path).convert('RGB')
            img = img.resize((144, 288), Image.LANCZOS)
            img = np.array(img)
            img = self.transform(img)
        except Exception as e:
            print(f"Error loading {path}: {e}")
            img = torch.zeros(3, 288, 144)

        return img, self.labels[index], self.cams[index]


class LLCM_Sampler(data.Sampler):
    """
    LLCM 专用采样器
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