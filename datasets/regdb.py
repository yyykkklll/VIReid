import os
import numpy as np
import torch.utils.data as data
from PIL import Image
from .data_process import get_train_transforms, get_test_transformer

class RegDB:
    def __init__(self, args):
        self.args = args
        self.trial = args.trial
        self.path = os.path.join(args.data_path, "RegDB/")
        self.num_workers = args.num_workers
        self.batch_size = args.batch_pidnum * args.pid_numsample # 直接使用总 Batch Size
        self.test_batch = args.test_batch

        # Train Set
        self.train_rgb = RegDB_train(args, self.path, self.trial, modal='rgb')
        self.train_ir = RegDB_train(args, self.path, self.trial, modal='ir')

        # Test Set
        self.query = RegDB_test(args, self.path, mode="thermal", trial=self.trial)
        self.gallery = RegDB_test(args, self.path, mode="visible", trial=self.trial)
        
        self.n_query = len(self.query)
        self.n_gallery = len(self.gallery)
        
        # Test Loaders
        self.query_loader = data.DataLoader(
            self.query, self.test_batch, shuffle=False, num_workers=self.num_workers, drop_last=False)
        self.gallery_loader = data.DataLoader(
            self.gallery, self.test_batch, shuffle=False, num_workers=self.num_workers, drop_last=False)
    
    def get_train_loader(self):
        # [修改] 移除 RegDB_Sampler，使用标准的 shuffle=True
        # 在无监督学习中，我们不应该使用 ID 信息来平衡 Batch
        
        train_rgb_loader = data.DataLoader(
            self.train_rgb, 
            batch_size=self.batch_size,
            shuffle=True,  # 开启随机打乱
            num_workers=self.num_workers, 
            drop_last=True
        )
        
        train_ir_loader = data.DataLoader(
            self.train_ir, 
            batch_size=self.batch_size,
            shuffle=True,  # 开启随机打乱
            num_workers=self.num_workers, 
            drop_last=True
        )
        return train_rgb_loader, train_ir_loader

# RegDB_train 和 RegDB_test 类保持不变 (除了不再需要 relabel 逻辑，但保留也无妨，作为伪标签初始化的参考)
# RegDB_Sampler 类可以删除
class RegDB_train(data.Dataset):
    def __init__(self, args, data_path, trial, modal=None):
        self.num_classes = args.num_classes
        self.data_path = data_path
        self.modal = modal
        self.trial = trial
        
        self.img_h = args.img_h
        self.img_w = args.img_w
        
        if modal == 'rgb':
            self.transform_normal, self.transform_aug = get_train_transforms(self.img_h, self.img_w, 'rgb')
        else:
            self.transform_normal, self.transform_aug = get_train_transforms(self.img_h, self.img_w, 'ir')
            
        self._init_data()

    def _init_data(self):
        if self.modal == 'rgb':
            train_list = self.data_path + 'idx/train_visible_{}'.format(self.trial) + '.txt'
        else:
            train_list = self.data_path + 'idx/train_thermal_{}'.format(self.trial) + '.txt'
            
        img_file, train_label = self._load_data(train_list)
        train_image = []
        for i in range(len(img_file)):
            img = Image.open(self.data_path + img_file[i])
            img = img.resize((self.img_w, self.img_h), Image.LANCZOS)
            pix_array = np.array(img)
            train_image.append(pix_array)

        self.train_image = np.array(train_image)
        # 即使是无监督，加载 Label 用于 Debug 或后续可能的伪标签评估也是可以的
        self.label = np.array(train_label)

    def _load_data(self, input_data_path):
        with open(input_data_path) as f:
            data_file_list = open(input_data_path, 'rt').read().splitlines()
            file_image = [s.split(' ')[0] for s in data_file_list]
            file_label = [int(s.split(' ')[1]) for s in data_file_list]
        return file_image, file_label
    
    def __len__(self):
        return len(self.train_image)
    
    def __getitem__(self, index):
        img_np = self.train_image[index]
        img = Image.fromarray(img_np)
        
        # 返回: (原图, 增强图, 索引/Label信息)
        # USL 训练主要用 img 和 aug_img
        return self.transform_normal(img), self.transform_aug(img), index

class RegDB_test(data.Dataset):
    def __init__(self, args, data_path, mode, trial):
        self.data_path = data_path
        self.mode = mode
        self.transform = get_test_transformer(args.img_h, args.img_w)
        self.img_h = args.img_h
        self.img_w = args.img_w
        
        test_img_file, test_label, test_cam = self._process_test_regdb(trial, mode)
        test_image = []
        for i in range(len(test_img_file)):
            img = Image.open(test_img_file[i])
            img = img.resize((self.img_w, self.img_h), Image.LANCZOS)
            pix_array = np.array(img)
            test_image.append(pix_array)
        self.test_image = np.array(test_image)
        self.test_label = test_label
        self.test_cam = test_cam

    def __len__(self):
        return len(self.test_label)

    def __getitem__(self, index):
        img_np = self.test_image[index]
        img = Image.fromarray(img_np)
        return self.transform(img), self.test_label[index]

    def _process_test_regdb(self, trial, mode):
        if mode == "visible":
            file_path = os.path.join(self.data_path, 'idx/test_visible_{}'.format(trial) + '.txt')
        elif mode == "thermal":
            file_path = os.path.join(self.data_path, 'idx/test_thermal_{}'.format(trial) + '.txt')

        with open(file_path) as f:
            data_file_list = open(file_path, 'rt').read().splitlines()
            file_image = [self.data_path + '/' + s.split(' ')[0] for s in data_file_list]
            file_label = [int(s.split(' ')[1]) for s in data_file_list]
            file_cam = [0 for i in range(len(file_label))]
        return file_image, np.array(file_label), np.array(file_cam)