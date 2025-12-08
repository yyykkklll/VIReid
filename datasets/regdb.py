import os
import numpy as np
import torch.utils.data as data
from PIL import Image
import random
from .data_process import get_train_transforms, get_test_transformer

class RegDB:
    def __init__(self, args):
        self.args = args
        self.trial = args.trial
        self.path = os.path.join(args.data_path, "RegDB/")
        self.num_workers = args.num_workers
        self.batch_size = args.pid_numsample * args.batch_pidnum
        self.test_batch = args.test_batch

        # [修改] Train Set
        self.train_rgb = RegDB_train(args, self.path, self.trial, modal='rgb')
        self.train_ir = RegDB_train(args, self.path, self.trial, modal='ir')

        self.rgb_relabel_dict = self.train_rgb.relabel_dict
        self.ir_relabel_dict = self.train_ir.relabel_dict

        # [修改] Test Set
        self.query = RegDB_test(args, self.path, mode="thermal", trial=self.trial)
        self.gallery = RegDB_test(args, self.path, mode="visible", trial=self.trial)
        
        self.n_query = len(self.query)
        self.n_gallery = len(self.gallery)
        
        self.query_loader = data.DataLoader(
            self.query, self.test_batch, shuffle=False, num_workers=self.num_workers, drop_last=False)
        self.gallery_loader = data.DataLoader(
            self.gallery, self.test_batch, shuffle=False, num_workers=self.num_workers, drop_last=False)
    
    def get_train_loader(self):
        self.train_rgb.load_mode = 'train'
        self.train_ir.load_mode = 'train'
        sampler = RegDB_Sampler(self.args, self.train_rgb.label, self.train_ir.label)
        
        self.train_rgb.sampler_idx = sampler.rgb_index
        self.train_ir.sampler_idx = sampler.ir_index
        
        train_rgb_loader = data.DataLoader(self.train_rgb, batch_size=self.batch_size,
                                           sampler=sampler, num_workers=self.num_workers, drop_last=True)
        train_ir_loader = data.DataLoader(self.train_ir, batch_size=self.batch_size,
                                          sampler=sampler, num_workers=self.num_workers, drop_last=True)
        return train_rgb_loader, train_ir_loader
    
    def get_normal_loader(self):
        self.train_rgb.load_mode = 'test'
        self.train_ir.load_mode = 'test'
        normal_rgb_loader = data.DataLoader(self.train_rgb, batch_size=self.test_batch,
                                       num_workers=self.num_workers, drop_last=False)
        normal_ir_loader = data.DataLoader(self.train_ir, batch_size=self.test_batch,
                                       num_workers=self.num_workers, drop_last=False)
        return normal_rgb_loader, normal_ir_loader

class RegDB_train(data.Dataset):
    def __init__(self, args, data_path, trial, modal=None):
        self.num_classes = args.num_classes
        self.relabel = args.relabel
        self.data_path = data_path
        self.modal = modal
        self.sampler_idx = None
        self.load_mode = None
        self.trial = trial
        
        # [修改] 动态获取 Transform
        self.img_h = args.img_h
        self.img_w = args.img_w
        
        if modal == 'rgb':
            self.transform_normal, self.transform_aug = get_train_transforms(self.img_h, self.img_w, 'rgb')
        else:
            self.transform_normal, self.transform_aug = get_train_transforms(self.img_h, self.img_w, 'ir')
            
        self.transform_test = get_test_transformer(self.img_h, self.img_w)

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
            # [修改] Resize 使用 args
            img = img.resize((self.img_w, self.img_h), Image.LANCZOS)
            pix_array = np.array(img)
            train_image.append(pix_array)

        train_image = np.array(train_image)
        length = len(train_label)
        train_label = np.array(train_label).reshape(length,1)
        train_idx = np.array([i for i in range(length)]).reshape(length,1)
        train_cam = np.array([0 for i in range(length)]).reshape(length,1)
        train_info = np.concatenate((train_idx,train_label,train_cam),axis=1)
        
        self.train_info, self.relabel_dict = self._relabel(train_info)
        self.train_image = train_image
        self.label = self.train_info[:,1]

    def _load_data(self, input_data_path):
        with open(input_data_path) as f:
            data_file_list = open(input_data_path, 'rt').read().splitlines()
            file_image = [s.split(' ')[0] for s in data_file_list]
            file_label = [int(s.split(' ')[1]) for s in data_file_list]
        return file_image, file_label
    
    def _relabel(self, info):
        label = info[:,1]
        gt = label.reshape(label.shape[0],-1)
        info = np.concatenate((info,gt),axis=1)
        pid_set = set(label)
        random_pid = list(range(len(pid_set)))
        random.shuffle(random_pid)
        pid2label = {pid:idx for idx, pid in enumerate(pid_set)}
        pid2random_label = {pid:idx for idx, pid in enumerate(random_pid)}
        for i in range(len(label)):
            labeled_id = pid2label[label[i]]
            info[:,-1][i] = labeled_id
            if self.relabel:
                info[:,1][i]= pid2random_label[labeled_id]
            else:
                info[:,1][i]= labeled_id
        return info, pid2random_label
    
    def __len__(self):
        return len(self.train_image)
    
    def __getitem__(self, index):
        if self.load_mode == 'train': 
            idx = self.sampler_idx[index]
            info = self.train_info[idx]
            img_np = self.train_image[idx]
            
            # [修改] 关键修复：将 numpy 数组转回 PIL Image
            img = Image.fromarray(img_np)
            
            return self.transform_normal(img), self.transform_aug(img), info
            
        else: # extract features
            img_np = self.train_image[index]
            
            # [修改] 关键修复：extract feature 也要转
            img = Image.fromarray(img_np)
            
            return self.transform_test(img), self.train_info[index]

class RegDB_test(data.Dataset):
    def __init__(self, args, data_path, mode, trial):
        self.data_path = data_path
        self.mode = mode
        
        # [修改] 动态获取 Transform
        self.transform = get_test_transformer(args.img_h, args.img_w)
        self.img_h = args.img_h
        self.img_w = args.img_w
        
        test_img_file, test_label, test_cam = self._process_test_regdb(trial, mode)
        test_image = []
        for i in range(len(test_img_file)):
            img = Image.open(test_img_file[i])
            # [修改] Resize 使用 args
            img = img.resize((self.img_w, self.img_h), Image.LANCZOS)
            pix_array = np.array(img)
            test_image.append(pix_array)
        test_image = np.array(test_image)

        self.test_image = test_image
        self.test_label = test_label
        self.test_cam = test_cam

    def __len__(self):
        return len(self.test_label)

    def __getitem__(self, index):
        img_np = self.test_image[index]
        
        # [修改] 关键修复：转为 PIL Image
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

class RegDB_Sampler(data.Sampler):
    # 保持不变
    def __init__(self, args, rgb, ir):
        self.rgb = rgb
        self.ir = ir
        self.len=max(len(rgb),len(ir))*5 # RegDB 比较小，这里扩大 epoch 长度
        self.num_classes = args.num_classes
        self.batch_pidnum = args.batch_pidnum
        self.pid_numsample = args.pid_numsample
        
        self.rgb_dict = {k:[] for k in range(self.num_classes)}
        self.ir_dict  = {k:[] for k in range(self.num_classes)}
        for i in range(len(rgb)):
            self.rgb_dict[int(rgb[i])].append(i)
            if i < len(ir):
                self.ir_dict[int(ir[i])].append(i)

        self._sampler()
        
    def _sampler(self):
        rgb_index = []
        ir_index = []
        batch_num = int(1+self.len/(self.batch_pidnum*self.pid_numsample))
        for i in range(batch_num):
            selected_id = random.sample(list(range(self.num_classes)),self.batch_pidnum)
            for each_id in selected_id:
                if min(len(self.rgb_dict[each_id]),len(self.ir_dict[each_id])) < self.pid_numsample:
                    selected_rgb = random.choices(self.rgb_dict[each_id],k=self.pid_numsample)
                    selected_ir = random.choices(self.ir_dict[each_id],k=self.pid_numsample)
                else:
                    selected_rgb = random.sample(self.rgb_dict[each_id],self.pid_numsample)
                    selected_ir = random.sample(self.ir_dict[each_id],self.pid_numsample)
                rgb_index.extend(selected_rgb)
                ir_index.extend(selected_ir)
        self.rgb_index = rgb_index
        self.ir_index = ir_index

    def __iter__(self):
        return iter(range(self.len))
            
    def __len__(self):
        return self.len