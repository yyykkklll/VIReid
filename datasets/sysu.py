import os
import numpy as np
import torch.utils.data as data
from PIL import Image
import random
from .data_process import get_train_transforms, get_test_transformer

class SYSU:
    def __init__(self, args):
        self.args = args
        self.path = os.path.join(args.data_path, "SYSU-MM01/")
        self.num_workers = args.num_workers
        self.batch_size = args.pid_numsample * args.batch_pidnum
        self.test_batch = args.test_batch

        # [修改] 传入 args
        self.train_rgb = SYSU_train(args, self.path, modal='rgb')
        self.train_ir = SYSU_train(args, self.path, modal='ir')

        self.rgb_relabel_dict = self.train_rgb.relabel_dict
        self.ir_relabel_dict = self.train_ir.relabel_dict

        # [修改] 传入 args
        self.query = SYSU_test(args, self.path, mode="query")
        self.n_query = len(self.query)
        self.n_gallery = None
        self.gallery_list = []
        for i in range(10):
            gallery = SYSU_test(args, self.path, mode="gallery", trial=i)
            if self.n_gallery == None:
                self.n_gallery = len(gallery) 
            self.gallery_list.append(gallery)
            
        self._get_query_loader()
        self._get_gallery_loader()
    
    def get_train_loader(self):
        self.train_rgb.load_mode = 'train'
        self.train_ir.load_mode = 'train'
        sampler = SYSU_Sampler(self.args, self.train_rgb.label, self.train_ir.label)
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
    
    def _get_query_loader(self):
        self.query_loader = data.DataLoader(
            self.query, self.test_batch, shuffle=False, num_workers=self.num_workers, drop_last=False)
            
    def _get_gallery_loader(self):
        self.gall_info = []
        self.gallery_loaders = []
        for i in range(10):
            self.gall_info.append((self.gallery_list[i].test_label, self.gallery_list[i].test_cam))
            gallery_loader = data.DataLoader(
                self.gallery_list[i], self.test_batch, shuffle=False, num_workers=self.num_workers, drop_last=False)
            self.gallery_loaders.append(gallery_loader)

class SYSU_train(data.Dataset):
    def __init__(self, args, data_path, modal=None):
        self.num_classes = args.num_classes
        self.relabel = args.relabel
        self.data_path = data_path
        self.modal = modal
        self.sampler_idx = None
        self.load_mode = None
        
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
        # 加载 .npy 文件 (注意：这里加载的是预处理后的数据，可能是 288x144)
        if self.modal == 'rgb':
            self.train_image = np.load(self.data_path + "train_rgb_modified_img.npy")
            train_info = np.load(self.data_path + "train_rgb_info.npy")
        elif self.modal == 'ir':
            self.train_image = np.load(self.data_path + "train_ir_modified_img.npy")
            train_info = np.load(self.data_path + "train_ir_info.npy")
        else:
            raise ValueError("modal should be rgb or ir")
            
        self.train_info, self.relabel_dict = self._relabel(train_info)
        self.label = self.train_info[:,1]
        
    def _relabel(self, info):
        label = info[:,1]
        gt = label.reshape(label.shape[0],-1)
        info = np.concatenate((info,gt),axis=1)
        
        # [修复核心] 强制排序，去随机化
        pid_list = sorted(list(set(label)))
        pid2label = {pid:idx for idx, pid in enumerate(pid_list)}
        
        for i in range(len(label)):
            original_pid = label[i]
            labeled_id = pid2label[original_pid]
            
            info[:,-1][i] = labeled_id
            
            if self.relabel:
                info[:,1][i]= labeled_id
            else:
                info[:,1][i]= labeled_id
                
        return info, pid2label
        
    def __len__(self):
        return len(self.train_image)

    def __getitem__(self, index):
        if self.load_mode == 'train': 
            idx = self.sampler_idx[index]
            info = self.train_info[idx]
            img_np = self.train_image[idx]
            
            # [修改] 转换为 PIL 并 Resize，确保输入 ViT 的尺寸正确
            img = Image.fromarray(img_np)
            img = img.resize((self.img_w, self.img_h), Image.LANCZOS)
            
            return self.transform_normal(img), self.transform_aug(img), info
            
        else: 
            img_np = self.train_image[index]
            img = Image.fromarray(img_np)
            img = img.resize((self.img_w, self.img_h), Image.LANCZOS)
            
            return self.transform_test(img), self.train_info[index]

class SYSU_test(data.Dataset):
    def __init__(self, args, data_path, mode, search_mode="all", gall_mode="single", trial=0):
        self.data_path = data_path
        self.mode = mode
        self.search_mode = args.search_mode
        self.gall_mode = args.gall_mode
        
        # [修改] 动态获取 Transform
        self.transform = get_test_transformer(args.img_h, args.img_w)
        self.img_h = args.img_h
        self.img_w = args.img_w
        
        if mode == "query":
            test_img_file, test_label, test_cam = self._process_query_sysu()
        elif mode == "gallery":
            test_img_file, test_label, test_cam = self._process_gallery_sysu(trial)
            
        test_image = []
        for i in range(len(test_img_file)):
            img = Image.open(test_img_file[i])
            # [修改] 使用参数 Resize
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
        return self.transform(self.test_image[index]), self.test_label[index]

    def _process_query_sysu(self):
        if self.search_mode == "all":
            ir_cameras = ["cam3", "cam6"]
        elif self.search_mode == "indoor":
            ir_cameras = ["cam3", "cam6"]

        file_path = os.path.join(self.data_path, "exp/test_id.txt")
        files_ir = []

        with open(file_path, "r") as file:
            ids = file.read().splitlines()
            ids = [int(y) for y in ids[0].split(",")]
            ids = ["%04d" % x for x in ids]

        for id in sorted(ids):
            for cam in ir_cameras:
                img_dir = os.path.join(self.data_path, cam, id)
                if os.path.isdir(img_dir):
                    new_files = sorted([img_dir + "/" + i for i in os.listdir(img_dir)])
                    files_ir.extend(new_files)
        query_img = []
        query_id = []
        query_cam = []
        for img_path in files_ir:
            camid, pid = int(img_path[-15]), int(img_path[-13:-9])
            query_img.append(img_path)
            query_id.append(pid)
            query_cam.append(camid)

        return query_img, np.array(query_id), np.array(query_cam)

    def _process_gallery_sysu(self, seed):
        random.seed(seed)
        if self.search_mode == 'all':
            rgb_cameras = ['cam1', 'cam2', 'cam4', 'cam5']
        elif self.search_mode == 'indoor':
            rgb_cameras = ['cam1', 'cam2']

        file_path = os.path.join(self.data_path, 'exp/test_id.txt')
        files_rgb = []
        with open(file_path, 'r') as file:
            ids = file.read().splitlines()
            ids = [int(y) for y in ids[0].split(',')]
            ids = ["%04d" % x for x in ids]

        for id in sorted(ids):
            for cam in rgb_cameras:
                img_dir = os.path.join(self.data_path, cam, id)
                if os.path.isdir(img_dir):
                    new_files = sorted([img_dir + '/' + i for i in os.listdir(img_dir)])
                    if self.gall_mode == 'single':
                        files_rgb.append(random.choice(new_files))
                    if self.gall_mode == 'multi':
                        files_rgb.append(np.random.choice(new_files, 10, replace=False))
        gall_img = []
        gall_id = []
        gall_cam = []

        for img_path in files_rgb:
            if self.gall_mode == 'single':
                camid, pid = int(img_path[-15]), int(img_path[-13:-9])
                gall_img.append(img_path)
                gall_id.append(pid)
                gall_cam.append(camid)

            if self.gall_mode == 'multi':
                for i in img_path:
                    camid, pid = int(i[-15]), int(i[-13:-9])
                    gall_img.append(i)
                    gall_id.append(pid)
                    gall_cam.append(camid)

        return gall_img, np.array(gall_id), np.array(gall_cam)
    
class SYSU_Sampler(data.Sampler):
    # 保持不变
    def __init__(self, args, rgb, ir):
        self.rgb = rgb
        self.ir = ir
        self.len=max(len(rgb),len(ir))
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