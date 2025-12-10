import os
import sys
import random
import numpy as np
import torch
import time

def makedir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print('make dir {} successful!'.format(path))

def time_now():
    return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())

def set_seed(seed):
    seed = int(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def os_walk(folder_dir):
    for root, dirs, files in os.walk(folder_dir):
        files = sorted(files, reverse=True)
        dirs = sorted(dirs, reverse=True)
        return root, dirs, files

def fliplr(img):
    '''flip horizontal'''
    # 增加 .to(img.device) 增强鲁棒性
    inv_idx = torch.arange(img.size(3)-1,-1,-1).long().to(img.device)
    img_flip = img.index_select(3,inv_idx)
    return img_flip

class Logger:
    def __init__(self, log_file):
        self.log_file = log_file

    def __call__(self, input):
        input = str(input)
        with open(self.log_file, 'a') as f:
            f.writelines(input+'\n')
        print(input)

    def clear(self):
        with open(self.log_file, 'w') as f:
            pass

class MultiItemAverageMeter:
    def __init__(self):
        self.content = {}

    def update(self, val):
        for key in list(val.keys()):
            value = val[key]
            if key not in list(self.content.keys()):
                self.content[key] = {'avg': value, 'sum': value, 'count': 1.0}
            else:
                self.content[key]['sum'] += value
                self.content[key]['count'] += 1.0
                self.content[key]['avg'] = self.content[key]['sum'] / self.content[key]['count']

    def get_val(self):
        keys = list(self.content.keys())
        values = []
        for key in keys:
            try:
                values.append(self.content[key]['avg'].data.cpu().numpy())
            except:
                values.append(self.content[key]['avg'])
        return keys, values

    def get_str(self):
        result = ''
        keys, values = self.get_val()

        for i,(key, value) in enumerate(zip(keys, values)):
            result += key
            result += ': '
            result += "{:.4f}".format(value)
            result += ';  '
            if i%2:
                result += '\n'

        return result