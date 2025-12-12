"""
Utility Functions and Classes for WSL-ReID
==========================================
Components:
- Checkpoint management
- Random seed setting
- Logging utilities
- Metric computation (AverageMeter, MultiItemAverageMeter)
- Evaluation functions (eval_regdb, eval_sysu)
- Helper functions
"""

import os
import sys
import random
import numpy as np
import torch
import time


# ============= Checkpoint Management =============
def save_checkpoint(args, model, epoch):
    """Save model checkpoint (legacy function, kept for compatibility)"""
    if args.dataset == 'regdb':
        path = './saved_pretrain_{}_{}_{}_{}/'.format(args.dataset, args.arch, args.trial, args.save_path)
    else:
        path = './saved_pretrain_{}_{}/'.format(args.dataset, args.arch)
    makedir(path)
    all_state_dict = {
        'backbone': model.model.state_dict(),
        'classifier1': model.classifier1.state_dict(), 
        'classifier2': model.classifier2.state_dict(),
        'classifier3': model.classifier3.state_dict()
    }
    torch.save(all_state_dict, path + 'model_{}.pth'.format(epoch))


def makedir(path):
    """Create directory if not exists"""
    if not os.path.exists(path):
        os.makedirs(path)
        print('make dir {} successful!'.format(path))


# ============= Time and Seed =============
def time_now():
    """Get current time string"""
    return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())


def set_seed(seed):
    """Set random seed for reproducibility"""
    seed = int(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============= File System =============
def os_walk(folder_dir):
    """Walk through directory and return sorted files/dirs"""
    for root, dirs, files in os.walk(folder_dir):
        files = sorted(files, reverse=True)
        dirs = sorted(dirs, reverse=True)
        return root, dirs, files


# ============= Image Processing =============
def fliplr(img):
    """Flip image horizontally"""
    inv_idx = torch.arange(img.size(3)-1, -1, -1).long()  # N x C x H x W
    img_flip = img.index_select(3, inv_idx)
    return img_flip


@torch.no_grad()
def infoEntropy(input):
    """Compute information entropy"""
    input = torch.nn.functional.softmax(input, dim=1)
    output = -torch.mean(input * torch.log2(input + 1e-10))
    return output


def pha_unwrapping(x):
    """Phase unwrapping using FFT"""
    fft_x = torch.fft.fft2(x.clone(), dim=(-2, -1))
    fft_x = torch.stack((fft_x.real, fft_x.imag), dim=-1)
    pha_x = torch.atan2(fft_x[:, :, :, :, 1], fft_x[:, :, :, :, 0])

    fft_clone = torch.zeros(fft_x.size(), dtype=torch.float).cuda()
    fft_clone[:, :, :, :, 0] = torch.cos(pha_x.clone())
    fft_clone[:, :, :, :, 1] = torch.sin(pha_x.clone())

    pha_unwrap = torch.fft.ifft2(
        torch.complex(fft_clone[:, :, :, :, 0], fft_clone[:, :, :, :, 1]),
        dim=(-2, -1)
    ).float()

    return pha_unwrap.to(x.device)


# ============= Logging =============
class Logger:
    """Simple file logger"""
    def __init__(self, log_file):
        self.log_file = log_file

    def __call__(self, input):
        input = str(input)
        with open(self.log_file, 'a') as f:
            f.writelines(input + '\n')
        print(input)

    def clear(self):
        with open(self.log_file, 'w') as f:
            pass


# ============= Metric Meters =============
class AverageMeter:
    """
    Computes and stores the average and current value
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class MultiItemAverageMeter:
    """
    Computes and stores average for multiple items
    """
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

        for i, (key, value) in enumerate(zip(keys, values)):
            result += key
            result += ': '
            if isinstance(value, float):
                result += f'{value:.4f}'
            else:
                result += str(value)
            result += ';  '
            if i % 2:
                result += '\n'

        return result


# ============= Evaluation Functions =============
def eval_regdb(distmat, q_pids, g_pids, max_rank=20):
    """
    Evaluation for RegDB dataset
    
    Args:
        distmat: Distance matrix [num_query, num_gallery]
        q_pids: Query person IDs
        g_pids: Gallery person IDs
        max_rank: Maximum rank for CMC computation
    """
    num_q, num_g = distmat.shape
    
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    
    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    # Compute CMC
    all_cmc = []
    all_AP = []
    all_INP = []
    num_valid_q = 0
    
    for q_idx in range(num_q):
        # Get query pid
        q_pid = q_pids[q_idx]
        
        # [FIX] Removed incorrect filtering of positive samples.
        # In cross-modal ReID (RegDB), positive matches in the gallery (different modality) are valid.
        # We should NOT remove g_pids == q_pid.
        
        # Compute CMC
        raw_cmc = matches[q_idx] # Use all matches without filtering
        
        if not np.any(raw_cmc):
            continue
        
        cmc = raw_cmc.cumsum()
        cmc[cmc > 1] = 1
        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1
        
        # Compute AP
        num_rel = raw_cmc.sum()
        tmp_cmc = raw_cmc.cumsum()
        tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * raw_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)
        
        # Compute INP
        INP = raw_cmc.cumsum()[np.where(raw_cmc == 1)[0][0]] / (np.where(raw_cmc == 1)[0][0] + 1.0)
        all_INP.append(INP)
    
    if num_valid_q == 0:
        raise RuntimeError("No valid query")
    
    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)
    mINP = np.mean(all_INP)
    
    return all_cmc, mAP, mINP


def eval_sysu(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=20):
    """
    Evaluation for SYSU-MM01 dataset
    """
    num_q, num_g = distmat.shape
    
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    
    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    # Compute CMC for each query
    all_cmc = []
    all_AP = []
    all_INP = []
    num_valid_q = 0
    
    for q_idx in range(num_q):
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]
        
        # Remove gallery samples from same camera
        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)
        
        # Compute CMC
        raw_cmc = matches[q_idx][keep]
        if not np.any(raw_cmc):
            continue
        
        cmc = raw_cmc.cumsum()
        cmc[cmc > 1] = 1
        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1
        
        # Compute AP
        num_rel = raw_cmc.sum()
        tmp_cmc = raw_cmc.cumsum()
        tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * raw_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)
        
        # Compute INP
        INP = raw_cmc.cumsum()[np.where(raw_cmc == 1)[0][0]] / (np.where(raw_cmc == 1)[0][0] + 1.0)
        all_INP.append(INP)
    
    if num_valid_q == 0:
        raise RuntimeError("No valid query")
    
    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)
    mINP = np.mean(all_INP)
    
    return all_cmc, mAP, mINP


# ============= Loss Utility Functions =============
def get_loss_annealing_weight(epoch, total_epochs, start_weight=1.0, end_weight=0.5):
    if epoch >= total_epochs:
        return end_weight
    progress = epoch / total_epochs
    weight = start_weight - (start_weight - end_weight) * progress
    return weight


def compute_loss_warmup_weight(epoch, warmup_epochs, target_weight=1.0):
    if epoch >= warmup_epochs:
        return target_weight
    return target_weight * (epoch / warmup_epochs)