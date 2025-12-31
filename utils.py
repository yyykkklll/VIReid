import os
import sys
import random
import numpy as np
import torch


def makedir(path):
    """Create directory if it does not exist"""
    if not os.path.exists(path):
        os.makedirs(path)


def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # Deterministic mode
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Logger(object):
    """
    File-only logger that does NOT intercept stdout.
    Preserves terminal output for tqdm and interactive messages.
    """
    def __init__(self, fpath=None):
        self.file = None
        if fpath is not None:
            makedir(os.path.dirname(fpath))
            self.file = open(fpath, 'a')  # Append mode for resume support
    
    def __del__(self):
        self.close()
    
    def __call__(self, msg):
        """
        Write message to log file only (not terminal).
        Use print() for terminal output.
        """
        if self.file is not None:
            self.file.write(str(msg) + '\n')
            self.file.flush()
            os.fsync(self.file.fileno())
    
    def close(self):
        """Close log file"""
        if self.file is not None:
            self.file.close()
            self.file = None
    
    def clear(self):
        """Clear log file content (for fresh training start)"""
        if self.file is not None:
            self.file.seek(0)
            self.file.truncate(0)
            self.file.flush()


class AverageMeter(object):
    """Computes and stores the average and current value"""
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


class MultiItemAverageMeter(object):
    """
    Manages multiple AverageMeters for dictionary-based metric tracking.
    """
    def __init__(self):
        self.meters = {}
    
    def update(self, val_dict):
        """Update multiple meters at once"""
        for k, v in val_dict.items():
            # Auto-convert tensors to python floats
            if isinstance(v, torch.Tensor):
                v = v.item()
            
            if k not in self.meters:
                self.meters[k] = AverageMeter()
            
            self.meters[k].update(v)
    
    def get_val(self):
        """Get current values"""
        return {k: m.val for k, m in self.meters.items()}
    
    def get_avg(self):
        """Get average values"""
        return {k: m.avg for k, m in self.meters.items()}
    
    def get_str(self):
        """Get formatted string of averages"""
        return ' '.join([f"{k}: {m.avg:.4f}" for k, m in self.meters.items()])


def infoEntropy(prob):
    """
    Calculate Information Entropy.
    Args:
        prob: [N, C] softmax probabilities
    """
    epsilon = 1e-10
    entropy = -torch.sum(prob * torch.log(prob + epsilon), dim=1)
    return entropy.mean()
