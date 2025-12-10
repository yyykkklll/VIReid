import torch

class MemoryBank(object):
    """
    简单的 FIFO 记忆库，用于存储历史特征
    """
    def __init__(self, size, dim, device):
        self.size = size
        self.dim = dim
        self.device = device
        self.ptr = 0
        self.features = torch.randn(size, dim).to(device)
        self.features = torch.nn.functional.normalize(self.features, dim=1)

    def update(self, feats):
        batch_size = feats.shape[0]
        assert batch_size < self.size
        
        # 指针移动逻辑
        if self.ptr + batch_size > self.size:
            # 简单处理：如果这就满了，这就回绕（Ring Buffer）
            # 这里写个简化版，满了就覆盖开头
            rem = self.size - self.ptr
            self.features[self.ptr:self.size] = feats[:rem]
            self.features[0:batch_size-rem] = feats[rem:]
            self.ptr = batch_size - rem
        else:
            self.features[self.ptr:self.ptr+batch_size] = feats
            self.ptr += batch_size