import torch
import torch.nn as nn

class Projector(nn.Module):
    def __init__(self, in_dim=2048, out_dim=256):
        super(Projector, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.BatchNorm1d(in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, out_dim)
        )

    def forward(self, x):
        return self.net(x)