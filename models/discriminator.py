import torch
import torch.nn as nn

class DomainDiscriminator(nn.Module):
    def __init__(self, in_dim=2048):
        super(DomainDiscriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)