"""
Models Package - Simple Strong Baseline
"""

from .pcb_backbone import PCBBackbone, PartPooling
from .pfmgcd_model import PF_MGCD
from .loss import (
    MultiPartIDLoss,
    TripletLossHardMining,
    TotalLoss
)

# 向后兼容别名
TripletLoss = TripletLossHardMining

__all__ = [
    'PCBBackbone',
    'PartPooling',
    'PF_MGCD',
    'MultiPartIDLoss',
    'TripletLoss',
    'TripletLossHardMining',
    'TotalLoss',
]