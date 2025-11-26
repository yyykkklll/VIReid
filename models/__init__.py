"""
Models Package
统一导出所有模型和模块
"""

# PCB Backbone
from .pcb_backbone import PCBBackbone, PartPooling

# ISG-DM
from .isg_dm import MultiPartISG_DM

# Memory Bank
from .memory_bank import MultiPartMemoryBank, AdaptiveMemoryBank

# Graph Propagation
from .graph_propagation import AdaptiveGraphPropagation

# Complete Model
from .pfmgcd_model import PF_MGCD

# Loss Functions
from .loss import (
    MultiPartIDLoss,
    GraphDistillationLoss,
    OrthogonalLoss,
    ModalityLoss,
    TotalLoss
)


__all__ = [
    # Backbone
    'PCBBackbone',
    'PartPooling',
    
    # ISG-DM
    'MultiPartISG_DM',
    
    # Memory
    'MultiPartMemoryBank',
    'AdaptiveMemoryBank',
    
    # Graph
    'AdaptiveGraphPropagation',
    
    # Complete Model
    'PF_MGCD',
    
    # Loss
    'MultiPartIDLoss',
    'GraphDistillationLoss',
    'OrthogonalLoss',
    'ModalityLoss',
    'TotalLoss',
]


def create_pfmgcd_model(num_parts=6, num_identities=395, **kwargs):
    """
    工厂函数: 创建PF-MGCD模型
    Args:
        num_parts: 部件数量
        num_identities: 身份数量
        **kwargs: 其他参数
    Returns:
        model: PF_MGCD模型实例
    """
    model = PF_MGCD(
        num_parts=num_parts,
        num_identities=num_identities,
        **kwargs
    )
    return model


# 版本信息
__version__ = '1.0.0'
__author__ = 'PF-MGCD Team'
