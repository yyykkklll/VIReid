"""
Models Package
统一导出所有模型和模块
"""

# PCB Backbone
from .pcb_backbone import PCBBackbone, PartPooling

# ISG-DM
from .isg_dm import ISG_DM, MultiPartISG_DM

# Memory Bank
from .memory_bank import MultiPartMemoryBank, AdaptiveMemoryBank

# Graph Propagation
from .graph_propagation import (
    GraphPropagation,
    AdaptiveGraphPropagation,
    GraphDistillationLoss
)

# Complete Model
from .pfmgcd_model import PF_MGCD

# Teacher Network
from .teacher_network import (
    TeacherNetwork,
    DualModalityTeacher,
    create_teacher_network
)

# Loss Functions
from .loss import (
    MultiPartIDLoss,
    OrthogonalLoss,
    ModalityLoss,
    GraphDistillationLoss,
    TotalLoss
)

# Classifiers (如果存在)
try:
    from .classifier import MultiPartClassifier
except ImportError:
    pass


__all__ = [
    # Backbone
    'PCBBackbone',
    'PartPooling',
    
    # ISG-DM
    'ISG_DM',
    'MultiPartISG_DM',
    
    # Memory
    'MultiPartMemoryBank',
    'AdaptiveMemoryBank',
    
    # Graph
    'GraphPropagation',
    'AdaptiveGraphPropagation',
    'GraphDistillationLoss',
    
    # Complete Model
    'PF_MGCD',
    
    # Teacher
    'TeacherNetwork',
    'DualModalityTeacher',
    'create_teacher_network',
    
    # Loss
    'MultiPartIDLoss',
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