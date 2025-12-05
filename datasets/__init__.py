"""
Datasets Package for VI-ReID
支持 SYSU-MM01, RegDB, LLCM 数据集
"""

import os

from .sysu import SYSU
from .llcm import LLCM
from .regdb import RegDB
from .dataloader_adapter import get_dataloader
from .data_process import (
    transform_color_normal,
    transform_color_ca,
    transform_color_sa,
    transform_infrared_normal,
    transform_infrared_sa,
    transform_test,
    transform_test_sa
)


_datasets = {
    'sysu': SYSU,
    'llcm': LLCM,
    'regdb': RegDB
}


def create(args):
    """
    创建数据集对象（兼容原有接口）
    Args:
        args: 参数配置
    Returns:
        dataset: 数据集对象
    """
    if args.dataset not in _datasets:
        raise KeyError("Unknown dataset:", args.dataset)
    print('Building {} dataset ...'.format(args.dataset))
    return _datasets[args.dataset](args)


__all__ = [
    'SYSU',
    'LLCM', 
    'RegDB',
    'create',
    'get_dataloader',
    'transform_color_normal',
    'transform_color_ca',
    'transform_color_sa',
    'transform_infrared_normal',
    'transform_infrared_sa',
    'transform_test',
    'transform_test_sa'
]


# 版本信息
__version__ = '1.0.0'
