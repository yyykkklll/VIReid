#!/bin/bash
# RegDB FD-Mamba (Frequency-Disentangled Mamba) Training Script
# 目标: 验证 Scheme A (Mamba) + Scheme C (FreqAug) 在 RegDB 上的性能
# 预期: 相比 ViT-Base，Mamba 参数更少且具备全局感受野，配合频域增强应能显著缓解过拟合。

# 自动切换到脚本所在目录
cd "$(dirname "$0")" || exit
export PYTHONPATH=$PYTHONPATH:.

echo "🚀 RegDB FD-Mamba (Scheme A + C) - Training Start..."

# 清理可能存在的缓存（可选）
rm -rf __pycache__

python main.py \
    --dataset regdb \
    --data-path ./datasets \
    --mode train \
    --device 0 \
    --seed 42 \
    \
    --arch vmamba \
    --feat-dim 384 \
    --img-h 256 \
    --img-w 128 \
    \
    --batch-pidnum 8 \
    --pid-numsample 4 \
    --test-batch 128 \
    --num-workers 8 \
    \
    --lr 0.0003 \
    --weight-decay 0.05 \
    --milestones 40 80 \
    \
    --stage1-epoch 20 \
    --stage2-epoch 120 \
    --trial 1 \
    \
    --save-path regdb_fd_mamba_v1 \
    --debug wsl \
    --relabel 1 \
    --weak-weight 0.25 \
    --tri-weight 0.25

echo "✅ Training Complete! Logs saved to saved_regdb_vmamba/regdb_fd_mamba_v1"