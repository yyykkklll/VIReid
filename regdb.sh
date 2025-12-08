#!/bin/bash
# RegDB ViT-Base Quick Verification - 60 Epochs
# 目标: 快速验证改进策略（如 Sinkhorn）是否生效，避免长时间无效等待

# 自动切换到脚本所在目录的上一级 (项目根目录)
cd "$(dirname "$0")" || exit

echo "🚀 RegDB SG-WSL (ViT-Base) - Quick Verification (60 Epochs)..."

# 设置 Python 路径
export PYTHONPATH=$PYTHONPATH:.

python main.py \
    --dataset regdb \
    --data-path ./datasets \
    --mode train \
    --device 0 \
    --seed 42 \
    \
    --arch vit \
    --feat-dim 768 \
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
    --milestones 30 50 \
    \
    --stage1-epoch 13 \
    --stage2-epoch 60 \
    --trial 1 \
    \
    --save-path regdb_vit_quick_v1 \
    --debug wsl \
    --relabel 1 \
    --weak-weight 0.25 \
    --tri-weight 0.25

echo "✅ Quick Verification Complete! Logs saved to saved_regdb_vit/regdb_vit_quick_v1"