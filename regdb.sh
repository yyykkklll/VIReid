#!/bin/bash
# RegDB ResNet50 Quick Training Script
# 目的: 快速验证 ResNet50 架构在 RegDB 上的训练流程和初步效果

# 自动切换到脚本所在目录
cd "$(dirname "$0")" || exit
export PYTHONPATH=$PYTHONPATH:.

echo "🚀 RegDB ResNet50 (Pretrained) - Start Training..."

# 显存清理 (可选)
# rm -rf __pycache__

python main.py \
    --dataset regdb \
    --data-path ./datasets \
    --mode train \
    --device 0 \
    --seed 42 \
    \
    --arch resnet50 \
    --feat-dim 2048 \
    --img-h 256 \
    --img-w 128 \
    \
    --batch-pidnum 8 \
    --pid-numsample 4 \
    --test-batch 128 \
    --num-workers 8 \
    \
    --lr 0.00035 \
    --weight-decay 0.05 \
    --milestones 40 70 \
    \
    --stage1-epoch 20 \
    --stage2-epoch 120 \
    --trial 1 \
    \
    --save-path regdb_resnet50_verify \
    --debug wsl \
    --relabel 1 \
    --weak-weight 0.25 \
    --tri-weight 0.25

echo "✅ Training Finished! Check logs in saved_regdb_resnet50/regdb_resnet50_verify"