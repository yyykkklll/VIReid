#!/bin/bash
# Unsupervised VI-ReID Baseline on RegDB

# 自动切换目录
cd "$(dirname "$0")" || exit
export PYTHONPATH=$PYTHONPATH:.

echo "🚀 USL-RegDB Training Start..."

# 清理缓存
find . -name "__pycache__" -type d -exec rm -rf {} +
find . -name "*.pyc" -delete

python main.py \
    --dataset regdb \
    --data-path ./datasets \
    --mode train \
    --device 0 \
    --seed 42 \
    \
    --arch resnet50 \
    --img-h 256 \
    --img-w 128 \
    \
    --batch-pidnum 8 \
    --pid-numsample 4 \
    --num-workers 8 \
    \
    --lr 0.00035 \
    --weight-decay 0.0005 \
    --milestones 40 70 \
    \
    --save-path regdb_usl_baseline \
    \
    --lambda-ot 0.1 \
    --lambda-adv 0.1 \
    --epochs 60

echo "✅ Training Finished!"