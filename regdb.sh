#!/bin/bash

cd "$(dirname "$0")" || exit
export PYTHONPATH=$PYTHONPATH:.

echo "🚀 [RegDB] Training Start (Fixed Version)..."

find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null
find . -name "*.pyc" -delete 2>/dev/null

# 检查磁盘空间
FREE_SPACE=$(df -h . | tail -1 | awk '{print $4}')
echo "Available disk space: $FREE_SPACE"

python main.py \
    --dataset regdb \
    --data-path ./datasets \
    --save-path regdb_wsl_fixed \
    --arch resnet \
    --trial 1 \
    --mode train \
    --device 0 \
    --seed 42 \
    --img-h 288 \
    --img-w 144 \
    --batch-pidnum 16 \
    --pid-numsample 4 \
    --lr 0.0003 \
    --weight-decay 0.0005 \
    --stage1-epoch 60 \
    --stage2-epoch 100 \
    --milestones 40 70 \
    --debug wsl \
    --use-clip \
    --use-sinkhorn \
    --w-clip 0.3 \
    --sinkhorn-reg 0.05 \
    --temperature 0.05 \
    --sigma 0.3

echo "✅ Training Finished!"
