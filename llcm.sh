#!/bin/bash
# LLCM ViT-Base Strong Baseline
cd "$(dirname "$0")" || exit
export PYTHONPATH=$PYTHONPATH:.

echo "🚀 LLCM SG-WSL (ViT-Base) Start..."

python main.py \
    --dataset llcm \
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
    --milestones 40 80 \
    \
    --stage1-epoch 30 \
    --stage2-epoch 120 \
    --trial 1 \
    \
    --save-path llcm_vit_baseline_v1 \
    --debug wsl \
    --relabel 1 \
    --weak-weight 0.25 \
    --tri-weight 0.25