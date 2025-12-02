#!/bin/bash
cd "$(dirname "$0")/.." || exit

# ====================================================================
# SYSU-MM01 Final Training Script
# Features: PCB + Transformer + Graph Loss (Lambda=0.5)
# ====================================================================

echo "Starting SYSU-MM01 Final Training (Transformer Enabled)..."

# 清理日志
rm -rf logs/sysu_final

python main.py \
  --dataset sysu \
  --mode train \
  --gpu 0 \
  --data-path ./datasets \
  \
  --backbone resnet50 \
  --pretrained \
  --amp \
  \
  --num-parts 6 \
  --feature-dim 512 \
  \
  --batch-size 48 \
  --num-workers 8 \
  --pid-numsample 6 \
  --batch-pidnum 8 \
  --test-batch 128 \
  \
  --img-w 144 \
  --img-h 288 \
  --relabel \
  --search-mode all \
  --gall-mode single \
  \
  --total-epoch 100 \
  --warmup-epochs 10 \
  --lr 0.00035 \
  --weight-decay 5e-4 \
  --lr-scheduler cosine \
  \
  --lambda-graph 0.5 \
  --lambda-triplet 1.0 \
  --label-smoothing 0.1 \
  \
  --init-memory \
  --memory-momentum 0.9 \
  --temperature 3.0 \
  --top-k 5 \
  \
  --save-epoch 10 \
  --eval-epoch 5 \
  --grad-clip 5.0 \
  \
  --save-dir ./checkpoints/sysu_final \
  --log-dir ./logs/sysu_final