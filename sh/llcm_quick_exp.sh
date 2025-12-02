#!/bin/bash
cd "$(dirname "$0")/.." || exit

# ====================================================================
# LLCM Quick Experiment (Full Model)
# Model: ISG-DM + Transformer + Graph Loss
# ====================================================================

echo "Starting LLCM Full Model Quick Exp..."

python main.py \
  --dataset llcm \
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
  --test-mode v2t \
  \
  --total-epoch 60 \
  --warmup-epochs 5 \
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
  --save-epoch 60 \
  --eval-epoch 5 \
  --grad-clip 5.0 \
  \
  --save-dir ./checkpoints/llcm_full_quick \
  --log-dir ./logs/llcm_full_quick