#!/bin/bash
cd "$(dirname "$0")/.." || exit

# ====================================================================
# RegDB Quick Experiment (Full Model)
# Model: ISG-DM + Transformer + Graph Loss
# ====================================================================

echo "Starting RegDB Full Model Quick Exp..."

python main.py \
  --dataset regdb \
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
  --batch-size 64 \
  --num-workers 8 \
  --pid-numsample 8 \
  --batch-pidnum 8 \
  --test-batch 128 \
  \
  --img-w 144 \
  --img-h 288 \
  \
  --total-epoch 60 \
  --warmup-epochs 5 \
  --lr 0.00035 \
  --weight-decay 1e-3 \
  --lr-scheduler cosine \
  \
  --lambda-graph 0.05 \
  --lambda-triplet 1.0 \
  --label-smoothing 0.1 \
  \
  --init-memory \
  --memory-momentum 0.9 \
  --temperature 3.0 \
  --top-k 5 \
  --trial 1 \
  \
  --save-epoch 60 \
  --eval-epoch 5 \
  --grad-clip 5.0 \
  \
  --save-dir ./checkpoints/regdb_full_quick \
  --log-dir ./logs/regdb_full_quick