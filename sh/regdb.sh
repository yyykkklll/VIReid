#!/bin/bash
# ====================================================================
# RegDB Boosted Training (Performance Focused)
# Strategy:
#   1. High Resolution (384x128) -> Better details
#   2. Graph Distillation (Enabled after warmup) -> Semi-supervised gain
#   3. Modality Adversarial (Low weight) -> Alignment
#   4. AMP Disabled -> Stability
# ====================================================================

# 自动切换到项目根目录
cd "$(dirname "$0")/.." || exit

echo "Running RegDB Boosted Training..."

python main.py \
  --dataset regdb \
  --mode train \
  --gpu 0 \
  --data-path ./datasets \
  \
  --backbone resnet50 \
  --pretrained \
  --use-ibn \
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
  --img-w 128 \
  --img-h 384 \
  \
  --total-epoch 100 \
  --warmup-epochs 10 \
  --lr 0.00035 \
  --weight-decay 5e-4 \
  --lr-scheduler cosine \
  \
  --use-adversarial \
  --lambda-adv 0.02 \
  \
  --lambda-graph 0.1 \
  --lambda-triplet 1.0 \
  --label-smoothing 0.1 \
  \
  --init-memory \
  --memory-momentum 0.9 \
  --temperature 3.0 \
  --top-k 5 \
  --trial 1 \
  \
  --save-epoch 10 \
  --eval-epoch 5 \
  --grad-clip 5.0 \
  \
  --save-dir ./checkpoints/regdb_boost \
  --log-dir ./logs/regdb_boost