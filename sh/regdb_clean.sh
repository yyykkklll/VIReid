#!/bin/bash

# ====================================================================
# RegDB Clean Baseline (Rescue Plan)
# 目标：验证模型核心能力 (BNNeck + Bias=True)，预期 Rank-1 > 60%
# ====================================================================

# 切换到项目根目录
cd "$(dirname "$0")/.." || exit

python main.py \
  --dataset regdb \
  --mode train \
  --gpu 0 \
  --data-path ./datasets \
  \
  --backbone resnet50 \
  --pretrained \
  --amp \
  --num-parts 6 \
  --feature-dim 256 \
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
  --total-epoch 80 \
  --warmup-epochs 5 \
  --lr 0.00035 \
  --weight-decay 5e-4 \
  --lr-scheduler step \
  --lr-step 40,70 \
  --lr-gamma 0.1 \
  \
  --lambda-graph 0.0 \
  --lambda-orth 0.0 \
  --lambda-mod 0.0 \
  --lambda-triplet 1.0 \
  --label-smoothing 0.1 \
  \
  --save-epoch 10 \
  --eval-epoch 5 \
  --grad-clip 5.0 \
  \
  --init-memory \
  --pool-parts \
  --distance-metric euclidean \
  --trial 1 \
  \
  --save-dir ./checkpoints/regdb_clean \
  --log-dir ./logs/regdb_clean

echo ""
echo "======================================"
echo "RegDB Clean Baseline Training"
echo "======================================"
echo "Changes:"
echo "  1. Bias=True in Classifiers (Crucial)"
echo "  2. Disabled Graph/Orth/Mod losses (Focus on ID+Tri)"
echo "  3. Batch Size = 64"
echo "======================================"