#!/bin/bash

# ====================================================================
# RegDB Training Script - A100 Optimized
# ====================================================================

# 切换到项目根目录
cd "$(dirname "$0")/.." || exit

python main.py \
  --dataset regdb \
  --mode train \
  --gpu 0 \
  --data-path ./datasets \
  \
  --backbone resnet101 \
  --pretrained \
  --amp \
  --num-parts 6 \
  --feature-dim 256 \
  \
  --batch-size 48 \
  --num-workers 8 \
  --pid-numsample 6 \
  --batch-pidnum 8 \
  --test-batch 128 \
  \
  --img-w 144 \
  --img-h 288 \
  \
  --total-epoch 80 \
  --warmup-epochs 5 \
  --lr 0.001 \
  --weight-decay 5e-4 \
  --lr-scheduler cosine \
  \
  --lambda-graph 0.5 \
  --lambda-orth 0.1 \
  --lambda-mod 0.5 \
  --lambda-triplet 0.5 \
  --label-smoothing 0.1 \
  \
  --save-epoch 10 \
  --eval-epoch 5 \
  --grad-clip 5.0 \
  \
  --init-memory \
  --pool-parts \
  --distance-metric cosine \
  --trial 1 \
  \
  --save-dir ./checkpoints/regdb_a100 \
  --log-dir ./logs/regdb_a100

echo ""
echo "======================================"
echo "RegDB A100 Training"
echo "======================================"
echo "Configuration:"
echo "  • Backbone: ResNet101 (pretrained)"
echo "  • Mixed Precision: Enabled"
echo "  • Epochs: 80"
echo "  • Batch Size: 48"
echo "  • Learning Rate: 0.001 (cosine decay)"
echo "  • Expected Performance:"
echo "    - Epoch 20: Rank-1 ~35-45%"
echo "    - Epoch 40: Rank-1 ~55-65%"
echo "    - Epoch 80: Rank-1 ~65-80%"
echo "======================================"
