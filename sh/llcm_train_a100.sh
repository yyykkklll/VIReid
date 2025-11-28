#!/bin/bash

# ====================================================================
# LLCM Training Script - A100 Optimized
# ====================================================================

# 切换到项目根目录
cd "$(dirname "$0")/.." || exit

python main.py \
  --dataset llcm \
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
  --test-mode v2t \
  \
  --total-epoch 120 \
  --warmup-epochs 20 \
  --lr 0.0006 \
  --weight-decay 1e-3 \
  --lr-scheduler cosine \
  \
  --lambda-graph 0.1 \
  --lambda-orth 0.1 \
  --lambda-mod 0.5 \
  --lambda-triplet 0.5 \
  --label-smoothing 0.1 \
  \
  --save-epoch 10 \
  --eval-epoch 10 \
  --grad-clip 5.0 \
  \
  --init-memory \
  --pool-parts \
  --distance-metric cosine \
  \
  --save-dir ./checkpoints/llcm_a100 \
  --log-dir ./logs/llcm_a100

echo ""
echo "======================================"
echo "LLCM A100 Training"
echo "======================================"
echo "Configuration:"
echo "  • Backbone: ResNet50 (pretrained)"
echo "  • Mixed Precision: Enabled"
echo "  • Epochs: 120"
echo "  • Batch Size: 48"
echo "  • Learning Rate: 0.0006 (cosine decay)"
echo "  • Expected Performance:"
echo "    - Epoch 40: Rank-1 ~35-45%"
echo "    - Epoch 80: Rank-1 ~50-60%"
echo "    - Epoch 120: Rank-1 ~60-75%"
echo "======================================"
