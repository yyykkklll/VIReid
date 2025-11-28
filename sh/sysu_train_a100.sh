#!/bin/bash

# ====================================================================
# SYSU-MM01 Training Script - A100 Optimized
# ====================================================================

# 切换到项目根目录
cd "$(dirname "$0")/.." || exit

python main.py \
  --dataset sysu \
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
  \
  --total-epoch 100 \
  --warmup-epochs 20 \
  --lr 0.0008 \
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
  --save-dir ./checkpoints/sysu_a100 \
  --log-dir ./logs/sysu_a100

echo ""
echo "======================================"
echo "SYSU-MM01 A100 Training"
echo "======================================"
echo "Configuration:"
echo "  • Backbone: ResNet50 (pretrained)"
echo "  • Mixed Precision: Enabled"
echo "  • Epochs: 100"
echo "  • Batch Size: 48"
echo "  • Learning Rate: 0.0008 (cosine decay)"
echo "  • Expected Performance:"
echo "    - Epoch 30: Rank-1 ~30-40%"
echo "    - Epoch 60: Rank-1 ~45-55%"
echo "    - Epoch 100: Rank-1 ~55-70%"
echo "======================================"
