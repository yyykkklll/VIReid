#!/bin/bash

# ====================================================================
# SYSU-MM01 Training Script - Standard Configuration (3090/V100)
# Expected Performance: Rank-1 ~50-60%, mAP ~45-55%
# Training Time: ~7 hours (100 epochs on 3090)
# ====================================================================

python main.py \
  --dataset sysu \
  --mode train \
  --gpu 0 \
  --data-path ./datasets \
  \
  --backbone resnet101 \
  --pretrained \
  --num-parts 6 \
  --feature-dim 256 \
  \
  --batch-size 32 \
  --num-workers 4 \
  --pid-numsample 4 \
  --batch-pidnum 8 \
  --test-batch 64 \
  \
  --img-w 144 \
  --img-h 288 \
  --relabel \
  --search-mode all \
  --gall-mode single \
  \
  --total-epoch 100 \
  --warmup-epochs 10 \
  --lr 0.0006 \
  --weight-decay 5e-4 \
  --lr-scheduler step \
  --lr-step 50 \
  --lr-gamma 0.1 \
  \
  --lambda-graph 0.5 \
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
  --save-dir ./checkpoints/sysu_3090 \
  --log-dir ./logs/sysu_3090

echo ""
echo "======================================"
echo "SYSU-MM01 Training (3090/Standard)"
echo "======================================"
echo "Configuration:"
echo "  • GPU: 3090/V100"
echo "  • Backbone: ResNet101 (pretrained)"
echo "  • Epochs: 100"
echo "  • Batch Size: 32"
echo "  • Learning Rate: 0.0006 (step decay at epoch 50)"
echo "  • Search Mode: all-search"
echo "  • Expected Performance:"
echo "    - Epoch 30: Rank-1 ~25-35%"
echo "    - Epoch 60: Rank-1 ~40-50%"
echo "    - Epoch 100: Rank-1 ~50-60%"
echo "======================================"
