#!/bin/bash

# ====================================================================
# LLCM Training Script - Standard Configuration (3090/V100)
# Expected Performance: Rank-1 ~55-65%, mAP ~50-60%
# Training Time: ~9 hours (120 epochs on 3090)
# ====================================================================

python main.py \
  --dataset llcm \
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
  --test-mode v2t \
  \
  --total-epoch 120 \
  --warmup-epochs 10 \
  --lr 0.0005 \
  --weight-decay 5e-4 \
  --lr-scheduler step \
  --lr-step 60 90 \
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
  --save-dir ./checkpoints/llcm_3090 \
  --log-dir ./logs/llcm_3090

echo ""
echo "======================================"
echo "LLCM Training (3090/Standard)"
echo "======================================"
echo "Configuration:"
echo "  • GPU: 3090/V100"
echo "  • Backbone: ResNet101 (pretrained)"
echo "  • Epochs: 120"
echo "  • Batch Size: 32"
echo "  • Learning Rate: 0.0005 (step decay at epoch 60, 90)"
echo "  • Test Mode: Visible to Thermal (v2t)"
echo "  • Expected Performance:"
echo "    - Epoch 40: Rank-1 ~30-40%"
echo "    - Epoch 80: Rank-1 ~45-55%"
echo "    - Epoch 120: Rank-1 ~55-65%"
echo "======================================"
