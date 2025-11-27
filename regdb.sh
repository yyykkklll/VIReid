#!/bin/bash

# ====================================================================
# RegDB Training Script - Standard Configuration (3090/V100)
# Expected Performance: Rank-1 ~60-70%, mAP ~55-65%
# Training Time: ~3 hours (80 epochs on 3090)
# ====================================================================

python main.py \
  --dataset regdb \
  --mode train \
  --gpu 0 \
  --data-path ./datasets \
  \
  --backbone resnet50 \
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
  \
  --total-epoch 80 \
  --warmup-epochs 5 \
  --lr 0.0008 \
  --weight-decay 5e-4 \
  --lr-scheduler step \
  --lr-step 40 \
  --lr-gamma 0.1 \
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
  --save-dir ./checkpoints/regdb_3090 \
  --log-dir ./logs/regdb_3090

echo ""
echo "======================================"
echo "RegDB Training (3090/Standard)"
echo "======================================"
echo "Configuration:"
echo "  • GPU: 3090/V100"
echo "  • Backbone: ResNet50 (pretrained)"
echo "  • Epochs: 80"
echo "  • Batch Size: 32"
echo "  • Learning Rate: 0.0008 (step decay at epoch 40)"
echo "  • Expected Performance:"
echo "    - Epoch 20: Rank-1 ~30-40%"
echo "    - Epoch 40: Rank-1 ~50-60%"
echo "    - Epoch 80: Rank-1 ~60-70%"
echo "======================================"
