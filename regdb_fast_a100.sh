#!/bin/bash

# RegDB Fast Training - ResNet50 with A100 Optimizations

python main.py \
  --dataset regdb \
  --mode train \
  --gpu 0 \
  --data-path ./datasets \
  --backbone resnet50 \
  --amp \
  --num-parts 6 \
  --feature-dim 256 \
  --batch-size 64 \
  --num-workers 8 \
  --pid-numsample 8 \
  --batch-pidnum 8 \
  --test-batch 128 \
  --img-w 144 \
  --img-h 288 \
  --total-epoch 60 \
  --warmup-epochs 5 \
  --lr 0.001 \
  --weight-decay 5e-4 \
  --lambda-graph 0.5 \
  --lambda-orth 0.1 \
  --lambda-mod 0.5 \
  --lambda-triplet 0.5 \
  --save-epoch 10 \
  --eval-epoch 5 \
  --init-memory \
  --pool-parts \
  --distance-metric cosine \
  --trial 1 \
  --save-dir ./checkpoints/regdb_fast_a100 \
  --log-dir ./logs/regdb_fast_a100

echo "================================"
echo "RegDB Fast Training (A100)"
echo "================================"
echo "Settings:"
echo "  ✓ Backbone: ResNet50"
echo "  ✓ Batch Size: 64 (Large!)"
echo "  ✓ Training Time: ~1.5 hours"
echo "================================"
