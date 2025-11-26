#!/bin/bash

# RegDB Training with ResNet101 Backbone

python main.py \
  --dataset regdb \
  --mode train \
  --gpu 0 \
  --data-path ./datasets \
  --backbone resnet101 \
  --num-parts 6 \
  --feature-dim 256 \
  --batch-size 24 \
  --num-workers 4 \
  --pid-numsample 4 \
  --batch-pidnum 6 \
  --test-batch 64 \
  --img-w 144 \
  --img-h 288 \
  --total-epoch 60 \
  --warmup-epochs 5 \
  --lr 0.0008 \
  --weight-decay 5e-4 \
  --lambda-graph 0.5 \
  --lambda-orth 0.1 \
  --lambda-mod 0.5 \
  --save-epoch 10 \
  --eval-epoch 5 \
  --init-memory \
  --pool-parts \
  --distance-metric cosine \
  --trial 1 \
  --save-dir ./checkpoints/regdb_resnet101 \
  --log-dir ./logs/regdb_resnet101

echo "================================"
echo "Training with ResNet101 Backbone"
echo "Parameters: ~75M (vs 30M for ResNet50)"
echo "Expected improvement: +3-8% Rank-1"
echo "================================"
