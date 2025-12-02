#!/bin/bash
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
  --batch-size 48 \
  --num-workers 8 \
  --pid-numsample 6 \
  --batch-pidnum 8 \
  --test-batch 128 \
  \
  --img-w 144 \
  --img-h 288 \
  \
  --total-epoch 100 \
  --warmup-epochs 10 \
  --lr 0.00035 \
  --weight-decay 1e-3 \
  --lr-scheduler step \
  --lr-step 40,70 \
  --lr-gamma 0.1 \
  \
  --lambda-graph 0.1 \
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

echo "Optimized RegDB Training Started..."