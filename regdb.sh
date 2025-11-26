#!/bin/bash

python main.py \
  --dataset regdb \
  --mode train \
  --gpu 0 \
  --data-path ./datasets \
  --num-parts 6 \
  --feature-dim 256 \
  --batch-size 32 \
  --num-workers 4 \
  --pid-numsample 4 \
  --batch-pidnum 8 \
  --test-batch 64 \
  --total-epoch 120 \
  --warmup-epochs 10 \
  --lr 0.00045 \
  --lambda-graph 1.0 \
  --lambda-orth 0.1 \
  --lambda-mod 0.5 \
  --save-epoch 10 \
  --eval-epoch 10 \
  --init-memory \
  --pool-parts \
  --distance-metric cosine \
  --trial 1
