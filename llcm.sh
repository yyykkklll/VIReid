#!/bin/bash

# LLCM Dataset Training Script for PF-MGCD

python main.py \
    --dataset llcm \
    --data-path ./datasets \
    --mode train \
    --gpu 0 \
    --seed 42 \
    --num-parts 6 \
    --feature-dim 256 \
    --memory-momentum 0.9 \
    --temperature 3.0 \
    --top-k 5 \
    --pretrained \
    --num-workers 4 \
    --pid-numsample 4 \
    --batch-pidnum 8 \
    --test-batch 64 \
    --img-w 144 \
    --img-h 288 \
    --relabel \
    --search-mode all \
    --gall-mode single \
    --test-mode v2t \
    --lambda-graph 1.0 \
    --lambda-orth 0.1 \
    --lambda-mod 0.5 \
    --label-smoothing 0.1 \
    --total-epoch 120 \
    --warmup-epochs 10 \
    --batch-size 32 \
    --lr 0.0003 \
    --weight-decay 5e-4 \
    --grad-clip 5.0 \
    --lr-scheduler step \
    --lr-step 40 \
    --lr-gamma 0.1 \
    --init-memory \
    --save-dir ./checkpoints/llcm \
    --log-dir ./logs/llcm \
    --save-epoch 10 \
    --eval-epoch 10