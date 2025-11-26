#!/bin/bash

# LLCM Fast Training - Optimized for Quick Convergence
# Expected time: ~2-3 hours for 60 epochs
# Expected Rank-1: 45-65%

python main.py \
  --dataset llcm \
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
  --img-w 144 \
  --img-h 288 \
  --relabel \
  --search-mode all \
  --gall-mode single \
  --test-mode v2t \
  --total-epoch 60 \
  --warmup-epochs 5 \
  --lr 0.001 \
  --weight-decay 5e-4 \
  --lambda-graph 0.5 \
  --lambda-orth 0.1 \
  --lambda-mod 0.5 \
  --save-epoch 10 \
  --eval-epoch 5 \
  --init-memory \
  --pool-parts \
  --distance-metric cosine \
  --save-dir ./checkpoints/llcm_fast \
  --log-dir ./logs/llcm_fast

echo "LLCM Fast Training Started!"
echo "Checkpoints: ./checkpoints/llcm_fast/"
echo "Logs: ./logs/llcm_fast/"
