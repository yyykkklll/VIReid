#!/bin/bash
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
  --total-epoch 120 \
  --warmup-epochs 10 \
  --lr 0.00035 \
  --weight-decay 1e-3 \
  --lr-scheduler step \
  --lr-step 60,90 \
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
  \
  --save-dir ./checkpoints/sysu_a100 \
  --log-dir ./logs/sysu_a100

echo "Optimized SYSU Training Started..."