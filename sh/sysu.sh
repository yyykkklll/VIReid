#!/bin/bash
# ====================================================================
# SYSU-MM01 Ultimate Training (All Features Enabled)
# Config: IBN + GeM + Adversarial + Graph Reasoning + AMP
# ====================================================================

cd "$(dirname "$0")/.." || exit

python main.py \
  --dataset sysu \
  --mode train \
  --gpu 0 \
  --data-path ./datasets \
  \
  --backbone resnet50 \
  --pretrained \
  --use-ibn \
  --amp \
  \
  --num-parts 6 \
  --feature-dim 512 \
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
  --use-adversarial \
  --use-graph-reasoning \
  \
  --total-epoch 120 \
  --warmup-epochs 10 \
  --lr 0.00035 \
  --weight-decay 5e-4 \
  --lr-scheduler cosine \
  \
  --lambda-graph 0.2 \
  --lambda-adv 0.1 \
  --lambda-triplet 1.0 \
  --label-smoothing 0.1 \
  \
  --init-memory \
  --memory-momentum 0.9 \
  --temperature 3.0 \
  --top-k 5 \
  \
  --save-epoch 10 \
  --eval-epoch 5 \
  --grad-clip 5.0 \
  \
  --save-dir ./checkpoints/sysu_ultimate \
  --log-dir ./logs/sysu_ultimate