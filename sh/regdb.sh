#!/bin/bash
# RegDB Simple Baseline - 60 Epochs 快速验证
# 目标: 验证简化模型的性能


# 自动切换到项目根目录
cd "$(dirname "$0")/.." || exit

echo "🚀 RegDB Baseline V3 (60 Epochs)"

python main.py \
    --dataset regdb \
    --data-path ./datasets \
    --mode train \
    --gpu 0 \
    --seed 42 \
    \
    --num-parts 6 \
    --feature-dim 512 \
    --pretrained \
    --backbone resnet50 \
    --amp \
    \
    --use-ibn \
    \
    --num-classes 206 \
    --num-workers 4 \
    --pid-numsample 8 \
    --batch-pidnum 8 \
    --test-batch 128 \
    --img-w 144 \
    --img-h 288 \
    --relabel \
    --trial 1 \
    \
    --lambda-triplet 1.0 \
    --label-smoothing 0.1 \
    \
    --total-epoch 60 \
    --warmup-epochs 5 \
    --batch-size 64 \
    --lr 0.0007 \
    --weight-decay 5e-4 \
    --grad-clip 5.0 \
    \
    --save-dir ./checkpoints/regdb_baseline_v3 \
    --log-dir ./logs/regdb_baseline_v3 \
    --save-epoch 20 \
    --eval-epoch 5

echo "✅ Training Complete!"