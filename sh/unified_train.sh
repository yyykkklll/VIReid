#!/bin/bash
# ====================================================================
# Unified MTRL-Gated Training Script
# 适用于: RegDB, SYSU-MM01, LLCM
# ====================================================================
cd "$(dirname "$0")/.." || exit

# Dataset Config
DATASET=$1  # sysu, regdb, llcm
GPU=$2

if [ -z "$DATASET" ]; then
    echo "Usage: bash unified_train.sh [sysu|regdb|llcm] [gpu_id]"
    exit 1
fi

echo "🚀 Starting Unified Training for $DATASET on GPU $GPU..."

# 通用参数
COMMON_ARGS="--backbone resnet50 --pretrained --use-ibn --num-parts 6 --feature-dim 512"
# 开启所有模块 (因为有 Gate 和 MTRL 保护，不会崩)
MODULE_ARGS="--use-adversarial --use-graph-reasoning --lambda-graph 0.1 --lambda-adv 0.1"

if [ "$DATASET" == "regdb" ]; then
    python main.py --dataset regdb --gpu $GPU --data-path ./datasets \
      $COMMON_ARGS $MODULE_ARGS \
      --batch-size 64 --num-workers 8 --test-batch 128 \
      --img-w 144 --img-h 288 \
      --total-epoch 100 --warmup-epochs 10 --lr 0.00035 \
      --save-dir ./checkpoints/regdb_unified --log-dir ./logs/regdb_unified

elif [ "$DATASET" == "sysu" ]; then
    python main.py --dataset sysu --gpu $GPU --data-path ./datasets \
      $COMMON_ARGS $MODULE_ARGS \
      --amp \
      --batch-size 48 --num-workers 8 --test-batch 128 \
      --img-w 144 --img-h 288 \
      --search-mode all --gall-mode single \
      --total-epoch 120 --warmup-epochs 10 --lr 0.00035 \
      --save-dir ./checkpoints/sysu_unified --log-dir ./logs/sysu_unified

elif [ "$DATASET" == "llcm" ]; then
    python main.py --dataset llcm --gpu $GPU --data-path ./datasets \
      $COMMON_ARGS $MODULE_ARGS \
      --amp \
      --batch-size 48 --num-workers 8 --test-batch 128 \
      --img-w 144 --img-h 288 \
      --test-mode v2t \
      --total-epoch 120 --warmup-epochs 10 --lr 0.00035 \
      --save-dir ./checkpoints/llcm_unified --log-dir ./logs/llcm_unified
fi