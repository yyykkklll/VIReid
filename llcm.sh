#!/bin/bash
# llcm.sh
# VI-ReID Training Script (Sinkhorn + CMCL Optimized)
# Dataset: LLCM

# Clean up cache
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true

# ==================== Configuration ====================
DATASET="llcm"
ARCH="resnet"
DEVICE=0

# Use local datasets directory
DATA_PATH="./datasets/"
SAVE_PATH="save_llcm_cmcl"

# ==================== Training Settings ====================
# [Phase 1] Standard Supervised Pre-training
# LLCM default: 80
STAGE1_EPOCH=80

# [Phase 2] Weakly Supervised Learning (Sinkhorn + CMCL)
# LLCM default: 120
STAGE2_EPOCH=120

# Learning Rate
LR=0.0003

# Milestones for LR decay
MILESTONES="30 70"

# ==================== Batch Settings ====================
BATCH_PIDNUM=8
PID_NUMSAMPLE=4
TRI_WEIGHT=0.25
WEAK_WEIGHT=0.25
SIGMA=0.8
TEMPERATURE=3

# ==================== Execute ====================
echo "=========================================="
echo "LLCM TRAINING (Sinkhorn + CMCL)"
echo "  - Total Epochs: $((STAGE1_EPOCH + STAGE2_EPOCH))"
echo "  - Phase 1: 1 - $STAGE1_EPOCH"
echo "  - Phase 2: $((STAGE1_EPOCH + 1)) - $((STAGE1_EPOCH + STAGE2_EPOCH))"
echo "=========================================="

mkdir -p ./logs
export TMPDIR=$(pwd)/local_tmp

# Save log with timestamp
LOG_FILE="./logs/llcm_train_$(date +%Y%m%d_%H%M%S).log"

python main.py \
    --dataset $DATASET \
    --arch $ARCH \
    --device $DEVICE \
    --mode train \
    --debug wsl \
    --data-path $DATA_PATH \
    --save-path $SAVE_PATH \
    \
    --lr $LR \
    --milestones $MILESTONES \
    --stage1-epoch $STAGE1_EPOCH \
    --stage2-epoch $STAGE2_EPOCH \
    \
    --batch-pidnum $BATCH_PIDNUM \
    --pid-numsample $PID_NUMSAMPLE \
    \
    --tri-weight $TRI_WEIGHT \
    --weak-weight $WEAK_WEIGHT \
    --sigma $SIGMA \
    -T $TEMPERATURE \
    2>&1 | tee $LOG_FILE
