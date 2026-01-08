#!/bin/bash
# llcm.sh
# VI-ReID Training Script (Sinkhorn + CMCL Optimized)
# Dataset: LLCM
# Full Pipeline: Enhanced Phase 1 (Label Smoothing + More Epochs) -> Stable Phase 2

# Clean up cache
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true

# ==================== Configuration ====================
DATASET="llcm"
ARCH="resnet"
DEVICE=0
DATA_PATH="./datasets/"
SAVE_PATH="save_llcm_cmcl"

# ==================== Training Settings ====================
# [Phase 1] Enhanced Supervised Pre-training
STAGE1_EPOCH=80

# [Phase 2] Weakly Supervised Learning
# LLCM usually needs more epochs
STAGE2_EPOCH=140

# Learning Rate
LR=0.0003
MILESTONES="40 100"

# Early Stopping
PATIENCE=15

# ==================== Batch Settings ====================
BATCH_PIDNUM=8
PID_NUMSAMPLE=4
TRI_WEIGHT=0.25
WEAK_WEIGHT=0.25

# [Phase 2 Stability]
SIGMA=0.2
TEMPERATURE=3

# ==================== Execute ====================
echo "=========================================="
echo "LLCM FULL TRAINING (ENHANCED)"
echo "  - Phase 1: 80 Epochs"
echo "  - Phase 2: 140 Epochs"
echo "  - Patience: $PATIENCE"
echo "=========================================="

mkdir -p ./logs
export TMPDIR=$(pwd)/local_tmp

LOG_FILE="./logs/llcm_full_enhanced_$(date +%Y%m%d_%H%M%S).log"

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
    --patience $PATIENCE \
    \
    --batch-pidnum $BATCH_PIDNUM \
    --pid-numsample $PID_NUMSAMPLE \
    \
    --tri-weight $TRI_WEIGHT \
    --weak-weight $WEAK_WEIGHT \
    --sigma $SIGMA \
    -T $TEMPERATURE \
    2>&1 | tee $LOG_FILE