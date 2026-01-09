#!/bin/bash
# llcm.sh
# VI-ReID Training Script for LLCM
# Template based on demo.sh

# Clean up cache
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true

# ==================== Configuration ====================
DATASET="llcm"
ARCH="resnet"
SEARCH_MODE="all"
GALL_MODE="single"
DATA_PATH="./datasets/"
SAVE_PATH="llcm_log"

# ==================== Training Settings ====================
# [Phase 1] Warm-up
# Optimization: LLCM is large, Phase 1 is crucial. Increased from default 20 to 40.
STAGE1_EPOCH=40

# [Phase 2] Main Training
STAGE2_EPOCH=140

# Learning Rate & Milestones
LR=0.0003
MILESTONES="40 100"

# ==================== Batch Settings ====================
BATCH_PIDNUM=8
PID_NUMSAMPLE=4
TRI_WEIGHT=0.25
WEAK_WEIGHT=0.25 
PATIENCE=15

# ==================== Execute ====================
echo "=========================================="
echo "LLCM TRAINING (Optimized)"
echo "  - Phase 1: $STAGE1_EPOCH Epochs"
echo "  - Phase 2: End at $STAGE2_EPOCH Epochs"
echo "=========================================="

python main.py \
    --dataset $DATASET \
    --arch $ARCH \
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
    --patience $PATIENCE \
    \
    --search-mode $SEARCH_MODE \
    --gall-mode $GALL_MODE \
    --test-mode t2v
