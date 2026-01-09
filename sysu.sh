#!/bin/bash
# sysu.sh
# VI-ReID Training Script for SYSU-MM01
# Template based on demo.sh

# Clean up cache
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true

# ==================== Configuration ====================
DATASET="sysu"
ARCH="resnet"
SEARCH_MODE="all"
GALL_MODE="single"
DATA_PATH="./datasets/"
SAVE_PATH="sysu_log"

# ==================== Training Settings ====================
# [Phase 1] Warm-up / Pre-training
# Optimization: Increased from 20 to 40 to ensure better initial features
STAGE1_EPOCH=40

# [Phase 2] Main Training
# Optimization: Increased to 140 to allow more convergence time after CMA
STAGE2_EPOCH=140

# Learning Rate & Milestones
LR=0.0003
# Decay adapted to new epoch structure
MILESTONES="40 100"

# ==================== Batch Settings ====================
BATCH_PIDNUM=8
PID_NUMSAMPLE=4
TRI_WEIGHT=0.25
WEAK_WEIGHT=0.25 
PATIENCE=15

# ==================== Execute ====================
echo "=========================================="
echo "SYSU-MM01 TRAINING (Optimized)"
echo "  - Phase 1: $STAGE1_EPOCH Epochs"
echo "  - Phase 2: End at $STAGE2_EPOCH Epochs"
echo "  - Patience: $PATIENCE"
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
    --gall-mode $GALL_MODE
