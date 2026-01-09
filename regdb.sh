#!/bin/bash
# regdb.sh
# VI-ReID Training Script for RegDB
# Template based on demo.sh

# Clean up cache
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true

# ==================== Configuration ====================
DATASET="regdb"
ARCH="resnet"
SEARCH_MODE="all" 
GALL_MODE="single" 
DATA_PATH="./datasets/"
SAVE_PATH="regdb_log"
TRIAL=1

# ==================== Training Settings ====================
# [Phase 1] Warm-up
# Optimization: Increased from 50 to 60
STAGE1_EPOCH=60

# [Phase 2] Main Training
# RegDB usually converges faster/differently, usually 120 is fine.
STAGE2_EPOCH=120

# Learning Rate & Milestones
LR=0.00045 # Higher LR for RegDB as per original config
MILESTONES="50 70"

# ==================== Batch Settings ====================
BATCH_PIDNUM=8
PID_NUMSAMPLE=4
TRI_WEIGHT=0.25
WEAK_WEIGHT=0.25 
PATIENCE=15

# ==================== Execute ====================
echo "=========================================="
echo "RegDB TRAINING (Optimized) - Trial $TRIAL"
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
    --trial $TRIAL \
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
    --test-mode t2v
