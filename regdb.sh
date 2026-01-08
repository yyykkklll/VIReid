#!/bin/bash
# regdb.sh
# VI-ReID Training Script (Sinkhorn + CMCL Optimized)
# Dataset: RegDB
# Full Pipeline: Enhanced Phase 1 (Label Smoothing + More Epochs) -> Stable Phase 2

# Clean up cache
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true

# ==================== Configuration ====================
DATASET="regdb"
ARCH="resnet"
DEVICE=0
TRIAL=1
DATA_PATH="./datasets/"
SAVE_PATH="save_regdb_cmcl"

# ==================== Training Settings ====================
# [Phase 1] Enhanced Supervised Pre-training
STAGE1_EPOCH=60

# [Phase 2] Weakly Supervised Learning
STAGE2_EPOCH=120

# Learning Rate
# RegDB uses higher LR
LR=0.00045
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
echo "RegDB FULL TRAINING (ENHANCED) - Trial $TRIAL"
echo "  - Phase 1: 60 Epochs"
echo "  - Phase 2: 120 Epochs"
echo "  - Patience: $PATIENCE"
echo "=========================================="

mkdir -p ./logs
export TMPDIR=$(pwd)/local_tmp

LOG_FILE="./logs/regdb_trial${TRIAL}_full_enhanced_$(date +%Y%m%d_%H%M%S).log"

python main.py \
    --dataset $DATASET \
    --arch $ARCH \
    --device $DEVICE \
    --mode train \
    --debug wsl \
    --trial $TRIAL \
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