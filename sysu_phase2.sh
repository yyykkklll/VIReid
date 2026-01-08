#!/bin/bash
# sysu_phase2.sh
# VI-ReID Training Script - Phase 2 ONLY (Optimized for Stability)
# Skips Phase 1 and loads a pre-trained Phase 1 model

# Clean up cache
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true

# ==================== Configuration ====================
DATASET="sysu"
ARCH="resnet"
DEVICE=0
SEARCH_MODE="all"
GALL_MODE="single"
DATA_PATH="./datasets/"
SAVE_PATH="save_sysu_cmcl"

# Path to the BEST Phase 1 Checkpoint
PHASE1_MODEL_PATH="./saved_sysu_resnet/save_sysu_cmcl/models/best_phase1.pth"

# ==================== Training Settings ====================
STAGE1_EPOCH=0
# Continue Phase 2 for enough epochs
STAGE2_EPOCH=120

# [OPTIMIZATION 1] Lower Learning Rate for Phase 2
# Previous 0.0003 caused severe oscillation. Lowering to 0.00015.
LR=0.00015
MILESTONES="40 80"

# ==================== Batch Settings ====================
BATCH_PIDNUM=8
PID_NUMSAMPLE=4
TRI_WEIGHT=0.25
WEAK_WEIGHT=0.25

# [OPTIMIZATION 2] Stabilize Memory Bank Update
# Previous sigma 0.8 (update 80% new) was too volatile.
# Changing to 0.2 (update 20% new, keep 80% old) to smooth out features.
SIGMA=0.2
TEMPERATURE=3

# ==================== Execute ====================
echo "=========================================="
echo "SYSU-MM01 TRAINING - PHASE 2 (STABLE)"
echo "  - LR: $LR (Reduced)"
echo "  - Sigma: $SIGMA (Stabilized)"
echo "========================================== "

mkdir -p ./logs
export TMPDIR=$(pwd)/local_tmp

LOG_FILE="./logs/sysu_phase2_stable_$(date +%Y%m%d_%H%M%S).log"

python main.py \
    --dataset $DATASET \
    --arch $ARCH \
    --device $DEVICE \
    --mode train \
    --debug wsl \
    --data-path $DATA_PATH \
    --save-path $SAVE_PATH \
    --model-path $PHASE1_MODEL_PATH \
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
    \
    --search-mode $SEARCH_MODE \
    --gall-mode $GALL_MODE \
    2>&1 | tee $LOG_FILE