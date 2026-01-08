#!/bin/bash
# sysu_phase2.sh
# VI-ReID Training Script - Phase 2 ONLY
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
# Adjust this path if your file is located elsewhere
PHASE1_MODEL_PATH="./saved_sysu_resnet/save_sysu_cmcl/models/best_phase1.pth"

# ==================== Training Settings ====================
# [Phase 1] Skipped (Set to 0)
STAGE1_EPOCH=0

# [Phase 2] Weakly Supervised Learning
# Total Epochs = STAGE2_EPOCH
# Since we skip Phase 1, start_epoch will be determined by the loaded model (e.g. 20)
# So it will run from 20 to 120.
STAGE2_EPOCH=120

# Learning Rate
LR=0.0003
MILESTONES="30 70"

# ==================== Batch Settings ====================
BATCH_PIDNUM=8
PID_NUMSAMPLE=4
TRI_WEIGHT=0.25
WEAK_WEIGHT=0.25
SIGMA=0.8
TEMPERATURE=3

# ==================== Execute ====================
echo "========================================== "
echo "SYSU-MM01 TRAINING - PHASE 2 ONLY"
echo "  - Loading Model: $PHASE1_MODEL_PATH"
echo "  - Phase 1: Skipped"
echo "  - Phase 2: Ends at Epoch $STAGE2_EPOCH"
echo "========================================== "

mkdir -p ./logs
export TMPDIR=$(pwd)/local_tmp

# Save log with timestamp
LOG_FILE="./logs/sysu_phase2_train_$(date +%Y%m%d_%H%M%S).log"

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
