#!/bin/bash
# sysu_phase2.sh
# Load Phase 1 model and start Phase 2 directly
# Useful for debugging Phase 2 collapse or continuing training

# Clean up cache
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true

# ==================== Configuration ====================
DATASET="sysu"
ARCH="resnet"
SEARCH_MODE="all"
GALL_MODE="single"
DATA_PATH="./datasets/"
# Using a specific log folder for Phase 2 debugging
SAVE_PATH="sysu_phase2_debug"

# ==================== Training Settings ====================
# We skip Phase 1 effectively by loading the model, 
# but we keep variables consistent.
STAGE1_EPOCH=0 
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

# ==================== Model Loading ====================
# Path to the pre-trained Phase 1 model
MODEL_PATH="./best_phase1.pth"

# ==================== Execute ====================
echo "========================================== "
echo "SYSU-MM01 PHASE 2 TRAINING"
echo "  - Loading from: $MODEL_PATH"
echo "  - Save Path: $SAVE_PATH"
echo "========================================== "

python main.py \
    --dataset $DATASET \
    --arch $ARCH \
    --mode train \
    --debug wsl \
    --data-path $DATA_PATH \
    --save-path $SAVE_PATH \
    --model-path $MODEL_PATH \
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
