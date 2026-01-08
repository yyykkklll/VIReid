#!/bin/bash
# sysu.sh
# VI-ReID Training Script (Sinkhorn + CMCL Optimized)
# Dataset: SYSU-MM01
# Full Pipeline: Enhanced Phase 1 (Label Smoothing + More Epochs) -> Stable Phase 2

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

# ==================== Training Settings ====================
# [Phase 1] Enhanced Supervised Pre-training
# Increased from 20 to 60 for better convergence
STAGE1_EPOCH=60

# [Phase 2] Weakly Supervised Learning (Sinkhorn + CMCL)
# Total Epochs = Phase 1 + Phase 2 (e.g. 60 + 120 = 180)
# Phase 2 runs for 120 epochs
STAGE2_EPOCH=120

# Learning Rate
# Start at 0.00035 (slightly higher for label smoothing)
LR=0.00035

# Milestones
# Phase 1 decays at 40. Phase 2 decays at 80 (relative to Phase 2 start) or global?
# The code uses global milestones if passed directly.
# Let's check main.py/scheduler logic. 
# Scheduler uses `current_epoch`.
# In Phase 1: 0 -> 60. Decay at 40 is good.
# In Phase 2: starts at 60 -> 180. 
# We want Phase 2 to start with decent LR again?
# The scheduler is re-initialized? No, model is re-loaded.
# Actually, in main.py, scheduler_phase1 and scheduler_phase2 are separate.
# scheduler_phase1 runs for 0-60. 
# scheduler_phase2 runs for 60-180.
# So we need milestones that cover both ranges appropriately.
# Milestones: "40 100" (Decay at 40 for Ph1, then at 100 (Ph2 epoch 40))
MILESTONES="40 100"

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
echo "SYSU-MM01 FULL TRAINING (ENHANCED)"
echo "  - Phase 1: 60 Epochs (Label Smoothing)"
echo "  - Phase 2: 120 Epochs (Stable Sinkhorn)"
echo "  - Milestones: $MILESTONES"
echo "=========================================="

mkdir -p ./logs
export TMPDIR=$(pwd)/local_tmp

LOG_FILE="./logs/sysu_full_enhanced_$(date +%Y%m%d_%H%M%S).log"

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
    \
    --search-mode $SEARCH_MODE \
    --gall-mode $GALL_MODE \
    2>&1 | tee $LOG_FILE