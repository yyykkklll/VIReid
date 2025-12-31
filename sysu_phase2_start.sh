#!/bin/bash
# sysu_phase2_start.sh
# Start Phase 2 directly using pretrained Phase 1 weights
# FIX: Adjusted Diffusion params to prevent backbone collapse

# Path to your best Phase 1 checkpoint
CHECKPOINT="/root/vireid/checkpoint.pth"

# ==================== Configuration ====================
DATASET="sysu"
DEVICE=0
DATA_PATH="./datasets/"

# ==================== Phase 2 Settings ====================
# Set Stage 1 to 0 to skip it
STAGE1_EPOCH=0
STAGE2_EPOCH=120
LR=0.00035

# ==================== CCPA Settings ====================
# Delay CCPA start to let Diffusion warm up.
CCPA_START_EPOCH=4
USE_CCPA=true
CCPA_WEIGHT=0.5
CCPA_THRESHOLD_MODE="hybrid"
PSEUDO_MOMENTUM=0.90

# ==================== Diffusion Settings (FIXED) ====================
USE_DIFFUSION=true
FEATURE_DIFFUSION_STEPS=5
SEMANTIC_DIFFUSION_STEPS=10
DIFFUSION_HIDDEN=1024

# 🔴 [CRITICAL FIX] Lower weight to protect backbone, higher LR to learn fast
DIFFUSION_WEIGHT=0.1    # 🔧 降至 0.1 (避免破坏主干特征)
DIFFUSION_LR=0.0001      # 🔧 提至 0.0001 (加速扩散模块收敛)

# ==================== Execute ====================
echo "=========================================="
echo "SYSU-MM01 DIRECT PHASE 2 START (FIXED)"
echo "  - Loading weights from: $CHECKPOINT"
echo "  - Diffusion Weight: $DIFFUSION_WEIGHT (Safe Mode)"
echo "  - Diffusion LR: $DIFFUSION_LR"
echo "=========================================="

python main.py \
    --dataset $DATASET \
    --device $DEVICE \
    --data-path $DATA_PATH \
    --arch resnet \
    --mode train \
    --debug wsl \
    --search-mode all \
    --gall-mode single \
    \
    --load-weights $CHECKPOINT \
    --stage1-epoch $STAGE1_EPOCH \
    --stage2-epoch $STAGE2_EPOCH \
    \
    --lr $LR \
    --batch-pidnum 12 \
    --pid-numsample 4 \
    --tri-weight 1.0 \
    --weak-weight 0.5 \
    --sigma 0.1 \
    -T 0.07 \
    --use-cosine-annealing \
    --eval-step 1 \
    \
    --use-diffusion \
    --feature-diffusion-steps $FEATURE_DIFFUSION_STEPS \
    --semantic-diffusion-steps $SEMANTIC_DIFFUSION_STEPS \
    --diffusion-hidden $DIFFUSION_HIDDEN \
    --diffusion-weight $DIFFUSION_WEIGHT \
    --diffusion-lr $DIFFUSION_LR \
    \
    --cross-attn-heads 8 \
    --cross-attn-dropout 0.15 \
    \
    --use-memory-bank \
    --memory-size-per-class 5 \
    \
    --ragm-temperature 0.12 \
    \
    --use-cycle-consistency \
    --ccpa-weight $CCPA_WEIGHT \
    --ccpa-start-epoch $CCPA_START_EPOCH \
    --ccpa-threshold-mode $CCPA_THRESHOLD_MODE \
    --pseudo-momentum $PSEUDO_MOMENTUM