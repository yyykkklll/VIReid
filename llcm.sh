#!/bin/bash
# llcm.sh
# Optimized for PRUD Framework (Identical config to SYSU)

find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true

DATASET="llcm"
DEVICE=0
SEARCH_MODE="all"
GALL_MODE="single"
DATA_PATH="./datasets/"

# Training Settings
STAGE1_EPOCH=50
STAGE2_EPOCH=120
LR=0.00035

# PRUD Settings
USE_CCPA=true
CCPA_START_EPOCH=60   # Stage 1 (50) + 10 Warmup
CCPA_WEIGHT=0.8       # High confidence weight
CCPA_THRESHOLD_MODE="hybrid"
PSEUDO_MOMENTUM=0.90

# Batch & Loss Settings
BATCH_PIDNUM=12
PID_NUMSAMPLE=4
TRI_WEIGHT=1.0
WEAK_WEIGHT=0.5
SIGMA=0.1
TEMPERATURE=0.07

# Diffusion Settings
USE_DIFFUSION=true
FEATURE_DIFFUSION_STEPS=5
SEMANTIC_DIFFUSION_STEPS=10
DIFFUSION_HIDDEN=1024
DIFFUSION_WEIGHT=0.1
DIFFUSION_LR=0.0001
CROSS_ATTN_HEADS=8
CROSS_ATTN_DROPOUT=0.15

# Memory Bank
USE_MEMORY_BANK=true
MEMORY_SIZE_PER_CLASS=5
RAGM_TEMPERATURE=0.12

echo "=========================================="
echo "LLCM FULL TRAINING (PRUD OPTIMIZED)"
echo "  - PRUD Start: Epoch $CCPA_START_EPOCH"
echo "  - PRUD Weight: $CCPA_WEIGHT"
echo "=========================================="

mkdir -p ./logs
export TMPDIR=$(pwd)/local_tmp
LOG_FILE="./logs/llcm_prud_$(date +%Y%m%d_%H%M%S).log"

python main.py \
    --dataset $DATASET \
    --device $DEVICE \
    --arch resnet \
    --mode train \
    --debug wsl \
    --search-mode $SEARCH_MODE \
    --gall-mode $GALL_MODE \
    --data-path $DATA_PATH \
    \
    --lr $LR \
    --stage1-epoch $STAGE1_EPOCH \
    --stage2-epoch $STAGE2_EPOCH \
    --batch-pidnum $BATCH_PIDNUM \
    --pid-numsample $PID_NUMSAMPLE \
    --tri-weight $TRI_WEIGHT \
    --weak-weight $WEAK_WEIGHT \
    --sigma $SIGMA \
    -T $TEMPERATURE \
    --use-cosine-annealing \
    \
    --use-diffusion \
    --feature-diffusion-steps $FEATURE_DIFFUSION_STEPS \
    --semantic-diffusion-steps $SEMANTIC_DIFFUSION_STEPS \
    --diffusion-hidden $DIFFUSION_HIDDEN \
    --diffusion-weight $DIFFUSION_WEIGHT \
    --diffusion-lr $DIFFUSION_LR \
    \
    --cross-attn-heads $CROSS_ATTN_HEADS \
    --cross-attn-dropout $CROSS_ATTN_DROPOUT \
    \
    --use-memory-bank \
    --memory-size-per-class $MEMORY_SIZE_PER_CLASS \
    \
    --ragm-temperature $RAGM_TEMPERATURE \
    \
    --use-cycle-consistency \
    --ccpa-weight $CCPA_WEIGHT \
    --ccpa-start-epoch $CCPA_START_EPOCH \
    --ccpa-threshold-mode $CCPA_THRESHOLD_MODE \
    --pseudo-momentum $PSEUDO_MOMENTUM \
    --eval-step 1 \
    2>&1 | tee $LOG_FILE