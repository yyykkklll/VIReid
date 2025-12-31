#!/bin/bash
# sysu.sh
# VI-ReID Full Training Script (Stability & Performance Optimized)
# 修复了 Phase 2 梯度爆炸和 CCPA 质量崩塌的问题

# Clean up cache to ensure code changes take effect
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true

# ==================== Configuration ====================
DATASET="sysu"
DEVICE=0
SEARCH_MODE="all"
GALL_MODE="single"

# Use local datasets directory
DATA_PATH="./datasets/"

# ==================== Training Settings ====================
# [Phase 1] Standard Supervised Pre-training
# 50 Epochs is sufficient for ResNet50 to converge on ID Loss
STAGE1_EPOCH=50

# [Phase 2] Weakly Supervised Learning + Diffusion
# Total training will last 50 + 120 = 170 Epochs
STAGE2_EPOCH=120

# Global Learning Rate (for Backbone)
LR=0.00035

# ==================== CCPA Settings (Safe Mode) ====================
USE_CCPA=true

# 🔧 [CRITICAL ADJUSTMENT] Delayed Start
# Phase 1 ends at 50. Phase 2 starts at 51.
# We set CCPA start to 75, giving Diffusion 25 Epochs (51-75) to warm up.
# Do NOT set this too early (e.g., 60), or you risk "Quality Collapse".
CCPA_START_EPOCH=65

CCPA_WEIGHT=0.5         
CCPA_THRESHOLD_MODE="hybrid"
PSEUDO_MOMENTUM=0.90

# ==================== Batch Settings ====================
BATCH_PIDNUM=12
PID_NUMSAMPLE=4
TRI_WEIGHT=1.0
WEAK_WEIGHT=0.5
SIGMA=0.1
TEMPERATURE=0.07

# ==================== Diffusion (Heteroscedastic UNet) ====================
USE_DIFFUSION=true
FEATURE_DIFFUSION_STEPS=5
SEMANTIC_DIFFUSION_STEPS=10
DIFFUSION_HIDDEN=1024

# 🔴 [CRITICAL FIX] Stability Settings
# Previous 0.5 was causing Gradient Explosion (Norm > 100).
# Lowering to 0.1 ensures the backbone features aren't corrupted by initial noise.
DIFFUSION_WEIGHT=0.1 

# Previous 0.0005/0.001 was too high for Transformer/UNet from scratch.
# 0.0001 is the standard safe LR for diffusion models in this architecture.
DIFFUSION_LR=0.0001     

# Cross-Attention
CROSS_ATTN_HEADS=8
CROSS_ATTN_DROPOUT=0.15

# ==================== Reliability Gating (Adaptive) ====================
USE_MEMORY_BANK=true
MEMORY_SIZE_PER_CLASS=5
RAGM_TEMPERATURE=0.12   # Adaptive scaling base temperature

# ==================== Execute ====================
echo "=========================================="
echo "SYSU-MM01 FULL TRAINING (STABLE)"
echo "  - Total Epochs: $((STAGE1_EPOCH + STAGE2_EPOCH))"
echo "  - Phase 1: 1 - $STAGE1_EPOCH (Supervised)"
echo "  - Phase 2: $((STAGE1_EPOCH + 1)) - $((STAGE1_EPOCH + STAGE2_EPOCH)) (WSL)"
echo "  - Diffusion Warmup: Epoch $((STAGE1_EPOCH + 1)) - $((CCPA_START_EPOCH - 1))"
echo "  - CCPA Active: Epoch $CCPA_START_EPOCH+"
echo "  - Diffusion Weight: $DIFFUSION_WEIGHT (Low for Stability)"
echo "  - Diffusion LR: $DIFFUSION_LR (Safe)"
echo "=========================================="

mkdir -p ./logs
export TMPDIR=$(pwd)/local_tmp

# Save log with timestamp
LOG_FILE="./logs/sysu_full_train_$(date +%Y%m%d_%H%M%S).log"

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