#!/bin/bash
# sysu_fast_debug_extended.sh
# Extended Verification: 4 Epoch Phase1 + 12 Epoch Phase2
# Goal: Verify Phase Transition fix & Diffusion Stability

# Clean up cache
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true

# ==================== Configuration ====================
DATASET="sysu"
DEVICE=0
SEARCH_MODE="all"
GALL_MODE="single"
DATA_PATH="./datasets/"

# ==================== Extended Debug Settings ====================
# 🔧 策略：快速通过 Phase 1，但在 Phase 2 停留足够久以观察稳定性
STAGE1_EPOCH=4
STAGE2_EPOCH=12
LR=0.00035

# ==================== CCPA Settings ====================
USE_CCPA=true
# 🔧 策略：Phase 2 从 Epoch 4 开始。
# 我们让 CCPA 在 Epoch 10 启动（即 Phase 2 运行 6 个 Epoch 后）。
# 这能测试“热身期”是否足够消除 Quality Collapse 警告。
CCPA_START_EPOCH=10
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

# ==================== Diffusion Settings (Optimized) ====================
USE_DIFFUSION=true
FEATURE_DIFFUSION_STEPS=5
SEMANTIC_DIFFUSION_STEPS=10
DIFFUSION_HIDDEN=1024

# ✅ 保持之前修复的参数：高权重，防止梯度饥饿
DIFFUSION_WEIGHT=0.5 
DIFFUSION_LR=0.0005

CROSS_ATTN_HEADS=8
CROSS_ATTN_DROPOUT=0.15

# ==================== Reliability Gating ====================
USE_MEMORY_BANK=true
MEMORY_SIZE_PER_CLASS=5
RAGM_TEMPERATURE=0.12

# ==================== Execute ====================
echo "=========================================="
echo "SYSU-MM01 EXTENDED DEBUG MODE"
echo "  - Phase 1: $STAGE1_EPOCH epochs"
echo "  - Phase 2: $STAGE2_EPOCH epochs"
echo "  - Transition at: Epoch $STAGE1_EPOCH"
echo "  - CCPA Start: Epoch $CCPA_START_EPOCH"
echo "=========================================="

mkdir -p ./local_debug_tmp
export TMPDIR=$(pwd)/local_debug_tmp

# Log filename with timestamp
LOG_FILE="debug_extended_$(date +%Y%m%d_%H%M%S).log"

python main.py \
    --dataset $DATASET \
    --device $DEVICE \
    --data-path $DATA_PATH \
    --arch resnet \
    --mode train \
    --debug wsl \
    --search-mode $SEARCH_MODE \
    --gall-mode $GALL_MODE \
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
    --eval-step 1 \
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
    2>&1 | tee $LOG_FILE

echo ""
echo "📝 Log saved to: $LOG_FILE"
echo "✅ Please check if 'Switching to Phase 2' appears after Epoch $STAGE1_EPOCH"