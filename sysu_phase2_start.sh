#!/bin/bash
# sysu_phase2_start.sh
# Start Phase 2 directly using pretrained Phase 1 weights
# Optimized for PRUD (Prototype-Rectified Unidirectional Distillation)

# ⚠️ 确保你的 Phase 1 权重路径正确
CHECKPOINT="/root/vireid/checkpoint.pth"

# ==================== Configuration ====================
DATASET="sysu"
DEVICE=0
DATA_PATH="./datasets/"

# ==================== Phase 2 Settings ====================
# Set Stage 1 to 0 to skip it (直接进入第二阶段)
STAGE1_EPOCH=0
STAGE2_EPOCH=120
LR=0.00035

# ==================== PRUD / CCPA Settings ====================
CCPA_START_EPOCH=5 

USE_CCPA=true

# 🔧 关键修改 2: 提高权重上限
# PRUD 自带置信度衰减，所以我们可以设置更高的基础权重 (0.8)，让高质量的伪标签发挥更大作用。
CCPA_WEIGHT=0.8

CCPA_THRESHOLD_MODE="hybrid"
PSEUDO_MOMENTUM=0.90

# ==================== Diffusion Settings (FIXED) ====================
USE_DIFFUSION=true
FEATURE_DIFFUSION_STEPS=5
SEMANTIC_DIFFUSION_STEPS=10
DIFFUSION_HIDDEN=1024

# 🔧 保持之前的修复 (保护 Backbone)
DIFFUSION_WEIGHT=0.1     # 保持 0.1，防止扩散 Loss 压倒 ID Loss
DIFFUSION_LR=0.0001      # 保持 0.0001，让扩散模块快速学习

# ==================== Execute ====================
echo "=========================================="
echo "SYSU-MM01 DIRECT PHASE 2 START (PRUD OPTIMIZED)"
echo "  - Loading weights from: $CHECKPOINT"
echo "  - Start Epoch: $CCPA_START_EPOCH"
echo "  - PRUD Weight: $CCPA_WEIGHT"
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