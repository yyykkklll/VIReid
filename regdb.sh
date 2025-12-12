#!/bin/bash

# ==================== Setup ====================
cd "$(dirname "$0")" || exit

export PYTHONPATH=$PYTHONPATH:.

# Clean Python cache
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null
find . -name "*.pyc" -delete 2>/dev/null

# ==================== Configuration ====================
TRIAL=${1:-1}

echo "=========================================="
echo "WSL-ReID Training on RegDB"
echo "=========================================="
echo "Trial: ${TRIAL}"
echo "Start Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="

# ==================== Training Command ====================
python3 main.py \
    \
    `# Basic Settings` \
    --dataset regdb \
    --arch resnet \
    --debug wsl \
    --save-path ./saved_regdb_resnet/regdb_${TRIAL} \
    --trial ${TRIAL} \
    \
    `# Training Phases` \
    --stage1-epoch 50 \
    --stage2-epoch 120 \
    \
    `# Data Settings` \
    --img-h 288 \
    --img-w 144 \
    --batch-pidnum 8 \
    --pid-numsample 4 \
    --test-batch 128 \
    --relabel 1 \
    \
    `# Optimizer and Scheduler` \
    --lr 0.00045 \
    --weight-decay 0.0005 \
    --milestones 50 70 \
    \
    `# Loss Function Settings` \
    --tri-weight 0.25 \
    --weak-weight 0.25 \
    \
    `# Cross-Modal Matching` \
    --sigma 0.8 \
    --temperature 3 \
    \
    `# Testing Settings` \
    --test-mode t2v \
    --search-mode all \
    --gall-mode single

# ==================== Training Complete ====================
echo "=========================================="
echo "Training Completed!"
echo "End Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Check results in ./saved_regdb_resnet/regdb_${TRIAL}/"
echo "=========================================="
