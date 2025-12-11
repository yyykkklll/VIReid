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
python main.py \
    \
    `# Basic Settings` \
    --dataset regdb \
    --arch resnet \
    --mode train \
    --data-path ./datasets \
    --save-path ./save \
    --seed 42 \
    --num-workers 8 \
    \
    `# Training Phases - Optimized for reasonable training time` \
    --stage1-epoch 40 \
    --stage2-epoch 60 \
    --trial ${TRIAL} \
    \
    `# Data Settings` \
    --img-h 288 \
    --img-w 144 \
    --batch-pidnum 16 \
    --pid-numsample 4 \
    --test-batch 128 \
    --relabel 1 \
    \
    `# Optimizer and Scheduler` \
    --lr 0.0003 \
    --weight-decay 0.0005 \
    --milestones 30 50 \
    --warmup-epochs 5 \
    \
    `# Loss Function Settings` \
    --tri-weight 0.25 \
    --weak-weight 0.25 \
    --label-smoothing 0.1 \
    --contrastive-temp 0.07 \
    --intra-contrastive-weight 0.1 \
    --contrastive-weight 0.3 \
    \
    `# Loss Schedule` \
    --triplet-warmup-epochs 8 \
    --contrastive-start-epoch 5 \
    --cross-contrastive-start 3 \
    --cmo-start-epoch 3 \
    --cmo-warmup 8 \
    --weak-start-epoch 5 \
    --weak-warmup 12 \
    \
    `# Cross-Modal Matching` \
    --sigma 0.3 \
    --temperature 0.05 \
    \
    `# Advanced Features (Optional - comment out if not needed)` \
    --use-clip \
    --w-clip 0.3 \
    --use-sinkhorn \
    --sinkhorn-reg 0.05 \
    \
    `# Testing Settings` \
    --test-mode all \
    --search-mode all \
    --gall-mode single \
    \
    `# Debug` \
    --debug wsl

# ==================== Training Complete ====================
echo "=========================================="
echo "Training Completed!"
echo "End Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Check results in ./save/"
echo "=========================================="
