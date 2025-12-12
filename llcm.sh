#!/bin/bash

# ==================== Setup ====================
cd "$(dirname "$0")" || exit
export PYTHONPATH=$PYTHONPATH:.

# Clean Python cache
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null
find . -name "*.pyc" -delete 2>/dev/null

# ==================== Configuration ====================
USE_DIFFUSION=${1:-0}  # 0: 不使用扩散桥, 1: 使用扩散桥
TRIAL_NAME=${2:-"trial1"}  # 实验名称

echo "=========================================="
echo "WSL-ReID Training on LLCM Dataset"
echo "=========================================="
echo "Diffusion Bridge: $([ ${USE_DIFFUSION} -eq 1 ] && echo 'ENABLED' || echo 'DISABLED')"
echo "Trial Name: ${TRIAL_NAME}"
echo "Start Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="

# ==================== LLCM Dataset Characteristics ====================
# - 713 identities (largest among three datasets)
# - Long-wavelength cross-modality (different from RegDB/SYSU)
# - Requires longer training due to larger identity space
# - Recommended settings: stage1=80, stage2=160, lr=0.0003
# ====================================================================

# ==================== Base Training Arguments ====================
BASE_ARGS="
    --dataset llcm \
    --arch resnet \
    --debug wsl \
    --mode train \
    --save-path llcm_${TRIAL_NAME}$([ ${USE_DIFFUSION} -eq 1 ] && echo '_diffusion' || echo '') \
    --data-path ./datasets/ \
    \
    $(: 'Training Phases - LLCM needs longer training') \
    --stage1-epoch 80 \
    --stage2-epoch 160 \
    \
    $(: 'Data Settings') \
    --img-h 288 \
    --img-w 144 \
    --batch-pidnum 8 \
    --pid-numsample 4 \
    --test-batch 64 \
    --num-workers 8 \
    --relabel 1 \
    \
    $(: 'Optimizer and Scheduler - Lower LR for stability') \
    --lr 0.0003 \
    --weight-decay 0.0005 \
    --milestones 60 120 \
    \
    $(: 'Loss Function Settings') \
    --tri-weight 0.25 \
    --weak-weight 0.25 \
    \
    $(: 'Cross-Modal Matching - Higher temperature for LLCM') \
    --sigma 0.8 \
    --temperature 3.5 \
    \
    $(: 'Testing Settings') \
    --test-mode t2v \
    --search-mode all \
    --gall-mode single \
    \
    $(: 'Other Settings') \
    --seed 1
"

# ==================== Diffusion-Specific Arguments ====================
if [ ${USE_DIFFUSION} -eq 1 ]; then
    DIFFUSION_ARGS="
        --use-diffusion \
        --diffusion-steps 10 \
        --diffusion-hidden 1024 \
        --diffusion-weight 0.08 \
        --diffusion-lr 0.0003 \
        --confidence-weight 0.08
    "
    echo "Diffusion Parameters:"
    echo "  - Steps: 10"
    echo "  - Hidden Dim: 1024"
    echo "  - Diffusion Weight: 0.08 (lower for stability)"
    echo "  - Confidence Weight: 0.08"
else
    DIFFUSION_ARGS=""
fi

# ==================== Execute Training ====================
echo "=========================================="
echo "Starting training..."
echo "=========================================="

python3 main.py ${BASE_ARGS} ${DIFFUSION_ARGS}

EXIT_CODE=$?

# ==================== Training Complete ====================
echo "=========================================="
if [ ${EXIT_CODE} -eq 0 ]; then
    echo "✓ Training Completed Successfully!"
    echo ""
    echo "Final Results Summary:"
    echo "  Check: ./saved_llcm_resnet/llcm_${TRIAL_NAME}$([ ${USE_DIFFUSION} -eq 1 ] && echo '_diffusion' || echo '')/log/log.txt"
else
    echo "✗ Training Failed with Exit Code: ${EXIT_CODE}"
    echo ""
    echo "Troubleshooting:"
    echo "  1. Check GPU memory (LLCM requires ~10GB VRAM)"
    echo "  2. Verify dataset path: ./datasets/LLCM/"
    echo "  3. Review log file for detailed errors"
fi
echo "End Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="

exit ${EXIT_CODE}
