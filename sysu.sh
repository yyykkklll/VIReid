#!/bin/bash

# ==================== Setup ====================
cd "$(dirname "$0")" || exit
export PYTHONPATH=$PYTHONPATH:.

# Clean Python cache
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null
find . -name "*.pyc" -delete 2>/dev/null

# ==================== Configuration ====================
USE_DIFFUSION=${1:-0}  # 0: 不使用扩散桥, 1: 使用扩散桥
SEARCH_MODE=${2:-"all"}  # all: 全场景搜索, indoor: 室内搜索
TRIAL_NAME=${3:-"trial1"}  # 实验名称

echo "=========================================="
echo "WSL-ReID Training on SYSU-MM01 Dataset"
echo "=========================================="
echo "Diffusion Bridge: $([ ${USE_DIFFUSION} -eq 1 ] && echo 'ENABLED' || echo 'DISABLED')"
echo "Search Mode: ${SEARCH_MODE}"
echo "Trial Name: ${TRIAL_NAME}"
echo "Start Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="

# ==================== SYSU Dataset Characteristics ====================
# - 395 identities
# - Indoor + Outdoor scenarios
# - Visible + Near-infrared (NIR)
# - Two testing modes: all-search (harder), indoor-search (easier)
# - Recommended: Use CLIP-ResNet for better cross-modal alignment
# - Settings: stage1=20, stage2=80-100, lr=0.0003
# ====================================================================

# ==================== Architecture Selection ====================
# SYSU benefits from CLIP-ResNet due to diverse scenarios
if [ ${USE_DIFFUSION} -eq 1 ]; then
    ARCH="resnet"  # 扩散桥暂时只支持 ResNet
    echo "⚠️  Note: Using ResNet (CLIP-ResNet support coming soon)"
else
    ARCH="clip-resnet"  # 推荐使用 CLIP-ResNet
    echo "✓ Using CLIP-ResNet for better cross-modal alignment"
fi

# ==================== Base Training Arguments ====================
BASE_ARGS="
    --dataset sysu \
    --arch ${ARCH} \
    --debug wsl \
    --mode train \
    --save-path sysu_${SEARCH_MODE}_${TRIAL_NAME}$([ ${USE_DIFFUSION} -eq 1 ] && echo '_diffusion' || echo '') \
    --data-path ./datasets/ \
    \
    $(: 'Training Phases - SYSU needs moderate training') \
    --stage1-epoch 20 \
    --stage2-epoch 100 \
    \
    $(: 'Data Settings') \
    --img-h 288 \
    --img-w 144 \
    --batch-pidnum 8 \
    --pid-numsample 4 \
    --test-batch 128 \
    --num-workers 8 \
    --relabel 1 \
    \
    $(: 'Optimizer and Scheduler') \
    --lr 0.0003 \
    --weight-decay 0.0005 \
    --milestones 30 70 \
    \
    $(: 'Loss Function Settings') \
    --tri-weight 0.25 \
    --weak-weight 0.25 \
    \
    $(: 'Cross-Modal Matching') \
    --sigma 0.8 \
    --temperature 3.0 \
    \
    $(: 'Testing Settings - SYSU Specific') \
    --search-mode ${SEARCH_MODE} \
    --gall-mode single \
    --test-mode t2v \
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
        --diffusion-weight 0.1 \
        --diffusion-lr 0.0003 \
        --confidence-weight 0.1
    "
    echo "Diffusion Parameters:"
    echo "  - Steps: 10"
    echo "  - Hidden Dim: 1024"
    echo "  - Diffusion Weight: 0.1"
    echo "  - Confidence Weight: 0.1"
else
    DIFFUSION_ARGS=""
fi

# ==================== Execute Training ====================
echo "=========================================="
echo "Starting training..."
echo "Dataset Info:"
echo "  - Identities: 395"
echo "  - Scenarios: Indoor + Outdoor"
echo "  - Search Mode: ${SEARCH_MODE}"
echo "=========================================="

python3 main.py ${BASE_ARGS} ${DIFFUSION_ARGS}

EXIT_CODE=$?

# ==================== Training Complete ====================
echo "=========================================="
if [ ${EXIT_CODE} -eq 0 ]; then
    echo "✓ Training Completed Successfully!"
    echo ""
    echo "Final Results Summary:"
    echo "  Check: ./saved_sysu_${ARCH}/sysu_${SEARCH_MODE}_${TRIAL_NAME}$([ ${USE_DIFFUSION} -eq 1 ] && echo '_diffusion' || echo '')/log/log.txt"
    echo ""
    echo "Recommended Next Steps:"
    if [ "${SEARCH_MODE}" == "all" ]; then
        echo "  1. Test on indoor-search mode for comparison"
        echo "     bash sysu.sh ${USE_DIFFUSION} indoor ${TRIAL_NAME}"
    else
        echo "  1. Test on all-search mode for comprehensive evaluation"
        echo "     bash sysu.sh ${USE_DIFFUSION} all ${TRIAL_NAME}"
    fi
else
    echo "✗ Training Failed with Exit Code: ${EXIT_CODE}"
    echo ""
    echo "Troubleshooting:"
    echo "  1. Check GPU memory (SYSU requires ~8-10GB VRAM)"
    echo "  2. Verify dataset path: ./datasets/SYSU-MM01/"
    echo "  3. Ensure preprocessing is done: python pre_process_sysu.py"
    echo "  4. Review log file for detailed errors"
fi
echo "End Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="

exit ${EXIT_CODE}
