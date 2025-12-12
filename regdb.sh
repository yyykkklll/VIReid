#!/bin/bash

# ==================== Setup ====================
cd "$(dirname "$0")" || exit
export PYTHONPATH=$PYTHONPATH:.

# Clean Python cache
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null
find . -name "*.pyc" -delete 2>/dev/null

# ==================== Configuration ====================
TRIAL=${1:-1}
USE_DIFFUSION=${2:-0}  # 0: 不使用扩散桥, 1: 使用扩散桥

echo "=========================================="
echo "WSL-ReID Training on RegDB"
echo "=========================================="
echo "Trial: ${TRIAL}"
echo "Diffusion Bridge: $([ ${USE_DIFFUSION} -eq 1 ] && echo 'ENABLED' || echo 'DISABLED')"
echo "Start Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="

# ==================== Base Training Arguments ====================
BASE_ARGS="
    --dataset regdb \
    --arch resnet \
    --debug wsl \
    --mode train \
    --save-path regdb_trial${TRIAL}$([ ${USE_DIFFUSION} -eq 1 ] && echo '_diffusion' || echo '') \
    --trial ${TRIAL} \
    --stage1-epoch 50 \
    --stage2-epoch 120 \
    --img-h 288 \
    --img-w 144 \
    --batch-pidnum 8 \
    --pid-numsample 4 \
    --test-batch 128 \
    --num-workers 8 \
    --relabel 1 \
    --lr 0.00045 \
    --weight-decay 0.0005 \
    --milestones 50 70 \
    --tri-weight 0.25 \
    --weak-weight 0.25 \
    --sigma 0.8 \
    --temperature 3 \
    --test-mode t2v \
    --search-mode all \
    --gall-mode single \
    --seed 1
"

# ==================== Diffusion-Specific Arguments ====================
if [ ${USE_DIFFUSION} -eq 1 ]; then
    DIFFUSION_ARGS="
        --use-diffusion \
        --diffusion-steps 10 \
        --diffusion-hidden 1024 \
        --diffusion-weight 0.1 \
        --diffusion-lr 0.00045 \
        --confidence-weight 0.1
    "
else
    DIFFUSION_ARGS=""
fi

# ==================== Execute Training ====================
python3 main.py ${BASE_ARGS} ${DIFFUSION_ARGS}

EXIT_CODE=$?

# ==================== Training Complete ====================
echo "=========================================="
if [ ${EXIT_CODE} -eq 0 ]; then
    echo "✓ Training Completed Successfully!"
else
    echo "✗ Training Failed with Exit Code: ${EXIT_CODE}"
fi
echo "End Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Results saved in: ./saved_regdb_resnet/regdb_trial${TRIAL}$([ ${USE_DIFFUSION} -eq 1 ] && echo '_diffusion' || echo '')/"
echo "=========================================="

exit ${EXIT_CODE}
