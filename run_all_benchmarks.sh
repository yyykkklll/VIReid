#!/bin/bash
# run_all_benchmarks.sh
# Optimized for PRUD Framework (High Weight, Smart Warmup)

# ==================== 全局设置 ====================
DEVICE=0
DATA_PATH="./datasets/"
SAVE_DIR="./logs"
mkdir -p $SAVE_DIR
export TMPDIR=$(pwd)/local_tmp
mkdir -p $TMPDIR

# 清理 Python 缓存函数
clean_cache() {
    echo "🧹 Cleaning pycache..."
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find . -type f -name "*.pyc" -delete 2>/dev/null || true
}

# 打印分隔符
print_header() {
    echo ""
    echo "######################################################################"
    echo "# STARTING TRAINING FOR DATASET: $1"
    echo "######################################################################"
    echo ""
}

# ==================== 任务 1: SYSU-MM01 (大数据集) ====================
clean_cache
DATASET="sysu"
print_header $DATASET

LOG_FILE="$SAVE_DIR/${DATASET}_prud_$(date +%Y%m%d_%H%M%S).log"

# 策略: 50轮预训练 + 10轮扩散热身 + 110轮PRUD强蒸馏
python main.py \
    --dataset $DATASET \
    --data-path $DATA_PATH \
    --device $DEVICE \
    --arch resnet \
    --mode train \
    --gall-mode single \
    --search-mode all \
    \
    --stage1-epoch 50 \
    --stage2-epoch 120 \
    --lr 0.00035 \
    \
    --use-diffusion \
    --feature-diffusion-steps 5 \
    --semantic-diffusion-steps 15 \
    --diffusion-hidden 1024 \
    --diffusion-weight 0.1 \
    --diffusion-lr 0.0001 \
    \
    --use-memory-bank \
    --memory-size-per-class 5 \
    \
    --use-cycle-consistency \
    --ccpa-weight 0.8 \
    --ccpa-start-epoch 60 \
    --ccpa-threshold-mode hybrid \
    --pseudo-momentum 0.9 \
    2>&1 | tee $LOG_FILE

echo "✅ $DATASET Training Completed."
sleep 10  # 释放显存

# ==================== 任务 2: LLCM (大数据集) ====================
clean_cache
DATASET="llcm"
print_header $DATASET

LOG_FILE="$SAVE_DIR/${DATASET}_prud_$(date +%Y%m%d_%H%M%S).log"

# 策略: 同 SYSU，保持稳健的大数据集参数
python main.py \
    --dataset $DATASET \
    --data-path $DATA_PATH \
    --device $DEVICE \
    --arch resnet \
    --mode train \
    --gall-mode single \
    --search-mode all \
    \
    --stage1-epoch 50 \
    --stage2-epoch 120 \
    --lr 0.00035 \
    \
    --use-diffusion \
    --feature-diffusion-steps 5 \
    --semantic-diffusion-steps 15 \
    --diffusion-hidden 1024 \
    --diffusion-weight 0.1 \
    --diffusion-lr 0.0001 \
    \
    --use-memory-bank \
    --memory-size-per-class 5 \
    \
    --use-cycle-consistency \
    --ccpa-weight 0.8 \
    --ccpa-start-epoch 60 \
    --ccpa-threshold-mode hybrid \
    --pseudo-momentum 0.9 \
    2>&1 | tee $LOG_FILE

echo "✅ $DATASET Training Completed."
sleep 10

# ==================== 任务 3: RegDB (小数据集) ====================
clean_cache
DATASET="regdb"
print_header $DATASET

LOG_FILE="$SAVE_DIR/${DATASET}_prud_$(date +%Y%m%d_%H%M%S).log"

# 策略: 40轮预训练 + 10轮扩散热身 + 70轮PRUD强蒸馏
# RegDB 收敛快，缩短周期
python main.py \
    --dataset $DATASET \
    --data-path $DATA_PATH \
    --device $DEVICE \
    --arch resnet \
    --mode train \
    --trial 1 \
    \
    --stage1-epoch 40 \
    --stage2-epoch 80 \
    --lr 0.00035 \
    \
    --use-diffusion \
    --feature-diffusion-steps 5 \
    --semantic-diffusion-steps 10 \
    --diffusion-hidden 1024 \
    --diffusion-weight 0.1 \
    --diffusion-lr 0.0001 \
    \
    --use-memory-bank \
    --memory-size-per-class 5 \
    \
    --use-cycle-consistency \
    --ccpa-weight 0.8 \
    --ccpa-start-epoch 50 \
    --ccpa-threshold-mode hybrid \
    --pseudo-momentum 0.9 \
    2>&1 | tee $LOG_FILE

echo "✅ $DATASET Training Completed."

echo ""
echo "🎉 ALL BENCHMARKS COMPLETED SUCCESSFULLY."