#!/bin/bash
# run_all_benchmarks.sh

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

# ==================== 任务 1: SYSU-MM01 ====================
clean_cache
DATASET="sysu"
print_header $DATASET

LOG_FILE="$SAVE_DIR/${DATASET}_train_$(date +%Y%m%d_%H%M%S).log"

# SYSU 配置: 标准大数据集策略
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
    --semantic-diffusion-steps 10 \
    --diffusion-hidden 1024 \
    --diffusion-weight 0.1 \
    --diffusion-lr 0.0001 \
    \
    --use-memory-bank \
    --memory-size-per-class 5 \
    \
    --use-cycle-consistency \
    --ccpa-weight 0.5 \
    --ccpa-start-epoch 65 \
    --ccpa-threshold-mode hybrid \
    2>&1 | tee $LOG_FILE

echo "✅ $DATASET Training Completed."
sleep 10  # 休息一下让显存释放

# ==================== 任务 2: LLCM ====================
clean_cache
DATASET="llcm"
print_header $DATASET

LOG_FILE="$SAVE_DIR/${DATASET}_train_$(date +%Y%m%d_%H%M%S).log"

# LLCM 配置: 与 SYSU 类似，LLCM 也是大数据集，使用相同的稳健参数
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
    --semantic-diffusion-steps 10 \
    --diffusion-hidden 1024 \
    --diffusion-weight 0.1 \
    --diffusion-lr 0.0001 \
    \
    --use-memory-bank \
    --memory-size-per-class 5 \
    \
    --use-cycle-consistency \
    --ccpa-weight 0.5 \
    --ccpa-start-epoch 65 \
    --ccpa-threshold-mode hybrid \
    2>&1 | tee $LOG_FILE

echo "✅ $DATASET Training Completed."
sleep 10

# ==================== 任务 3: RegDB ====================
clean_cache
DATASET="regdb"
print_header $DATASET

LOG_FILE="$SAVE_DIR/${DATASET}_train_$(date +%Y%m%d_%H%M%S).log"

# RegDB 配置: 小数据集策略
# 1. 总 Epoch 缩短 (40 + 80 = 120)
# 2. CCPA 提前介入 (60)，因为收敛较快
# 3. Batch Size 保持不变
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
    --ccpa-weight 0.5 \
    --ccpa-start-epoch 55 \
    --ccpa-threshold-mode hybrid \
    2>&1 | tee $LOG_FILE

echo "✅ $DATASET Training Completed."

echo ""
echo "🎉 ALL BENCHMARKS COMPLETED SUCCESSFULLY."