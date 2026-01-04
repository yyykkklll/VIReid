#!/bin/bash
# train_phase1_benchmarks.sh
# 仅训练第一阶段 (Supervised Pre-training Baseline)
# 关闭 Diffusion/CCPA/MemoryBank，只训练 ResNet Backbone + Classifier

# ==================== 全局设置 ====================
DEVICE=0
DATA_PATH="./datasets/"
SAVE_DIR="./logs_phase1"
mkdir -p $SAVE_DIR
export TMPDIR=$(pwd)/local_tmp
mkdir -p $TMPDIR

# 清理缓存
clean_cache() {
    echo "🧹 Cleaning pycache..."
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find . -type f -name "*.pyc" -delete 2>/dev/null || true
}

# 打印表头
print_header() {
    echo ""
    echo "######################################################################"
    echo "# STARTING PHASE 1 TRAINING FOR: $1"
    echo "######################################################################"
    echo ""
}

# ==================== 任务 1: SYSU-MM01 ====================
clean_cache
DATASET="sysu"
print_header $DATASET

LOG_FILE="$SAVE_DIR/${DATASET}_phase1_$(date +%Y%m%d_%H%M%S).log"

# 配置: 60 Epoch 纯监督训练 (足够收敛)
# 注意: 不包含 --use-diffusion, --use-memory-bank 等参数
python main.py \
    --dataset $DATASET \
    --data-path $DATA_PATH \
    --device $DEVICE \
    --arch resnet \
    --mode train \
    --gall-mode single \
    --search-mode all \
    \
    --stage1-epoch 60 \
    --stage2-epoch 0 \
    --lr 0.00035 \
    \
    --batch-pidnum 8 \
    --pid-numsample 4 \
    --tri-weight 1.0 \
    --use-cosine-annealing \
    --eval-step 2 \
    2>&1 | tee $LOG_FILE

echo "✅ $DATASET Phase 1 Completed."
sleep 5

# ==================== 任务 2: LLCM ====================
clean_cache
DATASET="llcm"
print_header $DATASET

LOG_FILE="$SAVE_DIR/${DATASET}_phase1_$(date +%Y%m%d_%H%M%S).log"

# 配置: 60 Epoch 纯监督训练
python main.py \
    --dataset $DATASET \
    --data-path $DATA_PATH \
    --device $DEVICE \
    --arch resnet \
    --mode train \
    --gall-mode single \
    --search-mode all \
    \
    --stage1-epoch 60 \
    --stage2-epoch 0 \
    --lr 0.00035 \
    \
    --batch-pidnum 8 \
    --pid-numsample 4 \
    --tri-weight 1.0 \
    --use-cosine-annealing \
    --eval-step 2 \
    2>&1 | tee $LOG_FILE

echo "✅ $DATASET Phase 1 Completed."
sleep 5

# ==================== 任务 3: RegDB ====================
clean_cache
DATASET="regdb"
print_header $DATASET

LOG_FILE="$SAVE_DIR/${DATASET}_phase1_$(date +%Y%m%d_%H%M%S).log"

# 配置: 60 Epoch 纯监督训练 (RegDB 数据少，60轮足够)
python main.py \
    --dataset $DATASET \
    --data-path $DATA_PATH \
    --device $DEVICE \
    --arch resnet \
    --mode train \
    --trial 1 \
    \
    --stage1-epoch 60 \
    --stage2-epoch 0 \
    --lr 0.00035 \
    \
    --batch-pidnum 8 \
    --pid-numsample 4 \
    --tri-weight 1.0 \
    --use-cosine-annealing \
    --eval-step 2 \
    2>&1 | tee $LOG_FILE

echo "✅ $DATASET Phase 1 Completed."

echo ""
echo "🎉 ALL PHASE 1 BASELINES COMPLETED."