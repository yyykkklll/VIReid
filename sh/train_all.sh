#!/bin/bash

# ====================================================================
# Train All Datasets Sequentially
# ====================================================================

# 切换到项目根目录
cd "$(dirname "$0")/.." || exit

MODE=${1:-"standard"}

echo "======================================"
echo "Starting Sequential Training"
echo "Mode: $MODE"
echo "======================================"

if [ "$MODE" = "a100" ]; then
    echo "Using A100 optimized configurations..."
    
    echo ""
    echo "[1/3] Training RegDB..."
    bash sh/regdb_train_a100.sh
    
    echo ""
    echo "[2/3] Training SYSU-MM01..."
    bash sh/sysu_train_a100.sh
    
    echo ""
    echo "[3/3] Training LLCM..."
    bash sh/llcm_train_a100.sh
else
    echo "Using standard configurations..."
    
    echo ""
    echo "[1/3] Training RegDB..."
    bash sh/regdb_train.sh
    
    echo ""
    echo "[2/3] Training SYSU-MM01..."
    bash sh/sysu_train.sh
    
    echo ""
    echo "[3/3] Training LLCM..."
    bash sh/llcm_train.sh
fi

echo ""
echo "======================================"
echo "All Training Completed!"
echo "======================================"
echo "Results:"
echo "  RegDB:   ./logs/regdb_${MODE}/"
echo "  SYSU:    ./logs/sysu_${MODE}/"
echo "  LLCM:    ./logs/llcm_${MODE}/"
echo "======================================"
