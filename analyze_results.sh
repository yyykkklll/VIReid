#!/bin/bash

# ====================================================================
# Comprehensive Training Results Analysis
# Analyzes both 3090 and A100 configurations
# ====================================================================

echo ""
echo "======================================================================"
echo "              Training Results Analysis                              "
echo "======================================================================"
echo ""

# 定义颜色
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# 分析单个日志文件
analyze_log() {
    local log_file=$1
    local dataset=$2
    local config=$3
    
    if [ ! -f "$log_file" ]; then
        echo "  ${RED}✗ Log file not found${NC}"
        return 1
    fi
    
    # 提取信息
    best_epoch=$(grep "Best Epoch:" "$log_file" | tail -1 | grep -oP '\d+')
    best_rank1=$(grep "Best Rank-1:" "$log_file" | tail -1 | grep -oP '\d+\.\d+')
    best_map=$(grep "Best mAP:" "$log_file" | tail -1 | grep -oP '\d+\.\d+')
    
    # 提取最后一次验证结果
    last_rank1=$(grep "Validation Results:" "$log_file" | tail -1 | grep -oP 'Rank-1 \K\d+\.\d+')
    last_map=$(grep "Validation Results:" "$log_file" | tail -1 | grep -oP 'mAP \K\d+\.\d+')
    last_minp=$(grep "Validation Results:" "$log_file" | tail -1 | grep -oP 'mINP \K\d+\.\d+')
    
    # 计算训练时间
    first_epoch_time=$(grep "Epoch 1/" "$log_file" | head -1 | grep -oP 'Time: \K\d+\.\d+')
    total_epochs=$(grep "Total Epochs:" "$log_file" | grep -oP '\d+')
    
    if [ -n "$best_rank1" ]; then
        echo "  ${GREEN}✓ Training completed${NC}"
        echo "    Best Epoch:  ${YELLOW}$best_epoch${NC} / $total_epochs"
        echo "    Best Rank-1: ${YELLOW}$best_rank1%${NC}"
        echo "    Best mAP:    ${YELLOW}$best_map%${NC}"
        
        if [ -n "$last_rank1" ]; then
            echo ""
            echo "    Latest Results (Last Validation):"
            echo "      Rank-1: $last_rank1%"
            echo "      mAP:    $last_map%"
            echo "      mINP:   $last_minp%"
        fi
        
        if [ -n "$first_epoch_time" ] && [ -n "$total_epochs" ]; then
            estimated_time=$(echo "$first_epoch_time * $total_epochs" | bc)
            hours=$(echo "$estimated_time / 3600" | bc)
            minutes=$(echo "($estimated_time % 3600) / 60" | bc)
            echo ""
            echo "    Est. Total Time: ~${hours}h ${minutes}m"
        fi
        
        return 0
    else
        echo "  ${YELLOW}⚠ Training in progress or incomplete${NC}"
        return 1
    fi
}

# ===== RegDB Results =====
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "1. RegDB Dataset"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

echo "[3090 Configuration]"
analyze_log "./logs/regdb_3090/train.log" "regdb" "3090"
echo ""

echo "[A100 Configuration]"
analyze_log "./logs/regdb_a100/train.log" "regdb" "a100"
echo ""

# ===== SYSU Results =====
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "2. SYSU-MM01 Dataset"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

echo "[3090 Configuration]"
analyze_log "./logs/sysu_3090/train.log" "sysu" "3090"
echo ""

echo "[A100 Configuration]"
analyze_log "./logs/sysu_a100/train.log" "sysu" "a100"
echo ""

# ===== LLCM Results =====
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "3. LLCM Dataset"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

echo "[3090 Configuration]"
analyze_log "./logs/llcm_3090/train.log" "llcm" "3090"
echo ""

echo "[A100 Configuration]"
analyze_log "./logs/llcm_a100/train.log" "llcm" "a100"
echo ""

echo "======================================================================"
