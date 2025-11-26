#!/bin/bash

# ====================================================================
# Analyze Training Results
# ====================================================================

# 切换到项目根目录
cd "$(dirname "$0")/.." || exit

echo "======================================"
echo "Training Results Summary"
echo "======================================"

for dataset in regdb sysu llcm; do
    for mode in standard a100; do
        log_file="./logs/${dataset}_${mode}/train.log"
        
        if [ -f "$log_file" ]; then
            echo ""
            echo "[$dataset - $mode]"
            echo "----------------------------------------"
            
            # 提取最佳性能
            best_rank1=$(grep "Best Rank-1" "$log_file" | tail -1 | grep -oP '\d+\.\d+')
            best_map=$(grep "Best mAP" "$log_file" | tail -1 | grep -oP '\d+\.\d+')
            best_epoch=$(grep "Best Epoch" "$log_file" | tail -1 | grep -oP '\d+')
            
            if [ -n "$best_rank1" ]; then
                echo "  Best Epoch: $best_epoch"
                echo "  Best Rank-1: $best_rank1%"
                echo "  Best mAP: $best_map%"
            else
                echo "  Training not completed or log not found"
            fi
        else
            echo ""
            echo "[$dataset - $mode]"
            echo "  Log file not found: $log_file"
        fi
    done
done

echo ""
echo "======================================"
