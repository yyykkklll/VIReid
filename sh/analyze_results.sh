#!/bin/bash

# ====================================================================
# Analyze Training Results (Auto-Detect Root Dir)
# 自动定位项目根目录，分析 logs 文件夹下的训练日志
# ====================================================================

# 1. 获取脚本所在的绝对路径
SCRIPT_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
CURRENT_FOLDER=$(basename "$SCRIPT_PATH")

# 2. 智能切换到项目根目录 (包含 logs/ 的那个目录)
if [ "$CURRENT_FOLDER" == "sh" ]; then
    # 如果脚本在 sh/ 子目录下，根目录是上一级
    PROJECT_ROOT="$SCRIPT_PATH/.."
else
    # 否则（在根目录下），根目录就是脚本所在目录
    PROJECT_ROOT="$SCRIPT_PATH"
fi

# 切换工作目录
cd "$PROJECT_ROOT" || exit

echo "======================================"
echo "Training Results Summary"
echo "Working Directory: $(pwd)"
echo "======================================"

# 3. 遍历所有数据集和模式进行分析
# 脚本会同时检查 standard 和 a100 的日志，如果不存在会自动跳过
for dataset in regdb sysu llcm; do
    for mode in standard a100; do
        log_file="./logs/${dataset}_${mode}/train.log"
        
        # 检查日志文件是否存在
        if [ -f "$log_file" ]; then
            echo ""
            echo "[$dataset - $mode]"
            echo "----------------------------------------"
            
            # 提取关键指标
            best_rank1=$(grep "Best Rank-1" "$log_file" | tail -1 | grep -oP '\d+\.\d+')
            best_map=$(grep "Best mAP" "$log_file" | tail -1 | grep -oP '\d+\.\d+')
            best_epoch=$(grep "Best Epoch" "$log_file" | tail -1 | grep -oP '\d+')
            
            # 提取最后一次验证结果 (可选，查看最新状态)
            last_rank1=$(grep "Validation Results:" "$log_file" | tail -1 | grep -oP 'Rank-1 \K\d+\.\d+')
            
            if [ -n "$best_rank1" ]; then
                echo "  Best Epoch:  $best_epoch"
                echo "  Best Rank-1: $best_rank1%"
                echo "  Best mAP:    $best_map%"
                
                # 如果有最新结果且不同于最佳结果，也可以显示
                if [ -n "$last_rank1" ] && [ "$last_rank1" != "$best_rank1" ]; then
                    echo "  Last Rank-1: $last_rank1%"
                fi
            else
                echo "  Training started but no results found yet."
            fi
        else
            # 如果是本地运行，通常只关心存在的日志，忽略不存在的报错会更清爽
            # 这里我们只在 Debug 时打印 "Log file not found"，或者保持静默
            # 为了保持原脚本逻辑，这里保留提示，但您可以注释掉下面两行以减少干扰
            # echo ""
            # echo "[$dataset - $mode]: Log not found"
            : # 空指令，占位符
        fi
    done
done

echo ""
echo "======================================"