#!/bin/bash

# ====================================================================
# Train All Datasets Sequentially (Auto-Detect Mode)
# 自动根据所在文件夹判断是 A100 模式还是 Standard 模式
# ====================================================================

# 1. 获取脚本所在的目录名称 (sh 还是 vireid)
# BASH_SOURCE[0] 代表脚本文件本身的路径
SCRIPT_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
CURRENT_FOLDER=$(basename "$SCRIPT_PATH")

echo "======================================"
echo "Initializing Training Pipeline..."
echo "Script Location: $SCRIPT_PATH"
echo "Current Folder:  $CURRENT_FOLDER"
echo "======================================"

# 2. 根据文件夹名称决定模式和工作目录
if [ "$CURRENT_FOLDER" == "sh" ]; then
    # 场景 A: 脚本在 sh/ 目录下 (A100环境)
    MODE="a100"
    # 工作目录需要切换到上一级 (项目根目录)
    WORK_DIR="$SCRIPT_PATH/.."
    
elif [ "$CURRENT_FOLDER" == "vireid" ] || [ -f "main.py" ]; then
    # 场景 B: 脚本在项目根目录下 (3090环境)
    # (判断依据：文件夹名叫 vireid 或者当前目录下有 main.py)
    MODE="standard"
    WORK_DIR="$SCRIPT_PATH"
    
else
    # 兜底：如果文件夹名改了，默认作为根目录处理
    echo "Warning: Unknown folder structure. Assuming Standard mode."
    MODE="standard"
    WORK_DIR="$SCRIPT_PATH"
fi

# 3. 切换到工作目录 (项目根目录)
cd "$WORK_DIR" || exit
echo "Working Directory: $(pwd)"
echo "Detected Mode:     $MODE"
echo "======================================"

# 4. 根据模式执行对应的训练脚本
if [ "$MODE" == "a100" ]; then
    echo ""
    echo "[1/3] Training RegDB (A100 Config)..."
    bash sh/regdb_train_a100.sh
    
    echo ""
    echo "[2/3] Training SYSU-MM01 (A100 Config)..."
    bash sh/sysu_train_a100.sh
    
    echo ""
    echo "[3/3] Training LLCM (A100 Config)..."
    bash sh/llcm_train_a100.sh

else
    echo ""
    echo "[1/3] Training RegDB (Standard Config)..."
    bash regdb.sh
    
    echo ""
    echo "[2/3] Training SYSU-MM01 (Standard Config)..."
    bash sysu.sh
    
    echo ""
    echo "[3/3] Training LLCM (Standard Config)..."
    bash llcm.sh
fi

echo ""
echo "======================================"
echo "All Training Completed!"
echo "======================================"
echo "Results Logs: ./logs/ (Check specific dataset folders)"
echo "======================================"