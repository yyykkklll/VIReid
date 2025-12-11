#!/bin/bash
# WSL-VI-ReID Training on RegDB with CLIP & Sinkhorn
# 修复记录：添加 --data-path 参数指向正确的数据集位置

# 1. 自动切换到脚本所在目录
cd "$(dirname "$0")" || exit
export PYTHONPATH=$PYTHONPATH:.

echo "🚀 [RegDB] Training Start with CLIP-Refereed & Sinkhorn Matching..."

# 2. 清理缓存
echo "🧹 Cleaning up cache..."
find . -name "__pycache__" -type d -exec rm -rf {} +
find . -name "*.pyc" -delete

# 3. 启动训练
# 关键修改：添加 --data-path ./datasets
# 代码逻辑会自动在 datasets 目录下寻找 "RegDB" 文件夹，所以这里只需指向 datasets

python main.py \
    --dataset regdb \
    --data-path ./datasets \
    --save-path regdb_wsl_clip_sinkhorn \
    --arch resnet \
    --trial 1 \
    \
    --mode train \
    --device 0 \
    --seed 42 \
    \
    --img-h 288 \
    --img-w 144 \
    --batch-pidnum 8 \
    --pid-numsample 4 \
    \
    --lr 0.00045 \
    --weight-decay 0.0005 \
    \
    --stage1-epoch 50 \
    --stage2-epoch 120 \
    --milestones 30 70 \
    \
    --debug wsl \
    \
    --use-clip \
    --use-sinkhorn \
    --w-clip 0.3 \
    --sinkhorn-reg 0.05

echo "✅ Training Finished! Check results in 'regdb_wsl_clip_sinkhorn/' directory."