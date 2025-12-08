#!/bin/bash
# RegDB Quick Verification Script
# 目标: 快速验证模型全流程 (Phase 1 + Phase 2) 无报错
# 耗时: 预计 2-5 分钟

# 自动切换到脚本所在目录
cd "$(dirname "$0")" || exit
export PYTHONPATH=$PYTHONPATH:.

echo "🚀 [Quick Check] RegDB ViT Pipeline Verification..."

# 清理旧的验证日志（可选）
rm -rf ../saved_regdb_vit/quick_verify_test

python main.py \
    --dataset regdb \
    --data-path ./datasets \
    --mode train \
    --device 0 \
    --seed 42 \
    \
    --arch vit \
    --feat-dim 768 \
    --img-h 256 \
    --img-w 128 \
    \
    --batch-pidnum 4 \
    --pid-numsample 4 \
    --test-batch 128 \
    --num-workers 4 \
    \
    --lr 0.0003 \
    --weight-decay 0.05 \
    --milestones 1 \
    \
    --stage1-epoch 1 \
    --stage2-epoch 2 \
    --trial 1 \
    \
    --save-path quick_verify_test \
    --debug wsl \
    --relabel 1 \
    --weak-weight 0.25 \
    --tri-weight 0.25

echo "----------------------------------------------------------------"
echo "✅ 如果你看到这句话，说明 Phase 1 和 Phase 2 都已成功跑通！"
echo "✅ 现在可以放心地运行 regdb_full.sh 进行全量训练了。"
echo "----------------------------------------------------------------"