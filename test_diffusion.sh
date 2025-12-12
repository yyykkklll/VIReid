#!/bin/bash
# 快速测试扩散模块是否正常工作（使用少量 epoch）

echo "=========================================="
echo "Quick Test: Feature Diffusion Bridge"
echo "=========================================="

python3 main.py \
    --dataset regdb \
    --arch resnet \
    --debug wsl \
    --trial 1 \
    --save-path test_diffusion_quick \
    --stage1-epoch 2 \
    --stage2-epoch 5 \
    --batch-pidnum 4 \
    --pid-numsample 2 \
    --lr 0.00045 \
    --milestones 3 \
    --use-diffusion \
    --diffusion-steps 5 \
    --diffusion-weight 0.1

EXIT_CODE=$?

if [ ${EXIT_CODE} -eq 0 ]; then
    echo "✓ Diffusion module test PASSED!"
else
    echo "✗ Diffusion module test FAILED!"
fi

exit ${EXIT_CODE}
