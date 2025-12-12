#!/bin/bash
# 消融实验：对比不同扩散步数的效果

echo "=========================================="
echo "RegDB Ablation Study: Diffusion Steps"
echo "=========================================="

TRIALS=(1 2 3)
DIFFUSION_STEPS=(5 10 15 20)

for TRIAL in "${TRIALS[@]}"; do
    for STEPS in "${DIFFUSION_STEPS[@]}"; do
        echo ""
        echo ">>> Running Trial ${TRIAL} with T=${STEPS} steps <<<"
        
        python3 main.py \
            --dataset regdb \
            --arch resnet \
            --debug wsl \
            --trial ${TRIAL} \
            --save-path regdb_ablation_T${STEPS}_trial${TRIAL} \
            --stage1-epoch 50 \
            --stage2-epoch 120 \
            --lr 0.00045 \
            --milestones 50 70 \
            --use-diffusion \
            --diffusion-steps ${STEPS} \
            --diffusion-weight 0.1 \
            > logs/ablation_T${STEPS}_trial${TRIAL}.log 2>&1
        
        echo "✓ Trial ${TRIAL} with T=${STEPS} completed"
    done
done

echo ""
echo "=========================================="
echo "All ablation experiments completed!"
echo "Check logs/ directory for results"
echo "=========================================="
