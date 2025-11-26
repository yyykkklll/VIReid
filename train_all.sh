#!/bin/bash

# ====================================================================
# Train All Datasets Sequentially - 3090/V100 Standard Configuration
# Estimated Total Time: ~19 hours (3h + 7h + 9h)
# ====================================================================

echo ""
echo "======================================================================"
echo "        Sequential Training for All Datasets (3090/Standard)         "
echo "======================================================================"
echo ""
echo "Hardware: 3090/V100"
echo "Configuration: ResNet50 + Standard Settings"
echo ""
echo "Training Schedule:"
echo "  [1/3] RegDB   - 80 epochs  (~3 hours)"
echo "  [2/3] SYSU    - 100 epochs (~7 hours)"
echo "  [3/3] LLCM    - 120 epochs (~9 hours)"
echo ""
echo "Total Estimated Time: ~19 hours"
echo "======================================================================"
echo ""

read -p "Press Enter to start training or Ctrl+C to cancel..."

# 记录开始时间
START_TIME=$(date +%s)

# ===== Training RegDB =====
echo ""
echo "======================================================================"
echo "[1/3] Training RegDB Dataset"
echo "======================================================================"
REGDB_START=$(date +%s)

bash regdb.sh

REGDB_END=$(date +%s)
REGDB_TIME=$((REGDB_END - REGDB_START))
echo ""
echo "✓ RegDB training completed in $(($REGDB_TIME / 3600))h $(($REGDB_TIME % 3600 / 60))m"
echo ""

# ===== Training SYSU =====
echo ""
echo "======================================================================"
echo "[2/3] Training SYSU-MM01 Dataset"
echo "======================================================================"
SYSU_START=$(date +%s)

bash sysu.sh

SYSU_END=$(date +%s)
SYSU_TIME=$((SYSU_END - SYSU_START))
echo ""
echo "✓ SYSU training completed in $(($SYSU_TIME / 3600))h $(($SYSU_TIME % 3600 / 60))m"
echo ""

# ===== Training LLCM =====
echo ""
echo "======================================================================"
echo "[3/3] Training LLCM Dataset"
echo "======================================================================"
LLCM_START=$(date +%s)

bash llcm.sh

LLCM_END=$(date +%s)
LLCM_TIME=$((LLCM_END - LLCM_START))
echo ""
echo "✓ LLCM training completed in $(($LLCM_TIME / 3600))h $(($LLCM_TIME % 3600 / 60))m"
echo ""

# ===== 总结 =====
END_TIME=$(date +%s)
TOTAL_TIME=$((END_TIME - START_TIME))

echo ""
echo "======================================================================"
echo "                    All Training Completed!                          "
echo "======================================================================"
echo ""
echo "Time Summary:"
echo "  RegDB:  $(($REGDB_TIME / 3600))h $(($REGDB_TIME % 3600 / 60))m"
echo "  SYSU:   $(($SYSU_TIME / 3600))h $(($SYSU_TIME % 3600 / 60))m"
echo "  LLCM:   $(($LLCM_TIME / 3600))h $(($LLCM_TIME % 3600 / 60))m"
echo "  ---------------------------------------"
echo "  Total:  $(($TOTAL_TIME / 3600))h $(($TOTAL_TIME % 3600 / 60))m"
echo ""
echo "Results Locations:"
echo "  RegDB:  ./logs/regdb_3090/  & ./checkpoints/regdb_3090/"
echo "  SYSU:   ./logs/sysu_3090/   & ./checkpoints/sysu_3090/"
echo "  LLCM:   ./logs/llcm_3090/   & ./checkpoints/llcm_3090/"
echo ""
echo "Next Steps:"
echo "  1. Run analysis: bash analyze_results.sh"
echo "  2. Compare with A100: bash compare_configs.sh"
echo "======================================================================"
