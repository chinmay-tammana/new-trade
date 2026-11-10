#!/usr/bin/env bash
# run_all.sh: End-to-end execution script reproducing all V2 results, tables, and figures.
set -eo pipefail

echo "======================================================================"
echo "GS AI Trading Indicator: End-to-End V2 Upgrade Execution Pipeline"
echo "IMPACT Lab (IIT Patna) Protocol"
echo "======================================================================"

# Determine Python command
if [ -d ".venv" ]; then
    PYTHON_CMD=".venv/bin/python3"
elif command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
else
    PYTHON_CMD="python"
fi

export MPLCONFIGDIR="/tmp/mpl"
mkdir -p results/logs results/tables results/figures models configs

echo "Using Python: $($PYTHON_CMD --version) at $PYTHON_CMD"

# Verify secrets and environment
echo "--- Checking Security & Environment ---"
$PYTHON_CMD src/check_secrets.py

# Step 1: Build 8-Year Multi-Asset Dataset with 51 Features
echo "--- 1. Building V2 Feature Matrix (51 Causal Features, 8 Assets) ---"
$PYTHON_CMD -m src.build_dataset_v2 | tee results/logs/build_dataset_v2.log

# Step 2: Walk-Forward OOF Experiments, Tournament, Horizon & Insight Heads
echo "--- 2. Running 20-Quarter Walk-Forward OOF Tournament & Insight Heads ---"
$PYTHON_CMD -m src.run_experiments_v2 | tee results/logs/experiments_v2.log

# Step 3: Single Read Research Holdout Evaluation
echo "--- 3. Evaluating on Pre-Registered Research Holdout (Strict Single Read) ---"
$PYTHON_CMD -m src.evaluate_holdout_v2 | tee results/logs/evaluate_holdout_v2.log

# Step 4: Final Refit on All Bars to Freeze Date
echo "--- 4. Refitting Final Model on All Data up to Freeze Date ---"
$PYTHON_CMD -m src.refit_final_v2 | tee results/logs/refit_final_v2.log

# Step 5: Day Replay Invariant Verification
echo "--- 5. Replaying Sample Trading Day (2026-11-06) ---"
$PYTHON_CMD -m src.predict_day --date 2026-11-06 | tee results/logs/predict_day.log

# Step 6: Comprehensive Integrity & Leak Detection Suite
echo "--- 6. Running Comprehensive Integrity & Leak Detection Suite ---"
$PYTHON_CMD -m src.checks | tee results/logs/checks_v2.log

# Step 7: Final Secret & Data File Security Scan
echo "--- 7. Performing Final Git Security & Leak Scan ---"
$PYTHON_CMD src/check_secrets.py

echo "======================================================================"
echo "V2 execution pipeline completed successfully with zero errors."
echo "Results, tables, and figures have been frozen to results/ and models/."
echo "======================================================================"
