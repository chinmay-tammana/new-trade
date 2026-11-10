#!/usr/bin/env bash
# run_all.sh: End-to-end execution script reproducing all results, tables, and figures.
set -eo pipefail

echo "======================================================================"
echo "GS AI Trading Indicator: End-to-End Execution Pipeline"
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
mkdir -p results/logs results/tables results/figures

echo "Using Python: $($PYTHON_CMD --version) at $PYTHON_CMD"

# Verify secrets and environment
echo "--- Checking Security & Environment ---"
$PYTHON_CMD src/check_secrets.py

# Step 1: Data Audit
echo "--- 1. Auditing Market Data & Parity Checks ---"
$PYTHON_CMD -m src.audit | tee results/logs/audit.log

# Step 2: Exploratory Data Analysis & Feature Drift
echo "--- 2. Computing Preprocessing & Exploratory Data Analysis ---"
$PYTHON_CMD -m src.eda | tee results/logs/eda.log

# Step 3: Chronological Split
echo "--- 3. Constructing Chronological Split (Base/FT/Val/Test) ---"
$PYTHON_CMD -m src.split | tee results/logs/split.log

# Step 4: Baseline Models
echo "--- 4. Evaluating Baseline Models on Validation Block ---"
$PYTHON_CMD -m src.baselines | tee results/logs/baselines.log

# Step 5: Fine-Tuning & Model Selection (Base & FT blocks only)
echo "--- 5. Hyperparameter Tuning & Model Selection ---"
$PYTHON_CMD -m src.finetune | tee results/logs/finetune.log

# Step 6: Probability Calibration & Signal Rule (Validation block)
echo "--- 6. Calibrating Probabilities & Searching Signal Rule ---"
$PYTHON_CMD -m src.calibrate | tee results/logs/calibration.log

# Step 7: Final Held-Out Evaluation (Test block read strictly once)
echo "--- 7. Evaluating on Held-Out Test Block (Single Read) ---"
$PYTHON_CMD -m src.evaluate | tee results/logs/evaluate.log

# Step 8: Error Analysis, SHAP & Robustness
echo "--- 8. Performing Error Analysis & Feature Attributions ---"
$PYTHON_CMD -m src.analysis | tee results/logs/analysis.log

# Step 9: Leakage & Integrity Verification Suite
echo "--- 9. Running Comprehensive Integrity & Leak Detection Suite ---"
$PYTHON_CMD -m src.checks | tee results/logs/checks.log

# Verify clean secrets after full execution
$PYTHON_CMD src/check_secrets.py

echo "======================================================================"
echo "Execution pipeline completed successfully with zero errors."
echo "Results, tables, and figures have been frozen to results/ and models/."
echo "======================================================================"
