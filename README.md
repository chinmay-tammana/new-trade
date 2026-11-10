# Goldman Sachs (GS) AI Directional Trading & Insight Engine (V2 Upgrade)

[![Model Release](https://img.shields.io/badge/Release-model--v2-blue.svg)](https://github.com/chinmay-tammana/new-trade/releases/tag/model-v2)
[![Historical Baseline](https://img.shields.io/badge/V1%20Baseline-v1--negative--result-orange.svg)](https://github.com/chinmay-tammana/new-trade/releases/tag/v1-negative-result)
[![Python Version](https://img.shields.io/badge/Python-3.11-green.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/Scope-Educational%20%2F%20Paper%20Trading-orange.svg)]()
[![Repository](https://img.shields.io/badge/GitHub-chinmay--tammana%2Fnew--trade-blue)](https://github.com/chinmay-tammana/new-trade)

An end-to-end machine learning system developed according to the **IMPACT Lab (IIT Patna) End-to-End ML Project Guidelines** to predict the 1-hour directional return of Goldman Sachs (`GS`) equity and emit multi-dimensional market insights (`VOL`, `BAND`, `REGIME`) with post-hoc probability calibration and mandatory two-sided signals under realistic transaction friction.

> **Disclaimer**: This repository is developed strictly for academic research and educational evaluation. It does NOT constitute investment advice or a recommendation to trade financial securities. Real capital trading is strictly outside the project scope; all evaluation is restricted to simulated paper trading.

---

## Architecture Overview

```
[ Historical 1-Min Bars: GS + SPY + XLF + JPM + MS + BAC + TLT + VIXY (8.5M+ Bars) ]
                                          |
                                          v
[ 1. Session-Anchored 1h Resampling: 14,084 Aligned Bars (2018-10 to 2026-11) ]
                                          |
                                          v
[ 2. Causal Feature Engineering: 51 Features across 9 Feature Groups ]
                                          |
                                          v
[ 3. Expanding Window Walk-Forward OOF (20 Quarters on 12,004 Pre-Holdout Bars) ]
   ├── Candidate Tournament: 3-Way Ensemble (XGBoost + HistGB + ElasticNet) -> Winner
   ├── Horizon Tournament: H1 (1-Hour Ahead) Selected as Primary Horizon
   ├── Class-Balanced Platt Calibration: ECE = 0.0191
   └── Two-Sided Quantile Thresholds: q_lo = 0.4904, q_hi = 0.5084 (Coverage c = 0.30)
                                          |
                                          v
[ 4. Research Holdout Single Read (1,872 Bars from 2025-10-16 to 2026-11-09) ]
   ├── Emitted Signals: 888 Up, 786 Down, 198 No-Signal (Two-Sided Assertion PASSED)
   ├── Accuracy: 52.25% (Wilson 95% CI: [49.65%, 54.84%])
   ├── Permutation Test (10,000 shuffles): p = 0.0417 (Statistically Significant at alpha=0.05)
   ├── Drift-Adjusted: Up +1.59% vs Base Rate, Down +3.20% vs Base Rate
   └── Chop Regime: 58.30% Accuracy, +2.28 bps Net Return @ 5 bps Friction
                                          |
                                          v
[ 5. Final Model Refit on All 13,884 Bars (models/model_v2.joblib, calib_v2.joblib) ]
                                          |
                                          v
[ 6. Production Reusable API (src/insights.py) & Day Replay (src/predict_day.py) ]
```

---

## 1. Executive Performance Summary: V1 Failure vs. V2 Upgrade

| Metric | V1 Architecture (4 Years, Fixed Val) | V2 Pre-Registered Target | V2 Research Holdout (Single Read) | Verdict |
| :--- | :--- | :--- | :--- | :--- |
| **History Span & Bars** | 4 Years (7,118 bars) | 8 Years (14,000+ bars) | **8 Years (14,084 bars)** | **Complete** |
| **Feature Dimension** | 22 basic technical features | 51 features in 9 groups | **51 features** (Microstructure, Peers, Levels) | **Complete** |
| **Validation Architecture**| Single Val block (602 bars) | 20-quarter walk-forward OOF | **20 expanding quarters (12,004 bars)** | **Complete** |
| **Signal Counts** | 188 Up / 0 Down (Up-only) | Two-sided ($n_{\text{up}}, n_{\text{down}} > 0$) | **888 Up / 786 Down / 198 No-Sig** (848/574 on labelled) | **RESOLVED** |
| **Signal Accuracy** | 46.28% [39.30%, 53.41%] | $\ge 53\%$ with $p < 0.10$ | **52.25% (743/1,422) [49.65%, 54.84%]** | **NOT MET (< 53.0%)** |
| **Drift-Adjusted Edge** | +0.00% vs Up-drift | Both sides beat base rates | **Up +1.59%, Down +3.20%** | **MET (Crit. C)** |
| **Calibration Error (ECE)**| 0.0454 | $\le 0.050$ | **0.0249** | **MET (Crit. D)** |
| **Chop Regime Accuracy** | Not segmented | - | **58.30% (Net Return +2.28 bps)** | **Strong Edge** |
| **Net Return @ 5 bps** | -7.24 bps | $> 0.0$ bps | **-3.53 bps** (Overall) / **+2.28 bps** (Chop) | **Regime-dependent** |
| **Integrity Audit** | 6 Checks Passed | All Checks Passed | **ALL 7 CHECKS PASSED** | **MET (Crit. E)** |

---

## 2. Multi-Head Insight Engine

Beyond simple binary classification, `src/insights.py` exposes a reusable production function `get_bar_insights(row)` that outputs seven fields:
1. `direction`: `Up`, `Down`, or `No-Signal` from the frozen two-sided quantile rule.
2. `confidence`: Calibrated probability for the indicated side ($\ge 50\%$).
3. `horizon`: Primary horizon (`H1`, 1-hour ahead).
4. `vol_class`: Expected forward volatility tercile (`Low`, `Normal`, `High`) from an XGBoost regressor (Spearman rank correlation = **0.666**).
5. `band_low`, `band_high`: Expected price excursion boundaries from quantile regression ($q=0.10, 0.90$) with average 82.7 bps width.
6. `regime`: Intraday / multi-day market regime (`Uptrend`, `Downtrend`, `Chop`).
7. `historical_hit_rate_for_this_hour_and_side`: Historical out-of-fold hit rate conditioned on the specific hour of day and direction.

---

## 3. Repository Layout

```text
├── AGENTS.md                  # Operating instructions and protocol
├── README.md                  # Project overview and reproduction instructions
├── RESULTS_V2.md              # Holdout evaluation results scorecard
├── requirements.txt           # Pinned Python package dependencies
├── run_all.sh                 # End-to-end V2 reproduction script
├── configs/
│   ├── config.py              # Centralized configuration and paths
│   ├── data_freeze.json       # Frozen last completed GS bar (2026-11-09)
│   └── model_v2_config.json   # Frozen hyperparameters, quantiles, and terciles
├── docs/
│   ├── problem_definition_v2.md # Pre-registered criteria A-E locked before holdout
│   ├── evaluation_protocol_v2.md# Walk-forward OOF & single-read protocol
│   ├── algorithm.md           # Formal mathematical algorithm specification
│   ├── dataset_card.md        # 8-year dataset card with 8 context symbols
│   ├── data_dictionary.md     # 51 features, targets, and insight outputs
│   ├── eda_findings.md        # Pre-holdout EDA, drift tests, and correlations
│   └── user_guide.md          # Trader guide and when NOT to trust insight heads
├── instructions/
│   ├── 00_rules_and_config.md .. 09_report_slides_freeze.md
│   ├── 10_v2_upgrade.md       # Verbatim V2 upgrade specification
│   └── progress.md            # Execution checklist, decisions, and fit logs
├── models/
│   ├── model_v2.joblib        # Final refit 3-way ensemble on 13,884 bars
│   ├── calib_v2.joblib        # Final Platt calibrator
│   ├── vol_v2.joblib          # Final volatility regressor
│   └── band_v2.joblib         # Final quantile band regressors
├── report/
│   └── report.md              # Full 10-section IMPACT technical report
├── slides/
│   └── outline.md             # 15-slide presentation outline
├── src/
│   ├── fetch_data.py          # Resumable Alpaca SIP data fetcher
│   ├── build_dataset_v2.py    # 1h resampling and 51-feature matrix builder
│   ├── features.py            # 51 stationary causal features in 9 groups
│   ├── labels.py              # Multi-horizon labels (H1, H3, HD, VOL)
│   ├── walk_forward.py        # 20-quarter expanding window OOF generator
│   ├── run_experiments_v2.py  # Model tournament, search, ablations, calibration
│   ├── evaluate_holdout_v2.py # Single-read holdout evaluation script
│   ├── refit_final_v2.py      # Final refit on all 13,884 bars
│   ├── insights.py            # Reusable get_bar_insights(row) function
│   ├── predict_day.py         # Bar-by-bar day replay CLI tool
│   ├── checks.py              # Comprehensive 7-check integrity suite
│   └── check_secrets.py       # Pre-commit secret and data file scanner
└── results/
    ├── tables/                # Tournament, ablation, holdout, and day CSVs
    ├── figures/               # Calibration curves, EDA, and day replay PNGs
    └── logs/                  # Execution logs and integrity reports
```

---

## 4. End-to-End Reproduction Instructions

### 4.1 Environment Setup
```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 4.2 Replay a Trading Day
To replay any historical trading day bar-by-bar using strictly closed-bar data:
```bash
python3 -m src.predict_day --date 2026-11-06
```
This generates:
- `results/tables/day_2026-11-06.csv`
- `results/figures/day_2026-11-06.png`

### 4.3 Run Full Pipeline
To execute the complete end-to-end workflow (dataset build, walk-forward tournament, holdout scoring, day replay, and integrity suite):
```bash
bash run_all.sh
```

### 4.4 Run Integrity & Leak Detection Suite
```bash
python3 -m src.checks
```
Verification checks include:
1. Shuffled-label null leakage test (~50% accuracy).
2. Time-shift degradation test.
3. Synthetic two-sided unit test.
4. 50-cutoff causal truncation test across all 51 features.
5. Pre-holdout walk-forward ledger audit.
6. Rolling window live-compatibility test.
7. Full git history security & parquet scan.

### 4.5 Run V3 Live Trading Dashboard & Server
```bash
./run_live.sh
# or
python3 -m src.live.server --port 8000
```
Open [http://localhost:8000](http://localhost:8000) for the live TradingView Lightweight Charts dashboard, real-time forming candle, and signal telemetry.

Accelerated historical day replay:
```bash
python3 -m src.live.server --replay 2026-11-06
```

### 4.6 Run Live Scoring CLI
```bash
python3 -m src.live.score --since 2026-11-10
```
Outputs cumulative performance to `results/tables/live_scoreboard.csv` and daily logs to `results/tables/live_{date}.csv`.
