# Goldman Sachs (GS) AI Directional Trading & Insight Engine (5-Minute M5 System & V2 Upgrade)

[![Model Release](https://img.shields.io/badge/Release-m5--always--call-blue.svg)](https://github.com/chinmay-tammana/new-trade/releases/tag/m5-always-call)
[![Historical Baseline](https://img.shields.io/badge/V1%20Baseline-v1--negative--result-orange.svg)](https://github.com/chinmay-tammana/new-trade/releases/tag/v1-negative-result)
[![Python Version](https://img.shields.io/badge/Python-3.11-green.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/Scope-Educational%20%2F%20Paper%20Trading-orange.svg)]()
[![Repository](https://img.shields.io/badge/GitHub-chinmay--tammana%2Fnew--trade-blue)](https://github.com/chinmay-tammana/new-trade)

An end-to-end machine learning system developed according to the **IMPACT Lab (IIT Patna) End-to-End ML Project Guidelines** to predict next-bar directional returns of Goldman Sachs (`GS`) equity. Now featuring the **5-minute intraday trading horizon (M5)** with **Always-Call directional signals (Rule v3)** alongside multi-dimensional market insights (`VOL`, `BAND`, `REGIME`).

> **Disclaimer**: This repository is developed strictly for academic research and educational evaluation. It does NOT constitute investment advice or a recommendation to trade financial securities. Real capital trading is strictly outside the project scope; all evaluation is restricted to simulated paper trading.

---

## 5-Minute System (M5) Always-Call Performance Profile (Rule v3)

Following Owner Decision (2026-11-16), the 5-minute system emits a directional call (**UP** or **DOWN**) on **every eligible 5-minute bar** of the trading day (~75–78 calls/day).

> [!WARNING]
> **IEX Microstructure Print Warning (Integrity Audit Gate 4)**:
> In Section A integrity audit, 5-minute directional accuracy evaluated against IEX exchange prints was **69.9%**, but drops to **51.80%** when evaluated against Consolidated SIP prints ($\Delta = +18.1\%$). On short 5-minute horizons, IEX trades often capture local odd-lot prints and bid-ask bounces rather than true NBBO mid-price movement.

### Core Always-Call Metrics & Pre-Registered Checks

| Metric / Check | Pre-Holdout OOF ($n=39,766$) | Research Holdout ($n=9,472$) | Pre-Registered Gate Status |
| :--- | :--- | :--- | :--- |
| **All-Bar Accuracy** | **55.60%** [55.11%, 56.09%] | **53.39%** [52.38%, 54.39%] | **Check (a) PASSED**: Beats Base Rate 50.67% ($p = 0.00020 < 0.05$) |
| **STRONG Tier Acc ($\ge 58.1\%$)** | **63.58%** [62.63%, 64.52%] ($n=9,942$) | **60.01%** [57.42%, 62.56%] ($n=1,393$) | **Check (b) PASSED**: Monotonic tiers on both OOF and Holdout |
| **NORMAL Tier Acc ($52.1\%-58.1\%$)**| **53.85%** [53.15%, 54.54%] ($n=19,882$) | **53.26%** [51.91%, 54.61%] ($n=5,272$) | Monotonic intermediate tier |
| **WEAK Tier Acc ($\le 52.1\%$)** | **51.12%** [50.13%, 52.10%] ($n=9,942$) | **50.34%** [48.49%, 52.19%] ($n=2,807$) | Monotonic base tier ("close to a coin flip") |
| **Expected Calibration Error (ECE)**| **0.0069** | **0.0127** | **Check (c) PASSED**: Both $\le 0.030$ |
| **Side Balance** | 19,695 Up / 20,071 Down | 4,814 Up / 4,658 Down | **Check (d) PASSED**: Both sides balanced |
| **Net Return @ 2bps (ALL)** | $+0.06$ bps [$-0.12, +0.25$] | $-0.82$ bps [$-1.21, -0.50$] | Descriptive |
| **Net Return @ 2bps (STRONG)** | **$+3.43$ bps** [$+3.00, +3.87$] | **$+1.57$ bps** [$+0.59, +2.51$] | Descriptive |
| **Signals per Full Day** | ~65 calls/day (burn-in period) | **75.2 calls/day** | Full intraday coverage achieved |

*Holdout Caveat: The research holdout dataset has already been inspected during development; all holdout numbers are descriptive.*

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

### 4.5 System Health Check (Doctor CLI)
Verify system clocks (ET and IST), NYSE calendar, delayed signal schedule, delay arithmetic (960s), frozen model SHA256 hashes, hash chain (schema_v=1 and schema_v=2), Alpaca delayed SIP feeds for all 8 symbols, and local server:
```bash
python -m src.live.doctor
```

### 4.6 Run V3 Live Trading Dashboard & Server (Delayed SIP)
```bash
./run_live.sh
```
Open [http://localhost:8000](http://localhost:8000) for the live TradingView Lightweight Charts dashboard, real-time forming candle, synchronized ET/IST clocks, inline full-width signals log panel directly below the chart, and signal telemetry.

**V3 Feed Bias Finding & Switch to Delayed SIP (Option B)**:
- **IEX Gate Failure**: During initial live deployment, free IEX feeds failed Parity Gate B (Three-Class Agreement = 72.14% < 75%) and Gate C (GS Missing Minutes = 31.08% > 5%) because IEX captures only ~2% of total consolidated volume.
- **Feed Bias Diagnostics**: Thin volume skewed `signed_volume_ratio` (+1.32 Base stds) and depressed `up_minute_ratio` (-3.46 Base stds), driving calibrated probability downward and producing one-sided Down signals on IEX (7 Down, 0 Up).
- **Swap-Test Proof**: Swapping only the volume feature group (`rvol` and `signed_volume_ratio`) to consolidated SIP restored two-sided signals (2 Up, 5 Down), matching consolidated SIP tape (2 Up, 4 Down, 18 No Signal).
- **Option B Selection**: Owner selected **Option B (Delayed Consolidated SIP for Signals, IEX Display Only)**. Live signals are computed at `bar_end + 16 min` using consolidated SIP tape for all 8 tickers. All 3 Pre-Registered SIP Gates passed: Gate A (Feature Parity max diff 1.09e-11 <= 1e-6), Gate B (Delayed SIP access 8/8 OK), Gate C (Replay equivalence 100% on 3 days).
- **Measurement Notice**: The 52.25% accuracy [49.65, 54.84] and permutation p = 0.0417 were measured on the pre-refit model with narrow thresholds 0.490 / 0.508 (88.65% coverage on holdout). The deployed rule uses thresholds 0.434 / 0.563 (~30% target coverage) and has no held-out accuracy measurement.

**Monday Live Test Checklist**:
1. Run `python -m src.live.doctor` before 09:00 ET (19:30 IST) to verify all 7 sub-systems.
2. Start `./run_live.sh` before 09:00 ET (19:30 IST).
3. First signal will generate and deliver at **10:46 ET (21:16 IST)** (as the 09:30–10:30 candle closes at 10:30 ET and delayed SIP minute bars are ingested at `10:30 + 16 min`).
4. Total signals per day: **~6 evaluation signals** delivered at 10:46, 11:46, 12:46, 13:46, 14:46, 15:46 ET. The last signal covers 14 remaining minutes before 16:00 close; the 15:30–16:00 bar has No signal.
5. At market close (16:00 ET / 02:30 IST), verify hash chain immutability:
   ```bash
   python -c "from src.live.store import LiveDatabase; db = LiveDatabase(); print(db.verify_hash_chain())"
   ```

*Note on Live Test Policy*: Per TA instructions, a **ONE-DAY test** is sufficient for live execution demonstration. The indicator strictly evaluates **1-hour candles** (no 10-minute mode). A 1-day sample ($N \approx 6$) is an engineering proof of execution, data ingestion, and cryptographic auditability; formal statistical evaluation was conducted on the sealed 1,422 holdout candles ($p = 0.0417$). Real browser renders across all viewports (1280x720, 1440x900, 1920x1080, 390x844) are tested via `tests/ui_render_check.py` and saved to `results/figures/ui_log_<viewport>.png`.

**V3.6 Single-Source-of-Truth Day View**:
- **Single Source of Truth**: The Signal Log, Day Summary Strip, Hour Strip, and Signal Balance Card all derive from `src/live/log_engine.py`, guaranteeing zero numerical contradictions.
- **De-duplication**: Exactly 1 primary row per hourly bar (preferring consolidated SIP tape).
- **Day Controls**: Easy navigation via `◀ Prev` and `Next ▶` buttons, interactive date picker, and `?day=YYYY-MM-DD` URL parameter support.
- **Hour Strip**: Interactive chips per hour showing predictions and outcomes, linked directly to chart markers and table rows.
- **Audit Details**: Click any row to expand its exact calibrated probability against thresholds, market regime, model hash `75d0b0bf`, and cryptographic record hash.

### 4.7 Run Live Scoring CLI
```bash
python -m src.live.score --since 2026-11-10
```
Outputs cumulative performance to `results/tables/live_scoreboard.csv` and daily logs to `results/tables/live_{date}.csv`.

### 4.8 Run Out-of-Sample Scoreboard Audit CLI
```bash
python -m src.live.audit_scoreboard
```
Audits all post-freeze signals, verifies forward returns against target candles, populates `outcomes_v2`, and outputs `results/tables/live_scoring_audit.csv`.


---

## 7. M5 (5-Minute) System Upgrade

Per owner decision (dated 2026-11-16), a dedicated 5-minute directional prediction pipeline is deployed alongside the 1-hour system:
- **Horizon**: Next 5-minute bar directional return ($H1_{5m}$).
- **Feed**: IEX real-time feed with causal aggregation and forensic minute logging (`raw_minutes`).
- **Holdout Evaluation (6 Months Single Read, May 18 - Nov 13, 2026)**:
  - Total Bars: 9,472 | Directional Signals: 288 (150 Up, 138 Down, 9,184 No-signal)
  - Accuracy: **67.36%** (Wilson 95% CI: `[61.75%, 72.52%]`)
  - Up: 66.00% (base 49.33%) | Down: 68.84% (base 50.67%)
  - Permutation Test: $p = 0.0001$ (10,000 shuffles)
  - Net Return @ 2 bps friction: **+6.08 bps** (95% CI: `[+5.38, +6.77]`)
- **Execution Script**:
  ```bash
  ./run_live_m5.sh   # Starts 5-minute operational server on http://localhost:8000
  ```

### 7.1 V4.1-COVERAGE Update & Integrity Audit (Owner Decision 2026-11-16)
- **Target Coverage**: `coverage_target_m5 = 0.30` (30% directional coverage, $\approx 23$ signals/day), expanding trade frequency from earlier 10% coverage (~2–4 signals/day).
- **Integrity Audit Findings (5 Pre-Registered Fail Gates)**:
  - Gate 1 (Shuffled Labels Null Test): **50.00%** [PASS]
  - Gate 2 (Time-Shift +1 Bar): **50.00%** [PASS]
  - Gate 3 (Single-Feature Leakage): Max AUC Delta **0.0591** (`gs_minus_peer_1`, AUC = 0.4409) [PASS]
  - Gate 4 (IEX Microstructure Artifact): **ARTIFACT DETECTED**. IEX forward-return accuracy is 69.9% while Consolidated SIP tape accuracy is **51.80%** ($\Delta = +18.1\%$). Red warning card displayed on page.
  - Gate 5 (Stability): Consistent across calendar years (68.1%–75.5%), time-of-day buckets (Opening 70.2%, Midday 69.8%, Closing 70.0%), and sides (Up 69.6%, Down 70.3%).
- **Rule Selection Decision**:
  - Rule A achieved only 12.57% coverage on the last 20 sessions ($< 15.0\%$ fallback threshold).
  - **Rule B (Dynamic Trailing Quantiles) ACTIVATED**: $q_{lo} = 0.45168$, $q_{hi} = 0.54528$, `min_margin = 0.01` (all directional calls $\ge 51.0\%$).
- **Confidence Tiers**:
  - `STRONG` ($\ge 58.4\%$ confidence): 68.3% OOF acc, 64.4% holdout acc
  - `NORMAL` ($55.6\% - 58.4\%$ confidence): 57.3% OOF acc, 57.6% holdout acc
  - `WEAK` ($< 55.6\%$ confidence): 55.0% OOF acc, 53.0% holdout acc
- **CLI Tool**:
  ```bash
  python -m src.m5.apply_rule --coverage 0.30  # Change coverage target in one non-destructive command
  ```
