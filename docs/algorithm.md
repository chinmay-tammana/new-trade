# Algorithm Specification: Multi-Horizon Calibrated Direction and Insight Engine for GS (V2 Upgrade)

This document provides a formal, code-independent algorithmic specification of the machine learning pipeline developed for predicting next-hour price direction and generating volatility/band/regime insights in Goldman Sachs (`GS`) equity. It details inputs, outputs, mathematical operations, training phases, tuning loops, decision rules, and evaluation protocols strictly adhering to the IMPACT Lab (IIT Patna) End-to-End ML Guidelines.

---

## 1. Mathematical Definitions and System Input/Output

### 1.1 Inputs
- **Market Data Feed**: Historical 1-minute OHLCV trade bars for target asset $S = \text{GS}$ and context benchmarks:
  - Benchmark Equities: $B_1 = \text{SPY}$ (S&P 500 ETF), $B_2 = \text{XLF}$ (Financial Select Sector SPDR)
  - Direct Banking Peers: $P_1 = \text{JPM}$, $P_2 = \text{MS}$, $P_3 = \text{BAC}$
  - Macro Factors: $M_1 = \text{TLT}$ (20+ Year Treasury Bond ETF), $M_2 = \text{VIXY}$ (VIX Short-Term Futures ETF)
  - Data span: October 15, 2018 to November 9, 2026 (14,084 session-anchored 1h bars).
- **Trading Session Hours**: Regular Trading Hours (RTH) on US equity markets, defined from 09:30 to 16:00 US Eastern Time (ET).
- **Cost Benchmark**: Bid-ask spread and transaction friction penalty fixed at $c = 5.0$ basis points (bps) per round-trip trade ($0.0005$).

### 1.2 Outputs
1. **Calibrated Posterior Direction Probability**: $\hat{p}_t = \mathbb{P}(Y_{t, H1} = 1 \mid \mathcal{F}_t)$, representing the model's posterior confidence that price moves upward over the subsequent 1-hour bar ($H1$).
2. **Directional Trading Signal**: $s_t \in \{\text{Up}, \text{Down}, \text{No-Signal}\}$, determined by two-sided quantile thresholds $q_{\text{lo}}$ and $q_{\text{hi}}$.
3. **Volatility Forecast Class (`vol_class`)**: $\in \{\text{Low}, \text{Normal}, \text{High}\}$, derived from XGBoost regression on forward log-range $\ln\left(\frac{High_{t+1} - Low_{t+1}}{Close_{t+1}}\right)$.
4. **Expected Price Bounds (`band_low`, `band_high`)**: Next-bar return bounds at $q=0.10$ and $q=0.90$ quantiles.
5. **Macro / Micro Regime (`regime`)**: $\in \{\text{Uptrend}, \text{Downtrend}, \text{Chop}\}$, determined by 20-day cumulative return and efficiency ratio.

---

## 2. Algorithmic Architecture and Pipeline Steps

```
[ Raw 1-Min Bars (GS + 7 Context) ]
             |
             v
[ 1. Session-Anchored 1h Resampling (7 bars/day, 09:30 open) ]
             |
             v
[ 2. Causal Feature Engineering (51 features across 9 groups) ]
             |
             v
[ 3. Multi-Horizon Labeling (H1, H3, HD, VOL) & Decisive Filtering ]
             |
             v
[ 4. Quarterly Walk-Forward OOF (20 Expanding Folds, 8-bar purge gap) ]
             |
             +---> [ Candidate Tournament: 3-Way Ensemble (XGB+HGB+ElasticNet) ]
             +---> [ Horizon Tournament: H1 Selected (Wilson LB: 49.83%) ]
             +---> [ Class-Balanced Platt Calibration (ECE: 0.0191) ]
             +---> [ Signal Quantiles: q_lo = 0.4904, q_hi = 0.5084 (c = 0.30) ]
             |
             v
[ 5. Pre-Registered Research Holdout Single Read (1,872 bars) ]
             | (Accuracy: 52.25%, p = 0.0417, 888 Up / 786 Down)
             v
[ 6. Final Model Refit on All Data to Freeze Date (model_v2.joblib) ]
             |
             v
[ 7. Replay & Live Reusable API (src/insights.py, src/predict_day.py) ]
```

---

## 3. Detailed Algorithmic Operations

### Step 1: Session-Anchored Resampling
1. Partition 1-minute bars into 7 session-anchored candles indexed $k \in \{0, 1, \dots, 6\}$:
   - Bar 0: 09:30–10:29 (60 min)
   - Bar 1: 10:30–11:29 (60 min)
   - Bar 2: 11:30–12:29 (60 min)
   - Bar 3: 12:30–13:29 (60 min)
   - Bar 4: 13:30–14:29 (60 min)
   - Bar 5: 14:30–15:29 (60 min)
   - Bar 6: 15:30–15:59 (30 min, session close)
2. Aggregate: $Open = O_0$, $High = \max_m H_m$, $Low = \min_m L_m$, $Close = C_{\text{last}}$, $Volume = \sum_m V_m$.
3. Synchronize `GS` and 7 context assets by inner timestamp join (14,084 aligned bars).

### Step 2: Causal Feature Construction (51 Features)
The feature matrix $X \in \mathbb{R}^{N \times 51}$ is partitioned into 9 groups:
1. **Returns (5)**: `ret_1`, `ret_2`, `ret_3`, `ret_6`, `ret_14`, `gap`.
2. **Momentum (3)**: `rsi` (Wilder 14), `macd_n`, `macd_hist_n`.
3. **Volatility (4)**: `atr_n`, `vol_20`, `bb_pos`, `bb_width`.
4. **Volume (1)**: `rvol`.
5. **Time Context (5)**: `hour`, `dow`, `is_first_bar`, `bars_since_open`, `bars_to_close`.
6. **Market Benchmarks (6)**: `spy_ret_1`, `spy_ret_3`, `xlf_ret_1`, `xlf_ret_3`, `rel_ret_spy`, `rel_ret_xlf`.
7. **Peers & Macro (12)**: `jpm_ret_1`, `jpm_ret_3`, `ms_ret_1`, `ms_ret_3`, `bac_ret_1`, `bac_ret_3`, `tlt_ret_1`, `tlt_ret_3`, `vixy_ret_1`, `vixy_ret_3`, `gs_minus_peers`, `rel_ret_xlf`.
8. **Intrabar Microstructure (5)**: `realized_vol_1m`, `up_minute_ratio`, `clv`, `signed_volume_ratio`, `last15_ret`.
9. **Levels & Trend (10)**: `dist_prev_close`, `dist_prev_high`, `dist_prev_low`, `dist_session_vwap`, `ret_1d`, `ret_5d`, `ret_20d`, `ma_dist_50h`, `ma_dist_200h`, `efficiency_ratio_20`.

### Step 3: Target Definitions and Decisive Filtering
1. **Primary Direction ($H1$)**: $F_{t, 1} = \frac{C_{t+1} - C_t}{C_t}$ for $t$ not the final bar of day. Decisive if $|F_{t, 1}| > 0.0010$ (10 bps).
2. **Multi-Hour Direction ($H3$)**: $F_{t, 3} = \frac{C_{t+3} - C_t}{C_t}$ (strictly same day). Decisive if $|F_{t, 3}| > 0.0010 \times \sqrt{3}$.
3. **Rest of Day ($HD$)**: $F_{t, D} = \frac{C_{\text{day\_close}} - C_t}{C_t}$ (for bars with $\ge 2$ bars remaining). Decisive if $|F_{t, D}| > 0.0010 \times \sqrt{\text{bars\_remaining}}$.
4. **Volatility Target (`target_vol`)**: $\ln\left(\frac{High_{t+1} - Low_{t+1}}{Close_{t+1}}\right)$.
5. Decisive filtering applies strictly to training rows ($Y_t \in \{0, 1\}$); validation and evaluation score **all** non-censored bars.

### Step 4: Out-of-Fold Walk-Forward Tournament & Selection
1. **Quarterly Expanding Window**: 20 folds spanning 12,004 pre-holdout bars. Initial window = 2 years; step = 63 trading days; gap = 8 bars.
2. **Model Tournament**:
   - Constant Prior: Log-Loss = $0.69272$
   - **Ensemble (3-way)**: Log-Loss = **$0.69279$** (Tournament Winner among estimators)
   - ElasticNet Logistic Regression: Log-Loss = $0.69281$
   - Shallow XGBoost ($d=2, \eta=0.01$): Log-Loss = $0.69345$
   - HistGradientBoosting: Log-Loss = $0.69638$
3. **Horizon Selection**:
   - $H1$: Pooled OOF accuracy 51.92% (Wilson CI [49.83%, 54.02%], $n_{\text{up}}=1,091, n_{\text{down}}=1,091$, coverage $c=0.30$) $\to$ **Selected as Primary Horizon**.
   - $H3$: Coverage 20.0%, accuracy 50.84% (Wilson CI [47.53%, 54.14%]).
   - $HD$: Coverage 20.0%, accuracy 50.51% (Wilson CI [46.90%, 54.12%]).
4. **Probability Calibration**: Class-balanced Platt scaling on pooled OOF predictions yields $\text{ECE} = 0.0191$.
5. **Two-Sided Quantile Thresholds**: $q_{\text{lo}} = 0.49039, q_{\text{hi}} = 0.50844$. Assertions enforce $n_{\text{up}} \ge 150, n_{\text{down}} \ge 150$.

### Step 5: Research Holdout Single Read
- Evaluated strictly once on 1,872 bars (1,604 labelled hourly bars, October 16, 2025 to November 9, 2026).
- Generated 888 Up, 786 Down, 198 No-Signal across all bars (848 Up, 574 Down, 182 No-Signal on labelled bars).
- Evaluated direction accuracy: **52.25%** (743 / 1,422; Wilson 95% CI: [49.65%, 54.84%]).
- Permutation test (10k shuffles): **$p = 0.0417$**.
- Drift-adjusted: Up +1.59 points vs base (51.65% vs 50.06%), Down +3.20 points vs base (53.14% vs 49.94%).
- Calibration: Holdout $\text{ECE} = 0.0249$ ($\le 0.05$).
- Pre-registered Criterion B: NOT MET (accuracy below 53.0% threshold).

### Step 6: Final Refit & Reusable API
- All 13,884 bars up to freeze date are fit into `models/model_v2.joblib`, `models/calib_v2.joblib`, `models/vol_v2.joblib`, and `models/band_v2.joblib`.
- Updated quantiles: $q_{\text{lo}} = 0.43428, q_{\text{hi}} = 0.56323$.
- `src/insights.py` exposes `get_bar_insights(row)` for live production ingestion.
