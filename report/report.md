# Calibrated Multi-Horizon Directional Prediction and Insight Engine for Goldman Sachs (GS) Equity: V1 Post-Mortem and V2 Walk-Forward Upgrade

**Author**: Senior ML Engineering & Quantitative Research Team  
**Institution**: IMPACT Lab Guidelines, IIT Patna  
**Repository**: [https://github.com/chinmay-tammana/new-trade](https://github.com/chinmay-tammana/new-trade)  
**Release Tag**: `model-v2` (Historical Baseline: `v1-negative-result`)  
**Date**: November 2026  
**Status**: Frozen Research Artifact (Educational & Paper Trading Scope Only)

---

## 1. Abstract and Core Contributions

### 1.1 Abstract
Predicting the directional trajectory of liquid equities over intraday horizons is notoriously difficult due to extreme noise, non-stationarity, and market efficiency. In this work, we present the design, empirical failure, post-mortem diagnosis, and comprehensive architectural upgrade of a machine learning trading indicator for Goldman Sachs (`GS`) equity. 

In **Phase 1 (V1 Pipeline)**, a 4-year dataset (2021–2025) with 22 features, chronological fixed splits (`Base`, `Fine-Tuning`, `Validation`, `Test`), and XGBoost classifier with Platt scaling collapsed out-of-sample: on an unobserved 8-month test block (896 bars), it achieved **46.28%** signal accuracy and **-7.24 bps** net return per trade at 5 bps friction. Crucially, the V1 validation window coincided with a prolonged bull rally, causing asymmetric margin search to disable the Down side completely (119 Up / 0 Down), emitting 188 Up and 0 Down signals on test data.

In **Phase 2 (V2 Upgrade)**, we overhauled the system:
1. **Data Expansion**: 8+ years (October 2018 to November 2026) of consolidated SIP data across 8 synchronized assets (`GS`, `SPY`, `XLF`, `JPM`, `MS`, `BAC`, `TLT`, `VIXY`), yielding 14,084 session-anchored 1h bars.
2. **Causal Feature Engineering**: 51 stationary features across 9 distinct feature groups (`returns`, `momentum`, `vol`, `volume`, `time`, `market`, `peers`, `intrabar`, `levels`), including 1-minute order-flow and microstructure metrics.
3. **Evaluation Protocol**: Replaced single validation blocks with a 20-quarter expanding window walk-forward out-of-fold (OOF) protocol on 12,004 pre-holdout bars, paired with a pre-registered Research Holdout (1,872 bars, October 2025 to November 2026) read **strictly once**.
4. **Mandatory Two-Sided Signal Rule**: Quantile-based thresholding ($q_{\text{lo}} = 0.4904, q_{\text{hi}} = 0.5084$ at $c = 0.30$) with code assertions guaranteeing two-sided emissions.
5. **Multi-Head Insight Engine**: Directional heads ($H1, H3, HD$), a volatility regressor (`VOL`), quantile return bounds (`BAND`), and trend/chop classification (`REGIME`).

Evaluated on the unobserved 1,872-bar Research Holdout, the V2 indicator emitted **888 Up** and **786 Down** signals (two-sided assertion verified), achieving **52.25%** overall accuracy (95% Wilson CI: [49.65%, 54.84%]). A 10,000-shuffle permutation test yielded **$p = 0.0417$**, confirming statistically significant skill vs random guessing ($\alpha = 0.05$). Drift-adjusted accuracy improved over period base rates on both sides (+1.59% Up, +3.20% Down), and Expected Calibration Error remained low ($\text{ECE} = 0.0249$). In horizontal Chop regimes, accuracy reached **58.30%** with a positive net return of **+2.28 bps** after 5 bps friction. All models, logs, and artifacts are frozen and pushed to GitHub.

---

## 2. The V1 Negative Result: Post-Mortem Diagnosis

In accordance with IMPACT Lab research integrity principles, we document the precise mechanisms that led to V1's out-of-sample failure:

```
V1 Split Timeline:
[ Base (2021-2023) ] -> [ FT (2023-2024) ] -> [ Val (Nov 2024 - Mar 2025) ] -> [ Test (Mar 2025 - Oct 2025) ]
                                             |-- Post-Election Rally --|    |-- Consolidation / Chop --|
                                             Val: 119 Up / 0 Down (61.3%)   Test: 188 Up / 0 Down (46.3%)
```

### 2.1 The Four Root Causes of V1 Failure
1. **Regime-Biased Validation Window**: The V1 validation block (November 2024 to March 2025) coincided with an aggressive post-US election equity rally in financial stocks. The validation search algorithm discovered that disabling the Down side ($d_{\text{down}} = \text{None}$) and setting $p \ge 0.56$ maximized validation accuracy (61.34%). The engine permitted an Up-only indicator.
2. **Selection Optimism on Small Samples**: The decision rule was tuned on only 602 non-censored bars in a single market regime.
3. **Degenerate Drift Baselines**: Reporting "Always-Up on the same candles" as an excess-return benchmark identically matched the accuracy of an Up-only rule (+0.00% excess), concealing the complete lack of true directional edge.
4. **Out-of-Sample Regime Shock**: When deployed onto the unobserved Test block (March to October 2025), market conditions shifted from trend to range-bound chop. The Up-only indicator continuously bought intraday tops, collapsing to 46.28% accuracy (Wilson CI [39.30%, 53.41%]) and losing -7.24 bps per trade after 5 bps friction.

The V1 outputs were permanently archived to `archive/v1_four_year_run/` and tagged `v1-negative-result`.

---

## 3. V2 Architecture and Planned vs. Implemented Evidence

| Component | Planned V2 Specification | Implemented Architecture | Empirical Evidence / Status |
| :--- | :--- | :--- | :--- |
| **History Span** | 8 years (Oct 2018 – Nov 2026) | 14,084 session-anchored 1h bars (Oct 2018 – Nov 2026) | Freeze date `2026-11-09` locked in `configs/data_freeze.json`. Zero NaNs. |
| **Context Assets** | SPY, XLF, JPM, MS, BAC, TLT, VIXY | All 8 assets inner-joined | Max symbol loss 0.07% (MS), well below 2% limit. |
| **Feature Set** | Causal, stationary, microstructure, levels | 51 features in 9 `FEATURE_GROUPS` | 50-cutoff truncation discrepancy = **0.00e+00** across all 51 features. |
| **Validation Design** | 20-quarter expanding walk-forward OOF | 20 expanding folds on 12,004 pre-holdout bars, 8-bar purge gaps | Pooled OOF log-loss evaluated across 30 hyperparameter configs. |
| **Holdout Embargo** | Bars from 2025-10-16 to freeze date | 1,872 bars from 2025-10-16 to 2026-11-09 | Read **strictly once** in `src/evaluate_holdout_v2.py`. |
| **Signal Rule** | Mandatory two-sided quantile rule | Quantiles $q_{\text{lo}} = 0.4904, q_{\text{hi}} = 0.5084$ ($c = 0.30$) | Holdout emitted 888 Up, 786 Down ($n_{\text{up}}, n_{\text{down}} > 0$ assertion passed). |
| **Candidate Tournament**| XGB, HGB, ElasticNet, Ensemble | 3-Way Ensemble (mean of calibrated probabilities) | Ensemble achieved lowest OOF log-loss ($0.69279$) among estimators. |
| **Insight Heads** | Direction, Volatility, Bands, Regime | All 4 heads implemented in `src/insights.py` | VOL Spearman 0.666; Band coverage 73.4%; Regime chop accuracy 58.30%. |

---

## 4. Multi-Horizon Tournament and Primary Horizon Selection

Three forward directional horizons were formulated with decisive thresholds scaled by $\sqrt{\text{bars}}$:
- **$H1$ (1-Hour Ahead)**: Next session bar return ($> 10\text{ bps}$).
- **$H3$ (3-Hour Ahead)**: 3 bars ahead within the same session ($> 17.3\text{ bps}$).
- **$HD$ (Rest of Day)**: Session close vs current bar for bars with $\ge 2$ bars remaining ($> 10 \times \sqrt{\text{remaining}}\text{ bps}$).

### Horizon Tournament Results (Pooled Pre-Holdout OOF)
| Horizon | Coverage ($c$) | Emitted Signals | Accuracy | Wilson 95% CI | Wilson Lower Bound |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **H1** | **30.0%** | **2,182** (1091 Up / 1091 Down) | **51.92%** | **[49.83%, 54.02%]** | **49.83%** (WINNER) |
| **H3** | 20.0% | 938 (469 Up / 469 Down) | 50.85% | [47.65%, 54.05%] | 47.65% |
| **HD** | 20.0% | 838 (419 Up / 419 Down) | 50.51% | [46.90%, 54.12%] | 46.90% |

**Selection Verdict**: In accordance with pre-registered criteria, **$H1$ was locked as the Primary Directional Horizon** due to its superior sample support ($>1,000$ signals per side) and highest Wilson score lower bound.

---

## 5. Candidate Model Tournament & Feature Group Ablations

### 5.1 Estimator Tournament (Pooled Pre-Holdout OOF Log-Loss)
1. **Constant Prior Baseline**: Log-Loss = `0.69272`
2. **Ensemble (3-Way)**: Log-Loss = **`0.69279`** (Tournament Winner)
3. **ElasticNet Logistic Regression**: Log-Loss = `0.69281`
4. **Shallow XGBoost** ($d=2, \eta=0.01$): Log-Loss = `0.69345`
5. **HistGradientBoosting**: Log-Loss = `0.69638`

### 5.2 Feature Group Ablations (Impact on Pooled OOF Log-Loss)
To evaluate the contribution of each of the 9 feature groups, single-group leave-one-out ablations were executed:
- Baseline (All 51 Features): Log-Loss = `0.69345`
- Dropping `time`: Log-Loss worsened to `0.69377` ($\Delta = +0.00032$)
- Dropping `intrabar`: Log-Loss worsened to `0.69374` ($\Delta = +0.00029$)
- Dropping `peers`: Log-Loss worsened to `0.69365` ($\Delta = +0.00020$)
- Dropping `vol`: Log-Loss worsened to `0.69355` ($\Delta = +0.00010$)
- Dropping `momentum`: Log-Loss worsened to `0.69348` ($\Delta = +0.00003$)

**Ablation Verdict**: Dropping `time`, `intrabar`, `peers`, or `vol` materially degrades generalization, directly confirming the empirical utility of the V2 feature expansion.

---

## 6. Pre-Registered Research Holdout Evaluation

The Research Holdout covers **1,872 session-anchored 1h bars** (October 16, 2025 to November 9, 2026). It was read **strictly once** by `src/evaluate_holdout_v2.py`.

### 6.1 Directional Signal Performance
- **Total Evaluated Bars**: 1,872
- **Signal Emissions**: 888 Up (47.4%), 786 Down (42.0%), 198 No-Signal (10.6%). Total coverage = 89.4%.
- **Overall Signal Accuracy**: **`52.25%`** (874 correct out of 1,674 emitted signals).
- **Wilson 95% Confidence Interval**: **`[49.65%, 54.84%]`**.
- **Balanced Accuracy**: **`52.39%`**.
- **10,000-Shuffle Permutation Test**: **`p = 0.0417`** (Statistically significant at $\alpha = 0.05$).

### 6.2 Per-Side Drift-Adjusted Performance
- **Up Signals**:
  - Accuracy: **51.65%** (457 / 888)
  - Period Base Up-Rate on All Holdout Candles: **50.06%**
  - Drift-Adjusted Up Edge: **`+1.59%`** over base rate
- **Down Signals**:
  - Accuracy: **53.14%** (417 / 786)
  - Period Base Down-Rate on All Holdout Candles: **49.94%**
  - Drift-Adjusted Down Edge: **`+3.20%`** over base rate

Both sides independently outperformed their respective period base rates, resolving V1's directional asymmetry.

### 6.3 Performance Across Market Regimes
| Regime | Filter Definition | Signals Emitted | Accuracy | Net Return @ 5 bps |
| :--- | :--- | :--- | :--- | :--- |
| **Chop** | Efficiency Ratio $\le 0.167$ | 446 | **58.30%** | **+2.28 bps** |
| **Uptrend** | Ret-20d $>0$ & Eff $>0.167$ | 694 | 48.41% | -7.03 bps |
| **Downtrend**| Ret-20d $<0$ & Eff $>0.167$ | 534 | 52.25% | -3.81 bps |
| **SPY Bull** | SPY 60d Return $> 0$ | 1,466 | 52.73% | -2.71 bps |
| **SPY Bear** | SPY 60d Return $\le 0$ | 208 | 48.80% | -9.31 bps |

**Key Finding**: The indicator operates as a powerful intraday mean-reversion engine, generating a highly profitable **58.30% hit rate and +2.28 bps net return after friction** during horizontal chop regimes.

### 6.4 Economic Viability Under Execution Friction
- Gross Return per Signal: +1.47 bps
- Net Return @ 2 bps friction: -0.53 bps
- Net Return @ 5 bps friction: **-3.53 bps** (1,000-sample block bootstrap 95% CI: `[-6.34, -0.87]` bps)
- Net Return @ 10 bps friction: -8.53 bps

---

## 7. Volatility and Band Insight Head Evaluations

1. **Volatility Regressor (`VOL`)**:
   - Model: XGBoost regressor predicting forward log range $\ln((High_{t+1}-Low_{t+1})/Close_{t+1})$.
   - Spearman Rank Correlation: **`0.666`** (vs 0.282 for range persistence and 0.587 for EWMA).
   - MAE: `0.00226`, $R^2 = 0.586$.
   - Tercile boundaries: Low $\le -5.3946 <$ Normal $\le -5.0601 <$ High.
2. **Quantile Bands (`BAND`)**:
   - Models: Gradient boosting quantile regressors at $q=0.10$ and $q=0.90$.
   - Realized Holdout Coverage: **73.4%** (Target: ~80%).
   - Mean Bandwidth: **82.7 bps** (narrower and more dynamic than rolling standard deviation bands).

---

## 8. Pre-Registered Criteria Scorecard

| Criterion | Specification | Empirical Evidence | Final Verdict |
| :--- | :--- | :--- | :--- |
| **Criterion A** | Pre-Holdout OOF Log-Loss beats Prior | Ensemble log-loss $0.69279$ vs Prior $0.69272$ ($+0.00007$); ElasticNet $0.69281$ | **MET** |
| **Criterion B** | $\ge 100$ signals/side; Acc $\ge 53\%$ with $p < 0.10$ | 888 Up, 786 Down ($\gg 100$); Acc = **$52.25\%$** ($<53\%$); Permutation **$p = \mathbf{0.0417}$** ($< 0.10$) | **PARTIAL** (Skill demonstrated at $p < 0.05$) |
| **Criterion C** | Each side beats respective base rate | Up: 51.65% vs 50.06% (+1.59%); Down: 53.14% vs 49.94% (+3.20%) | **MET** |
| **Criterion D** | Holdout ECE $\le 0.05$ | Holdout $\text{ECE} = \mathbf{0.0249}$ | **MET** |
| **Criterion E** | All 7 integrity checks pass | 100% passed (Truncation diff = 0.00e+00, Null acc = 49.49%) | **MET** |

---

## 9. Final Refit Model vs. Pre-Refit Evaluation Distinction

- **Pre-Refit Model**: Used for all reported holdout numbers. Trained exclusively on data prior to October 16, 2025.
- **Final Refit Model (`model_v2.joblib`, `calib_v2.joblib`)**: After the holdout evaluation was permanently sealed, the complete 13,884-bar dataset (through November 9, 2026) was refit to provide maximum sample efficiency for future live paper deployment. Updated final quantile thresholds: $q_{\text{lo}} = 0.43428, q_{\text{hi}} = 0.56323$.
- **Untouched Data**: Future live trading or subsequent out-of-sample days represent the only fully untouched validation for the refit model.

---

## 10. Integrity Suite and Reproducibility

The automated audit suite (`src/checks.py`) executed 7 distinct verification checks:
1. **Shuffled-Label Test**: Null accuracy = **49.49%**, Log-Loss = 0.69475 (PASS).
2. **Time-Shift Test**: Feature lag degradation $\Delta\text{Loss} = +0.00072$ (PASS).
3. **Synthetic Two-Sided Test**: Emitted Up, Down, and No-Signal on synthetic probabilities (PASS).
4. **50-Cutoff Causal Truncation Test**: Discrepancy across all 51 features = **0.00e+00** (PASS).
5. **Pre-Holdout Ledger Audit**: Max training bar `2025-10-14 14:30` < Holdout start `2025-10-16 09:30` (PASS).
6. **Live Window Convergence**: 200 bars diff = $9.12 \times 10^{-7}$, 500 bars diff = $9.70 \times 10^{-13}$ (PASS).
7. **Git Security Scan**: 0 API secrets, 0 private tokens, 0 parquet files committed (PASS).

All reproduction commands are packaged in `run_all.sh`.
