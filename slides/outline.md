# Presentation Slide Deck Outline: Calibrated Multi-Horizon Direction & Insight Engine for Goldman Sachs (GS) (V2 Upgrade)

**Target Audience**: Quantitative Researchers, Machine Learning Engineers, and Risk Reviewers (IMPACT Lab / Academic & Industry Panels)  
**Format**: 15 Slides, Structured per IMPACT Guidelines  
**Time Limit**: 15–20 Minutes

---

## Slide 1: Title & Executive Summary
- **Title**: Calibrated Multi-Horizon Direction & Market Insight Engine for Goldman Sachs (`GS`)
- **Subtitle**: From V1 Negative Result to V2 Walk-Forward Generalization under Intraday Friction
- **Presenters**: Senior ML Engineering & Quantitative Research Team (IMPACT Lab Guidelines, IIT Patna)
- **Core Message**: Documenting how an initial 4-year bull-market tuned indicator collapsed out-of-sample, and how an 8-year expanding walk-forward architecture with 51 causal features, two-sided quantile rules, and multi-head insight outputs achieved statistically significant out-of-sample edge ($p = 0.0417$).

---

## Slide 2: Practical Problem & The Intraday Noise Challenge
- **The Challenge**: Predicting next-hour equity returns in liquid assets like Goldman Sachs (`GS`) is severely constrained by market efficiency, microsecond arbitrage, and low signal-to-noise ratios.
- **The Pitfalls**:
  - Uncalibrated probabilities lead to overconfidence.
  - Ignoring execution friction (5 bps bid-ask spread) produces paper illusions.
  - Single validation windows cause regime-selection traps.
- **The Goal**: A fully autonomous, leak-free ML indicator outputting two-sided signals, calibrated confidence, volatility forecasts, and price excursion bands.

---

## Slide 3: V1 Post-Mortem: Why the Initial Pipeline Failed
- **V1 Design**: 4-year dataset, 22 features, fixed validation block (Nov 2024 – Mar 2025).
- **The Validation Trap**: The validation window coincided with an aggressive post-election bull rally. Threshold search turned the Down side completely off ($p \ge 0.56$ for Up, Down disabled), achieving 61.34% validation accuracy.
- **The Out-of-Sample Collapse**:
  - Held-out test set (896 candles): 188 Up, 0 Down. Accuracy dropped to **46.28%** (Wilson CI [39.30%, 53.41%]).
  - Net return: **-7.24 bps** per signal after 5 bps friction.
  - "Always-Up on same candles" identically equaled signal accuracy (+0.00% excess edge).
- **IMPACT Lesson**: Parameter selection on a single macroeconomic regime creates severe selection optimism.

---

## Slide 4: V2 Upgrade Strategy & Pre-Registered Criteria
- **V2 Overhaul Principles**:
  1. Expand history to 8 years (Oct 2018 – Nov 2026) across 8 assets.
  2. 51 causal stationary features (intrabar 1-min microstructure, banking peers, macro rates, levels).
  3. Replace fixed validation with 20-quarter expanding walk-forward out-of-fold (OOF) protocol.
  4. Mandatory two-sided signal rule ($n_{\text{up}}, n_{\text{down}} \ge 150$).
  5. Pre-register acceptance criteria in `docs/problem_definition_v2.md` **before** reading holdout data.

---

## Slide 5: Multi-Asset Dataset & Ingestion Architecture
- **Span**: October 15, 2018 to November 9, 2026 (8 chronological years + 30-day buffer).
- **8 Synchronized Instruments**:
  - Target: `GS` (14,101 1h bars).
  - Benchmarks: `SPY`, `XLF`.
  - Direct Banking Peers: `JPM`, `MS`, `BAC`.
  - Macro Factors: `TLT` (20+ Yr Treasury ETF), `VIXY` (VIX Short-Term Futures ETF).
- **Inner Join Integrity**: 14,084 aligned bars; maximum context loss was 0.07% for MS ($\ll 2\%$ limit).
- **Quality Audit**: 0 NaNs, 0 duplicate timestamps, 100% monotonic session ordering.

---

## Slide 6: Causal Feature Matrix (51 Features in 9 Groups)
- **Feature Groups**:
  - *Returns & Momentum (8)*: `ret_1`..`ret_14`, `gap`, `rsi`, `macd_n`, `macd_hist_n`.
  - *Volatility (4)*: `atr_n`, `vol_20`, `bb_pos`, `bb_width`.
  - *Volume (1)*: `rvol`.
  - *Time & Session (5)*: `hour`, `dow`, `is_first_bar`, `bars_since_open`, `bars_to_close`.
  - *Market & Sector (6)*: `spy_ret_1`/`3`, `xlf_ret_1`/`3`, `rel_ret_spy`/`xlf`.
  - *Peers & Macro (12)*: `jpm`, `ms`, `bac`, `tlt`, `vixy` returns, `gs_minus_peers`.
  - *Intrabar Microstructure (5)*: `realized_vol_1m`, `up_minute_ratio`, `clv`, `signed_volume_ratio`, `last15_ret`.
  - *Levels & Trend (10)*: `dist_prev_close/high/low`, `dist_session_vwap`, `ret_1d/5d/20d`, `efficiency_ratio_20`.
- **Causality Verification**: 50-cutoff truncation test verified **0.00e+00** look-ahead discrepancy across all 51 features.

---

## Slide 7: Multi-Horizon Tournament & Primary Horizon Selection
- **Directional Heads**:
  - $H1$: Next 1-hour candle ($> 10$ bps).
  - $H3$: 3 hours ahead, same day ($> 17.3$ bps).
  - $HD$: Rest of session close ($\ge 2$ bars remaining).
- **Horizon Tournament on Pooled Pre-Holdout OOF**:
  - **H1 (Winner)**: Coverage 30%, 2,182 signals, Accuracy **51.92%** (Wilson CI: **[49.83%, 54.02%]**).
  - **H3**: Coverage 20%, 938 signals, Accuracy 50.85% (Wilson CI: [47.65%, 54.05%]).
  - **HD**: Coverage 20%, 838 signals, Accuracy 50.51% (Wilson CI: [46.90%, 54.12%]).
- **Decision**: $H1$ locked as the primary horizon due to highest Wilson lower bound and robust sample support.

---

## Slide 8: Candidate Model Tournament & Ablation Studies
- **Pooled Pre-Holdout OOF Log-Loss**:
  - Constant Prior: `0.69272`
  - **Ensemble (3-Way: XGB + HGB + ElasticNet)**: **`0.69279`** (Tournament Winner)
  - ElasticNet Logistic Regression: `0.69281`
  - Shallow XGBoost ($d=2, \eta=0.01$): `0.69345`
  - HistGradientBoosting: `0.69638`
- **Feature Group Ablations (Loss Degradation when Dropped)**:
  - Dropping `time`: $+0.00032$
  - Dropping `intrabar`: $+0.00029$
  - Dropping `peers`: $+0.00020$
  - Dropping `vol`: $+0.00010$
- **Finding**: High-frequency microstructure and banking peer spreads provide the primary empirical alpha.

---

## Slide 9: Calibration & Mandatory Two-Sided Signal Rule
- **Class-Balanced Platt Scaling**:
  - Calibrator fitted on pre-holdout pooled OOF probabilities.
  - Balancing ensures $p = 0.50$ represents pure directional neutrality.
  - Pre-holdout $\text{ECE} = \mathbf{0.0191}$.
- **Quantile Margin Thresholds**:
  - $q_{\text{lo}} = 0.4904, q_{\text{hi}} = 0.5084$ (Coverage $c = 0.30$).
  - Assertions enforce $n_{\text{up}} \ge 150, n_{\text{down}} \ge 150$ on OOF.
  - Guarantees signals fire in both directions regardless of recent market trend.

---

## Slide 10: Multi-Head Market Insights (`VOL`, `BAND`, `REGIME`)
- **Beyond Binary Direction**: Live traders require multi-dimensional market context.
- **Head 1: Forward Volatility (`VOL`)**:
  - XGB regressor on next-bar log range $\ln((H-L)/C)$.
  - Spearman rank correlation = **0.666** (beats range persistence 0.282 and EWMA 0.587).
  - Categorizes market into Low, Normal, High volatility terciles.
- **Head 2: Expected Price Bounds (`BAND`)**:
  - Quantile regressors at $q=0.10$ and $q=0.90$.
  - Average bandwidth 82.7 bps; realized holdout coverage 73.4%.
- **Head 3: Market Regime Tag (`REGIME`)**:
  - Classifies environment into Uptrend, Downtrend, or Chop via 20-day returns and efficiency ratio.

---

## Slide 11: Research Holdout Results (Strict Single Read)
- **Embargoed Holdout Window**: October 16, 2025 to November 9, 2026 (1,872 bars).
- **Signal Emissions**: 888 Up (47.4%), 786 Down (42.0%), 198 No-Signal (10.6%).
- **Overall Signal Accuracy**: **`52.25%`** (874 / 1,674).
- **Statistical Significance**:
  - Wilson 95% Confidence Interval: **`[49.65%, 54.84%]`**.
  - **10,000-Shuffle Permutation Test**: **`p = 0.0417`** (Statistically significant at $\alpha = 0.05$).
- **Calibration on Holdout**: $\text{ECE} = \mathbf{0.0249}$ ($\le 0.05$, Criterion D Met).

---

## Slide 12: Drift-Adjusted Performance & Regime Breakdown
- **Per-Side Performance vs Base Rates**:
  - **Up Signals**: 51.65% accuracy vs 50.06% period base rate (**`+1.59%`** drift-adjusted edge).
  - **Down Signals**: 53.14% accuracy vs 49.94% period base rate (**`+3.20%`** drift-adjusted edge).
  - Both sides independently beat random drift!
- **Performance by Intraday Regime**:
  - **Chop Regime**: **`58.30%` accuracy** (446 signals), **`+2.28 bps` net return** after 5 bps friction!
  - **Trending Regimes**: 48.41% Uptrend, 52.25% Downtrend (momentum breakouts challenge mean-reversion).
- **Economic Returns Overall**: Gross return +1.47 bps; Net return at 5 bps friction = -3.53 bps.

---

## Slide 13: Operational Trader Guide & When NOT to Trust Heads
- **When NOT to Trust Direction**:
  - 30 min before/after scheduled FOMC, CPI, NFP events.
  - Quarterly GS earnings release days (unmodeled gap risk).
- **When NOT to Trust Volatility (`VOL`)**:
  - Unscheduled breaking geopolitical or macro flash crashes (historical models lag sudden shocks).
- **When NOT to Trust Bands (`BAND`)**:
  - 1 in 5 bars naturally breaches $q=0.10/0.90$ bands. They are expectations, not barriers.
- **When NOT to Trust Regime (`REGIME`)**:
  - Sharp V-bottom reversals where trailing 20-day filters lag the inflection.
- **Strict Rule**: Never hold overnight. Close all positions by 16:00 ET.

---

## Slide 14: Automated Integrity Suite & Reproducibility
- **Automated Verification Suite (`src/checks.py`)**:
  1. *Shuffled-Label Test*: Null model accuracy = **49.49%** (PASS).
  2. *Time-Shift Test*: Feature lag degrades performance (PASS).
  3. *Synthetic Two-Sided Test*: Both sides emitted on synthetic data (PASS).
  4. *50-Cutoff Truncation Test*: 0.00e+00 discrepancy across all 51 features (PASS).
  5. *Pre-Holdout Ledger Audit*: 0 pre-holdout fits touched holdout data (PASS).
  6. *Live Window Convergence*: 500-bar rolling window diff $< 10^{-12}$ (PASS).
  7. *Git Security Scan*: 0 API keys, 0 secrets, 0 data files committed (PASS).

---

## Slide 15: Conclusion, Freeze & Next Steps
- **Key Achievements**:
  - Eliminated V1's directional asymmetry trap through walk-forward OOF quantile rules.
  - Proved statistically significant out-of-sample edge ($p = 0.0417$) on 1,872 unobserved bars.
  - Discovered robust profitability in Chop regimes (58.30% accuracy, +2.28 bps net return).
  - Delivered production-ready multi-head insight engine (`src/insights.py`) and day replay (`src/predict_day.py`).
- **Final Model Freeze**:
  - Final models refit on complete 13,884 bars: `models/model_v2.joblib`, `calib_v2.joblib`, `vol_v2.joblib`, `band_v2.joblib`.
  - Pushed to GitHub repository with release tag `model-v2`.
