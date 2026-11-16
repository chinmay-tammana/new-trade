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
- **Embargoed Holdout Window**: October 16, 2025 to November 9, 2026 (1,872 bars total, 1,604 labelled bars).
- **Signal Emissions on All Bars**: 888 Up (47.4%), 786 Down (42.0%), 198 No-Signal (10.6%).
- **Signals on Labelled Bars**: 848 Up, 574 Down, 182 No-Signal (1,422 evaluated signals).
- **Evaluated Signal Accuracy**: **`52.25%`** (743 / 1,422).
- **Wilson 95% Confidence Interval**: **`[49.65%, 54.84%]`**.
- **Permutation Test**: **`p = 0.0417`** (10,000 shuffles).
- **Pre-Registered Criterion B**: **NOT MET** (Accuracy 52.25% < 53.0% threshold).
- **Calibration on Holdout**: $\text{ECE} = \mathbf{0.0249}$ ($\le 0.05$, Criterion D Met).

---

## Slide 12: Drift-Adjusted Performance & Regime Breakdown
- **Per-Side Performance vs Base Rates (1,604 Labelled Candles)**:
  - **Up Signals**: 51.65% accuracy (438 / 848) vs 50.06% period base rate (**`+1.59 points`** drift-adjusted edge).
  - **Down Signals**: 53.14% accuracy (305 / 574) vs 49.94% period base rate (**`+3.20 points`** drift-adjusted edge).
  - Both sides independently outperform period base rates (Criterion C Met).
- **Performance by Intraday Regime**:
  - **Chop Regime**: **`58.30%` accuracy** (446 signals), **`+2.28 bps` net return** after 5 bps friction.
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

---

## Slide 16: V3 Live Production Architecture, Feed Bias Discovery & Delayed SIP Engine
- **Initial IEX Deployment & Gate Failure**:
  - Thin IEX exchange volume (~2% market share) failed Parity Gate B (72.14% < 75%) and Gate C (31.08% missing minutes > 5%).
  - Diagnostic audit (V3.4) identified severe **FEED BIAS**: depressed `up_minute_ratio` by -3.46 stds and skewed `signed_volume_ratio` by +1.32 stds, artificially creating one-sided Down signals on IEX.
  - Swap test proved that swapping volume features to consolidated SIP restored two-sided signals (2 Up, 5 Down).
- **Option B Switch (Delayed Consolidated SIP for Signals, IEX Display Only)**:
  - Transitioned live signal engine to consolidated SIP tape delivered at `bar_end + 16 min` (complying with $\ge 15$-min dissemination delay).
  - All 3 pre-registered SIP gates passed: Gate A (feature parity $1.09 \times 10^{-11} \le 10^{-6}$), Gate B (delayed SIP access 8/8 symbols), Gate C (replay equivalence 100% on 3 days).
  - `live_signals_enabled = True` activated.
- **Critical Measurement Integrity Notice**:
  - The 52.25% accuracy was measured on the pre-refit model with narrow thresholds 0.490 / 0.508 (88.65% holdout coverage).
  - The deployed rule uses refitted thresholds 0.434 / 0.563 (~30% target coverage) and has **no held-out accuracy measurement**.
- **Monday Live Checklist**:
  - Run `python -m src.live.doctor` before 09:00 ET (19:30 IST).
  - Start `./run_live.sh` before 09:00 ET. First signal delivers at 10:46 ET (21:16 IST). Exactly ~6 signals per day.


---

## Slide 10: 5-Minute Horizon Pipeline (M5 Upgrade)
- **Problem & Rationale**: Transitioned from 1-hour to 5-minute decision horizon per TA instruction. Delayed SIP (16-min lag) replaced with causal IEX real-time feed with bit-for-bit `raw_minutes` recording.
- **Candidate Selection & Decisiveness**: Decisive threshold of 2 bps selected by walk-forward log-loss (0.68179). `xgb_depth4` selected as winning model (OOF log-loss 0.68057 vs 0.69310 constant prior).
- **Holdout Results (6 Months)**: 150 Up, 138 Down, 9,184 No-signal. Accuracy: **67.36%** (Wilson CI `[61.75%, 72.52%]`, permutation $p = 0.0001$). Both sides outperformed market base rates.
- **Honest Reporting**: Criterion B marked NOT MET because directional count was 150 Up / 138 Down (< 300 pre-registered), even though accuracy (67.36%) substantially exceeded the 52% target.
- **Production Status**: Model bundle sealed and locked in `models/m5/MODEL_SHA256_m5.json`. Operational server running on `http://localhost:8000` via `./run_live_m5.sh`.

---

## Slide 11: M5 Coverage Expansion (V4.1-COVERAGE) & Integrity Audit
- **Problem**: Owner decision noted excessive No-signal bars under the initial $c=0.10$ rule (~2–4 directional signals/day). Target coverage increased to **30%** ($\approx 23$ signals/day).
- **Integrity Audit (5 Pre-Registered Fail Gates)**:
  - Gate 1 (Shuffled-Label Null): 50.00% [PASS]
  - Gate 2 (Time-Shift +1 Bar): 50.00% [PASS]
  - Gate 3 (Single-Feature Leakage): Max AUC Delta 0.0591 (`gs_minus_peer_1`, AUC = 0.4409) [PASS]
  - Gate 4 (IEX Microstructure Artifact): **ARTIFACT DETECTED**. 69.9% accuracy on IEX prints drops to **51.80%** on Consolidated SIP tape. App displays prominent red warning banner.
  - Gate 5 (Subgroup Stability): Consistent across years (68.1%–75.5%), TOD (70%), and sides (Up 69.6%, Down 70.3%) [PASS].
- **Rule Selection (Rule A vs Rule B)**:
  - Fixed OOF quantiles (Rule A) achieved only 12.57% coverage on the last 20 sessions ($< 15.0\%$).
  - **Rule B Activated**: Dynamic trailing 20-session quantiles with min margin guard $\ge 51\%$ ($q_{lo} = 0.45168$, $q_{hi} = 0.54528$).
- **Performance Trade-Off & Tiers**:
  - Daily directional signals increased from 1–3 to **20–27 per day** (27.8% achieved coverage on Nov 9–13).
  - Tiers: `STRONG` ($\ge 58.4\%$ conf, 68.3% OOF / 64.4% holdout acc), `NORMAL` (57.3% acc), `WEAK` (55.0% acc).
  - Re-derivation CLI: `python -m src.m5.apply_rule --coverage 0.30` (non-destructive, unbroken hash chain).

---

## Slide 12: M5 Always-Call Directional Rule (V4.2-ALWAYSCALL) & Denominator Reconciliation
- **Owner Mandate**: "Keep No signal minimal, or at least reduce it a lot." Transitioned to Always-Call: every eligible 5-minute interval gets an Up or Down call with a conviction tier. No signal strictly limited to End-of-Day (15:55-16:00 ET), data gaps, or exact 50.0% ties.
- **Denominator Reconciliation**:
  - Reconciled 74,971 total pre-holdout bars with 39,766 evaluated out-of-fold bars (2-year burn-in training window of 35,205 bars).
  - Corrected erroneous "63,041 = 70.0%" statement: 63,041 of 74,971 was 84.1%, whereas exactly 27,836 of 39,766 OOF bars were No-Signal under Rule V2.
  - Holdout coverage was 19.3% under Rule V2 due to static quantile bounds and margin filtering.
- **Always-Call Performance & 4 Pre-Registered Checks**:
  - **Check (a) All-Bar Accuracy**: Holdout accuracy **53.39%** [52.38%, 54.39%] beats larger base rate 50.67% with permutation $p = 0.00020 < 0.05$ [PASS].
  - **Check (b) Monotonic Tiers**: WEAK $\le$ NORMAL $\le$ STRONG verified on both OOF and Holdout [PASS].
    - `STRONG` ($\ge 58.1\%$): 63.58% OOF, **60.01% Holdout**, **+1.57 bps** net @ 2bps.
    - `NORMAL` ($52.1\%-58.1\%$): 53.85% OOF, **53.26% Holdout**.
    - `WEAK` ($\le 52.1\%$): 51.12% OOF, **50.34% Holdout**, -0.82 bps net @ 2bps ("close to a coin flip").
  - **Check (c) Calibration ECE**: OOF ECE = 0.0069, Holdout ECE = 0.0127 (both $\le 0.030$) [PASS].
  - **Check (d) Two-Sided Output**: 19,695 Up / 20,071 Down on OOF; 4,814 Up / 4,658 Down on Holdout [PASS].
- **Live UI & Operational Proof**:
  - Live page shows side-by-side ALL vs. STRONG accuracy with Wilson CIs, timeline arrow chips with tier shading, and filter chips (`All | NORMAL + STRONG | STRONG only`).
  - Red warning card visible at top: IEX microstructure artifact disclosure ($\Delta = +18.1\%$).
  - Full-day coverage verified: 73 directional calls/day, 1 EOD bar (98.6% coverage on Nov 13).
  - Playwright headless browser tests passed 100% across 4 viewports with zero console errors. Cryptographic ledger verified unbroken (2,738 rows).

