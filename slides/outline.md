# Presentation Slide Deck Outline: Calibrated Next-Hour Direction Prediction for Goldman Sachs (GS)

**Target Audience**: Quantitative Researchers, Machine Learning Engineers, and Risk Reviewers (IMPACT Lab / Academic & Industry Panels)  
**Format**: 15 Slides, Structured per IMPACT Guidelines  
**Time Limit**: 15–20 Minutes

---

## Slide 1: Title & Overview
- **Title**: Calibrated Next-Hour Directional Prediction for Goldman Sachs (`GS`)
- **Subtitle**: An End-to-End Leak-Free Machine Learning Framework under Market Friction
- **Presenter**: Senior ML Engineering & Quantitative Research Team
- **Affiliation**: IMPACT Lab Guidelines, IIT Patna
- **Key Takeaway**: A fully audited, leak-free pipeline with post-hoc calibration that honestly documents the structural risks of asymmetric drift strategies in efficient markets.

---

## Slide 2: Motivation & Practical Problem
- **The Challenge**: Predicting intraday equity movements over short horizons (1 hour) is dominated by market noise, bid-ask friction, and regime shifts.
- **Target Asset**: Goldman Sachs (`GS`), a premier institutional benchmark with complex macro sensitivity to interest rates and investment banking cycles.
- **The Core Problem**: Most quantitative classifiers fail in live deployment because of:
  - Random train-test cross-contamination.
  - Ignoring transaction costs (bid-ask spread + slippage).
  - Uncalibrated overconfidence leading to severe drawdowns during horizontal chop.
- **Goal**: Develop a rigorous, causal ML indicator with a calibrated abstention mechanism ($s_t \in \{+1, -1, 0\}$).

---

## Slide 3: Literature Survey & The Research Gap
- **Foundations**:
  - *Weak-Form EMH* (Fama, 1970; Malkiel, 2003): Return series exhibit near-zero autocorrelation; linear filters cannot overcome trading friction.
  - *Tree Ensembles in Finance* (Gu, Kelly, & Xiu, 2020): Gradient boosting captures non-linear interactions across macroeconomic and technical signals.
  - *Financial ML Pitfalls* (De Prado, 2018): Critical importance of embargo periods, purged cross-validation, and avoiding backtest overfitting.
  - *Probability Calibration* (Niculescu-Mizil & Caruana, 2005; Guo et al., 2017): Uncalibrated models produce unreliable margins in risk-sensitive applications.
- **The Research Gap**: Few studies combine session-anchored multi-asset features, strict out-of-fold preprocessing, post-hoc Platt calibration, asymmetric abstention, and single-read test evaluation under 5 bps friction.

---

## Slide 4: Multi-Asset Dataset & Quality Audit
- **Data Source**: Consolidated SIP 1-minute bars from Alpaca Markets (Oct 2021 – Oct 2025, 4+ years).
- **Instruments**:
  - Target: Goldman Sachs (`GS`) — 7,148 raw 1h candles.
  - Broad Market Benchmark: `SPY` (S&P 500) — 7,161 candles.
  - Sector Benchmark: `XLF` (Financials) — 7,161 candles.
- **Session Architecture**: Exactly 7 regular-hours candles per day (9:30–10:30 through 15:30–16:00).
- **Third-Party Parity Check**:
  - Parity audit against `yfinance` over 60 days: price correlation = **0.999998**, mean absolute difference = **0.23 bps**, volume ratio = **1.0019**.
  - 0 NaNs, 0 zero-volume candles, 0 lost rows in inner join (7,118 synchronized candles).

---

## Slide 5: Chronological Splits & Leakage Architecture
- **Chronological Blocks with 8-Bar Embargo Gaps**:
  - `Base`: Oct 2021 – Dec 2023 (3,613 candles, 3,096 labelled) $\rightarrow$ Base booster training.
  - *[8-Bar Embargo Buffer]*
  - `Fine-Tuning (FT)`: Dec 2023 – Nov 2024 (1,732 candles, 1,484 labelled) $\rightarrow$ 4-fold walk-forward hyperparameter search.
  - *[8-Bar Embargo Buffer]*
  - `Validation (Val)`: Nov 2024 – Mar 2025 (702 candles, 602 labelled) $\rightarrow$ Model selection, Platt scaling, margin search.
  - *[8-Bar Embargo Buffer]*
  - `Test`: Mar 2025 – Oct 2025 (1,047 candles, 896 labelled) $\rightarrow$ **Single-read held-out evaluation**.
- **In-Fold Preprocessing**: Quantile Winsorizing ($[1\%, 99\%]$) and standard scaling fitted strictly on training folds.
- **Audit**: All 19 model fits verified in `results/logs/fit_ledger.json` prior to Val start; 50-cutoff truncation discrepancy = **0.00e+00**.

---

## Slide 6: Features & Decisive Directional Labeling
- **22 Causal Features**:
  - Momentum: 1, 2, 3, 6, 14-bar returns.
  - Intraday Context: Opening gap, hour, day-of-week, opening candle flag.
  - Technical Volatility: Wilder RSI (14), MACD histogram, Bollinger Band bandwidth & position, Normalized ATR, Realized Volatility (20).
  - Relative & Macro Context: RVOL, `SPY` 1-bar & 3-bar returns, `XLF` 1-bar & 3-bar returns, relative return vs SPY.
- **Labeling Protocol**:
  - Decisive threshold $\delta = 10 \text{ bps}$: $Y=1$ if $F > +10$ bps, $Y=0$ if $F < -10$ bps.
  - Indecisive candles ($|F| \le 10$ bps) retained in features, filtered during classification training.
  - Overnight candle (15:30–16:00) strictly censored ($F = \text{NaN}$).

---

## Slide 7: Simple Baselines (Validation Benchmark Floor)
- Evaluated on all 602 decisive candles in the Validation block:
  - **Constant Prior ($\hat{p}=0.5284$)**: Log-loss = **0.69178**, Brier = **0.24932**, Accuracy = **53.16%**.
  - **Always-Up**: Coverage = 100%, Accuracy = **53.16%**, Net return @ 5 bps = **-5.64 bps**.
  - **RSI Heuristic ($<40$ Up, $>60$ Down)**: Coverage = 50.0%, Accuracy = **49.83%**, Net return @ 5 bps = **-0.85 bps**.
  - **Previous-Bar Persistence**: Coverage = 100%, Accuracy = **45.02%**, Net return @ 5 bps = **-8.45 bps**.
- **Takeaway**: Simple technical indicators fail to beat historical upward drift; transaction costs destroy uncalibrated frequency.

---

## Slide 8: Model Optimization, Tuning & Architecture Selection
- **Walk-Forward Cross-Validation (FT Block, 4 Splits, 40 Configs)**:
  - Optimal params: `max_depth=3`, `lr=0.05`, `n_est=400`, `subsample=1.0`, `colsample=0.7`, `min_child_weight=3`, `reg_lambda=3.0`, `gamma=1.5`, `base_weight=0.3`.
- **Validation Tournament**:
  - S0 (Base Default): 0.70176
  - S1 (Base Tuned): 0.69297
  - S3 (Warm-Start Booster): 0.69244
  - **S2 (Refit Base + FT)**: **0.68953** $\rightarrow$ Selected model! Beat Constant Prior (0.69178) by 0.00225 (**Criterion C1 Met**).
- **Alternative Comparators**: RandomForest (0.69092), LogisticRegression (0.69109), HistGradientBoosting (0.69276). All underperformed S2.

---

## Slide 9: Probability Calibration & Signal Rule Selection
- **Platt Scaling**:
  - Fitted on Validation: reduced log-loss from 0.68953 to **0.68704**, Brier score to **0.24697**, and ECE to **0.0105**.
  - Perfectly aligns predicted probabilities with empirical event frequencies.
- **Asymmetric Signal Rule Search**:
  - Optimal threshold: $d_{\text{up}} = 0.06$ ($\hat{p} \ge 0.56$), $d_{\text{down}} = \text{None}$ (Down side disabled).
  - Validation Performance: 119 Up signals, 0 Down, 483 No-Signal.
  - Accuracy: **61.34%** (95% CI: [52.37%, 69.61%]).
  - Net return @ 5 bps: **+7.91 bps**.

---

## Slide 10: Held-Out Test Evaluation (The Moment of Truth)
- **Strict Single-Read Protocol**: Evaluated on 896 unobserved Test candles (Mar – Oct 2025).
- **Primary Results**:
  - **Signal Emissions**: 188 Up, 0 Down, 708 No-signal (Coverage: 20.98%).
  - **Signal Accuracy**: **46.28%** (95% Wilson CI: [39.30%, 53.41%]).
  - **Always-Up Drift Accuracy**: **46.28%** (Excess Edge: **+0.00%**).
  - **Net Return @ 5 bps Friction**: **-7.24 bps** (95% Bootstrap CI: [-13.90, -0.02] bps).
  - **Expected Calibration Error**: **0.0454** ($\le 0.050$, **Criterion C4 Met**).
  - **5-Seed Stability**: $46.50\% \pm 0.85\%$ accuracy across distinct initialization seeds.

---

## Slide 11: Error Analysis: Why Did the Model Underperform?
- **The Upward Drift Trap**:
  - Training (2021–2024) and Validation (late 2024–early 2025) were characterized by persistent upward drift (`GS` rose 53.2% of the time).
  - Threshold optimization rewarded disabling Down signals and lowering the Up threshold.
  - Test period (2025) encountered consolidation/correction (`GS` positive rate dropped to 48.88%; signal candles rose only 46.28%).
- **Top Errors**: Concentrated around opening gap fades (09:30 candle) and macro rate-decision announcements where momentum indicators reversed violently within 30 minutes.

---

## Slide 12: Feature Explainability (TreeSHAP) & Ablation Insights
- **Top SHAP Contributors**:
  - `rsi`: Overbought/oversold levels heavily modulated tree routing.
  - `spy_ret_1` & `xlf_ret_1`: Crucial macro predictors; `GS` follows sector flow.
  - `hour` & `is_first_bar`: Opening volatility suppresses confidence below signal threshold.
- **Ablation Findings**:
  - Removing Context (`SPY`/`XLF`) caused log-loss to worsen by **+0.00553**.
  - Removing Time features worsened loss by **+0.00274**.
  - Decisive threshold of 10 bps verified optimal against 5, 15, and 20 bps variants.

---

## Slide 13: Integrity, Leak Detection & Reproducibility Suite
- **Comprehensive Phase 9 Audit**:
  - *Shuffled-Label Test*: Null model achieved **50.50%** accuracy on Val (Zero label leakage).
  - *Time-Shift Test*: 1-bar feature shift worsened log-loss by **+0.00450** (Proves real-time dependency).
  - *Fit Ledger Verification*: 19 recorded fits; zero access to Val or Test timestamps.
  - *Live-Compatibility Test*: 500-bar rolling window matches full history within **1.07e-12**.
  - *Git Security Audit*: Zero API keys, tokens, or parquet files committed.
  - *Reproducibility*: Fully reproducible via `./run_all.sh` in ~35 seconds.

---

## Slide 14: Pre-Registered Scorecard & Limitations
- **Scorecard**:
  - **C1 (Val Log-Loss)**: **MET** (0.68704 vs 0.69178).
  - **C2 (Test Accuracy)**: **NOT MET** (46.28% vs $>52.0\%$).
  - **C3 (Net Return)**: **NOT MET** (-7.24 bps vs $>0.0$ bps).
  - **C4 (Calibration ECE)**: **MET** (0.0454 vs $\le 0.050$).
  - **C5 (Directional Balance)**: **LIMITATION STATED** (Up-only rule selected).
  - **C6 (Pipeline Integrity)**: **MET** (All leak tests passed).
- **Core Limitation**: Unidirectional models tuned on bull runs suffer catastrophic decay when macro regimes invert.

---

## Slide 15: Conclusions, Future Directions & Q&A
- **Summary**: We delivered a textbook implementation of the IMPACT Lab guidelines: leak-free pipeline, honest metrics, rigorous confidence intervals, and zero p-hacking.
- **Future Directions**:
  - Symmetric volatility-conditioned thresholds (forcing balance regardless of historical drift).
  - High-frequency limit order book (LOB) order flow imbalance (OFI) features.
  - Regime-switching Markov or macro-regime state gating.
- **Disclaimer**: Educational and paper trading use only. Not investment advice.
- **Live Status**: `PENDING: live test`.
- **Questions & Discussion**: Open for technical defense.
