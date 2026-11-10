# Calibrated Directional Prediction for Goldman Sachs (GS) Equity: An End-to-End Machine Learning Investigation

**Author**: Senior ML Engineering & Quantitative Research Team  
**Institution**: IMPACT Lab Guidelines, IIT Patna  
**Repository**: [https://github.com/chinmay-tammana/new-trade](https://github.com/chinmay-tammana/new-trade)  
**Release Tag**: `model-v1`  
**Date**: November 2026  
**Status**: Frozen Research Artifact (Educational & Paper Trading Scope Only)

---

## 1. Abstract and Core Contributions

### 1.1 Abstract
Predicting the directional trajectory of liquid equity equities over intraday horizons is notoriously difficult due to low signal-to-noise ratios, non-stationarity, and market efficiency. In this work, we develop, validate, audit, and freeze a complete end-to-end machine learning system predicting the next-hour price direction of Goldman Sachs (`GS`) equity. Utilizing 4+ years (October 2021 to October 2025) of SIP-consolidated trade bars, we construct 22 causal, stationary features, establish strict chronological embargoed splits, implement in-fold preprocessing, and optimize gradient-boosted decision trees via walk-forward cross-validation. Calibrated via Platt scaling on a validation partition, an asymmetric threshold rule ($p \ge 0.56$ for Up, Down disabled) was locked and evaluated **strictly once** on an unobserved 8-month held-out test block (896 candles). 

While the model achieved superior validation log-loss (0.68704 vs 0.69178 prior) and tight probability calibration (ECE = 0.0454), held-out test evaluation revealed a signal accuracy of **46.28%** (95% Wilson CI: [39.30%, 53.41%]) across 188 emitted signals (20.98% coverage), matching Always-Up drift on the identical candles (+0.00% excess edge) and resulting in a net return of **-7.24 bps** per signal under a 5.0 bps transaction cost regime. In accordance with the IMPACT Lab research integrity protocol, we present these negative findings without post-hoc p-hacking or parameter retuning, diagnose the underlying structural regime shift from upward drift to mean-reverting chop, document the methodological overlap with prior 2-year experiments, and outline necessary architectural prerequisites for production-grade intraday edge.

### 1.2 Core Contributions
1. **Leak-Free Intraday ML Pipeline**: End-to-end pipeline featuring session-anchored hour aggregation, in-fold Winsorization and standard scaling, 8-bar embargo buffers, and a complete fit ledger timestamp audit.
2. **Post-Hoc Probability Calibration & Abstention**: Formalization of asymmetric decision margins allowing systematic abstention on low-confidence regimes, achieving low calibration error (ECE $\le 0.05$).
3. **Rigorous Held-Out Evaluation Protocol**: Strict single-read test set discipline with pre-registered success criteria, Wilson score confidence intervals, and stationary block-bootstrap return intervals.
4. **Structural Regime Drift Analysis**: In-depth empirical demonstration of how asymmetric directional filters trained during sustained macro bull regimes fail catastrophically during horizontal consolidation regimes.

---

## 2. Introduction and Problem Statement

### 2.1 Problem Formulation
We investigate the 1-hour intraday directional return of Goldman Sachs equity (`GS`) traded during Regular Trading Hours (RTH, 09:30 to 16:00 ET). Let $C_t$ denote the closing price of session-anchored candle $t \in \{0, 1, \dots, 5\}$. The forward simple return is defined as:
$$F_t = \frac{C_{t+1} - C_t}{C_t}$$
For the final intra-day candle ($t = 6$, 15:30 to 16:00 ET), $F_t$ is censored ($\text{NaN}$) because holding overnight introduces unmodeled gap risk outside the 1-hour trading scope.

Given a decisive price threshold $\delta = 10 \text{ bps} = 0.0010$, observations are assigned binary directional labels:
$$Y_t = \begin{cases} 1 & \text{if } F_t > +\delta \quad (\text{Up}) \\ 0 & \text{if } F_t < -\delta \quad (\text{Down}) \\ \text{Indecisive} & \text{if } |F_t| \le \delta \end{cases}$$

The objective is to learn a decision mapping $s_t = \mathcal{S}(\hat{p}_t) \in \{+1, -1, 0\}$ where $\hat{p}_t = \mathbb{P}(Y_t = 1 \mid \mathcal{F}_t)$ is the calibrated posterior probability given the historical information filtration $\mathcal{F}_t$, while maintaining an abstention option ($s_t = 0$) when directional confidence is insufficient.

### 2.2 Pre-Registered Acceptance Criteria
Before accessing validation or testing data, six formal success criteria were registered:
- **C1 (Validation Log-Loss)**: Tuned model validation log-loss must improve upon Constant Prior log-loss ($\le 0.69178$).
- **C2 (Held-Out Test Accuracy)**: Signal accuracy on the held-out Test block must strictly exceed 52.0% with lower 95% Wilson CI bound $> 50.0\%$.
- **C3 (Economic Viability)**: Mean net return per trade at $c = 5.0$ bps friction must be strictly positive ($> 0.0$ bps) with lower bootstrap CI bound $> 0.0$ bps.
- **C4 (Calibration Fidelity)**: Expected Calibration Error (ECE) over 10 bins must satisfy $\text{ECE} \le 0.05$.
- **C5 (Directional Balance)**: Signal emissions should ideally not be exclusively unidirectional unless explicitly dictated by validation regime tuning.
- **C6 (Data & Pipeline Integrity)**: Shuffled-label test null accuracy $\in [45\%, 55\%]$, time-shift degradation verified, 50-cutoff truncation discrepancy $\le 10^{-8}$, and zero test rows accessed prior to final evaluation.

---

## 3. Related Work and Research Gap

Systematic equity prediction on intraday horizons has been widely studied across empirical finance and machine learning literature:
- **Market Efficiency & Noise**: Fama (1970) and Malkiel (2003) establish the weak-form Efficient Market Hypothesis, showing that serial correlation in raw liquid equity returns is indistinguishable from zero after transaction costs.
- **Ensemble Tree Models in Quant Finance**: Gu, Kelly, and Xiu (2020) demonstrated that gradient-boosted decision trees (XGBoost, LightGBM) and shallow neural networks dominate linear models in cross-sectional equity return prediction due to their capacity to capture non-linear interactions without severe overfitting.
- **Financial ML Methodologies & Overfitting**: De Prado (2018) highlighted the severe risks of cross-contamination in financial time series, formalizing purged k-fold cross-validation, embargo periods, and the dangers of backtest overfitting.
- **Probability Calibration & Abstention**: Niculescu-Mizil & Caruana (2005) and Guo et al. (2017) formalized modern calibration frameworks (Platt scaling and isotonic regression). In quantitative trading, abstention mechanisms allow algorithmic strategies to preserve capital during turbulent or ambiguous regimes (Chow, 1970).

**Research Gap**: Existing quantitative literature frequently evaluates uncalibrated classifiers on random k-fold cross-validation or ignores intraday bid-ask friction, leading to inflated hit-rate claims that fail in live execution. This work addresses that gap by strictly isolating historical blocks, enforcing causal in-fold pipelines, calibrating probabilities, and accounting for round-trip execution drag.

---

## 4. Dataset and Quality Audit

### 4.1 Ingestion & Session Architecture
We ingested 4+ years of 1-minute consolidated Securities Information Processor (SIP) trade bars from Alpaca Markets for Goldman Sachs (`GS`), the S&P 500 ETF (`SPY`), and the Financial Select Sector SPDR ETF (`XLF`) spanning October 1, 2021 to October 15, 2025.
1-minute bars were aggregated into exactly seven session-anchored regular-hours candles per day:
- Candles 0 to 5: 60-minute duration (09:30-10:30 through 14:30-15:30 ET).
- Candle 6: 30-minute duration (15:30-16:00 ET).

### 4.2 Quality Audit & Independent Parity
An exhaustive data audit confirmed:
- **Missingness**: Exactly 0 missing values, 0 zero-volume candles, and 0 duplicate timestamps across all three series (7,148 `GS` bars, 7,161 `SPY` bars, 7,161 `XLF` bars).
- **Synchronization**: Inner timestamp join between `GS`, `SPY`, and `XLF` preserved 7,118 synchronized bars with zero missing data rows.
- **Third-Party Parity**: Cross-verification against independent `yfinance` market feeds over a 60-day window yielded a price correlation of **0.999998**, a mean absolute close difference of **0.23 bps**, and a volume ratio of **1.0019**, confirming institutional data fidelity.

---

## 5. Chronological Split and Leakage Control Protocol

To ensure zero information leakage across time, the 7,118 synchronized candles were strictly partitioned in chronological order, with **8-candle embargo gaps** (exceeding one full trading day) separating adjacent blocks:

| Block | Start Timestamp | End Timestamp | Total Candles | Decisive Labelled | Purpose |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Base** | 2021-10-06 10:30 | 2023-12-14 14:30 | 3,613 | 3,096 | Initial feature learning & base booster fit |
| *Embargo* | *8 candles discarded* | | 8 | - | Autocorrelation insulation |
| **Fine-Tuning (FT)** | 2023-12-15 15:30 | 2024-11-08 14:30 | 1,732 | 1,484 | 4-fold walk-forward hyperparameter search |
| *Embargo* | *8 candles discarded* | | 8 | - | Autocorrelation insulation |
| **Validation (Val)** | 2024-11-11 11:30 | 2025-03-24 13:30 | 702 | 602 | Model selection, Platt calibration, signal rule search |
| *Embargo* | *8 candles discarded* | | 8 | - | Autocorrelation insulation |
| **Test** | 2025-03-25 14:30 | 2025-10-15 15:30 | 1,047 | 896 | Single-read held-out evaluation (Sealed) |

**Leakage Controls**:
- Quantile Winsorizing ($[1\%, 99\%]$) and standard scaling are fitted exclusively on training blocks (`Base` and `FT`).
- All 19 model fits recorded in `results/logs/fit_ledger.json` terminated prior to `2024-11-11 11:30:00-05:00` (Val block start).
- Causal truncation test with 50 random cutoffs produced an absolute feature discrepancy of **0.00e+00**.

---

## 6. Experimental Setup and Baselines

### 6.1 Baseline Models on Validation
On the 602 decisive validation candles, classical heuristic baselines established the benchmark floor:
- **Constant Prior**: Predicts historical training marginal probability ($\hat{p} = 0.5284$). Log-loss = **0.69178**, Brier = **0.24932**, Accuracy = **53.16%**.
- **Always-Up**: Always predicts upward continuation. Accuracy = **53.16%**, Net return @ 5 bps = **-5.64 bps**.
- **RSI Heuristic** ($<40$ Up, $>60$ Down): Coverage = 50.0%, Accuracy = **49.83%**, Net return @ 5 bps = **-0.85 bps**.
- **Previous-Bar Persistence**: Coverage = 100%, Accuracy = **45.02%**, Net return @ 5 bps = **-8.45 bps**.

### 6.2 Candidate Architectures & Fine-Tuning
Using 4-fold walk-forward cross-validation on the FT block (40 parameter evaluations), the optimal XGBoost configuration was identified:
`max_depth=3`, `learning_rate=0.05`, `n_estimators=400`, `subsample=1.0`, `colsample_bytree=0.7`, `min_child_weight=3`, `reg_lambda=3.0`, `gamma=1.5`, and `base_weight=0.3`.

Comparing training strategies on the Validation block:
- **S0 (Base Default)**: Log-loss = 0.70176
- **S1 (Base Tuned)**: Log-loss = 0.69297
- **S3 (Warm-Start Booster)**: Log-loss = 0.69244
- **S2 (Refit Base + FT)**: Log-loss = **0.68953** (Selected model, beat Constant Prior by 0.00225)
- **Comparators**: RandomForest (0.69092), LogisticRegression (0.69109), HistGradientBoosting (0.69276). All underperformed S2.

### 6.3 Calibration & Signal Selection
Platt scaling fitted on Validation reduced log-loss to **0.68704**, Brier score to **0.24697**, and ECE to **0.0105**.
Grid search across directional margins selected an asymmetric rule:
$$s_t = \begin{cases} +1 \text{ (Up)} & \text{if } \hat{p}_t \ge 0.56 \quad (d_{\text{up}} = 0.06) \\ 0 \text{ (No Signal)} & \text{otherwise} \quad (d_{\text{down}} = \text{None, Down side disabled}) \end{cases}$$
On Validation, this achieved **61.34%** accuracy across 119 signals (19.77% coverage) and **+7.91 bps** net return.

---

## 7. Held-Out Evaluation Results (Test Block)

Following complete freezing of code, parameters, calibrator, and thresholds, the held-out Test block (896 candles, March to October 2025) was read **strictly once**.

### 7.1 Quantitative Performance Summary
| Metric | Pre-Registered Target | Validation Result | Held-Out Test Result | Verdict |
| :--- | :--- | :--- | :--- | :--- |
| **Log-Loss** | $< 0.69178$ | 0.68704 | 0.70027 | Degraded on Test |
| **Brier Score** | - | 0.24697 | 0.25350 | - |
| **Expected Calibration Error (ECE)** | $\le 0.050$ | 0.0105 | **0.0454** | **MET (C4)** |
| **Signal Coverage** | $\ge 15.0\%$ | 19.77% (119/602) | **20.98%** (188/896) | Consistent coverage |
| **Signal Direction Breakdown** | Balanced | 119 Up / 0 Down | **188 Up / 0 Down** | Up-only rule locked |
| **Signal Accuracy** | $> 52.0\%$ (CI $> 50\%$) | 61.34% [52.4, 69.6]% | **46.28% [39.30, 53.41]%** | **NOT MET (C2)** |
| **Always-Up Drift Accuracy** | - | 53.16% | **46.28%** | Identical on signals |
| **Excess Edge over Drift** | $> 0.0\%$ | +8.18% | **+0.00%** | Zero excess edge |
| **Net Return @ 5 bps Friction** | $> 0.0$ bps | +7.91 bps | **-7.24 bps [-13.9, -0.0]** | **NOT MET (C3)** |
| **Hit Rate @ 5 bps** | - | 55.46% | **40.43%** | Unfavorable |
| **Max Drawdown (bps)** | - | 345.2 bps | **1500.6 bps** | Severe drawdown |
| **5-Seed Variance (Acc)** | - | - | $46.50\% \pm 0.85\%$ | Highly stable |

### 7.2 Scorecard Against Pre-Registered Criteria
- **C1 (Validation Log-Loss)**: **MET**. Model S2 achieved 0.68953 (calibrated 0.68704), outperforming Constant Prior (0.69178).
- **C2 (Held-Out Test Accuracy)**: **NOT MET**. Test accuracy was 46.28% (95% CI: [39.30%, 53.41%]), falling well short of 52.0%.
- **C3 (Economic Viability)**: **NOT MET**. Mean net return was -7.24 bps with 95% bootstrap interval [-13.90, -0.02] bps.
- **C4 (Calibration Fidelity)**: **MET**. Test ECE of 0.0454 satisfies the $\le 0.05$ threshold.
- **C5 (Directional Balance)**: **LIMITATION EXPLICITLY STATED**. The validation search locked an Up-only rule; zero Down signals were emitted.
- **C6 (Pipeline Integrity)**: **MET**. Shuffled-label test yielded 50.50% null accuracy, fit ledger passed all timestamp assertions, and truncation tests verified zero look-ahead.

---

## 8. Error Analysis, Explainability, and Ablations

### 8.1 Structural Diagnosis: The Upward Drift Trap
Why did a model with 61.34% validation accuracy collapse to 46.28% on the held-out test set?
1. **Macro Regime Transition**: The Base and FT training periods (2021–2024) experienced significant upward trend, with positive decisive candle rates of 51.4% and 52.9% respectively. On the Validation block, upward drift reached 53.16%. As a direct mathematical consequence, the threshold grid search on Validation determined that disabling the Down side ($d_{\text{down}} = \text{None}$) and lowering the Up margin to $p \ge 0.56$ yielded maximum accuracy and return.
2. **Mean-Reverting Chop in 2025**: During the held-out Test block (March to October 2025), `GS` entered a consolidation and pullback phase where the ground-truth Up rate dropped to **48.88%**, and on the specific 188 candles where the model signaled Up, the actual market rose only **46.28%** of the time.
3. **Absence of Reversal Mechanism**: Because the Down side was disabled, the strategy had no mechanism to capitalize on sell-offs or hedge against sustained downtrends.

### 8.2 Feature Importance & SHAP Attribution
SHAP (TreeSHAP) analysis on Validation and gain-based importance revealed:
- **Dominant Predictive Features**: `rsi` (Wilder 14), `ret_1` (1-bar return), `spy_ret_1` (broad market momentum), and `xlf_ret_1` (financial sector context).
- **Opening Context**: `is_first_bar` and `hour` played pivotal roles in scaling probabilities during opening price discovery.
- **Ablation Evidence**: Removing context features (`SPY`/`XLF`) increased validation log-loss by **+0.00553**, confirming that idiosyncratic `GS` price action is deeply dependent on macro financial sector movements.

### 8.3 Prior 2-Year Implementation Disclosure & Test Overlap
An earlier 2-year experimental implementation yielded the following motivating baseline:
> *"Earlier 2-year run: test set 587 eligible hourly candles; 114 Up signals, 0 Down signals, 473 No signal; signal accuracy 53.5% (about 61 of 114 correct); thresholds were $p \ge 0.56$ (Up) and $p \le 0.44$ (Down). Likely causes: (a) Down threshold was never reached, so the indicator was effectively Up-only and mostly rode upward drift; (b) 114 signals gives a 95% interval of roughly 44%–63%, i.e. no demonstrable edge; (c) only about 2 years of data."*

**Methodological Disclosure**: The 4-year dataset constructed in this work expands historical depth to October 2021. However, the held-out Test block (March to October 2025) overlaps the newest segment of that earlier test window. Because model architecture and validation rules were refined with prior knowledge of this era, the true out-of-sample judge is exclusively future live forward paper testing.

**Validation Optimism Disclosure**: The Validation partition was used sequentially for: (1) selecting fine-tuning strategy S2 over S0/S1/S3; (2) fitting the Platt scaling sigmoid; and (3) searching the asymmetric signal margins ($d_{\text{up}}, d_{\text{down}}$). While mathematically separate from model parameter fitting, this joint usage introduces a modest degree of optimistic selection bias on validation metrics.

---

## 9. Operational Guardrails, Ethics, and Limitations

### 9.1 Trader Operating Rules (When NOT to Trust the Indicator)
Traders utilizing this model in paper trading must observe strict operational circuit breakers:
1. **Opening Auction**: The 09:30–10:30 candle contains extreme opening price discovery and spread widening.
2. **Scheduled Macro Events**: Never trade through FOMC rate announcements, CPI releases, or GS quarterly earnings announcements.
3. **Session Close**: The final candle (15:30–16:00) must never be traded overnight; all positions must be closed prior to 16:00 ET.
4. **Regime Misalignment**: In confirmed bear markets or when `GS` is trading below its 50-day moving average, the Up-only signal must be overridden.

### 9.2 Ethical & Regulatory Declarations
- **Educational Scope**: This project is developed strictly for academic and educational evaluation under IMPACT Lab guidelines. It does NOT constitute financial advice, an investment recommendation, or a live execution system.
- **Paper Trading Only**: The model is restricted to simulated paper trading. Capital allocation based on these signals is strictly prohibited.
- **Data Compliance**: Market data was accessed via authorized Alpaca Markets API credentials. Raw and aggregated market data parquet files are excluded from public git distribution in strict compliance with data licensing agreements.
- **Fairness & Privacy**: The system operates exclusively on anonymous public financial price and volume ticks; no personal, demographic, or private user data is processed.

---

## 10. Reproduction Protocol, Environment, and Live Status

### 10.1 Live Paper Test Status
- **Current Live Status**: `PENDING: live test`. No live forward execution results have been recorded or simulated beyond historical candle replay.

### 10.2 Software and Hardware Environment
- **Operating System**: macOS 15.3 (Darwin 24.3.0, Apple Silicon ARM64)
- **Python Version**: Python 3.11.18 (`.venv/bin/python3`)
- **Key Libraries**: `xgboost==3.1.1`, `scikit-learn==1.6.1`, `pandas==2.2.3`, `numpy==1.26.4`, `pyarrow==17.0.0`, `joblib==1.5.3`, `scipy==1.15.2`
- **Execution Script**: Complete pipeline reproduces via `./run_all.sh` in approximately 35 seconds.
- **Random Seed**: Fixed globally to `42` (`CONFIG["seed"] = 42`).

---

## 11. References
1. Chow, C. K. (1970). On optimum recognition error and reject tradeoff. *IEEE Transactions on Information Theory*, 16(1), 41-46.
2. De Prado, M. L. (2018). *Advances in Financial Machine Learning*. John Wiley & Sons.
3. Fama, E. F. (1970). Efficient capital markets: A review of theory and empirical work. *The Journal of Finance*, 25(2), 383-417.
4. Gu, S., Kelly, B., & Xiu, D. (2020). Empirical asset pricing via machine learning. *The Review of Financial Studies*, 33(5), 2223-2273.
5. Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On calibration of modern neural networks. *International Conference on Machine Learning (ICML)*, 1321-1330.
6. Malkiel, B. G. (2003). The efficient market hypothesis and its critics. *Journal of Economic Perspectives*, 17(1), 59-82.
7. Niculescu-Mizil, A., & Caruana, R. (2005). Predicting good probabilities with supervised learning. *Proceedings of the 22nd International Conference on Machine Learning*, 625-632.
8. Wilson, E. B. (1927). Probable inference, the law of population, and the frequency single-event distribution. *Journal of the American Statistical Association*, 22(158), 209-212.
