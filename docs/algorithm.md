# Algorithm Specification: Calibrated Next-Hour Direction Prediction for GS

This document provides a formal, code-independent algorithmic specification of the machine learning pipeline developed for predicting next-hour price direction in Goldman Sachs (`GS`) equity. It details inputs, outputs, mathematical operations, training phases, tuning loops, decision rules, and evaluation protocols strictly adhering to the IMPACT Lab (IIT Patna) End-to-End ML Guidelines.

---

## 1. Mathematical Definitions and System Input/Output

### 1.1 Inputs
- **Market Data Feed**: Historical 1-minute OHLCV trade bars for target asset $S = \text{GS}$ and context benchmarks $B_1 = \text{SPY}$ and $B_2 = \text{XLF}$ spanning October 1, 2021 to October 15, 2025.
- **Trading Session Hours**: Regular Trading Hours (RTH) on US equity markets, defined from 09:30 to 16:00 US Eastern Time (ET).
- **Hyperparameter Space**: Predefined configuration grids for model structure, regularization, learning rate, and signal margins.
- **Cost Benchmark**: Bid-ask spread and transaction friction penalty fixed at $c = 5.0$ basis points (bps) per round-trip trade ($0.0005$).

### 1.2 Outputs
- **Calibrated Posterior Probability**: $\hat{p}_t = \mathbb{P}(Y_t = 1 \mid \mathcal{F}_t)$, representing the model's posterior confidence that price moves upward by more than 10 bps over the subsequent hour.
- **Directional Trading Signal**: $s_t \in \{+1, -1, 0\}$, denoting an **Up** signal, **Down** signal, or **No Signal** (abstention).
- **Held-Out Generalization Metrics**: Final log-loss, Brier score, Expected Calibration Error (ECE), signal accuracy, and net return under friction.

---

## 2. Algorithmic Steps

### Step 1: Session-Anchored Resampling
1. For each trading day, partition 1-minute bars into exactly seven session-anchored candles indexed $k \in \{0, 1, \dots, 6\}$:
   - Candle 0: 09:30:00 to 10:29:59 (60 minutes)
   - Candle 1: 10:30:00 to 11:29:59 (60 minutes)
   - Candle 2: 11:30:00 to 12:29:59 (60 minutes)
   - Candle 3: 12:30:00 to 13:29:59 (60 minutes)
   - Candle 4: 13:30:00 to 14:29:59 (60 minutes)
   - Candle 5: 14:30:00 to 15:29:59 (60 minutes)
   - Candle 6: 15:30:00 to 15:59:59 (30 minutes)
2. Aggregate bar prices such that Open is the open of the first minute, High is the maximum high across the interval, Low is the minimum low across the interval, Close is the close of the final minute, and Volume is the sum of 1-minute volumes.
3. Synchronize `GS`, `SPY`, and `XLF` candles by inner timestamp join, discarding any timestamp where any of the three instruments lacks an aggregated candle.

---

### Step 2: Causal Feature Construction
At each candle $t$, construct a feature vector $\mathbf{x}_t \in \mathbb{R}^{22}$ using strictly backward-looking information:
1. **Multi-Horizon Momentum**: Compute simple returns over past 1, 2, 3, 6, and 14 candles:
   $$R_{t, k} = \frac{C_t - C_{t-k}}{C_{t-k}} \quad \text{for } k \in \{1, 2, 3, 6, 14\}$$
2. **Opening Gap**: For candle 0 of each day, compute $G_t = (O_t - C_{t-1}) / C_{t-1}$; for intraday candles, set $G_t = 0.0$.
3. **Relative Strength Index (RSI)**: Compute 14-period Wilder RSI based on upward and downward close-to-close changes.
4. **Moving Average Convergence Divergence (MACD)**:
   $$\text{MACD}_t = \text{EMA}_{12}(C) - \text{EMA}_{26}(C), \quad \text{Signal}_t = \text{EMA}_9(\text{MACD}), \quad \text{Hist}_t = \text{MACD}_t - \text{Signal}_t$$
5. **Bollinger Bands**: Compute 20-period moving average $\mu_{20}$ and standard deviation $\sigma_{20}$. Compute bandwidth $(\mu_{20} + 2\sigma_{20} - (\mu_{20} - 2\sigma_{20})) / \mu_{20}$ and normalized position $(C_t - (\mu_{20} - 2\sigma_{20})) / (4\sigma_{20})$.
6. **Normalized Average True Range (NATR)**: Compute 14-period True Range divided by $C_t$.
7. **Realized Volatility**: Compute annualized sample standard deviation of 1-bar returns over a 20-candle rolling window.
8. **Relative Volume (RVOL)**: Compute current candle volume divided by the 20-candle moving average volume.
9. **Calendar and Time Context**: Integer hour of day, day of week, and binary indicator $\mathbb{I}_{\{\text{hour}=9\}}$ for the opening candle.
10. **Benchmark Context**: 1-bar and 3-bar returns of `SPY` and `XLF`, and relative return $(R_{t, 1}^{\text{GS}} - R_{t, 1}^{\text{SPY}})$.

---

### Step 3: Directional Labeling with Decisive Return Filtering
1. For candle $t$ occurring at intra-session index $k \in \{0, 1, 2, 3, 4, 5\}$, compute the forward 1-hour return:
   $$F_t = \frac{C_{t+1} - C_t}{C_t}$$
2. For candle $t$ occurring at the final session index $k = 6$ (15:30 to 16:00), assign $F_t = \text{NaN}$ and censor from directional training, as holding over the overnight session introduces jump risk outside the 1-hour intraday trading horizon.
3. Apply a decisive return threshold $\delta = 10 \text{ bps} = 0.0010$:
   - If $F_t > +\delta$: assign binary label $Y_t = 1$ (Up) and decisive flag $D_t = \text{True}$.
   - If $F_t < -\delta$: assign binary label $Y_t = 0$ (Down) and decisive flag $D_t = \text{True}$.
   - If $|F_t| \le \delta$: assign decisive flag $D_t = \text{False}$ (Indecisive noise).

---

### Step 4: Chronological Partitioning and Embargo Gaps
Partition the sequence of 7,118 candles into four chronological blocks separated by 8-candle embargo buffers (equivalent to more than one full trading day of 7 candles) to eliminate autocorrelation leakage:
1. **Base Block**: Candles 0 to 3,612 (October 2021 to December 2023). Contains 3,096 labelled candles.
   *(Embargo: Candles 3,613 to 3,620 discarded)*
2. **Fine-Tuning (FT) Block**: Candles 3,621 to 5,352 (December 2023 to November 2024). Contains 1,484 labelled candles.
   *(Embargo: Candles 5,353 to 5,360 discarded)*
3. **Validation (Val) Block**: Candles 5,361 to 6,062 (November 2024 to March 2025). Contains 602 labelled candles.
   *(Embargo: Candles 6,063 to 6,070 discarded)*
4. **Test Block**: Candles 6,071 to 7,117 (March 2025 to October 2025). Contains 896 labelled candles. Strictly frozen and sealed until final evaluation.

---

### Step 5: In-Fold Preprocessing Pipeline
To prevent distribution look-ahead:
1. For any training fold or combination of folds $\mathcal{D}_{\text{train}}$, fit a quantile Winsorizer computing the 1st percentile $q_{0.01, j}$ and 99th percentile $q_{0.99, j}$ for each feature $j$.
2. Clip feature values: $\tilde{x}_{t, j} = \min(\max(x_{t, j}, q_{0.01, j}), q_{0.99, j})$.
3. Fit a standard scaler on the winsorized training data computing empirical mean $\mu_j$ and standard deviation $\sigma_j$. Standardize: $z_{t, j} = (\tilde{x}_{t, j} - \mu_j) / \sigma_j$.
4. Apply the learned $(q_{0.01, j}, q_{0.99, j}, \mu_j, \sigma_j)$ parameters out-of-fold to validation or test data without re-estimating statistics.

---

### Step 6: Walk-Forward Cross-Validation and Model Optimization
1. Within the Fine-Tuning (FT) block, define an expanding-window walk-forward cross-validation scheme with 4 splits:
   - Fold 1: Train on initial 20% of FT; validate on next 20%.
   - Fold 2: Train on initial 40% of FT; validate on next 20%.
   - Fold 3: Train on initial 60% of FT; validate on next 20%.
   - Fold 4: Train on initial 80% of FT; validate on final 20%.
2. Train XGBoost gradient boosted trees on decisive candles ($D_t = \text{True}$) minimizing weighted binary cross-entropy (log-loss).
3. Apply sample weighting: observations from the historical Base block receive weight $w_{\text{base}} \in [0.1, 1.0]$, while observations from the FT block receive unit weight $1.0$, allowing the model to adapt to regime shifts while retaining sample depth.
4. Search 40 hyperparameter configurations varying tree depth, learning rate, subsampling, column subsampling, minimum child weight, and L2 regularization. Select the parameter tuple $\theta^*$ minimizing out-of-fold cross-validation log-loss.

---

### Step 7: Model Architecture Selection on Validation
Fit candidate architectures using data strictly prior to the Validation block and evaluate probability log-loss on all 602 Validation candles:
1. **Strategy S0**: Default XGBoost fitted on Base block alone.
2. **Strategy S1**: Tuned XGBoost fitted on Base block alone.
3. **Strategy S2**: Tuned XGBoost refitted on combined Base + FT blocks with $w_{\text{base}} = 0.3$.
4. **Strategy S3**: Warm-start booster starting from Base trees and adding 100 trees on FT.
5. **Alternative Architectures**: L2-regularized Logistic Regression, Random Forest, and Histogram-based Gradient Boosting.
6. **Selection Rule**: Select Strategy S2 (Refit Base+FT), which achieved the lowest log-loss (0.68953), outperforming all baseline comparators and the Constant Prior (0.69178).

---

### Step 8: Post-Hoc Probability Calibration
1. Generate raw uncalibrated probabilities $p_t^{\text{raw}}$ from the selected S2 model for all candles in the Validation block.
2. Fit Platt scaling (univariate logistic regression on logit features) strictly on Validation observations:
   $$\hat{p}_t = \sigma(\alpha \cdot \text{logit}(p_t^{\text{raw}}) + \beta)$$
3. Calibrator maps raw probabilities into calibrated posteriors, reducing Validation log-loss from 0.68953 to 0.68704, Brier score to 0.24697, and Expected Calibration Error (ECE) to 0.0105.
4. Freeze the calibration function $\mathcal{C}(\cdot)$.

---

### Step 9: Asymmetric Margin Signal Rule Search
1. Using the calibrated probabilities $\hat{p}_t$ on the Validation block, search directional decision thresholds:
   - Candidate Up-margin $d_{\text{up}} \in [0.01, 0.08]$ with step 0.005 ($p \ge 0.50 + d_{\text{up}}$).
   - Candidate Down-margin $d_{\text{down}} \in [0.01, 0.08] \cup \{\text{None}\}$ ($p \le 0.50 - d_{\text{down}}$).
2. Compute coverage, signal accuracy, and net return at $c = 5$ bps across all combinations satisfying pre-registered constraints:
   - Minimum coverage: $\ge 15\%$ of eligible candles.
   - Net return: $> 0$ bps after transaction costs.
3. **Rule Selection**:
   - The optimal rule on Validation is the asymmetric Up-only rule:
     $$d_{\text{up}} = 0.06 \implies \hat{p}_t \ge 0.56 \implies s_t = +1 \text{ (Up)}$$
     $$d_{\text{down}} = \text{None} \implies s_t = 0 \text{ (Down side disabled)}$$
     $$\text{Otherwise} \implies s_t = 0 \text{ (No Signal)}$$
   - On Validation, this rule achieved 61.34% accuracy across 119 signals (19.77% coverage) and +7.91 bps net return.
4. Lock the decision rule $s_t = \mathcal{S}(\hat{p}_t; d_{\text{up}}=0.06, d_{\text{down}}=\text{None})$.

---

### Step 10: Held-Out Test Evaluation (Strict Single Read)
1. Read the held-out Test block (896 candles) **strictly once** using frozen preprocessing parameters, frozen model S2, frozen calibrator $\mathcal{C}$, and frozen decision thresholds.
2. For each candle $t$ in the Test block:
   - Compute feature vector $\mathbf{x}_t$.
   - Standardize using training fold statistics.
   - Infer raw probability $p_t^{\text{raw}}$.
   - Calibrate $\hat{p}_t = \mathcal{C}(p_t^{\text{raw}})$.
   - Generate trading signal $s_t = \mathcal{S}(\hat{p}_t)$.
3. Record predictions to `results/tables/test_predictions.csv` and compute frozen performance metrics:
   - Signal Coverage: $\frac{1}{N} \sum_{t} \mathbb{I}_{\{s_t \ne 0\}}$
   - Signal Accuracy with Wilson 95% Score Confidence Interval:
     $$\text{Acc} = \frac{\sum_{t, s_t \ne 0} \mathbb{I}_{\{s_t = 2Y_t - 1\}}}{\sum_{t} \mathbb{I}_{\{s_t \ne 0\}}}$$
   - Benchmark Drift Accuracy: Directional accuracy of Always-Up strategy on the identical subset of signal candles.
   - Friction-Adjusted Net Return with Stationary Block-Bootstrap 95% Confidence Interval:
     $$\bar{R}_{\text{net}} = \frac{1}{N_{\text{signal}}} \sum_{t, s_t \ne 0} \left( s_t \cdot F_t - c \right)$$
   - Out-of-sample Log-Loss, Brier Score, and Expected Calibration Error (ECE).
4. Seal Test block data from any further modification, tuning, or secondary evaluation.
