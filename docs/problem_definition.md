# Problem Definition: Next-Hour Direction Prediction for Goldman Sachs Stock

## 1. Project Title
**Next-Hour Direction Prediction for Goldman Sachs Stock Using Calibrated XGBoost on Hourly OHLCV Data**

---

## 2. Application Context and Target Users
- **Application**: An AI-driven directional decision-support indicator designed to assist intraday discretionary and systematic equity traders. The indicator monitors Goldman Sachs Group Inc. (NYSE: `GS`) on an hourly candle timeframe and emits calibrated directional signals (Up, Down, or Abstain / No Signal) at the close of each candle for the subsequent hour.
- **Target Users**: Intraday quantitative researchers and institutional/retail equity traders seeking systematic, probability-calibrated directional filters that reduce trade over-frequency, control transaction friction, and avoid subjective heuristics.
- **Trading Environment**: Strictly simulated paper trading and academic evaluation. This system does not execute automated orders with real capital.

---

## 3. Formal Problem Statement
> **Problem Statement Template Formulation**:  
> "Given **the preceding 200–500 closed 1-hour regular-session OHLCV bars of Goldman Sachs (GS) and aligned market/sector context bars of SPY (S&P 500) and XLF (Financial Select Sector SPDR)**,  
> predict **the next-hour price direction \(y_{t+1} \in \{0, 1\}\) defined as whether the forward simple return \(fwd = \frac{Close_{t+1}}{Close_t} - 1\) is positive (\(y=1\)) or non-positive (\(y=0\))**  
> for **regular US equity trading session hourly candles of GS (09:30 to 16:00 ET)**  
> under **strict causal constraints (zero look-ahead, zero repainting, no overnight prediction holding, features computable live from rolling closed bars, paper trading only)**,  
> achieving **validation log-loss \(\ge 0.002\) below constant prior, held-out test signal accuracy \(\ge 53\%\) across \(\ge 100\) active signals, outperforming prevailing drift (always-Up baseline), Expected Calibration Error (ECE) \(\le 0.05\), and positive net simulated return after 5 bps round-trip transaction costs**."

---

## 4. Task Classification and Mathematical Formulation
- **Task Class**: Binary classification with selective abstention (selective classification / reject option).
- **Label Definition**:
  \[
  fwd_t = \frac{Close_{t+1}}{Close_t} - 1
  \]
  - For intra-day candles: \(y_t = \mathbb{I}(fwd_t > 0)\).
  - For the final candle of the trading day (15:30–16:00 ET) and any candle immediately prior to a non-contiguous session (e.g. holiday or weekend), \(fwd_t = \text{NaN}\) and no label is generated. Overnight holding risk is completely eliminated.
  - Training decison filter: A candle is deemed decisive if \(|fwd_t| > \delta\) where \(\delta = 0.0010\) (10 bps). Training occurs only on decisive rows to focus gradient updates on actionable moves; evaluation is performed on **all** eligible labelled candles.
- **Inputs**: Stationary engineered features computed exclusively from historical closed bars \(1, \dots, t\):
  - Return horizons: `ret_1`, `ret_2`, `ret_3`, `ret_6`, `ret_14`.
  - Overnight move: `gap`.
  - Momentum: `rsi` (Wilder 14), `macd_n`, `macd_hist_n`.
  - Volatility & Dispersion: `bb_pos`, `bb_width`, `atr_n`, `vol_20`.
  - Volume dynamics: `rvol` (relative volume against 20-bar rolling mean).
  - Temporal context: `hour`, `dow`, `is_first_bar`.
  - Market context: `spy_ret_1`, `spy_ret_3`, `xlf_ret_1`, `xlf_ret_3`, `rel_ret_1`.
- **Output**:
  - Raw model probability: \(\hat{p}_t = P(y_t = 1 \mid x_t)\).
  - Calibrated probability: \(p_{\text{cal}, t} = \sigma(a \cdot \text{logit}(\hat{p}_t) + b)\).
  - Discrete signal rule:
    \[
    \text{Signal}_t = \begin{cases}
    \text{Up (confidence } p_{\text{cal}, t}\text{)}, & \text{if } p_{\text{cal}, t} \ge 0.5 + d_{\text{up}} \\
    \text{Down (confidence } 1 - p_{\text{cal}, t}\text{)}, & \text{if } p_{\text{cal}, t} \le 0.5 - d_{\text{down}} \\
    \text{No Signal}, & \text{otherwise}
    \end{cases}
    \]

---

## 5. Constraints and Assumptions
1. **No Look-Ahead / Zero Leakage**: All feature transformations, scalers, and normalizations must be fitted strictly inside training folds. The test block is evaluated strictly once.
2. **No Repainting**: Signals emitted at bar close \(t\) depend only on bars closed at or before \(t\). Once a candle closes, the historical signal is permanently immutable.
3. **Overnight Risk Avoidance**: Positions are liquidated at market close (16:00 ET). No directional predictions bridge across session gaps.
4. **Market Microstructure & Friction**: Execution is assumed at candle close with realistic transaction costs (modeled at 2, 5, and 10 bps; primary benchmark at 5 bps).
5. **Stationarity Assumption**: Financial returns exhibit low signal-to-noise ratio and time-varying distributions. Feature sets must be rigorously stationary (bounded oscillators, normalized differences, ratios).

---

## 6. Hypotheses
- **Null Hypothesis (\(H_0\))**: Calibrated XGBoost confidence signals provide no statistically significant predictive edge over market drift or random guessing on held-out data; specifically, test signal accuracy \(\le 50\%\) or \(\le\) always-Up benchmark on identical candles, or net return after 5 bps costs \(\le 0\).
- **Alternative Hypothesis (\(H_1\))**: Calibrated XGBoost probability signals, conditioned on asymmetric confidence margin filters (\(d_{\text{up}}, d_{\text{down}}\)), achieve a statistically detectable directional edge on held-out hourly data (\(>\) 50% accuracy, \(>\) always-Up drift accuracy, and positive net expectancy after transaction costs).

---

## 7. Pre-Registered Success Criteria (Fixed Protocol)
The following pre-registered criteria are established prior to training any model and cannot be altered post-hoc:
1. **Validation Log-Loss**: The proposed model achieves a validation log-loss on all labelled validation candles at least **0.002 lower** than the constant-prior baseline.
2. **Held-Out Test Accuracy & Coverage**: On the unseen Test block, active signal accuracy achieves \(\ge \mathbf{53.0\%}\) across \(\ge \mathbf{100}\) total signals, AND exceeds the always-Up baseline accuracy evaluated on those identical signal candles.
3. **Simulated Net Expectancy**: Net return per active signal strictly \(> \mathbf{0.0}\) basis points after deducting **5 bps** primary transaction costs.
4. **Calibration Quality**: Expected Calibration Error (ECE) with 10 uniform bins on the Test block \(\le \mathbf{0.05}\).
5. **Directional Integrity & Balance**: Both Up and Down signal channels remain active with \(\ge \mathbf{25}\) signals each on the Test block; if market regime suppresses one side, the constraint limitation must be explicitly documented and disclosed.
6. **Reproducibility & Leak Checks**: 100% passage across all integrity checks (shuffled-label test \(\approx 50\%\), time-shift feature drop, truncation reproducibility, zero-secret leak).
