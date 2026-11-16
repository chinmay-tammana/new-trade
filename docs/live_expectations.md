# Live Test Expectations and Statistical Power Analysis

This document formalizes the statistical reality, evaluation policy, and sample-size constraints of the live evaluation for the Goldman Sachs (`GS`) AI Trading Indicator (`model-v2`).

---

## 1. Evaluation Policy (TA Guideline)

- **Test Duration**: A **ONE-DAY live test** is fully sufficient.
- **Bar Horizon & Delivery Mechanism**: The indicator operates strictly on **1-hour candles** (09:30 to 16:00 ET). There is **NO 10-minute mode** anywhere in the system.
- **Delayed SIP Delivery Schedule**: The live system operates under **Option B (Delayed Consolidated SIP for Signals, IEX for Display Only)**. In accordance with Alpaca's delayed SIP data policy (>= 15-minute delay), minute bars for a closed 1-hour candle are fetched and processed at **`bar_end + 16 min`**.
  - A full session produces **about 6 signals** delivered at:
    1. **10:46 ET (21:16 IST)** (Bar 09:30–10:30 closed at 10:30 ET; covers remaining 44 min of the 10:30–11:30 hour)
    2. **11:46 ET (22:16 IST)** (Bar 10:30–11:30 closed at 11:30 ET)
    3. **12:46 ET (23:16 IST)** (Bar 11:30–12:30 closed at 12:30 ET)
    4. **13:46 ET (00:16 IST)** (Bar 12:30–13:30 closed at 13:30 ET)
    5. **14:46 ET (01:16 IST)** (Bar 13:30–14:30 closed at 14:30 ET)
    6. **15:46 ET (02:16 IST)** (Bar 14:30–15:30 closed at 15:30 ET; covers remaining 14 min of trading)
  - The final end-of-day bar (15:30–16:00 ET) closes the session and is **unlabelled with No signal (end of day)**.
  - **Notice for the TA**: Please inform the Teaching Assistant that live signals are computed from consolidated SIP with a **15-minute regulatory delay**, so each signal is delivered 16 minutes after the candle close.
- **Purpose**: The 1-day live test is a demonstration of **end-to-end engineering integrity**:
  - Live bar ingestion via Alpaca Consolidated Tape (SIP) with >= 15-minute lag compliance
  - Real-time forming candle and price ticker display via IEX
  - Streaming multi-asset feature pipeline (8 tickers, intrabar stats, order-flow proxies)
  - Frozen model inference with locked SHA256 integrity
  - Cryptographically chained, append-only SQLite audit ledger with `schema_v=2`
  - Responsive web dashboard with live ET/IST dual clocks and countdown timer
- **Statistical Proof vs Engineering Demonstration**: A 1-day test is **not** a statistically powered evaluation. A full day gives ~6 signals, which is a demonstration of mechanics, not statistical proof. Statistical significance was pre-registered and evaluated on the sealed 1,422 holdout candles.

---

## 1.1 Critical Statistical Notice Regarding Deployed Thresholds

> [!IMPORTANT]
> **Measurement Disconnect Disclosure**:
> The 52.25% accuracy [49.65, 54.84] and permutation $p = 0.0417$ were measured on the **pre-refit model** with narrow thresholds $0.490 / 0.508$ (88.65% coverage on holdout). The **deployed production rule** uses refitted thresholds **$0.434 / 0.563$** (calibrated to restore ~30% target coverage) and has **no held-out accuracy measurement**. The earlier 52.25% figure must never be presented as the accuracy of the deployed live signals.

---

## 2. Statistical Reality of an $N = 6$ Live Sample

For a single trading day ($N = 6$ candidate evaluation candles):
- Even if coverage is 100%, $N = 6$.
- If the model hits 4 out of 6 (66.7%):
  $$\text{Wilson 95\% CI} = [30.00\%, 90.32\%] \quad (\text{Width: } 60.32\%)$$
- If the model hits 3 out of 6 (50.0%):
  $$\text{Wilson 95\% CI} = [18.76\%, 81.24\%] \quad (\text{Width: } 62.48\%)$$
- If the model hits 2 out of 6 (33.3%):
  $$\text{Wilson 95\% CI} = [9.68\%, 70.00\%] \quad (\text{Width: } 60.32\%)$$

A confidence interval spanning over **60 percentage points** cannot statistically distinguish an elite trading system from an inverted coin toss. Sampling variance dominates completely at $N = 6$.

---

## 3. Verified Holdout Benchmark ($N = 1,422$)

The true empirical performance of frozen `model-v2` was evaluated on the sealed Research Holdout (October 16, 2025 to November 9, 2026, 268 trading days):

- **Total Chronological Bars**: 1,872 bars.
- **Labelled Hourly Bars**: 1,604 bars.
- **Evaluated Signals**: 1,422 signals.
  - **Up Signals**: 848 emitted $\rightarrow$ 438 hits $\rightarrow$ **51.65%** (Wilson 95% CI: `[48.29%, 55.00%]`).
  - **Down Signals**: 574 emitted $\rightarrow$ 305 hits $\rightarrow$ **53.14%** (Wilson 95% CI: `[49.05%, 57.18%]`).
  - **Pooled Signal Accuracy**: 743 hits out of 1,422 $\rightarrow$ **52.25%** (Wilson 95% CI: `[49.65%, 54.84%]`).
  - **Permutation Test (10,000 shuffles)**: $p = \mathbf{0.0417}$ ($\alpha = 0.05$).
  - **Holdout Base Rates**: Up = 50.06%, Down = 49.94%.
  - **Drift-Adjusted Edge**: Up: **+1.59 points**, Down: **+3.20 points**.
- **Coverage**: 88.65% of labelled bars received a directional signal under the narrow OOF corridor, whereas production refit restored ~30% coverage on subsequent data.
- **Criterion (b) Verdict**: NOT MET. Measured holdout signal accuracy is **52.25%** (Wilson 95% CI: `[49.65%, 54.84%]`, permutation $p = 0.0417$), which is **below the pre-registered accuracy bar of 53.0%**.

---

## 4. Operational Interpretation for Demonstrators & Evaluators

1. **Do not draw conclusions from single-day hit rates**: A score of 2/6, 3/6, or 5/6 on Monday is purely stochastic noise around the expected 52% mean.
2. **Focus on verification checks**:
   - Did the system run without exceptions or missed bars?
   - Did the bar closes occur exactly at :30 past the hour?
   - Did the hash chain maintain integrity from bar 0 to bar 6?
   - Did the UI reflect live state truthfully with clear ET and IST timestamps?

---

## 5. Audit of STRONG Tier & Model Abstentions (V3.3 Audit)

### STRONG Tier Audit
- **Out-of-Sample Directional Signals**: 8 total directional signals were emitted across the 4 out-of-sample trading days (Nov 10–13, 2026).
- **STRONG Directional Signals**: **8 of 8 (100.0%)** have confidence $\ge 0.520414$ (range 57.1% to 69.5%).
- **Non-Discrimination Finding**: Because 100.0% of out-of-sample directional signals qualify as STRONG, the frozen threshold ($0.520414$) is **not discriminating** on this recent regime sample. As mandated by project integrity rules, the frozen threshold is **strictly preserved without post-hoc tuning**.
- **No-Signal Rule**: A No-signal bar must never receive a STRONG tag under any circumstances.

### Abstention Breakdown
- In the out-of-sample period (Nov 10–13, 2026), 16 of the 24 candidate regular trading-hour candles (66.7%) abstained (`No Signal`).
- Breakdown (`results/tables/abstention_breakdown.csv`):
  - **Inside No-Signal Band**: 16 candidate candles (calibrated probability fell within neutral interval $[q_{\text{lo}}=0.434, q_{\text{hi}}=0.563]$).
  - **End of Day**: 4 session-closing candles (15:30 ET bars with no forward target hour).
  - **Data Gaps / Other**: 0 candles.
- This 66.7% abstention rate aligns with the ~30% coverage ($[q_{\text{lo}}, q_{\text{hi}}]$) target established in V2 model calibration.

---

## 6. One-Sided Signals & Feed Bias Diagnostic (V3.4 Diagnostic Audit)

- **Nov 10–13 Signal Breakdown**:
  - **IEX Feed (Real-Time Live)**: 7 Down, 0 Up, 17 No Signal (8 directional signals in database, all Down).
  - **SIP Feed (Consolidated Tape)**: 4 Down, 2 Up, 18 No Signal (Two-sided distribution).
  - **Exact Side Agreement**: **79.2%**.
- **Root Cause (Pre-Registered Rule 2: FEED BIAS)**:
  - LiveEngine code parity confirmed within $5.23 \times 10^{-12}$ (< 1e-6); Platt calibrators are strictly monotonic.
  - IEX feed experiences 31.08% missing minutes and captures only ~2% of total consolidated volume.
  - This missing volume skews `up_minute_ratio` by -3.46 Base standard deviations and `signed_volume_ratio` by +1.32 standard deviations.
  - Because `signed_volume_ratio` is the #1 Down contributor in XGBoost (SHAP contribution = -0.0286), the thin volume on IEX artificially depresses $p_{\text{cal}}$, shifting potential Up signals into No-Signal and neutral bars into Down signals.
- **Coverage Contradiction Resolved**:
  - Pre-holdout OOF had low variance ($\sigma \approx 0.015$), yielding narrow thresholds $[0.490, 0.508]$ for 30% coverage.
  - When applied to the holdout test set with higher variance ($\sigma \approx 0.040$), 88.65% of holdout candles fell outside the narrow corridor.
  - In production refit, thresholds were recalibrated on all data to $[q_{\text{lo}}=0.43428, q_{\text{hi}}=0.56323]$, restoring the intended ~30% target (8 of 24 = 33.3% on recent days).
- **Protocol Decision**: The model remains strictly FROZEN. No artificial balancing quotas, threshold shifts, or recalibration are permitted.
