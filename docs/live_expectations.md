# Live Test Statistical Expectations and Power Analysis

This document formalizes the statistical reality and sample-size constraints of the upcoming 5-trading-day live evaluation for the Goldman Sachs (`GS`) AI Trading Indicator (`model-v2`).

---

## 1. Verified Baseline Holdout Metrics

On the sealed 1,872-bar Research Holdout (October 16, 2025 to November 9, 2026), the frozen V2 model produced the following verified performance:

- **Total Chronological Bars**: 1,872 bars (268 trading days).
- **Labelled Hourly Bars**: 1,604 bars (the 15:30–16:00 closing bar of each session is censored from directional evaluation, as it has no subsequent intraday bar).
- **Signals on All 1,872 Bars**: 888 Up (47.44%), 786 Down (42.00%), 198 No-Signal (10.58%).
- **Signals Evaluated on 1,604 Labelled Bars**:
  - **Up Signals**: 848 emitted $\rightarrow$ 438 hits $\rightarrow$ **51.65%** accuracy (95% Wilson CI: `[48.29%, 55.00%]`).
  - **Down Signals**: 574 emitted $\rightarrow$ 305 hits $\rightarrow$ **53.14%** accuracy (95% Wilson CI: `[49.05%, 57.18%]`).
  - **Total Evaluated Signals**: 1,422 emitted $\rightarrow$ 743 hits $\rightarrow$ **52.25%** pooled accuracy (95% Wilson CI: `[49.65%, 54.84%]`).
  - **Permutation Test (10,000 shuffles)**: $p = \mathbf{0.0417}$ ($\alpha = 0.05$).
  - **Holdout Base Rates**: Up Base Rate = 50.06%, Down Base Rate = 49.94%.
  - **Drift-Adjusted Edge**: Up: **+1.59 points**, Down: **+3.20 points**.

---

## 2. Statistical Constraints of a 5-Day Live Test

In a standard 5-trading-day live evaluation:
- Each regular trading day contains 7 hourly candles (09:30 to 16:00 ET).
- The final bar of each day (15:30 to 16:00 ET) is unlabelled because trading does not hold overnight.
- Therefore, exactly **6 labelled hourly candles** exist per full day.
- Over 5 trading days, exactly **30 candidate evaluation candles** will be generated.
- Given the model's observed coverage (~88.7% on labelled candles), approximately **26 to 28 signals** (Up or Down) are expected to fire.

### Wilson 95% Confidence Interval for $N = 30$ Signals
If the model performs exactly at its empirical holdout accuracy (~52.25%, yielding approximately 15 or 16 hits out of 30):
- **16 Hits out of 30 (53.33%)**:
  $$\text{Wilson 95\% CI} = [36.14\%, 69.77\%] \quad (\text{Width: } 33.63\%)$$
- **15 Hits out of 30 (50.00%)**:
  $$\text{Wilson 95\% CI} = [33.15\%, 66.85\%] \quad (\text{Width: } 33.69\%)$$

---

## 3. Methodological Implication: A Limit of Sample Size, Not a Model Defect

A confidence interval spanning 33.6 percentage points demonstrates that:
1. **A 5-day live test CANNOT distinguish a 52% true hit rate from a 50% coin flip.**
2. A random series of 30 coin tosses will frequently produce 12 hits (40.0%) or 18 hits (60.0%) purely by sampling variance ($p > 0.25$).
3. **Operational Interpretation**:
   - The live test serves to verify **system integrity, feed parity, contiguity, and error-free real-time execution**.
   - It cannot statistically prove or disprove an intraday alpha of 150–200 basis points of directional hit rate.
   - Any attempt to "tune" or "adjust" model thresholds based on the outcome of 5 live days would represent severe backtest overfitting and must be strictly avoided.

---

## 4. STRONG Confidence Tier

To provide traders with a secondary high-conviction filter, the **STRONG tier** was defined on pre-holdout out-of-fold data:
- **Threshold**: Displayed confidence $\ge 0.520414$ (the 75th percentile of displayed confidence among pooled pre-holdout OOF signals).
- **Holdout Performance**:
  - **All Signals**: 1,422 signals $\rightarrow$ **52.25%** accuracy (Wilson CI: `[49.65%, 54.84%]`).
  - **STRONG Signals**: 1,187 signals $\rightarrow$ **51.56%** accuracy (Wilson CI: `[48.72%, 54.39%]`).
- **Conclusion**: The STRONG tier does not meaningfully increase accuracy on the holdout, confirming that individual hourly signals carry high idiosyncratic noise and must be treated as weak probabilistic evidence on their own.
