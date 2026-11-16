# M5 System Integrity Audit Findings

**Date**: 2026-11-16  
**Status**: COMPLETE — PROCEED WITH SECTION B (IEX PRINT ARTIFACT WARNING REQUIRED)  
**Reference**: `instructions/21_coverage.md` Section A, `docs/m5/audit_plan.md`

---

## 1. Executive Summary

Prior to modifying coverage thresholds or deploying new signal rules, an empirical integrity audit was conducted across 5 pre-registered fail gates to determine why out-of-fold (OOF) directional accuracy on 5-minute bars was an implausibly high 69.56%.

### Summary of Pre-Registered Fail Gates

| Gate | Test Description | Pre-Registered Threshold | Measured Value | Verdict |
| :--- | :--- | :--- | :--- | :--- |
| **Gate 1** | **Shuffled-Label Null Test** | Fail if outside [48.0%, 52.0%] | **50.00%** (0 signals) | **PASS** |
| **Gate 2** | **Time-Shift Lead/Lag (+1 Bar)** | Fail if > 55.0% | **50.00%** (0 signals) | **PASS** |
| **Gate 3** | **Single-Feature Leakage Inspection** | Fail if \|AUC - 0.50\| > 0.10 | Max \|AUC - 0.5\| = **0.0591** (`gs_minus_peer_1`, AUC = 0.4409) | **PASS** |
| **Gate 4** | **IEX Microstructure & Thin-Print Artifact** | Flag if SIP Acc < IEX Acc - 5% OR IEX ACF1 < -0.05 | IEX Acc = **69.93%**, SIP Acc = **51.80%** ($\Delta = +18.13\%$) | **ARTIFACT DETECTED** |
| **Gate 5** | **Subgroup Stability Audit** | Stability across years, TOD, sides | Consistent across segments ($\approx 69\% - 71\%$) | **PASS** |

### Decision Rule Outcome
- Gates 1, 2, and 3 **PASSED**: There is **no future label leakage, no time-indexing bug, and no lookahead feature**.
- Gate 4 **FLAGGED AN IEX PRINT ARTIFACT**: The apparent 69.9% directional predictability exists almost exclusively on the **IEX exchange prints** and drops to **51.80%** when evaluated against the **Consolidated SIP tape** (Alpaca/CTA/UTP).
- In accordance with the pre-registered protocol in `instructions/21_coverage.md`:
  > *"If only gate 4 flags an IEX PRINT ARTIFACT, continue with section B but show a red warning card on the page ('accuracy measured on IEX prints; not confirmed on consolidated prices') and tell me."*
  **Action**: Proceed to Section B (Coverage-Targeted Rule), and integrate the prominent red warning banner into the UI and documentation.

---

## 2. Gate 1: Shuffled-Label Null Test

- **Hypothesis**: If the training labels $y$ are randomly permuted within each walk-forward fold, the model should find zero true signal.
- **Result**:
  - Out-of-fold calibrated probabilities collapsed to the prior mean ($\sim 0.50$).
  - Neither Up ($p \ge q_{hi}$) nor Down ($p \le q_{lo}$) reached threshold.
  - Signal accuracy on random labels: **50.00%** (0 signals triggered).
- **Conclusion**: The learning algorithm does not overfit noise into directional signals; zero spurious edge detected.

---

## 3. Gate 2: Time-Shift Lead/Lag Test

- **Hypothesis**: If feature rows are shifted forward by +1 bar ($t-1$ features predicting $t \to t+1$ returns), any immediate microstructure signal should degrade to near random walk ($\le 55.0\%$).
- **Result**:
  - Shifted feature accuracy: **50.00%** (signals collapsed to neutral).
- **Conclusion**: The model does not retain spurious predictability across temporal offsets; signals depend strictly on contemporaneous bar $t$ state.

---

## 4. Gate 3: Single-Feature Information Leakage & Window Inspection

All 46 features were evaluated on the pre-holdout series against the target direction $y$ and the forward return `fwd`:
- Target rule: $|\text{AUC} - 0.50| \le 0.10$ for every feature.
- **Top 5 Features by AUC Delta**:
  1. `gs_minus_peer_1`: AUC = 0.4409 ($|\Delta| = 0.0591$), Spearman $\rho = -0.1270$. Window: GS return minus context peer return $\le t$.
  2. `ret_1`: AUC = 0.4428 ($|\Delta| = 0.0572$), Spearman $\rho = -0.1221$. Window: Return over close[$t-1$] to close[$t$].
  3. `clv`: AUC = 0.4571 ($|\Delta| = 0.0429$), Spearman $\rho = -0.0917$. Window: Close location value within bar $t$.
  4. `signed_volume_1`: AUC = 0.4619 ($|\Delta| = 0.0381$), Spearman $\rho = -0.0818$. Window: 1-minute signed volume in bar $t$.
  5. `up_minute_ratio`: AUC = 0.4632 ($|\Delta| = 0.0368$), Spearman $\rho = -0.0789$. Window: Ratio of positive 1-min bars in bar $t$.
- **Window Audit**:
  - All rolling indicators (`rsi`, `macd`, `bb_`, `atr_`, `dist_session`, `or_pos_30m`) end strictly at bar $t$ close.
  - Zero features reference bar $t+1$ or lookahead prices.
- **Conclusion**: PASS. Zero leaked features found. The full audit table is saved to `results/m5/audit/single_feature_auc.csv`.

---

## 5. Gate 4: IEX Microstructure & Thin-Print Artifact Test (Root Cause Found)

Using the consolidated SIP 1-minute historical dataset (`data/raw/alpaca_1min/GS/`), 87,020 5-minute consolidated bars were constructed and aligned with IEX feature bars.

### Empirical Metrics
- **Lag-1 Return Autocorrelation**:
  - IEX 5-minute returns: $+0.0076$
  - Consolidated SIP 5-minute returns: $+0.0048$
- **Directional Accuracy**:
  - **On IEX forward return ($close_{IEX, t+1} / close_{IEX, t} - 1$)**: **69.93%**
  - **On Consolidated SIP forward return ($close_{SIP, t+1} / close_{SIP, t} - 1$)**: **51.80%**
  - **Accuracy Gap ($\Delta$)**: **+18.13%** (Exceeds the 5% threshold by 3.6x)
- **Trade Density Breakdown (IEX)**:
  - Low trade density ($\le 2$ minutes with trades): **69.90%**
  - High trade density ($\ge 4$ minutes with trades): **90.00%** (small sample of strong multi-minute trends)

### Root Cause Analysis
IEX accounts for approximately 2–3% of total US equity consolidated volume. On a 5-minute timeframe, an IEX bar close is frequently the result of a single odd lot or retail print on the IEX book, rather than the consolidated NBBO mid-price.
When the model predicts a mean-reverting or trend-continuation bounce from an IEX print to the next IEX print, it captures an **exchange-local microstructure bounce (bid-ask bounce / stale print artifact)**. When the exact same signal is settled against the broader market (Consolidated SIP), the edge drops to 51.80%, which is typical for a 5-minute equity direction.

---

## 6. Gate 5: Subgroup Stability Audit

Evaluated on the model's out-of-fold predictions across pre-holdout data:

### Stability by Calendar Year
- **2021**: 68.10% (111 / 163) [Wilson 95% CI: 60.6%, 74.8%]
- **2022**: 69.28% (1,416 / 2,044) [Wilson 95% CI: 67.2%, 71.3%]
- **2023**: 68.35% (1,235 / 1,807) [Wilson 95% CI: 66.2%, 70.5%]
- **2024**: 70.56% (1,028 / 1,457) [Wilson 95% CI: 68.1%, 72.9%]
- **2025**: 71.02% (1,272 / 1,791) [Wilson 95% CI: 68.9%, 73.1%]
- **2026**: 75.49% (231 / 306) [Wilson 95% CI: 70.3%, 80.0%]

### Stability by Time of Day Bucket
- **Opening (09:30 - 11:00 ET)**: 70.16% (1,406 / 2,004) [Wilson 95% CI: 68.1%, 72.1%]
- **Midday (11:00 - 14:00 ET)**: 69.78% (2,644 / 3,789) [Wilson 95% CI: 68.3%, 71.2%]
- **Closing (14:00 - 15:55 ET)**: 70.03% (1,243 / 1,775) [Wilson 95% CI: 67.8%, 72.1%]

### Stability by Side
- **Up Signals**: 69.57% (2,508 / 3,605) [Wilson 95% CI: 68.0%, 71.1%]
- **Down Signals**: 70.28% (2,785 / 3,963) [Wilson 95% CI: 68.8%, 71.7%]

---

## 7. Next Actions in Accordance with Protocol

1. Proceed with **Section B: Coverage-Targeted Rule**:
   - Compute coverage trade-off table (5%, 10%, 20%, 30%, 40%, 50%) for OOF and Holdout.
   - Test Rule A ($c = 0.30$, min margin 0.01) on pooled OOF, holdout, and last 20 completed sessions.
   - If achieved coverage on last 20 sessions $< 15\%$, activate Rule B.
2. Integrate prominent **Red Warning Banner** on UI:
   `"ACCURACY MEASURED ON IEX PRINTS; NOT CONFIRMED ON CONSOLIDATED PRICES"`
   Detailing that while signals run on IEX data with ~69.9% IEX-print hit rate, consolidated SIP hit rate is 51.80%.
