# Read-Only Performance Report: 5-Minute (M5) Goldman Sachs Model (Always-Call Rule v3)

**Date of Evaluation**: 2026-11-16  
**Status**: FROZEN AUDIT (Read-Only Mode)  
**Active Rule**: `models/m5/rule_m5_v3.json` (pointed to by `models/m5/active_rule.json`)  
**Database**: `data/live/live_m5.db` (opened strictly via SQLite URI `file:data/live/live_m5.db?mode=ro`)  

---

## Mandatory Disclosures & Risk Caveats

> [!IMPORTANT]
> **Holdout Caveat**: The research holdout period (`2026-05-18` to `2026-11-13`) was inspected during exploratory feature engineering, threshold analysis, and model selection. Consequently, all metrics derived from this dataset are **descriptive historical summaries**, not a clean, virgin out-of-sample forward test.

> [!WARNING]
> **IEX Print Artifact Warning**: In the rigorous M5 data source audit (`docs/m5/audit_findings.md`), the apparent ~69.93% accuracy against IEX exchange prints collapsed to **51.80%** when evaluated against the official Consolidated SIP tape (CTA/UTP). IEX prints suffer from odd-lot latency, sparse quote matching, and print timing artifacts that inflate historical backtest returns. Real-world forward execution performance will be substantially lower than numbers shown here.

> [!NOTE]
> **Single-Day Small-Sample Caveat**: A single trading day contains only ~74 to 76 five-minute bars. Due to small sample size ($n \approx 75$), 95% Wilson confidence intervals are extremely wide ($\pm 11$ to $12$ percentage points). Performance on any single session (such as Nov 12 or Nov 13) serves as a **functional demonstration, not statistical proof** of edge.

---

## 1. Frozen Model & Rule Cryptographic Verification

Before executing any evaluation logic, all production model weights, calibrator parameters, and rule definition files were verified byte-for-byte against the pre-registered cryptographic manifest `models/m5/MODEL_SHA256_m5.json`:

| Component File | Expected SHA-256 Digest | Actual SHA-256 Digest | Status |
| :--- | :--- | :--- | :---: |
| `model_m5.joblib` | `bb018397a7561c630ae4...` | `bb018397a7561c630ae4...` | **VERIFIED MATCH** |
| `calib_m5.json` | `6f65c1d2a74871c676d9...` | `6f65c1d2a74871c676d9...` | **VERIFIED MATCH** |
| `rule_m5_v2.json` | `80ffe1deca487305d674...` | `80ffe1deca487305d674...` | **VERIFIED MATCH** |
| `active_rule.json` | `a1781bf6ec6c3764875c...` | `a1781bf6ec6c3764875c...` | **VERIFIED MATCH** |
| `rule_m5_v3.json` | `3dd1a08043c8f1c3dca5...` | `3dd1a08043c8f1c3dca5...` | **VERIFIED MATCH** |

### Active Rule Configuration (`rule_m5_v3.json`)
- **Rule Type**: Always-Call Directional Protocol
- **Abstention Boundaries**: `q_lo = 0.50`, `q_hi = 0.50` (Directional call emitted on every eligible bar unless $p_{cal} == 0.500000$)
- **Conviction Tiers** (derived from validation OOF confidence distribution):
  - **`WEAK`**: $\text{Confidence} \le 0.520765$ (~bottom quartile)
  - **`NORMAL`**: $0.520765 < \text{Confidence} < 0.581116$ (~middle 50%)
  - **`STRONG`**: $\text{Confidence} \ge 0.581116$ (~top quartile)
- **Ground Truth Rule**: For bar $t$, label $y = 1$ if $\text{close}(t+1) > \text{close}(t)$, else $0$. The final bar of each session ($15:55$ ET) has no forward intraday bar and is classified as `No signal (end of day)`.

---

## 2. Cross-Slice Performance Summary

### 2.1 Signal Counts & Coverage Breakdown

| Metric | Slice A: Holdout (126d) | Slice B: Nov 12 (1d) | Slice C: Nov 13 (1d) | Slice B + C Combined (2d) |
| :--- | :---: | :---: | :---: | :---: |
| **Calendar Sessions** | 126 | 1 | 1 | 2 |
| **Theoretical Bars** (78/day) | 9,828 | 78 | 78 | 156 |
| **Recorded Bars in Tape** | 9,598 | 76 | 74 | 150 |
| **Data Gaps** (Missing Tape Bars) | 230 (2.34%) | 2 (2.56%) | 4 (5.13%) | 6 (3.85%) |
| **Up Calls** | 4,679 (48.75%) | 39 (51.32%) | 31 (41.89%) | 70 (46.67%) |
| **Down Calls** | 4,793 (49.94%) | 36 (47.37%) | 42 (56.76%) | 78 (52.00%) |
| **No Signal (EOD 15:55)** | 126 (1.31%) | 1 (1.32%) | 1 (1.35%) | 2 (1.33%) |
| **No Signal (Exact Ties)** | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) |
| **Total No Signal in Recorded Tape** | 126 (1.31%) | 1 (1.32%) | 1 (1.35%) | 2 (1.33%) |
| **Sum Check** (`UP + DOWN + NoSig == Total`) | **PASSED** (9598 == 9598) | **PASSED** (76 == 76) | **PASSED** (74 == 74) | **PASSED** (150 == 150) |

### 2.2 Directional Accuracy & Wilson 95% Confidence Intervals

| Metric | Slice A: Holdout (126d) | Slice B: Nov 12 (1d) | Slice C: Nov 13 (1d) | Slice B + C Combined (2d) |
| :--- | :---: | :---: | :---: | :---: |
| **Labelled Bars Evaluated** | 9,472 | 75 | 73 | 148 |
| **Overall Directional Accuracy** | **53.93%** (5,108/9,472) | **57.33%** (43/75) | **58.90%** (43/73) | **58.11%** (86/148) |
| **Wilson 95% CI (Overall)** | `[52.92%, 54.93%]` | `[46.05%, 67.90%]` | `[47.45%, 69.47%]` | `[50.05%, 65.75%]` |
| **Up Call Accuracy** | **53.30%** (2,494/4,679) | **58.97%** (23/39) | **58.06%** (18/31) | **58.57%** (41/70) |
| **Wilson 95% CI (Up)** | `[51.87%, 54.73%]` | `[43.42%, 72.92%]` | `[40.77%, 73.58%]` | `[46.88%, 69.37%]` |
| **Down Call Accuracy** | **54.54%** (2,614/4,793) | **55.56%** (20/36) | **59.52%** (25/42) | **57.69%** (45/78) |
| **Wilson 95% CI (Down)** | `[53.13%, 55.94%]` | `[39.58%, 70.46%]` | `[44.49%, 72.96%]` | `[46.62%, 68.04%]` |
| **Market Base Rate ($y=1$ Up)** | 49.33% | 52.00% | 47.95% | 50.00% |
| **Larger Base Rate (Majority Class)**| 50.67% (Down) | 52.00% (Up) | 52.05% (Down) | 50.00% (Even) |
| **Edge Over Majority Base Rate** | **+3.26%** | **+5.33%** | **+6.85%** | **+8.11%** |

### 2.3 Conviction Tier Performance & Monotonicity Audit

| Slice | Tier | Count ($n$) | Correct | Accuracy | Wilson 95% CI | Up / Down Split | Monotonicity ($W \le N \le S$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **A** (Research Holdout) | `WEAK` | 2,962 | 1,463 | **49.39%** | `[47.59%, 51.19%]` | 1504 Up / 1458 Dn | (Base) |
| | `NORMAL` | 5,485 | 3,002 | **54.73%** | `[53.41%, 56.04%]` | 2647 Up / 2838 Dn | - |
| | `STRONG` | 1,025 | 643 | **62.73%** | `[59.73%, 65.64%]` | 528 Up / 497 Dn | **HOLDS (PASS)** |
| **B** (Thursday 2026-11-12) | `WEAK` | 20 | 10 | **50.00%** | `[29.93%, 70.07%]` | 12 Up / 8 Dn | (Base) |
| | `NORMAL` | 50 | 29 | **58.00%** | `[44.23%, 70.62%]` | 24 Up / 26 Dn | - |
| | `STRONG` | 5 | 4 | **80.00%** | `[37.55%, 96.38%]` | 3 Up / 2 Dn | **HOLDS (PASS)** |
| **C** (Friday 2026-11-13) | `WEAK` | 22 | 13 | **59.09%** | `[38.73%, 76.74%]` | 11 Up / 11 Dn | (Base) |
| | `NORMAL` | 41 | 23 | **56.10%** | `[41.04%, 70.11%]` | 15 Up / 26 Dn | - |
| | `STRONG` | 10 | 7 | **70.00%** | `[39.68%, 89.22%]` | 5 Up / 5 Dn | Non-Monotonic (Noise) |
| **BC** (Slice B + C Combined) | `WEAK` | 42 | 23 | **54.76%** | `[39.95%, 68.78%]` | 23 Up / 19 Dn | (Base) |
| | `NORMAL` | 91 | 52 | **57.14%** | `[46.89%, 66.82%]` | 39 Up / 52 Dn | - |
| | `STRONG` | 15 | 11 | **73.33%** | `[48.05%, 89.10%]` | 8 Up / 7 Dn | **HOLDS (PASS)** |

### 2.4 Confusion Matrices

| Slice | Predicted \ Actual | Actual Up ($y=1$) | Actual Down ($y=0$) | Total Predicted |
| :--- | :--- | :---: | :---: | :---: |
| **A** | **Predicted Up** | **2,494** (TP) | **2,185** (FP) | 4,679 |
| | **Predicted Down** | **2,179** (FN) | **2,614** (TN) | 4,793 |
| | **Total Actual** | 4,673 | 4,799 | 9,472 |
| **B** | **Predicted Up** | **23** (TP) | **16** (FP) | 39 |
| | **Predicted Down** | **16** (FN) | **20** (TN) | 36 |
| | **Total Actual** | 39 | 36 | 75 |
| **C** | **Predicted Up** | **18** (TP) | **13** (FP) | 31 |
| | **Predicted Down** | **17** (FN) | **25** (TN) | 42 |
| | **Total Actual** | 35 | 38 | 73 |
| **BC** | **Predicted Up** | **41** (TP) | **29** (FP) | 70 |
| | **Predicted Down** | **33** (FN) | **45** (TN) | 78 |
| | **Total Actual** | 74 | 74 | 148 |

### 2.5 Holdout Statistical Calibration & Significance (Slice A Only)

- **Permutation-Test p-value** (5,000 Shuffles, seeded `np.random.default_rng(42)`): **$p = 0.00020$** ($p < 0.001$, statistically distinguishable from coin flip).
- **Expected Calibration Error (ECE, 10 Bins)**: **0.0079** (Well below the $\le 0.030$ reliability threshold).
- **Brier Score**: **0.2464** (Superior to an uninformative 50/50 prior of $0.2500$).

### 2.6 Net Return Estimates per Call (IEX Next-Bar Return in Basis Points)

> [!NOTE]
> Returns are calculated using the 5-minute IEX close-to-close returns. Long on `Up`, short on `Down`. Transaction cost is deducted per trade.

| Execution Cost | Holdout: All Tiers | Holdout: STRONG Only | Nov 12: All Tiers | Nov 12: STRONG | Nov 13: All Tiers | Nov 13: STRONG | B+C: All Tiers | B+C: STRONG |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0 bps** | +1.45 bps | **+4.85 bps** | +2.18 bps | **+5.81 bps** | +2.33 bps | **+2.73 bps** | +2.25 bps | **+3.76 bps** |
| **2 bps** | -0.55 bps | **+2.85 bps** | +0.18 bps | **+3.81 bps** | +0.33 bps | **+0.73 bps** | +0.25 bps | **+1.76 bps** |
| **3 bps** | -1.55 bps | **+1.85 bps** | -0.82 bps | **+2.81 bps** | -0.67 bps | **-0.27 bps** | -0.75 bps | **+0.76 bps** |
| **4 bps** | -2.55 bps | **+0.85 bps** | -1.82 bps | **+1.81 bps** | -1.67 bps | **-1.27 bps** | -1.75 bps | **-0.24 bps** |
| **5 bps** | -3.55 bps | **-0.15 bps** | -2.82 bps | **+0.81 bps** | -2.67 bps | **-2.27 bps** | -2.75 bps | **-1.24 bps** |

---

## 3. Thursday 2026-11-12 Session Audit (Slice B)

### 3.1 Hourly Accuracy Breakdown

| Hour Window (ET) | Total Bars | Directional Signals | Correct Calls | Accuracy (%) | Up Calls | Down Calls |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **09:30 – 09:55** | 6 | 6 | 4 | **66.7%** | 5 | 1 |
| **10:30 – 10:55** | 12 | 12 | 6 | **50.0%** | 3 | 9 |
| **11:30 – 11:55** | 12 | 12 | 7 | **58.3%** | 9 | 3 |
| **12:30 – 12:55** | 12 | 12 | 7 | **58.3%** | 7 | 5 |
| **13:30 – 13:55** | 11 | 11 | 5 | **45.5%** | 4 | 7 |
| **14:30 – 14:55** | 11 | 11 | 6 | **54.5%** | 5 | 6 |
| **15:30 – 15:55** | 12 | 11 | 8 | **72.7%** | 6 | 5 |

### 3.2 Bar-by-Bar Signal & Outcome Audit (Nov 12)

**Data Tape Gaps Noted**: 2 missing bars on IEX feed (`13:45`, `14:20`). Total recorded bars = 76 of 78 theoretical.

| Bar Start (ET) | Signal Direction | Confidence | Conviction Tier | Actual Direction ($y$) | Correct? | Notes |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| 09:30 | **Up** | 0.5002 | `WEAK` | Up | **YES** | - |
| 09:35 | **Down** | 0.5418 | `NORMAL` | Down | **YES** | - |
| 09:40 | **Up** | 0.5437 | `NORMAL` | Up | **YES** | - |
| 09:45 | **Up** | 0.5124 | `WEAK` | Up | **YES** | - |
| 09:50 | **Up** | 0.5473 | `NORMAL` | Down | **NO** | - |
| 09:55 | **Up** | 0.5154 | `WEAK` | Down | **NO** | - |
| 10:00 | **Up** | 0.5031 | `WEAK` | Up | **YES** | - |
| 10:05 | **Up** | 0.5122 | `WEAK` | Up | **YES** | - |
| 10:10 | **Down** | 0.5161 | `WEAK` | Up | **NO** | - |
| 10:15 | **Down** | 0.5658 | `NORMAL` | Down | **YES** | - |
| 10:20 | **Down** | 0.5233 | `NORMAL` | Up | **NO** | - |
| 10:25 | **Up** | 0.5795 | `NORMAL` | Up | **YES** | - |
| 10:30 | **Down** | 0.5157 | `WEAK` | Down | **YES** | - |
| 10:35 | **Down** | 0.5029 | `WEAK` | Up | **NO** | - |
| 10:40 | **Down** | 0.5309 | `NORMAL` | Up | **NO** | - |
| 10:45 | **Down** | 0.5145 | `WEAK` | Up | **NO** | - |
| 10:50 | **Down** | 0.5251 | `NORMAL` | Up | **NO** | - |
| 10:55 | **Down** | 0.5193 | `WEAK` | Down | **YES** | - |
| 11:00 | **Up** | 0.5251 | `NORMAL` | Up | **YES** | - |
| 11:05 | **Down** | 0.5246 | `NORMAL` | Up | **NO** | - |
| 11:10 | **Up** | 0.5503 | `NORMAL` | Up | **YES** | - |
| 11:15 | **Down** | 0.5274 | `NORMAL` | Down | **YES** | - |
| 11:20 | **Up** | 0.5632 | `NORMAL` | Down | **NO** | - |
| 11:25 | **Down** | 0.5210 | `NORMAL` | Down | **YES** | - |
| 11:30 | **Up** | 0.5220 | `NORMAL` | Down | **NO** | - |
| 11:35 | **Up** | 0.5025 | `WEAK` | Down | **NO** | - |
| 11:40 | **Up** | 0.5814 | `STRONG` | Up | **YES** | - |
| 11:45 | **Up** | 0.5340 | `NORMAL` | Down | **NO** | - |
| 11:50 | **Up** | 0.5324 | `NORMAL` | Up | **YES** | - |
| 11:55 | **Up** | 0.5428 | `NORMAL` | Up | **YES** | - |
| 12:00 | **Up** | 0.5143 | `WEAK` | Down | **NO** | - |
| 12:05 | **Up** | 0.5513 | `NORMAL` | Down | **NO** | - |
| 12:10 | **Up** | 0.5557 | `NORMAL` | Up | **YES** | - |
| 12:15 | **Down** | 0.5806 | `NORMAL` | Down | **YES** | - |
| 12:20 | **Up** | 0.5922 | `STRONG` | Up | **YES** | - |
| 12:25 | **Up** | 0.5640 | `NORMAL` | Up | **YES** | - |
| 12:30 | **Down** | 0.5299 | `NORMAL` | Up | **NO** | - |
| 12:35 | **Down** | 0.5234 | `NORMAL` | Up | **NO** | - |
| 12:40 | **Down** | 0.5470 | `NORMAL` | Down | **YES** | - |
| 12:45 | **Up** | 0.5265 | `NORMAL` | Up | **YES** | - |
| 12:50 | **Down** | 0.5334 | `NORMAL` | Down | **YES** | - |
| 12:55 | **Up** | 0.5226 | `NORMAL` | Down | **NO** | - |
| 13:00 | **Up** | 0.5238 | `NORMAL` | Down | **NO** | - |
| 13:05 | **Down** | 0.5374 | `NORMAL` | Up | **NO** | - |
| 13:10 | **Down** | 0.5063 | `WEAK` | Up | **NO** | - |
| 13:15 | **Up** | 0.5426 | `NORMAL` | Up | **YES** | - |
| 13:20 | **Down** | 0.5336 | `NORMAL` | Down | **YES** | - |
| 13:25 | **Up** | 0.5163 | `WEAK` | Down | **NO** | - |
| 13:30 | **Up** | 0.5424 | `NORMAL` | Down | **NO** | - |
| 13:35 | **Down** | 0.5461 | `NORMAL` | Down | **YES** | - |
| 13:40 | **Down** | 0.5340 | `NORMAL` | Down | **YES** | - |
| 13:50 | **Down** | 0.5351 | `NORMAL` | Up | **NO** | - |
| 13:55 | **Down** | 0.5660 | `NORMAL` | Down | **YES** | - |
| 14:00 | **Up** | 0.5227 | `NORMAL` | Up | **YES** | - |
| 14:05 | **Up** | 0.5631 | `NORMAL` | Down | **NO** | - |
| 14:10 | **Up** | 0.5689 | `NORMAL` | Up | **YES** | - |
| 14:15 | **Up** | 0.6081 | `STRONG` | Up | **YES** | - |
| 14:25 | **Down** | 0.7068 | `STRONG` | Down | **YES** | - |
| 14:30 | **Up** | 0.5242 | `NORMAL` | Down | **NO** | - |
| 14:35 | **Down** | 0.5352 | `NORMAL` | Up | **NO** | - |
| 14:40 | **Down** | 0.5259 | `NORMAL` | Down | **YES** | - |
| 14:45 | **Down** | 0.5582 | `NORMAL` | Up | **NO** | - |
| 14:50 | **Down** | 0.6149 | `STRONG` | Up | **NO** | - |
| 14:55 | **Down** | 0.5677 | `NORMAL` | Down | **YES** | - |
| 15:00 | **Up** | 0.5139 | `WEAK` | Down | **NO** | - |
| 15:05 | **Down** | 0.5249 | `NORMAL` | Down | **YES** | - |
| 15:10 | **Up** | 0.5030 | `WEAK` | Up | **YES** | - |
| 15:15 | **Down** | 0.5102 | `WEAK` | Down | **YES** | - |
| 15:20 | **Up** | 0.5461 | `NORMAL` | Up | **YES** | - |
| 15:25 | **Down** | 0.5153 | `WEAK` | Down | **YES** | - |
| 15:30 | **Up** | 0.5252 | `NORMAL` | Up | **YES** | - |
| 15:35 | **Down** | 0.5223 | `NORMAL` | Down | **YES** | - |
| 15:40 | **Up** | 0.5122 | `WEAK` | Up | **YES** | - |
| 15:45 | **Up** | 0.5082 | `WEAK` | Down | **NO** | - |
| 15:50 | **Down** | 0.5357 | `NORMAL` | Up | **NO** | - |
| 15:55 | `No signal` | 0.5000 | `NORMAL` | N/A (EOD) | N/A | End-of-Day Bar (no forward interval) |

---

## 4. Friday 2026-11-13 Session Audit (Slice C)

### 4.1 Hourly Accuracy Breakdown

| Hour Window (ET) | Total Bars | Directional Signals | Correct Calls | Accuracy (%) | Up Calls | Down Calls |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **09:30 – 09:55** | 5 | 5 | 2 | **40.0%** | 3 | 2 |
| **10:30 – 10:55** | 12 | 12 | 8 | **66.7%** | 7 | 5 |
| **11:30 – 11:55** | 12 | 12 | 5 | **41.7%** | 5 | 7 |
| **12:30 – 12:55** | 12 | 12 | 9 | **75.0%** | 4 | 8 |
| **13:30 – 13:55** | 12 | 12 | 8 | **66.7%** | 6 | 6 |
| **14:30 – 14:55** | 9 | 9 | 3 | **33.3%** | 3 | 6 |
| **15:30 – 15:55** | 12 | 11 | 8 | **72.7%** | 3 | 8 |

### 4.2 Bar-by-Bar Signal & Outcome Audit (Nov 13)

**Data Tape Gaps Noted**: 4 missing bars on IEX feed (`09:50`, `14:20`, `14:25`, `14:35`). Total recorded bars = 74 of 78 theoretical.  
**Flat Close Bars**: 2 bars had zero price movement (close $t+1 == t$): `12:10` ($909.41) and `14:30` ($911.59). Under the strict prompt specification ($y = 1$ if $\text{close}_{t+1} > \text{close}_t$ else $0$), these are labelled $y = 0$ (Down). (In `log_engine.py`, flat bars are treated as neutral/miss; here we report the exact label behavior).

| Bar Start (ET) | Signal Direction | Confidence | Conviction Tier | Actual Direction ($y$) | Correct? | Notes |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| 09:30 | **Up** | 0.5832 | `STRONG` | Up | **YES** | - |
| 09:35 | **Down** | 0.5866 | `STRONG` | Up | **NO** | - |
| 09:40 | **Up** | 0.5168 | `WEAK` | Down | **NO** | - |
| 09:45 | **Up** | 0.5042 | `WEAK` | Down | **NO** | - |
| 09:55 | **Down** | 0.5545 | `NORMAL` | Down | **YES** | - |
| 10:00 | **Up** | 0.5207 | `WEAK` | Down | **NO** | - |
| 10:05 | **Up** | 0.5215 | `NORMAL` | Down | **NO** | - |
| 10:10 | **Down** | 0.5795 | `NORMAL` | Down | **YES** | - |
| 10:15 | **Up** | 0.5130 | `WEAK` | Up | **YES** | - |
| 10:20 | **Up** | 0.5261 | `NORMAL` | Up | **YES** | - |
| 10:25 | **Down** | 0.5511 | `NORMAL` | Up | **NO** | - |
| 10:30 | **Down** | 0.5186 | `WEAK` | Down | **YES** | - |
| 10:35 | **Down** | 0.5042 | `WEAK` | Down | **YES** | - |
| 10:40 | **Up** | 0.5319 | `NORMAL` | Up | **YES** | - |
| 10:45 | **Up** | 0.5715 | `NORMAL` | Up | **YES** | - |
| 10:50 | **Down** | 0.6419 | `STRONG` | Down | **YES** | - |
| 10:55 | **Up** | 0.6483 | `STRONG` | Down | **NO** | - |
| 11:00 | **Up** | 0.5382 | `NORMAL` | Down | **NO** | - |
| 11:05 | **Up** | 0.7567 | `STRONG` | Up | **YES** | - |
| 11:10 | **Down** | 0.5987 | `STRONG` | Down | **YES** | - |
| 11:15 | **Down** | 0.5201 | `WEAK` | Up | **NO** | - |
| 11:20 | **Up** | 0.5182 | `WEAK` | Up | **YES** | - |
| 11:25 | **Down** | 0.5095 | `WEAK` | Up | **NO** | - |
| 11:30 | **Down** | 0.5486 | `NORMAL` | Down | **YES** | - |
| 11:35 | **Down** | 0.5020 | `WEAK` | Down | **YES** | - |
| 11:40 | **Up** | 0.5487 | `NORMAL` | Down | **NO** | - |
| 11:45 | **Down** | 0.6293 | `STRONG` | Up | **NO** | - |
| 11:50 | **Down** | 0.5469 | `NORMAL` | Up | **NO** | - |
| 11:55 | **Up** | 0.5085 | `WEAK` | Down | **NO** | - |
| 12:00 | **Down** | 0.5504 | `NORMAL` | Up | **NO** | - |
| 12:05 | **Up** | 0.5531 | `NORMAL` | Up | **YES** | - |
| 12:10 | **Down** | 0.6072 | `STRONG` | Down | **YES** | Flat bar: close 909.41 -> next 909.41 ($y=0$) |
| 12:15 | **Down** | 0.5143 | `WEAK` | Down | **YES** | - |
| 12:20 | **Up** | 0.5364 | `NORMAL` | Up | **YES** | - |
| 12:25 | **Down** | 0.5306 | `NORMAL` | Down | **YES** | - |
| 12:30 | **Up** | 0.5334 | `NORMAL` | Up | **YES** | - |
| 12:35 | **Down** | 0.5655 | `NORMAL` | Up | **NO** | - |
| 12:40 | **Up** | 0.5407 | `NORMAL` | Down | **NO** | - |
| 12:45 | **Down** | 0.5070 | `WEAK` | Down | **YES** | - |
| 12:50 | **Down** | 0.5289 | `NORMAL` | Down | **YES** | - |
| 12:55 | **Down** | 0.5695 | `NORMAL` | Down | **YES** | - |
| 13:00 | **Up** | 0.5373 | `NORMAL` | Up | **YES** | - |
| 13:05 | **Down** | 0.5563 | `NORMAL` | Down | **YES** | - |
| 13:10 | **Up** | 0.5112 | `WEAK` | Up | **YES** | - |
| 13:15 | **Up** | 0.5985 | `STRONG` | Up | **YES** | - |
| 13:20 | **Down** | 0.5601 | `NORMAL` | Up | **NO** | - |
| 13:25 | **Up** | 0.5737 | `NORMAL` | Down | **NO** | - |
| 13:30 | **Up** | 0.5112 | `WEAK` | Up | **YES** | - |
| 13:35 | **Down** | 0.5240 | `NORMAL` | Up | **NO** | - |
| 13:40 | **Down** | 0.5361 | `NORMAL` | Down | **YES** | - |
| 13:45 | **Down** | 0.5243 | `NORMAL` | Down | **YES** | - |
| 13:50 | **Up** | 0.5281 | `NORMAL` | Down | **NO** | - |
| 13:55 | **Down** | 0.5159 | `WEAK` | Down | **YES** | - |
| 14:00 | **Up** | 0.5713 | `NORMAL` | Up | **YES** | - |
| 14:05 | **Down** | 0.5399 | `NORMAL` | Down | **YES** | - |
| 14:10 | **Down** | 0.5451 | `NORMAL` | Up | **NO** | - |
| 14:15 | **Up** | 0.5069 | `WEAK` | Down | **NO** | - |
| 14:30 | **Up** | 0.5118 | `WEAK` | Down | **NO** | Flat bar: close 911.59 -> next 911.59 ($y=0$) |
| 14:40 | **Down** | 0.5251 | `NORMAL` | Up | **NO** | - |
| 14:45 | **Down** | 0.5458 | `NORMAL` | Up | **NO** | - |
| 14:50 | **Down** | 0.5235 | `NORMAL` | Up | **NO** | - |
| 14:55 | **Down** | 0.5148 | `WEAK` | Down | **YES** | - |
| 15:00 | **Up** | 0.5613 | `NORMAL` | Up | **YES** | - |
| 15:05 | **Down** | 0.5195 | `WEAK` | Up | **NO** | - |
| 15:10 | **Down** | 0.5376 | `NORMAL` | Down | **YES** | - |
| 15:15 | **Down** | 0.5175 | `WEAK` | Down | **YES** | - |
| 15:20 | **Down** | 0.5289 | `NORMAL` | Up | **NO** | - |
| 15:25 | **Down** | 0.5522 | `NORMAL` | Up | **NO** | - |
| 15:30 | **Down** | 0.5786 | `NORMAL` | Down | **YES** | - |
| 15:35 | **Up** | 0.5189 | `WEAK` | Up | **YES** | - |
| 15:40 | **Down** | 0.5274 | `NORMAL` | Down | **YES** | - |
| 15:45 | **Up** | 0.6760 | `STRONG` | Up | **YES** | - |
| 15:50 | **Down** | 0.5227 | `NORMAL` | Down | **YES** | - |
| 15:55 | `No signal` | 0.5000 | `NORMAL` | N/A (EOD) | N/A | End-of-Day Bar (no forward interval) |

---

## 5. Cross-Checks & Data Integrity Verifications

1. **Replay Equivalence (`predict_day_m5.py` vs SQLite Ledger `live_m5.db`)**:
   - **Nov 12, 2026**: 76 ledger rows compared against fresh pipeline replay. **Mismatches = 0**. Status: All rows marked `late = 1` (re-derived).
   - **Nov 13, 2026**: 74 ledger rows compared against fresh pipeline replay. **Mismatches = 0**. Status: All rows marked `late = 1` (re-derived).
2. **Cryptographic Hash Chain Integrity**:
   - Total rows in `data/live/live_m5.db`: **2,738**
   - Broken links found: **0** (Entire chain from genesis to latest record verified mathematically intact).
3. **Coverage Sum Consistency**:
   - Slice A (Holdout): $4,679\text{ Up} + 4,793\text{ Down} + 126\text{ No-Sig} = 9,598\text{ recorded bars}$ (**100.0% sum check verified**).
   - Slice B (Nov 12): $39\text{ Up} + 36\text{ Down} + 1\text{ No-Sig} = 76\text{ recorded bars}$ (**100.0% sum check verified**).
   - Slice C (Nov 13): $31\text{ Up} + 42\text{ Down} + 1\text{ No-Sig} = 74\text{ recorded bars}$ (**100.0% sum check verified**).
   - Combined B + C: $70\text{ Up} + 78\text{ Down} + 2\text{ No-Sig} = 150\text{ recorded bars}$ (**100.0% sum check verified**).