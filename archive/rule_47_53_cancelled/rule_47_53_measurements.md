# Measurement Report: Signal Rule 2 (No-Signal Band 47% to 53%)

**Date**: 2026-11-15  
**Owner Decision**: *"The no-signal band must be 47% to 53%."*  
**Rule Status**: Signal-rule change (not a model retraining or feature change). Deployed thresholds: $q_{\text{lo}} = 0.47$, $q_{\text{hi}} = 0.53$, $\text{weak\_below\_conf} = 0.5632$, $\text{strong\_conf} = 0.60$.

---

## 1. Coverage Comparison: Old Rule (43.4%–56.3%) vs. New Rule (47%–53%)

All measurements use the deployed calibrator on SIP consolidated tape features for the live and 60-session periods. Indicative comparisons for earlier research periods are explicitly noted.

| Evaluation Period | Status / Calibrator | Total Eligible Bars | Old Rule (Up / Down / No-Sig) | Old Coverage | New Rule (Up / Down / No-Sig) | New Coverage |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Last 60 Completed Sessions** | Deployed model & calibrator (SIP) | 360 | 55 / 50 / 255 | **29.17%** | 107 / 97 / 156 | **56.67%** |
| **Nov 10–13 Out-of-Sample** | Deployed model & calibrator (SIP) | 24 | 2 / 4 / 18 | **25.00%** | 6 / 7 / 11 | **54.17%** |
| **V2 Research Holdout** | Pre-refit model *(indicative only)* | 1,605 | 285 / 276 / 1,044 | 34.95% | 585 / 425 / 595 | 62.93% |
| **Pre-Holdout Base Reference** | Pre-refit model *(indicative only)* | 10,289 | 1,561 / 1,514 / 7,214 | 29.89% | 3,400 / 2,945 / 3,944 | 61.67% |

*Data source table: `results/tables/rule_47_53_coverage.csv`.*

---

## 2. Descriptive Accuracy by Confidence Tier Under New Rule

### Confidence Tier Definitions:
* **WEAK**: Confidence $53.0\% \le \text{conf} < 56.3\%$ (just above the $0.53$ / below the $0.47$ threshold). Weaker statistical evidence.
* **NORMAL**: Confidence $56.3\% \le \text{conf} < 60.0\%$ (standard directional conviction, exceeding the old $q_{\text{hi}}$).
* **STRONG**: Confidence $\ge 60.0\%$ (high conviction, exceeding the strong threshold).

> [!IMPORTANT]
> **Strict Disclosure**: No clean held-out accuracy exists for the deployed rule because the final production model was refit on all available data through the freeze date. The numbers below are descriptive only.

### (a) Pre-Holdout Base Reference (IN-SAMPLE, Optimistic)
Evaluated on 10,289 pre-holdout hourly bars:

| Confidence Tier | Side | Sample Size ($n$) | Realized Hits | Accuracy (%) | Wilson 95% CI |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **WEAK** | Up | 1,839 | 1,068 | 58.08% | [55.81%, 60.31%] |
| **WEAK** | Down | 1,367 | 779 | 56.99% | [54.35%, 59.59%] |
| **WEAK** | **Total** | **3,206** | **1,847** | **57.61%** | **[55.89%, 59.31%]** |
| **NORMAL** | Up | 1,043 | 660 | 63.28% | [60.31%, 66.15%] |
| **NORMAL** | Down | 895 | 545 | 60.89% | [57.66%, 64.04%] |
| **NORMAL** | **Total** | **1,938** | **1,205** | **62.18%** | **[60.00%, 64.31%]** |
| **STRONG** | Up | 518 | 401 | 77.41% | [73.62%, 80.80%] |
| **STRONG** | Down | 683 | 462 | 67.64% | [64.04%, 71.04%] |
| **STRONG** | **Total** | **1,201** | **863** | **71.86%** | **[69.25%, 74.33%]** |
| **ALL DIRECTIONAL** | **Total** | **6,345** | **3,915** | **61.70%** | **[60.50%, 62.89%]** |

*Monotonic Validation*: In-sample hit rates scale monotonically with conviction: WEAK (57.61%) < NORMAL (62.18%) < STRONG (71.86%).

### (b) Nov 10–13 Out-of-Sample Period (24 Candidate Hourly Bars)
Evaluated on the 24 recent out-of-sample candles using consolidated SIP tape:

| Confidence Tier | Side | Sample Size ($n$) | Realized Hits | Accuracy (%) | Wilson 95% CI |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **WEAK** | Up | 4 | 2 | 50.00% | [15.00%, 85.00%] |
| **WEAK** | Down | 3 | 3 | 100.00% | [43.85%, 100.00%] |
| **WEAK** | **Total** | **7** | **5** | **71.43%** | **[35.89%, 91.78%]** |
| **NORMAL** | Up | 1 | 1 | 100.00% | [20.65%, 100.00%] |
| **NORMAL** | Down | 1 | 1 | 100.00% | [20.65%, 100.00%] |
| **NORMAL** | **Total** | **2** | **2** | **100.00%** | **[34.24%, 100.00%]** |
| **STRONG** | Up | 1 | 1 | 100.00% | [20.65%, 100.00%] |
| **STRONG** | Down | 3 | 1 | 33.33% | [6.15%, 79.23%] |
| **STRONG** | **Total** | **4** | **2** | **50.00%** | **[15.00%, 85.00%]** |
| **ALL DIRECTIONAL** | Up | 6 | 4 | 66.67% | [30.00%, 90.32%] |
| **ALL DIRECTIONAL** | Down | 7 | 5 | 71.43% | [35.89%, 91.78%] |
| **ALL DIRECTIONAL** | **Total** | **13** | **9** | **69.23%** | **[42.37%, 87.32%]** |

*Findings on Nov 10–13*:
* The 47%–53% rule produces **13 directional signals (6 Up, 7 Down)** across the 4 sessions (balanced side distribution).
* Realized directional accuracy was **9 / 13 (69.23%)**. Due to small sample size ($n=13$), the Wilson 95% interval is broad: `[42.37%, 87.32%]`.

*Data source table: `results/tables/rule_47_53_tier_accuracy.csv`.*

---

## 3. Expected Signals Per Day (Last 60 Completed Sessions)

Daily directional signal frequency across the 60 sessions under both rules (excluding the 15:30 EOD bar):

| Rule Specification | Lower ($q_{\text{lo}}$) | Upper ($q_{\text{hi}}$) | Mean Signals / Day | Median | Range (Min – Max) | Std Dev |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Old Rule (Narrow Coverage)** | 0.4343 | 0.5632 | **1.75** | 2.0 | **0 to 5** | 1.08 |
| **New Rule (Owner 47%–53%)** | 0.4700 | 0.5300 | **3.40** | 3.0 | **1 to 6** | 1.08 |

*Key Takeaway*:
* Under the Old Rule, a typical 7-hour session produced only 1 to 2 directional signals, with zero signals occurring on slow days.
* Under the New Rule, an auditor can expect an average of **3.4 directional signals per session** (median: 3 signals, range: 1 to 6), guaranteeing that virtually every session yields tradeable signals for live verification.

*Data source table: `results/tables/rule_47_53_signals_per_day.csv`.*
