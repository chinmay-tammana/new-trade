# Coverage & Denominator Reconciliation (5-Minute System)

**Date**: 2026-11-16  
**Status**: COMPLETE & VERIFIED  

---

## 1. Executive Summary & Root Causes of Previous Discrepancies

This document reconciles all denominator counts and coverage metrics across the pre-holdout walk-forward out-of-fold (OOF) evaluation dataset and the research holdout dataset.

### A. The 74,971 vs 39,766 (or 39,778) Bar Discrepancy
- **Pre-Holdout Universe Total**: There are **77,511 total 5-minute bars** (spanning November 2021 to May 2026 across 1,137 trading sessions). 
- After dropping end-of-day bars (which have no forward 5-minute return in the same session) and initial warmup missing feature bars, there are **74,971 valid labelled bars** in the entire pre-holdout series.
- **Walk-Forward Burn-In Window**: The walk-forward cross-validation protocol (`build_walk_forward_folds`) enforces a strict **2-year initial training burn-in period** (~35,205 bars) before the first validation fold begins.
- Therefore, out-of-fold evaluation was conducted on exactly **39,766 bars** across all expanding validation folds.
- **Conclusion**: 74,971 was the total pre-holdout time series length, whereas 39,766 was the evaluated out-of-fold sample size.

### B. Why "No-Signal 63,041 bars (70.0%)" Was Mathematically Wrong
- In the initial V4.1 calculation, 11,930 signals were emitted out of the 39,766 OOF bars ($11,930 / 39,766 = 30.00\%$).
- The earlier script erroneously subtracted the emitted signals ($11,930$) from the **total pre-holdout universe** ($74,971$), yielding $74,971 - 11,930 = 63,041$ bars, and labeled this "70.0%".
- However, $63,041 / 74,971 = 84.09\%$, not $70.0\%$. Mixing the total series denominator with the OOF evaluation target generated the contradictory statement.
- Under the proper denominator of 39,766 OOF bars, exactly **27,836 bars (70.0%)** were No-Signal.

### C. Why Holdout Coverage Was 19.3% Against the 30% Target
- Under Rule V2 (30% coverage), the quantile thresholds ($q_{lo} = 0.4296$, $q_{hi} = 0.5727$) were computed and frozen from the pre-holdout OOF distribution.
- When applied to the holdout dataset (May–Nov 2026, 9,472 labelled bars):
  1. **Fixed Quantiles vs Distribution Shift**: The holdout period exhibited slightly lower prediction variance around 0.50, meaning fewer raw model predictions crossed the static $q_{lo}$ / $q_{hi}$ bounds.
  2. **Margin Guard Filter**: Rule V2 applied a dynamic margin guard ($|p_{cal} - 0.5| > margin$), which filtered out borderline bars during high-spread or uncertain regimes.
  3. Consequently, only 1,827 signals crossed the thresholds ($1,827 / 9,472 = 19.29\%$), lower than the 30% in-sample target.

---

## 2. Comprehensive Reconciliation Table

Below is the single reconciled ledger derived from `results/m5/coverage_reconciliation.csv`:

| Metric | Pre-Holdout OOF (Val Folds) | Research Holdout (Descriptive) | Total Universe Combined |
| :--- | :--- | :--- | :--- |
| **Total Calendar Sessions** | 511 val sessions (1,137 total) | 126 sessions | 1,263 sessions |
| **Total Raw Bars** | 39,766 val bars (77,511 total) | 9,598 bars | 87,109 bars |
| **End-of-Day Bars (Unlabelled)** | N/A (excluded from val folds) | 126 bars (1.31%) | 1,263 bars (1.45%) |
| **Data Gap Bars (Missing features)**| N/A (excluded from val folds) | 0 bars (0.00%) | 2,540 bars (2.92%) |
| **Labelled Evaluated Bars** | **39,766 bars** (100.0%) | **9,472 bars** (100.0%) | 49,238 bars |
| **Rule V1 (10% Target) Signals** | 3,978 signals (10.00%) | 288 signals (3.04%) | 4,266 signals (8.66%) |
| **Rule V1 No-Signal** | 35,788 bars (90.00%) | 9,184 bars (96.96%) | 44,972 bars (91.34%) |
| **Rule V2 (30% Target) Signals** | 11,930 signals (30.00%) | 1,827 signals (19.29%) | 13,757 signals (27.94%) |
| **Rule V2 No-Signal** | 27,836 bars (70.00%) | 7,645 bars (80.71%) | 35,481 bars (72.06%) |
| **Rule V3 (Always-Call) Signals** | **39,766 signals (100.00%)** | **9,472 signals (100.00%)** | **49,238 signals (100.00%)** |
| **Rule V3 No-Signal (on valid bars)**| **0 bars (0.00%)** | **0 bars (0.00%)** | **0 bars (0.00%)** |

*Note: In Rule V3 Always-Call, the only No-Signal bars across the entire trading calendar are the 126 End-of-Day bars (15:55-16:00, which have no subsequent 5-minute bar in the session) and any rare upstream exchange data outages.*

---

## 3. Disaggregation of No-Signal Reasons

### Under Rule V2 (30% Target):
- **Inside Band ($q_{lo} < p_{cal} < q_{hi}$)**:
  - OOF: 27,836 bars (70.00%)
  - Holdout: 7,512 bars (79.31%)
- **Margin Guard Filter**:
  - OOF: 0 bars (quantiles defined the margin)
  - Holdout: 133 bars (1.40%)
- **End-of-Day**:
  - Holdout: 126 bars (1.31% of raw bars)
- **Data Gap**:
  - Holdout: 0 bars

### Under Rule V3 (Always-Call):
- **Inside Band**: 0 bars (Eliminated by Owner Decision).
- **Margin Guard**: 0 bars (Eliminated).
- **Exact Tie ($p_{cal} = 0.50000000$)**: 0 bars observed in both OOF and Holdout.
- **End-of-Day**: 126 bars in Holdout (1 per session, 15:55-16:00).
- **Data Gap**: 0 bars.
