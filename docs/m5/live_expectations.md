# Live Statistical Expectations: M5 (5-Minute) System (Always-Call Rule)

**Updated**: 2026-11-16  
**Status**: ACTIVE & OPERATIONAL (Rule v3 Always-Call)  

---

## 1. Intraday Frequency & Signal Counts (Always-Call Rule v3)
- **Total 5-Minute Intervals**: Exactly 78 bars per standard trading day (09:30 to 15:55 start).
  - **77 intervals eligible for forward predictions** (09:35 to 15:55 delivery).
  - **1 interval reserved for end-of-day closure** (15:55-16:00 ET produces `No signal (End of day)`).
- **Expected Directional Signals per Day**:
  - Under the deployed **Always-Call Rule** ($c = 1.0$), **every eligible bar gets an Up or Down call** ($\approx 75 - 77$ directional calls per full session).
  - Recent completed sessions (Rule v3):
    - 2026-11-09: 77 calls (36 Up, 41 Down, 1 No-Sig, 98.7% coverage)
    - 2026-11-10: 76 calls (38 Up, 38 Down, 1 No-Sig, 98.7% coverage)
    - 2026-11-11: 77 calls (35 Up, 42 Down, 1 No-Sig, 98.7% coverage)
    - 2026-11-12: 75 calls (39 Up, 36 Down, 1 No-Sig, 98.7% coverage)
    - 2026-11-13: 73 calls (31 Up, 42 Down, 1 No-Sig, 98.6% coverage)

---

## 2. Statistical Sample Size & Daily Confidence Interval Width
- For a single trading session with $n \approx 73$ signals, sampling variance is substantial:
  - If accuracy on that day is 57.5% (42/73), the 95% Wilson confidence interval is **[46.1%, 68.2%]** (width $\approx \pm 11.1\%$).
  - Even if accuracy on that day is 53.4% (39/73), the 95% Wilson CI is **[42.0%, 64.5%]**.
- **Key Takeaway**: Daily session scoring is an operational health check, not statistical proof of alpha. Evaluating indicator skill requires aggregation over tens of sessions ($n > 1,000$).

---

## 3. Systematic Performance Profile (OOF & Holdout)

| Horizon / Tier | Sample Size | Directional Accuracy | 95% Wilson CI | Net Return @ 2bps | Permutation p-value |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **OOF (All Calls)** | 39,766 | 55.60% | [55.11%, 56.09%] | +0.06 bps | 0.00020 |
| **OOF (STRONG)** | 9,942 | 63.58% | [62.63%, 64.52%] | +3.43 bps | < 0.0001 |
| **Holdout (All Calls)** | 9,472 | 53.39% | [52.38%, 54.39%] | -0.82 bps | 0.00020 (Beats Base Rate 50.67%) |
| **Holdout (STRONG)** | 1,393 | 60.01% | [57.42%, 62.56%] | +1.57 bps | < 0.0001 |
| **Holdout (WEAK)** | 2,807 | 50.34% | [48.49%, 52.19%] | -3.82 bps | 0.48 (Close to a coin flip) |

*Caveat: Research holdout data has already been inspected; reported metrics are descriptive.*
