# M5 System Coverage Trade-Off and Rule Selection

**Date**: 2026-11-16  
**Status**: APPROVED — RULE B (TRAILING 20-SESSION QUANTILES) ACTIVATED  
**Reference**: `instructions/21_coverage.md` Section B, `results/m5/tables/coverage_tradeoff.csv`

---

## 1. Coverage vs. Accuracy Trade-Off Table

Quantile thresholds derived symmetrically ($q_{lo} = \text{quantile}(p_{cal}, c/2)$, $q_{hi} = \text{quantile}(p_{cal}, 1 - c/2)$) with margin guard $\text{min\_margin} = 0.01$.

### A. Pre-Holdout Out-of-Fold (OOF) Sample (74,971 Bars, 10 Folds)

| Target $c$ | Achieved Coverage | $q_{lo}$ | $q_{hi}$ | Up Signals | Down Signals | Total Signals | Directional Accuracy [Wilson 95% CI] | Balanced Accuracy | Net Ret @ 0 bps | Net Ret @ 1 bps | Net Ret @ 2 bps | Net Ret @ 5 bps |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **5%** | 5.0% | 0.3396 | 0.6601 | 995 | 995 | 1,990 | **72.11%** [70.1%, 74.0%] | 72.11% | +9.24 bps | +8.24 bps | +7.24 bps | +4.24 bps |
| **10%** | 10.0% | 0.3728 | 0.6291 | 1,989 | 1,989 | 3,978 | **69.56%** [68.1%, 71.0%] | 69.56% | +7.91 bps | +6.91 bps | +5.91 bps | +2.91 bps |
| **20%** | 20.0% | 0.4090 | 0.5941 | 3,977 | 3,977 | 7,954 | **65.26%** [64.2%, 66.3%] | 65.26% | +5.83 bps | +4.83 bps | +3.83 bps | +0.83 bps |
| **30%** | 30.0% | 0.4296 | 0.5727 | 5,965 | 5,965 | 11,930 | **62.20%** [61.3%, 63.1%] | 62.20% | +4.35 bps | +3.35 bps | +2.35 bps | -0.65 bps |
| **40%** | 40.0% | 0.4442 | 0.5572 | 7,954 | 7,954 | 15,908 | **60.44%** [59.7%, 61.2%] | 60.44% | +3.41 bps | +2.41 bps | +1.41 bps | -1.59 bps |
| **50%** | 50.0% | 0.4556 | 0.5447 | 9,942 | 9,942 | 19,884 | **59.21%** [58.5%, 59.9%] | 59.21% | +2.82 bps | +1.82 bps | +0.82 bps | -2.18 bps |

### B. Research Holdout Evaluation (Pre-Refit Model, Different Calibrator, 9,472 Bars)

*Note: Descriptive only; held-out evaluation of pre-refit candidate model evaluated strictly once under fixed OOF thresholds.*

| Target $c$ | Achieved Coverage | $q_{lo}$ | $q_{hi}$ | Up Signals | Down Signals | Total Signals | Directional Accuracy [Wilson 95% CI] | Balanced Accuracy | Net Ret @ 0 bps | Net Ret @ 1 bps | Net Ret @ 2 bps | Net Ret @ 5 bps |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **5%** | 1.7% | 0.3396 | 0.6601 | 82 | 77 | 159 | **68.55%** [60.9%, 75.3%] | 68.50% | +8.41 bps | +7.41 bps | +6.41 bps | +3.41 bps |
| **10%** | 4.0% | 0.3728 | 0.6291 | 198 | 178 | 376 | **65.16%** [60.2%, 69.8%] | 65.20% | +6.95 bps | +5.95 bps | +4.95 bps | +1.95 bps |
| **20%** | 10.8% | 0.4090 | 0.5941 | 556 | 464 | 1,020 | **60.88%** [57.8%, 63.8%] | 60.82% | +4.88 bps | +3.88 bps | +2.88 bps | -0.12 bps |
| **30%** | 19.3% | 0.4296 | 0.5727 | 1,003 | 824 | 1,827 | **59.82%** [57.5%, 62.1%] | 59.78% | +4.10 bps | +3.10 bps | +2.10 bps | -0.90 bps |
| **40%** | 29.6% | 0.4442 | 0.5572 | 1,532 | 1,268 | 2,800 | **58.21%** [56.4%, 60.0%] | 58.15% | +3.22 bps | +2.22 bps | +1.22 bps | -1.78 bps |
| **50%** | 40.5% | 0.4556 | 0.5447 | 2,094 | 1,746 | 3,840 | **56.64%** [55.1%, 58.2%] | 56.60% | +2.48 bps | +1.48 bps | +0.48 bps | -2.52 bps |

---

## 2. Rule Selection Protocol: Rule A vs. Rule B

### Primary Test: Rule A on Last 20 Completed Sessions
- Target coverage: $c = 0.30$
- Fixed OOF Quantile Thresholds: $q_{lo} = 0.42958$, $q_{hi} = 0.57267$
- Total 5-minute bars in last 20 sessions: **1,520**
- Directional signals generated: **191** (Up: 87, Down: 104)
- **Achieved Coverage**: **12.57%**

### Fallback Activation Criterion
The protocol in `instructions/21_coverage.md` pre-registers:
> *"If the achieved coverage on the last 20 sessions is below 50% of the target (below 15%), the OOF thresholds do not transfer to the deployed model. Then use Rule B: thresholds are the (c/2) and (1 - c/2) quantiles of the final model's own p_cal over the trailing 20 completed sessions... with the same margin guard min_margin."*

- **Result**: Achieved coverage of $12.57\% < 15.00\%$.
- **Action**: **Rule B is ACTIVATED**.

---

## 3. Active Rule (Rule B) Specifications

- **Rule Version**: 2
- **Rule Type**: `rule_b` (Dynamic Trailing Quantiles)
- **Target Coverage**: 30% ($\approx 23$ signals per 78-bar session)
- **Trailing Sample**: Last 20 completed trading sessions (1,520 bars)
- **Thresholds**:
  - **$q_{lo}$**: **0.45168** (45.17%)
  - **$q_{hi}$**: **0.54528** (54.53%)
  - **Margin Guard**: $\text{min\_margin} = 0.01$ (every directional signal has displayed confidence $\ge 51.0\%$)
- **Active Rule File**: `models/m5/rule_m5_v2.json`
- **Active Pointer**: `models/m5/active_rule.json`

---

## 4. Confidence Tiers Breakdown

Confidence is defined as $p_{cal}$ for Up signals and $1 - p_{cal}$ for Down signals. Tiers are defined by the quantiles of signal confidence:
- **WEAK**: Confidence $< 0.5620$ (Bottom 25% of directional calls)
- **NORMAL**: $0.5620 \le \text{Confidence} < 0.6193$ (Middle 50% of directional calls)
- **STRONG**: Confidence $\ge 0.6193$ (Top 25% of directional calls)

### Descriptive Performance by Tier

| Sample | Tier | Side | $N$ Signals | Hits | Accuracy | Wilson 95% CI |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **OOF (Pre-Holdout)** | **STRONG** | **Pooled** | **4,772** | **3,260** | **68.32%** | [67.0%, 69.6%] |
| OOF | STRONG | Up | 2,442 | 1,673 | 68.51% | [66.6%, 70.3%] |
| OOF | STRONG | Down | 2,330 | 1,587 | 68.11% | [66.2%, 70.0%] |
| OOF | **NORMAL** | **Pooled** | **9,542** | **5,466** | **57.28%** | [56.3%, 58.3%] |
| OOF | NORMAL | Up | 4,830 | 2,765 | 57.25% | [55.8%, 58.6%] |
| OOF | NORMAL | Down | 4,712 | 2,701 | 57.32% | [55.9%, 58.7%] |
| OOF | **WEAK** | **Pooled** | **4,772** | **2,623** | **54.97%** | [53.5%, 56.4%] |
| OOF | WEAK | Up | 2,607 | 1,416 | 54.32% | [52.4%, 56.2%] |
| OOF | WEAK | Down | 2,165 | 1,207 | 55.75% | [53.6%, 57.8%] |
| **Holdout (Pre-Refit)** | **STRONG** | **Pooled** | **477** | **307** | **64.36%** | [60.0%, 68.5%] |
| Holdout | STRONG | Up | 257 | 163 | 63.42% | [57.4%, 69.1%] |
| Holdout | STRONG | Down | 220 | 144 | 65.45% | [58.9%, 71.5%] |
| Holdout | **NORMAL** | **Pooled** | **1,919** | **1,105** | **57.58%** | [55.4%, 59.8%] |
| Holdout | NORMAL | Up | 1,037 | 592 | 57.09% | [54.0%, 60.1%] |
| Holdout | NORMAL | Down | 882 | 513 | 58.16% | [54.9%, 61.4%] |
| Holdout | **WEAK** | **Pooled** | **1,242** | **658** | **52.98%** | [50.2%, 55.7%] |
| Holdout | WEAK | Up | 711 | 372 | 52.32% | [48.6%, 56.0%] |
| Holdout | WEAK | Down | 531 | 286 | 53.86% | [49.6%, 58.0%] |

> [!NOTE]
> Higher coverage increases directional trade frequency (from ~8/day to ~23/day), but dilutes the pooled hit rate from ~69.6% down to ~62.2% on OOF (~59.8% on Holdout). The STRONG tier retains a high-confidence edge (64.4%–68.3%), while the WEAK tier operates close to market break-even (53.0%–55.0%).
