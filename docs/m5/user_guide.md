# User Guide: Interpreting the 5-Minute AI Trading Indicator (Always-Call Rule)

**Updated**: 2026-11-16  
**Status**: ACTIVE & OPERATIONAL (Rule v3 Always-Call)  

---

## 1. Overview & Deployed Rule (V4.2-ALWAYSCALL)

Following Owner Decision (2026-11-16), the 5-minute Goldman Sachs indicator operates in **Always-Call Mode**:
- **Every eligible 5-minute interval receives an explicit UP or DOWN directional call** alongside a calibrated confidence level and a confidence tier chip.
- **Expected Call Volume**: Approximately **75 to 78 calls per full trading session** (~100% directional coverage across eligible bars).
- **Allowed Abstentions Only**: A bar is marked `No signal` **strictly** for:
  1. **End-of-Day bar (15:55 to 16:00 ET)**: There is no forward 5-minute bar within the same regular trading session to score against.
  2. **Data Gaps**: Missing minute bars from upstream exchange feeds.
  3. **Exact Tie**: An exact posterior tie at $P(\text{Up}) = 0.50000000$ (statistically rare; 0 observed).

> [!WARNING]
> **IEX Microstructure Print Warning (Integrity Audit Gate 4)**:
> In the pre-registered Section A integrity audit, 5-minute directional accuracy evaluated against IEX exchange prints was **69.9%**, but drops to **51.80%** when evaluated against the Consolidated SIP tape ($\Delta = +18.1\%$). On short 5-minute horizons, IEX trades often capture local odd-lot prints and bid-ask bounces rather than true NBBO mid-price movement.

---

## 2. Confidence Tiers & How to Read Them

Because calls are emitted on every bar, conviction varies widely across the day. Calls are categorized into three empirical confidence quartiles:

| Conviction Tier | Confidence Cutoff | OOF Accuracy ($n=39,766$) | Holdout Accuracy ($n=9,472$) | Operational Meaning & Guidance |
| :--- | :--- | :--- | :--- | :--- |
| **`STRONG`** | $\ge 58.1\%$ (Top 25%) | **63.58%** [62.6%, 64.5%] | **60.01%** [57.4%, 62.6%] | **High Conviction**: Statistically significant edge. Generates $+1.57$ bps net return on holdout after 2 bps round-trip friction. Focus execution on this tier. |
| **`NORMAL`** | $52.1\% - 58.1\%$ (Middle 50%) | **53.85%** [53.2%, 54.5%] | **53.26%** [51.9%, 54.6%] | **Moderate Lean**: Directional bias aligns with short-term order flow, but edge is small ($+3.3\%$ over base rate). |
| **`WEAK`** | $\le 52.1\%$ (Bottom 25%) | **51.12%** [50.1%, 52.1%] | **50.34%** [48.5%, 52.2%] | **Close to a Coin Flip**: Barely distinguishes from pure noise ($50.3\%$ vs $50.7\%$ base rate). Negative net return after friction ($-0.82$ bps). Do NOT execute standalone discretionary trades on WEAK calls. |

*Note on Monotonicity: The tier check confirmed strict monotonicity ($\text{WEAK} \le \text{NORMAL} \le \text{STRONG}$) on both OOF and Holdout datasets.*

---

## 3. Statistical Notices & Single-Day Demonstration

- **Holdout Caveat**: The research holdout dataset (May–Nov 2026) has already been inspected during model selection and coverage tuning; all metrics reported for it are **descriptive summaries**, not a clean uncontaminated out-of-sample statistical test.
- **Daily Performance**: With $n \approx 73$ signals per day, a single session's accuracy has a wide 95% Wilson confidence interval (width $\pm 11\%$). A single day of live signals is an operational demonstration, not statistical proof of edge.
