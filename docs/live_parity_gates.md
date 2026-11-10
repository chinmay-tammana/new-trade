# Live Feed Parity Verification Gates & Empirical Results

This document formalizes the pre-registered parity gates and records the empirical comparison between the real-time **IEX feed** (`feed="iex"`) and the historical **SIP consolidated feed** (`feed="sip"`) over the last 60 completed trading days up to `freeze_date` (August 17, 2026 to November 9, 2026).

---

## 1. Rationale and Operational Context

Free consolidated SIP data is subject to a 15-minute delay on broker APIs (e.g., Alpaca free tier). Under a 15-minute delay, an hourly signal for the candle closing at 10:30 ET would only become available at 10:45 ET, missing the market open of candle $t+1$ and making real-time execution impossible.

The live production system therefore utilizes the **IEX feed end-to-end** (`feed="iex"`), including historical rolling window backfill, to eliminate all feed lag while keeping relative volume and price metrics internally consistent.

---

## 2. Pre-Registered Parity Acceptance Gates & Empirical Verdicts

| Gate | Criterion Description | Threshold | Empirical Result (60 Days / 420 Bars) | Verdict |
| :--- | :--- | :--- | :--- | :--- |
| **Gate (a)** | **Direction Signal Agreement** | $\ge \mathbf{85.0\%}$ | **97.44%** (76 / 78 active signals agreed) | **PASS** |
| **Gate (b)** | **Three-Class Overall Agreement** | $\ge \mathbf{75.0\%}$ | **72.14%** (303 / 420 bars agreed) | **FAIL** |
| **Gate (c)** | **Missing Minute Bar Limit (`GS`)** | $\le \mathbf{5.0\%}$ | **31.08%** missing session minutes | **FAIL** |

**Artifact References**:
- Feature Parity Data Table: `results/tables/live_parity.csv`
- Parity Audit Visualizations: `results/figures/live_parity.png`

---

## 3. Detailed Empirical Analysis

### 3.1 Direction Agreement (Gate a: PASS at 97.44%)
When both feeds produced an active trading signal ($p \ge q_{\text{hi}}$ or $p \le q_{\text{lo}}$), the two feeds agreed on the direction **97.44%** of the time. The directional bias of the underlying price movements is virtually identical across both venues.

### 3.2 Three-Class Agreement (Gate b: FAIL at 72.14%)
Across all 420 bars, three-class agreement reached 72.14% (below the 75.0% threshold). This discrepancy is driven by boundary cases: because IEX has lower volume and sparse minutes, predictions on IEX bars have slightly less conviction and drift toward the neutral zone, producing `No Signal` on bars where SIP triggered a weak signal.

### 3.3 Microstructure & Missing Minutes (Gate c: FAIL at 31.08%)
- **Data Completeness by Asset**:
  - `SPY`: 0.00% missing (trades every minute on IEX).
  - `BAC`: 0.80% missing.
  - `XLF`: 2.86% missing.
  - `JPM`: 7.68% missing.
  - `TLT`: 10.01% missing.
  - `MS`: 21.70% missing.
  - `GS`: **31.08% missing**.
  - `VIXY`: **61.49% missing**.
- **Root Cause**: IEX is an individual alternative trading system / lit exchange executing approximately 2% to 3% of total US equity market share. High-volume index ETFs like SPY execute trades constantly on IEX. In contrast, institutional stocks like `GS` and low-liquidity derivative ETFs like `VIXY` only print bars when institutional routing orders specifically interact with the IEX book.
- **Impact on Features**:
  - Price-based returns (`ret_1`..`ret_14`), `rsi`, `macd`, `bb_pos` maintain Spearman correlations **$> 0.99$** with median differences $< 3$ bps.
  - Intrabar volume microstructure (`up_minute_ratio`, `signed_volume_ratio`) exhibit low correlation ($0.19 - 0.24$) due to trade sparsity.

---

## 4. Fallback Protocol Triggered

In accordance with Section 2 of this specification and `instructions/11_v3_live.md`:
1. **NO RETRAINING OR RETUNING**: Models and thresholds remain frozen.
2. **DEFAULT SAFETY LOCK**: The live server sets `live_signals_enabled = False` by default in `src/live/server.py`.
3. **UI WARNING BANNER**: The frontend displays a prominent status badge: `SIGNALS DISABLED: IEX Parity Gate Failed`.
4. **DATA GAP HANDLING**: If a context symbol lacks any trade prints in an hour, the signal emits `No signal (data gap)`.
