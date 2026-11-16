# Problem Definition: 5-Minute Directional Trading Indicator (M5)

## 1. Executive Summary & Objective
At the request of the Teaching Assistant (TA) on 2026-11-16, the AI trading indicator was transitioned from a 1-hour decision horizon to a **5-minute operational horizon** ($H1_{5m}$).
The goal of the system is to predict the directional movement ($y \in \{0, 1\}$, where $y=1$ represents $close[t+1] > close[t]$) of Goldman Sachs Group Inc. (NYSE: `GS`) for the immediately following 5-minute bar of the regular US trading session (09:30 to 16:00 ET).

## 2. Rationale for Feed Architecture (IEX vs Delayed SIP)
Under the 1-hour system (V3.5), delayed SIP data with a 16-minute delivery lag was utilized. In a 5-minute operational framework, a 16-minute delivery delay exceeds the 5-minute prediction horizon by 3.2x, rendering delayed SIP mathematically and operationally obsolete. Consequently:
- **IEX 1-Minute Aggregated Feed** is used for both historical training and live real-time operational signaling.
- Bars are aggregated causally via `aggregate_1min_to_5min` with zero lookahead.
- Minute data received at bar close ($t_{\text{close}} + 8\text{s}$ to $25\text{s}$) is persisted into `raw_minutes` for bit-for-bit forensic replay.

## 3. High-Confidence Selective Abstention
Consistent with quantitative trading principles, the system does not force continuous predictions. Under the frozen decision rule ($c = 0.10$):
- Predictions are generated only when calibrated probability $p_{\text{cal}} \ge q_{\text{hi}}$ (Up) or $p_{\text{cal}} \le q_{\text{lo}}$ (Down).
- In neutral ranges ($q_{\text{lo}} < p_{\text{cal}} < q_{\text{hi}}$), during the final bar of the trading day (15:55-16:00 close), or upon data gaps, the indicator explicitly abstains with `No signal`.
