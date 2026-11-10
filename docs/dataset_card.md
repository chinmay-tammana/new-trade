# Dataset Card: Goldman Sachs Intraday Hourly OHLCV with Sector & Market Context

## 1. Dataset Overview
- **Dataset Name**: Goldman Sachs (GS) 1-Hour Intraday Market Dataset with SPY & XLF Context
- **Primary Asset**: Goldman Sachs Group, Inc. (`GS`, NYSE)
- **Context Assets**: 
  - SPDR S&P 500 ETF Trust (`SPY`) - Broad market benchmark
  - Financial Select Sector SPDR Fund (`XLF`) - Direct banking & financial sector peer benchmark
- **Collection Source**: Alpaca Market Data API (SIP full consolidated tape feed).
- **Data Terms & Licensing**: Alpaca Market Data Basic Tier terms of service. For personal, educational, and academic research purposes only. Commercial redistribution of raw tick/minute parquet feeds is prohibited.
- **Collection Period**: October 12, 2022 through November 10, 2026 (4 chronological years + 30-day initial historical buffer).
- **Unit of Observation**: A single session-anchored 1-hour regular trading session candle (09:30 to 16:00 ET).
- **Session Convention**:
  - 09:30 to 10:29 ET -> Bar timestamp `09:30` (60 min)
  - 10:30 to 11:29 ET -> Bar timestamp `10:30` (60 min)
  - 11:30 to 12:29 ET -> Bar timestamp `11:30` (60 min)
  - 12:30 to 13:29 ET -> Bar timestamp `12:30` (60 min)
  - 13:30 to 14:29 ET -> Bar timestamp `13:30` (60 min)
  - 14:30 to 15:29 ET -> Bar timestamp `14:30` (60 min)
  - 15:30 to 15:59 ET -> Bar timestamp `15:30` (30 min, session close)
  - Normal full trading days contain exactly 7 bars. Scheduled half-days (e.g. day after Thanksgiving, Christmas Eve) terminate early (e.g. 13:00 close with 4 bars).

---

## 2. Dataset Size and Structure
- **Raw Observations**: 421,893 one-minute raw bars per symbol.
- **Aggregated Sample Size**: Approximately 7,148 session-anchored 1-hour candles for GS, fully inner-joined with SPY and XLF.
- **Grouping**: Grouped naturally by trading day (`date`), with session timestamps preserving chronological intraday order.

---

## 3. Target Definition and Block Class Balance
- **Target Horizon**: Next immediate 1-hour bar within the same regular trading day.
- **Target Formulation**:
  \[
  fwd_t = \frac{Close_{t+1}}{Close_t} - 1
  \]
  \[
  y_t = \begin{cases} 1, & \text{if } fwd_t > 0 \\ 0, & \text{if } fwd_t \le 0 \end{cases}
  \]
- **Decisive Threshold**: \(|fwd_t| > 0.0010\) (10 basis points).
- **Overnight Treatment**: The final bar of each trading day (15:30 ET) and any session preceding a holiday or weekend has \(fwd_t = \text{NaN}\) and is strictly unlabelled. The system never holds or evaluates overnight gap risk.
- **Class Balance (Base Block Only)**: Evaluated strictly within the Base training block (~3,000 bars) to maintain test-set discipline.

---

## 4. Privacy, Fairness, and Ethical Review
- **Personal Data**: Zero Personally Identifiable Information (PII). All series consist of public, anonymized market transactions.
- **Fairness & Demographic Parity**: Inapplicable (financial asset pricing problem).
- **Target Proxies**: No unintended target proxies exist; all features are computed strictly from data at or prior to the candle close \(t\).

---

## 5. Known Data Limitations
1. **Unadjusted Cash Dividends**: Raw pricing data is unadjusted (`adjustment=raw`), causing quarterly cash dividends to appear as small overnight gaps at the market open. Since overnight moves are explicitly filtered from forward labels (\(fwd = \text{NaN}\)), dividends do not contaminate training labels.
2. **Consolidated SIP vs. Live Broker Feeds**: Alpaca's SIP feed incorporates consolidated volume across all US exchanges (CTA and UTP plans). In live trading with free-tier broker APIs (e.g., IEX-only feeds), observed volume may represent a sub-sample of consolidated volume; relative volume (`rvol`) is flagged for ablation.
3. **Session Anchoring Discrepancy**: Standard third-party retail charting platforms (e.g. standard TradingView 1h default) frequently anchor 1h candles at 09:00 or 10:00, blending pre-market minutes with regular hours. This dataset strictly anchors at 09:30 to match true exchange auction open times.
