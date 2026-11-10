# Dataset Card: Goldman Sachs Intraday Hourly OHLCV with Sector, Peer, Rates & Volatility Context (V2)

## 1. Dataset Overview
- **Dataset Name**: Goldman Sachs (`GS`) 8-Year Session-Anchored 1-Hour Intraday Dataset with Multi-Asset Context
- **Primary Asset**: The Goldman Sachs Group, Inc. (`GS`, NYSE)
- **Context Assets**: 
  - **Market & Sector**: `SPY` (S&P 500 ETF), `XLF` (Financial Select Sector SPDR)
  - **Direct Peers**: `JPM` (JPMorgan Chase), `MS` (Morgan Stanley), `BAC` (Bank of America)
  - **Macro Rates & Volatility**: `TLT` (iShares 20+ Year Treasury Bond ETF), `VIXY` (ProShares VIX Short-Term Futures ETF)
- **Data Source**: Alpaca Market Data API (SIP consolidated tape feed).
- **Data Terms & Licensing**: Educational, academic, and paper trading research only under Alpaca Market Data Terms of Service. Commercial redistribution of raw tick/minute parquet feeds is strictly prohibited.
- **Span**: October 15, 2018 through November 9, 2026 (8 chronological years + 30-day buffer up to frozen end date `2026-11-09`).
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

## 2. Dataset Size and Synchronization
- **Raw Minute Bars Ingested**: Over 8.5 million 1-minute bars across 8 symbols.
- **Hourly Bar Counts**:
  - `GS`: 14,101 session-anchored 1h bars
  - `SPY`: 14,144 bars (0 lost in join, 0.00%)
  - `XLF`: 14,139 bars (0 lost in join, 0.00%)
  - `JPM`: 14,172 bars (0 lost in join, 0.00%)
  - `MS`: 14,153 bars (10 lost in join, 0.07%)
  - `BAC`: 14,171 bars (6 lost in join, 0.04%)
  - `TLT`: 14,185 bars (5 lost in join, 0.04%)
  - `VIXY`: 14,177 bars (5 lost in join, 0.04%)
- **Total Synchronized 1h Bars (Inner Join)**: **14,084 bars** (Zero context symbol lost > 2% of GS bars).
- **Data Quality Audit**: 0 NaNs, 0 zero volume, 0 duplicate timestamps, 100% monotonic increasing ordering.

---

## 3. Special Asset Considerations
- **VIXY (VIX Short-Term Futures ETF)**: Operates on rolling 1st and 2nd month Cboe VIX futures contracts. It is subject to structural negative roll yield (contango drag) over multi-month horizons. For our 1-hour to multi-hour intraday prediction tasks, short-term returns capture instantaneous implied volatility spikes without being distorted by multi-week roll decay.
- **TLT (Treasury Duration)**: Reflects interest rate expectations and discount rate shocks directly affecting bank net interest margin (NIM) and equity valuations.

---

## 4. Privacy, Fairness, and Ethical Review
- **Personal Data**: Zero Personally Identifiable Information (PII). All series consist of public, anonymized market transactions.
- **Fairness & Demographic Parity**: Inapplicable (financial asset pricing problem).
- **Causality & Look-Ahead**: All features and intrabar metrics are strictly causal (information filtration at or before bar close $t$).

---

## 5. Known Data Limitations
1. **Unadjusted Cash Dividends**: Raw pricing data is unadjusted (`adjustment=raw`), causing quarterly cash dividends to appear as small overnight gaps at market open. Overnight returns are explicitly censored from intraday training labels ($fwd = \text{NaN}$).
2. **Consolidated SIP vs. Live Broker Feeds**: Alpaca's SIP feed incorporates consolidated volume across all US exchanges (CTA and UTP plans). In live trading with free-tier broker APIs (e.g. IEX-only feeds), observed volume represents a fraction of consolidated volume.
3. **Session Anchoring Discrepancy**: Standard retail charting platforms (e.g. standard TradingView 1h default) frequently anchor 1h candles at 09:00 or 10:00, blending pre-market minutes with regular hours. This dataset strictly anchors at 09:30 to match true exchange auction open times.
