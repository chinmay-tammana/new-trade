# Data Dictionary: Raw and Engineered Features

This document details all columns present in the raw aggregated dataset and engineered feature matrix.

---

## 1. Raw Session-Anchored 1-Hour Bar Columns

| Column | Data Type | Units / Range | Description |
| :--- | :--- | :--- | :--- |
| `timestamp` | `datetime64[ns, America/New_York]` | 2022-10 to 2026-11 | Session-anchored candle start timestamp (09:30, 10:30, ..., 15:30 ET). |
| `open` | `float64` | USD ($) | Price of the first executed trade within the 1-hour interval. |
| `high` | `float64` | USD ($) | Maximum trade price executed during the interval. |
| `low` | `float64` | USD ($) | Minimum trade price executed during the interval. |
| `close` | `float64` | USD ($) | Price of the final executed trade within the interval. |
| `volume` | `float64` | Share count | Cumulative trading volume across all consolidated SIP venues. |
| `trade_count`| `float64` | Count | Cumulative number of distinct trades executed during the interval. |
| `vwap` | `float64` | USD ($) | Volume-Weighted Average Price across all 1-minute bars in the hour. |
| `symbol` | `string` | GS, SPY, XLF | Ticker identifier. |

---

## 2. Engineered Stationary Feature Columns

All engineered features are strictly causal (computed using data at or before candle close \(t\)) and stationary (no raw price levels).

| Feature Name | Data Type | Formula / Definition | Lookback | Description |
| :--- | :--- | :--- | :--- | :--- |
| `ret_1` | `float64` | \(\frac{Close_t}{Close_{t-1}} - 1\) | 1 bar (1h) | 1-bar simple percentage return of GS. |
| `ret_2` | `float64` | \(\frac{Close_t}{Close_{t-2}} - 1\) | 2 bars (2h) | 2-bar cumulative return. |
| `ret_3` | `float64` | \(\frac{Close_t}{Close_{t-3}} - 1\) | 3 bars (3h) | 3-bar cumulative return. |
| `ret_6` | `float64` | \(\frac{Close_t}{Close_{t-6}} - 1\) | 6 bars (~1 day) | 6-bar cumulative return. |
| `ret_14` | `float64` | \(\frac{Close_t}{Close_{t-14}} - 1\) | 14 bars (~2 days) | 14-bar cumulative return. |
| `gap` | `float64` | \(\frac{Open_t}{Close_{t-1}} - 1\) | 1 bar | Overnight opening jump (nonzero only on first bar of session). |
| `rsi` | `float64` | Wilder's RSI (14) | 14 bars | Relative Strength Index bounded in \([0, 100]\). |
| `macd_n` | `float64` | \(\frac{\text{EMA}_{12} - \text{EMA}_{26}}{Close_t}\) | 26 bars | Moving Average Convergence Divergence normalized by closing price. |
| `macd_hist_n` | `float64` | \(\frac{\text{MACD} - \text{Signal}_{9}}{Close_t}\) | 35 bars | Normalized MACD histogram oscillator. |
| `bb_pos` | `float64` | \(\frac{Close_t - \text{Lower}_{20,2}}{\text{Upper}_{20,2} - \text{Lower}_{20,2}}\) | 20 bars | Bollinger Band %B position (typically bounded \([0, 1]\)). |
| `bb_width` | `float64` | \(\frac{\text{Upper}_{20,2} - \text{Lower}_{20,2}}{\text{SMA}_{20}}\) | 20 bars | Bollinger Band bandwidth (normalized dispersion metric). |
| `atr_n` | `float64` | \(\frac{\text{ATR}_{14}}{Close_t}\) | 14 bars | Average True Range normalized by current price. |
| `vol_20` | `float64` | Rolling 20-bar std of `ret_1` | 20 bars | Intraday historical realized volatility. |
| `rvol` | `float64` | \(\frac{Volume_t}{\text{SMA}_{20}(Volume)}\) | 20 bars | Relative volume compared to 20-bar rolling average. |
| `hour` | `float64` | \(\text{Hour} + \frac{\text{Minute}}{60}\) | Current | Continuous intraday decimal time (e.g. 9.5 for 09:30, 10.5 for 10:30). |
| `dow` | `int64` | Day of week \([0, 4]\) | Current | Day of week integer (0 = Monday, 4 = Friday). |
| `is_first_bar`| `int64` | \(\mathbb{I}(\text{hour} == 9.5)\) | Current | Indicator for opening session auction bar. |
| `spy_ret_1` | `float64` | \(\frac{Close^{\text{SPY}}_t}{Close^{\text{SPY}}_{t-1}} - 1\) | 1 bar | 1-bar return of SPY (S&P 500 benchmark). |
| `spy_ret_3` | `float64` | \(\frac{Close^{\text{SPY}}_t}{Close^{\text{SPY}}_{t-3}} - 1\) | 3 bars | 3-bar return of SPY. |
| `xlf_ret_1` | `float64` | \(\frac{Close^{\text{XLF}}_t}{Close^{\text{XLF}}_{t-1}} - 1\) | 1 bar | 1-bar return of XLF (Financial sector benchmark). |
| `xlf_ret_3` | `float64` | \(\frac{Close^{\text{XLF}}_t}{Close^{\text{XLF}}_{t-3}} - 1\) | 3 bars | 3-bar return of XLF. |
| `rel_ret_1` | `float64` | `ret_1` - `spy_ret_1` | 1 bar | Sector/Stock relative alpha return over broad market. |

---

## 3. Forward Target & Metadata Columns

| Column | Data Type | Domain | Description |
| :--- | :--- | :--- | :--- |
| `fwd` | `float64` | Real / NaN | Forward 1-bar simple return (\(\frac{Close_{t+1}}{Close_t} - 1\)). NaN on final bar of day. |
| `y` | `int64` | \(\{0, 1\}\) | Binary target: 1 if \(fwd > 0\), 0 if \(fwd \le 0\). |
| `decisive` | `bool` | True / False | Filter flag: True if \(|fwd| > 0.0010\) (10 bps threshold). |
| `block` | `string` | Base, FT, Val, Test | Chronological partition identifier. |
