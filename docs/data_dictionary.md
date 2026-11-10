# Data Dictionary: Raw, Engineered Features, Targets & Insight Heads (V2 Upgrade)

This document details all columns present in the raw aggregated dataset, engineered feature matrix, multi-horizon targets, and insight outputs.

---

## 1. Raw Session-Anchored 1-Hour Bar Columns

| Column | Data Type | Units / Range | Description |
| :--- | :--- | :--- | :--- |
| `timestamp` | `datetime64[ns, America/New_York]` | 2018-10 to 2026-11 | Session-anchored candle start timestamp (09:30, 10:30, ..., 15:30 ET). |
| `open` | `float64` | USD ($) | Price of the first executed trade within the 1-hour interval. |
| `high` | `float64` | USD ($) | Maximum trade price executed during the interval. |
| `low` | `float64` | USD ($) | Minimum trade price executed during the interval. |
| `close` | `float64` | USD ($) | Price of the final executed trade within the interval. |
| `volume` | `float64` | Share count | Cumulative trading volume across all consolidated SIP venues. |
| `trade_count`| `float64` | Count | Cumulative number of distinct trades executed during the interval. |
| `vwap` | `float64` | USD ($) | Volume-Weighted Average Price across all 1-minute bars in the hour. |
| `symbol` | `string` | GS, SPY, XLF, JPM, MS, BAC, TLT, VIXY | Ticker identifier. |

---

## 2. Engineered Stationary Feature Columns (By Feature Group)

All engineered features are strictly causal (computed using data at or before candle close $t$) and stationary (no raw un-normalized price levels).

### Group 1: Returns & Momentum (`returns`, `momentum`)
- `ret_1`: 1-bar simple return of GS ($\frac{Close_t}{Close_{t-1}} - 1$).
- `ret_2`: 2-bar cumulative return ($\frac{Close_t}{Close_{t-2}} - 1$).
- `ret_3`: 3-bar cumulative return ($\frac{Close_t}{Close_{t-3}} - 1$).
- `ret_6`: 6-bar cumulative return ($\frac{Close_t}{Close_{t-6}} - 1$, ~1 trading day).
- `ret_14`: 14-bar cumulative return ($\frac{Close_t}{Close_{t-14}} - 1$, ~2 trading days).
- `gap`: Overnight opening jump ($\frac{Open_t}{Close_{t-1}} - 1$ on opening candle, 0.0 intraday).
- `rsi`: Wilder's 14-period Relative Strength Index bounded in $[0, 100]$.
- `macd_n`: MACD line normalized by current close: $\frac{\text{EMA}_{12}(Close) - \text{EMA}_{26}(Close)}{Close_t}$.
- `macd_hist_n`: Normalized MACD histogram oscillator.

### Group 2: Volatility & Dispersion (`vol`)
- `atr_n`: 14-period Average True Range normalized by closing price ($\frac{\text{ATR}_{14}}{Close_t}$).
- `vol_20`: Rolling 20-bar sample standard deviation of `ret_1`.
- `bb_pos`: Bollinger Band %B position: $\frac{Close_t - \text{Lower}_{20, 2}}{\text{Upper}_{20, 2} - \text{Lower}_{20, 2}}$.
- `bb_width`: Bollinger Band normalized bandwidth: $\frac{\text{Upper}_{20, 2} - \text{Lower}_{20, 2}}{\text{SMA}_{20}(Close)}$.

### Group 3: Volume Dynamics (`volume`)
- `rvol`: Relative volume compared to 20-bar moving average volume: $\frac{Volume_t}{\text{SMA}_{20}(Volume)}$.

### Group 4: Time & Calendar Context (`time`)
- `hour`: Continuous intraday decimal time (e.g., 9.5 for 09:30, 10.5 for 10:30, 15.5 for 15:30).
- `dow`: Day of week integer ($0 = \text{Monday}, \dots, 4 = \text{Friday}$).
- `is_first_bar`: Binary indicator $\mathbb{I}(\text{hour} == 9.5)$ for opening auction bar.
- `bars_since_open`: Integer count of bars elapsed in current session ($0, 1, \dots, 6$).
- `bars_to_close`: Integer count of bars remaining in session before market close ($6, 5, \dots, 0$).

### Group 5: Market & Sector Benchmarks (`market`)
- `spy_ret_1`, `spy_ret_3`: 1-bar and 3-bar returns of S&P 500 ETF (`SPY`).
- `xlf_ret_1`, `xlf_ret_3`: 1-bar and 3-bar returns of Financial Select Sector SPDR (`XLF`).
- `rel_ret_spy`: Relative return of GS over SPY (`ret_1` - `spy_ret_1`).
- `rel_ret_xlf`: Relative return of GS over XLF (`ret_1` - `xlf_ret_1`).

### Group 6: Banking Peers, Rates & Implied Volatility (`peers`)
- `jpm_ret_1`, `jpm_ret_3`: 1-bar and 3-bar returns of JPMorgan Chase (`JPM`).
- `ms_ret_1`, `ms_ret_3`: 1-bar and 3-bar returns of Morgan Stanley (`MS`).
- `bac_ret_1`, `bac_ret_3`: 1-bar and 3-bar returns of Bank of America (`BAC`).
- `tlt_ret_1`, `tlt_ret_3`: 1-bar and 3-bar returns of 20+ Year Treasury Bond ETF (`TLT`).
- `vixy_ret_1`, `vixy_ret_3`: 1-bar and 3-bar returns of VIX Short-Term Futures ETF (`VIXY`).
- `gs_minus_peers`: GS 1-bar return minus equal-weighted average return of banking peers: $\text{ret\_1} - \frac{\text{jpm\_ret\_1} + \text{ms\_ret\_1} + \text{bac\_ret\_1}}{3}$.

### Group 7: Intrabar 1-Minute Microstructure (`intrabar`)
*Computed from the sixty 1-minute bars comprising each hourly candle:*
- `realized_vol_1m`: Sample standard deviation of 1-minute returns within the candle.
- `up_minute_ratio`: Fraction of 1-minute bars within the hour where $Close_{1m} > Open_{1m}$.
- `clv`: Close Location Value for the hour: $\frac{(Close - Low) - (High - Close)}{High - Low} \in [-1.0, +1.0]$.
- `signed_volume_ratio`: $\frac{Vol_{\text{up}} - Vol_{\text{down}}}{TotalVol}$ where up-minute is $Close_{1m} \ge Open_{1m}$.
- `last15_ret`: Simple return over the final 15 minutes of the hour ($C_{59m} / C_{45m} - 1$).

### Group 8: Multi-Day Levels & Trend Efficiency (`levels`)
- `dist_prev_close`: Distance from previous day's close in ATR units: $\frac{Close_t - Close_{\text{prev\_day}}}{\text{ATR}_{14}}$.
- `dist_prev_high`: Distance from previous day's high in ATR units: $\frac{Close_t - High_{\text{prev\_day}}}{\text{ATR}_{14}}$.
- `dist_prev_low`: Distance from previous day's low in ATR units: $\frac{Close_t - Low_{\text{prev\_day}}}{\text{ATR}_{14}}$.
- `dist_session_vwap`: Fractional distance from current session cumulative VWAP: $\frac{Close_t - \text{VWAP}_{\text{session}}}{\text{VWAP}_{\text{session}}}$.
- `ret_1d`, `ret_5d`, `ret_20d`: Trailing 1-day, 5-day, and 20-day returns up to the previous day's regular market close.
- `ma_dist_50h`: Normalized distance from 50-bar moving average: $\frac{Close_t - \text{SMA}_{50}(Close)}{\text{SMA}_{50}(Close)}$.
- `ma_dist_200h`: Normalized distance from 200-bar moving average: $\frac{Close_t - \text{SMA}_{200}(Close)}{\text{SMA}_{200}(Close)}$.
- `efficiency_ratio_20`: Kaufman Efficiency Ratio over 20 bars: $\frac{|Close_t - Close_{t-20}|}{\sum_{i=0}^{19} |Close_{t-i} - Close_{t-i-1}|} \in [0.0, 1.0]$.

---

## 3. Multi-Horizon Directional Targets & Insight Heads

### Directional Target Heads
1. **H1 (Next 1-Hour Bar)**: $F_{t}^{\text{H1}} = \frac{Close_{t+1} - Close_t}{Close_t}$. Decisive threshold $\delta_{\text{H1}} = 10\text{ bps}$. Censors final session bar (15:30) and overnight moves ($F = \text{NaN}$).
2. **H3 (Next 3-Hour Bars)**: $F_{t}^{\text{H3}} = \frac{Close_{t+3} - Close_t}{Close_t}$. Valid only if bar $t+3$ falls on the same calendar day (bars $t \in \{0, 1, 2, 3\}$). Decisive threshold $\delta_{\text{H3}} = 10 \cdot \sqrt{3} \approx 17.32\text{ bps}$.
3. **HD (Rest-of-Day Session Close)**: $F_{t}^{\text{HD}} = \frac{Close_{\text{15:30}} - Close_t}{Close_t}$. Valid for intraday bars with at least 2 bars remaining ($t \in \{0, 1, 2, 3, 4\}$). Decisive threshold $\delta_{\text{HD}} = 10 \cdot \sqrt{bars\_to\_close}\text{ bps}$.

### Insight Heads
1. **VOL (Next-Bar High-Low Range Regressor)**:
   - Target: $\log\left(\frac{High_{t+1} - Low_{t+1}}{Close_{t+1}}\right)$.
   - Classification: Discretized into `High`, `Normal`, `Low` volatility classes based on pre-holdout OOF terciles ($q_{0.33}, q_{0.67}$).
2. **BAND (Next-Bar Quantile Return Band)**:
   - Quantile regression models predicting 10th percentile ($q = 0.10$) and 90th percentile ($q = 0.90$) of forward 1-hour return. Target coverage: ~80%.
3. **REGIME (Macro Trend & Chop Classification)**:
   - Categorical state tagged as `Uptrend`, `Downtrend`, or `Chop` derived from the sign of trailing 20-day return (`ret_20d`) and market efficiency ratio (`efficiency_ratio_20`).
