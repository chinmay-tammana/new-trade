# Data Dictionary: M5 (5-Minute) Features and Targets

All features are causal, stationary, and leak-free. Truncation invariance verified within $10^{-6}$.

| Feature Group | Column | Description | Computation Formula / Source |
| :--- | :--- | :--- | :--- |
| **Returns** | `ret_1` | 5-minute return | $close[t] / close[t-1] - 1$ |
| | `ret_2` | 10-minute return | $close[t] / close[t-2] - 1$ |
| | `ret_3` | 15-minute return | $close[t] / close[t-3] - 1$ |
| | `ret_6` | 30-minute return | $close[t] / close[t-6] - 1$ |
| | `ret_12` | 60-minute return | $close[t] / close[t-12] - 1$ |
| | `ret_since_open` | Session return | $close[t] / open_{\text{session}} - 1$ |
| | `gap_prev_close` | Overnight gap | $open_{\text{session}} / close_{\text{prev\_day}} - 1$ |
| | `ret_1d` | Prior 1-day return | $close_{\text{prev\_day}} / close_{\text{prev\_day-1}} - 1$ |
| | `ret_5d` | Prior 5-day return | $close_{\text{prev\_day}} / close_{\text{prev\_day-5}} - 1$ |
| | `ret_20d` | Prior 20-day return | $close_{\text{prev\_day}} / close_{\text{prev\_day-20}} - 1$ |
| **Momentum** | `rsi_14` | Normalized Wilder's RSI | $\text{RSI}(14) / 100.0 \in [0, 1]$ |
| | `macd_diff_norm` | MACD histogram normalized | $(\text{MACD}(12,26) - \text{Signal}(9)) / close[t]$ |
| | `bb_pct_b` | Bollinger %B | $(close - \text{lower}) / (\text{upper} - \text{lower})$ over 20 bars |
| | `bb_width` | Bollinger Bandwidth | $(\text{upper} - \text{lower}) / \text{sma20}$ |
| **Volatility** | `atr_14_norm` | ATR normalized by close | $\text{ATR}(14) / close[t]$ |
| | `rolling_std_12` | 1-hour return volatility | Rolling std of 5-min returns over 12 bars |
| | `rolling_std_48` | 4-hour return volatility | Rolling std of 5-min returns over 48 bars |
| **Intrabar** | `realized_vol_intrabar` | Intrabar realized volatility | $\sqrt{\sum_{i=1}^5 r_{1m,i}^2}$ from 1-min returns inside bar |
| | `signed_volume_ratio` | Intrabar signed volume | $\sum (\text{sign}(r_{1m}) \times v_{1m}) / \sum v_{1m}$ |
| | `up_minute_ratio` | Positive minute share | Fraction of 1-min bars with $close > open$ |
| | `clv` | Close Location Value | $((close - low) - (high - close)) / (high - low)$ |
| | `trade_share_intrabar` | Trade minute density | $n_{\text{minutes\_with\_trades}} / 5.0$ |
| **Volume** | `rvol_tod` | Relative volume by time-of-day | $v[t] / \text{mean}(v_{\text{tod}} \text{ over past 20 sessions})$ |
| **Position** | `dist_session_vwap` | Distance to session VWAP | $(close - \text{session\_vwap}) / \text{ATR}$ |
| | `dist_session_high` | Distance to session high | $(close - \text{high}_{\text{session\_so\_far}}) / \text{ATR}$ |
| | `dist_session_low` | Distance to session low | $(close - \text{low}_{\text{session\_so\_far}}) / \text{ATR}$ |
| | `dist_prev_high` | Distance to prior day high | $(close - \text{high}_{\text{prev\_day}}) / \text{ATR}$ |
| | `dist_prev_low` | Distance to prior day low | $(close - \text{low}_{\text{prev\_day}}) / \text{ATR}$ |
| | `dist_prev_close` | Distance to prior day close | $(close - \text{close}_{\text{prev\_day}}) / \text{ATR}$ |
| | `or_pos_30m` | Opening range position | Position relative to 09:30-10:00 high/low range |
| **Time** | `mins_since_open` | Minutes from 09:30 | $(hour - 9) \times 60 + (min - 30)$ |
| | `mins_to_close` | Minutes to 16:00 | $390 - \text{mins\_since\_open}$ |
| | `tod_sin`, `tod_cos` | Cyclical time coordinates | $\sin, \cos(2\pi \times \text{mins\_since\_open} / 390)$ |
| **Context** | `{sym}_ret_1`, `_6` | Context symbol returns | 5-min and 30-min returns for SPY, XLF, JPM, MS, BAC, TLT |
| | `gs_minus_spy_1`, `_6` | Equity index spread | $ret_{\text{GS}} - ret_{\text{SPY}}$ |
| | `gs_minus_peer_1`, `_6`| Peer bank basket spread | $ret_{\text{GS}} - \text{mean}(ret_{\text{JPM}}, ret_{\text{MS}}, ret_{\text{BAC}})$ |
| **Targets** | `fwd` | Next 5-minute forward return | $close[t+1] / close[t] - 1$ (same session only) |
| | `y` | Primary directional label | $1 \text{ if } fwd > 0 \text{ else } 0$ (NaN on EOD) |
