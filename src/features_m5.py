"""
Feature engineering and labelling for M5 (5-minute) pipeline.
Strictly causal, stationary, leak-free.
Features pass truncation test.
"""
import datetime
import numpy as np
import pandas as pd
from configs.config import CONFIG_M5


FEATURE_GROUPS = {
    "returns": [
        "ret_1", "ret_2", "ret_3", "ret_6", "ret_12",
        "ret_since_open", "gap_prev_close",
        "ret_1d", "ret_5d", "ret_20d"
    ],
    "momentum": [
        "rsi_14", "macd_diff_norm", "bb_pct_b", "bb_width"
    ],
    "vol": [
        "atr_14_norm", "rolling_std_12", "rolling_std_48"
    ],
    "intrabar": [
        "realized_vol_intrabar", "signed_volume_ratio", 
        "up_minute_ratio", "clv", "trade_share_intrabar"
    ],
    "volume": [
        "rvol_tod"
    ],
    "position": [
        "dist_session_vwap", "dist_session_high", "dist_session_low",
        "dist_prev_high", "dist_prev_low", "dist_prev_close",
        "or_pos_30m"
    ],
    "time": [
        "mins_since_open", "mins_to_close", "tod_sin", "tod_cos"
    ],
    "context": [
        "spy_ret_1", "spy_ret_6",
        "xlf_ret_1", "xlf_ret_6",
        "jpm_ret_1", "jpm_ret_6",
        "ms_ret_1", "ms_ret_6",
        "bac_ret_1", "bac_ret_6",
        "gs_minus_spy_1", "gs_minus_spy_6",
        "gs_minus_peer_1", "gs_minus_peer_6"
    ]
}

ALL_FEATURE_COLS = [col for grp in FEATURE_GROUPS.values() for col in grp]


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    
    rs = avg_gain / (avg_loss + 1e-12)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi / 100.0 # Normalized [0, 1]


def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(com=period - 1, min_periods=period).mean()
    return atr


def make_features_m5(df_gs: pd.DataFrame, context_dfs: dict = None) -> pd.DataFrame:
    """
    Compute causal features and forward labels for 5-minute bars.
    
    Parameters:
    -----------
    df_gs: 5-minute bars for GS (sorted by timestamp ET)
    context_dfs: dict of symbol -> 5-minute bars DataFrame
    
    Returns:
    --------
    DataFrame containing all features, labels, and metadata.
    """
    df = df_gs.copy().sort_index()
    c = df["close"]
    o = df["open"]
    h = df["high"]
    l = df["low"]
    v = df["volume"]
    
    dates = df.index.date
    unique_dates = sorted(list(set(dates)))
    
    # 1. Daily Summary for Multi-day lags and Prior-day levels
    daily_records = []
    for d in unique_dates:
        mask = (dates == d)
        day_df = df[mask]
        daily_records.append({
            "date": d,
            "d_open": day_df["open"].iloc[0],
            "d_high": day_df["high"].max(),
            "d_low": day_df["low"].min(),
            "d_close": day_df["close"].iloc[-1],
            "d_volume": day_df["volume"].sum(),
        })
    daily_df = pd.DataFrame(daily_records).set_index("date").sort_index()
    daily_df["prev_close"] = daily_df["d_close"].shift(1)
    daily_df["prev_high"] = daily_df["d_high"].shift(1)
    daily_df["prev_low"] = daily_df["d_low"].shift(1)
    daily_df["ret_1d"] = daily_df["d_close"].shift(1) / daily_df["d_close"].shift(2) - 1.0
    daily_df["ret_5d"] = daily_df["d_close"].shift(1) / daily_df["d_close"].shift(6) - 1.0
    daily_df["ret_20d"] = daily_df["d_close"].shift(1) / daily_df["d_close"].shift(21) - 1.0

    # Map daily prior stats back to intraday rows
    df["date"] = dates
    df = df.join(daily_df[["prev_close", "prev_high", "prev_low", "ret_1d", "ret_5d", "ret_20d"]], on="date")
    
    # 2. Returns
    df["ret_1"] = c / c.shift(1) - 1.0
    df["ret_2"] = c / c.shift(2) - 1.0
    df["ret_3"] = c / c.shift(3) - 1.0
    df["ret_6"] = c / c.shift(6) - 1.0
    df["ret_12"] = c / c.shift(12) - 1.0
    
    # Session open return
    session_opens = df.groupby("date")["open"].transform("first")
    df["ret_since_open"] = c / session_opens - 1.0
    df["gap_prev_close"] = (session_opens / df["prev_close"]) - 1.0

    # 3. Momentum & Technicals
    df["rsi_14"] = compute_rsi(c, 14)
    
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    macd_diff = macd - signal
    df["macd_diff_norm"] = macd_diff / c

    # Bollinger Bands (20)
    sma20 = c.rolling(20, min_periods=20).mean()
    rstd20 = c.rolling(20, min_periods=20).std()
    bb_upper = sma20 + 2.0 * rstd20
    bb_lower = sma20 - 2.0 * rstd20
    df["bb_pct_b"] = (c - bb_lower) / ((bb_upper - bb_lower).replace(0, np.nan))
    df["bb_width"] = (bb_upper - bb_lower) / sma20

    # ATR(14)
    atr14 = compute_atr(h, l, c, 14)
    df["atr_14_norm"] = atr14 / c
    atr_safe = atr14.replace(0, np.nan).ffill().bfill()

    # Rolling Return Volatility
    df["rolling_std_12"] = df["ret_1"].rolling(12, min_periods=12).std()
    df["rolling_std_48"] = df["ret_1"].rolling(48, min_periods=48).std()

    # 4. Intrabar metrics (already present from aggregate_1min_to_5min or defaulted)
    if "realized_vol_intrabar" not in df.columns:
        df["realized_vol_intrabar"] = 0.001
    if "signed_volume_ratio" not in df.columns:
        df["signed_volume_ratio"] = 0.0
    if "up_minute_ratio" not in df.columns:
        df["up_minute_ratio"] = 0.5
    if "clv" not in df.columns:
        df["clv"] = 0.0
    if "trade_share_intrabar" not in df.columns:
        if "n_minutes_with_trades" in df.columns:
            df["trade_share_intrabar"] = df["n_minutes_with_trades"] / 5.0
        elif "n_minutes" in df.columns:
            df["trade_share_intrabar"] = df["n_minutes"] / 5.0
        else:
            df["trade_share_intrabar"] = 1.0

    # 5. Volume with Time-of-Day normalisation (rvol_tod: causal 20 sessions)
    df["bar_time"] = df.index.time
    # Shift daily by time-of-day: rolling mean of previous 20 sessions for same time bar
    # Using pivot on date x bar_time
    vol_piv = df.pivot(index="date", columns="bar_time", values="volume")
    vol_mean_20 = vol_piv.shift(1).rolling(20, min_periods=5).mean()
    vol_mean_melted = vol_mean_20.unstack().reset_index(name="vol_mean_20")
    df = df.reset_index().merge(vol_mean_melted, on=["bar_time", "date"], how="left").set_index("bar_start")
    df["rvol_tod"] = df["volume"] / df["vol_mean_20"].replace(0, np.nan)
    df["rvol_tod"] = df["rvol_tod"].fillna(1.0).clip(upper=10.0)

    # 6. Position Features
    # Session cumulative VWAP
    df["_pv"] = df["vwap"] * df["volume"]
    cum_vol = df.groupby("date")["volume"].cumsum()
    cum_pv = df.groupby("date")["_pv"].cumsum()
    session_vwap = cum_pv / cum_vol.replace(0, np.nan)
    df["dist_session_vwap"] = (c - session_vwap) / atr_safe
    df.drop(columns=["_pv"], inplace=True)

    # Session high and low so far
    session_h_so_far = df.groupby("date")["high"].cummax()
    session_l_so_far = df.groupby("date")["low"].cummin()
    df["dist_session_high"] = (c - session_h_so_far) / atr_safe
    df["dist_session_low"] = (c - session_l_so_far) / atr_safe

    # Distance to previous day levels
    df["dist_prev_high"] = (c - df["prev_high"]) / atr_safe
    df["dist_prev_low"] = (c - df["prev_low"]) / atr_safe
    df["dist_prev_close"] = (c - df["prev_close"]) / atr_safe

    # Opening range position (first 30 min: 09:30 to 10:00)
    # 09:30, 09:35, 09:40, 09:45, 09:50, 09:55
    is_or = df.index.time < datetime.time(10, 0)
    or_highs = df[is_or].groupby("date")["high"].max()
    or_lows = df[is_or].groupby("date")["low"].min()
    df["or_high"] = df["date"].map(or_highs)
    df["or_low"] = df["date"].map(or_lows)
    # If before 10:00, use session high/low so far
    df.loc[is_or, "or_high"] = session_h_so_far[is_or]
    df.loc[is_or, "or_low"] = session_l_so_far[is_or]
    or_range = (df["or_high"] - df["or_low"]).replace(0, np.nan)
    df["or_pos_30m"] = ((c - df["or_low"]) / or_range).fillna(0.5).clip(lower=0.0, upper=1.0)

    # 7. Time Features
    minutes_from_930 = (df.index.hour - 9) * 60 + (df.index.minute - 30)
    df["mins_since_open"] = minutes_from_930.astype(float)
    df["mins_to_close"] = (390.0 - df["mins_since_open"]).clip(lower=0.0)
    df["tod_sin"] = np.sin(2.0 * np.pi * df["mins_since_open"] / 390.0)
    df["tod_cos"] = np.cos(2.0 * np.pi * df["mins_since_open"] / 390.0)

    # 8. Context Features
    max_ffill = CONFIG_M5.get("max_context_ffill_bars", 1)
    if context_dfs:
        for sym in ["SPY", "XLF", "JPM", "MS", "BAC"]:
            if sym in context_dfs and not context_dfs[sym].empty:
                cdf = context_dfs[sym]
                # Forward-fill missing context bars up to limit from config
                c_c = cdf["close"].reindex(df.index).ffill(limit=max_ffill)
                df[f"{sym.lower()}_ret_1"] = c_c / c_c.shift(1) - 1.0
                df[f"{sym.lower()}_ret_6"] = c_c / c_c.shift(6) - 1.0
            else:
                df[f"{sym.lower()}_ret_1"] = np.nan
                df[f"{sym.lower()}_ret_6"] = np.nan
        
        # GS minus SPY
        df["gs_minus_spy_1"] = df["ret_1"] - df["spy_ret_1"]
        df["gs_minus_spy_6"] = df["ret_6"] - df["spy_ret_6"]
        
        # GS minus peer mean
        peer_ret_1 = (df["jpm_ret_1"] + df["ms_ret_1"] + df["bac_ret_1"]) / 3.0
        peer_ret_6 = (df["jpm_ret_6"] + df["ms_ret_6"] + df["bac_ret_6"]) / 3.0
        df["gs_minus_peer_1"] = df["ret_1"] - peer_ret_1
        df["gs_minus_peer_6"] = df["ret_6"] - peer_ret_6
    else:
        for col in FEATURE_GROUPS["context"]:
            df[col] = np.nan

    # 9. Regime Tag
    # Efficiency ratio over 48 bars: |close[t] - close[t-48]| / sum(|close[i] - close[i-1]|)
    abs_diff = c.diff().abs()
    sum_diff_48 = abs_diff.rolling(48, min_periods=48).sum()
    net_diff_48 = (c - c.shift(48)).abs()
    eff_ratio = net_diff_48 / sum_diff_48.replace(0, np.nan)
    df["efficiency_ratio_48"] = eff_ratio
    
    # Regime: Uptrend if ret_20d > 0 and eff_ratio > 0.25, Downtrend if ret_20d < 0 and eff_ratio > 0.25, else Chop
    is_trend = eff_ratio > 0.25
    df["regime_tag"] = "Chop"
    df.loc[is_trend & (df["ret_20d"] > 0), "regime_tag"] = "Uptrend"
    df.loc[is_trend & (df["ret_20d"] < 0), "regime_tag"] = "Downtrend"

    # 10. Labels (Forward Returns)
    # Next 5-minute bar of the SAME session
    next_c = c.shift(-1)
    next_date = df["date"].shift(-1)
    same_session_mask = (df["date"] == next_date)
    
    # Forward return over 1 bar
    fwd = next_c / c - 1.0
    fwd[~same_session_mask] = np.nan
    df["fwd"] = fwd
    df["y"] = np.where(fwd > 0, 1, 0)
    df.loc[fwd.isna(), "y"] = np.nan

    # Secondary heads
    # 15m (3 bars) and 60m (12 bars) same session
    fwd_15m = c.shift(-3) / c - 1.0
    same_session_15m = (df["date"] == df["date"].shift(-3))
    fwd_15m[~same_session_15m] = np.nan
    df["fwd_15m"] = fwd_15m
    df["y_15m"] = np.where(fwd_15m > 0, 1, np.where(fwd_15m.isna(), np.nan, 0))

    fwd_60m = c.shift(-12) / c - 1.0
    same_session_60m = (df["date"] == df["date"].shift(-12))
    fwd_60m[~same_session_60m] = np.nan
    df["fwd_60m"] = fwd_60m
    df["y_60m"] = np.where(fwd_60m > 0, 1, np.where(fwd_60m.isna(), np.nan, 0))

    # Next-bar volatility head: log((high[t+1] - low[t+1]) / close[t+1])
    next_hl_range = (h.shift(-1) - l.shift(-1)).clip(lower=1e-4)
    vol_head = np.log(next_hl_range / next_c)
    vol_head[~same_session_mask] = np.nan
    df["vol_head"] = vol_head

    return df


def load_and_build_features_m5(
    target_date: datetime.date,
    db_path: str = "data/live/live_m5.db",
    lookback_days: int = 30
) -> pd.DataFrame:
    """
    Unified multi-bar feature preparation for both offline replay and live engine.
    Ensures that live and replay feature paths are 100% identical.
    Combines frozen parquet history with today's live/raw minute data across all tickers.
    """
    import os
    import sqlite3
    from src.aggregate_m5 import aggregate_1min_to_5min
    
    symbols = ["GS", "SPY", "XLF", "JPM", "MS", "BAC", "TLT"]
    dfs = {}
    
    conn = None
    if os.path.exists(db_path):
        try:
            conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        except Exception:
            conn = None

    for sym in symbols:
        p = f"data/processed/m5/bars_5m_{sym}.parquet"
        if os.path.exists(p):
            p_df = pd.read_parquet(p)
            past_dates = sorted([d for d in set(p_df.index.date) if d < target_date])
            lookback_dates = past_dates[-lookback_days:] if len(past_dates) >= lookback_days else past_dates
            hist_slice = p_df[p_df.index.date >= lookback_dates[0]].copy() if lookback_dates else p_df.copy()
            
            # Check if target_date is in parquet (e.g. historical day replay)
            if any(d == target_date for d in set(p_df.index.date)):
                today_slice = p_df[p_df.index.date == target_date]
                comb = pd.concat([hist_slice[hist_slice.index.date < target_date], today_slice]).sort_index()
            elif conn is not None:
                # Query raw_minutes from live DB
                cur = conn.execute(
                    "SELECT timestamp, open, high, low, close, volume, vwap, trade_count FROM raw_minutes "
                    "WHERE symbol = ? AND timestamp LIKE ? ORDER BY timestamp;",
                    (sym, f"{target_date.isoformat()}%")
                )
                rows = cur.fetchall()
                if rows:
                    cols = [d[0] for d in cur.description]
                    r_df = pd.DataFrame(rows, columns=cols).set_index("timestamp")
                    r_df.index = pd.to_datetime(r_df.index)
                    b_5m = aggregate_1min_to_5min(r_df, symbol=sym)
                    comb = pd.concat([hist_slice, b_5m]).sort_index()
                    comb = comb[~comb.index.duplicated(keep="last")]
                else:
                    comb = hist_slice
            else:
                comb = hist_slice
            dfs[sym] = comb
        else:
            dfs[sym] = pd.DataFrame()

    if conn is not None:
        conn.close()

    if "GS" not in dfs or dfs["GS"].empty:
        return pd.DataFrame()

    context_dfs = {s: dfs[s] for s in symbols if s != "GS"}
    df_feat = make_features_m5(dfs["GS"], context_dfs=context_dfs)
    return df_feat

