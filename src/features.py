"""Feature engineering module for Goldman Sachs and multi-asset context (V2 Upgrade).

Implements 44 stationary, strictly causal technical, intrabar, peer, rate, and volatility features
grouped into FEATURE_GROUPS for group ablations.
"""

from typing import Dict, List
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd


FEATURE_GROUPS: Dict[str, List[str]] = {
    "returns": [
        "ret_1", "ret_2", "ret_3", "ret_6", "ret_14", "gap"
    ],
    "momentum": [
        "rsi", "macd_n", "macd_hist_n"
    ],
    "vol": [
        "bb_pos", "bb_width", "atr_n", "vol_20"
    ],
    "volume": [
        "rvol"
    ],
    "time": [
        "hour", "dow", "is_first_bar", "bars_since_open", "bars_to_close"
    ],
    "market": [
        "spy_ret_1", "spy_ret_3", "xlf_ret_1", "xlf_ret_3", "rel_ret_spy", "rel_ret_xlf"
    ],
    "peers": [
        "jpm_ret_1", "jpm_ret_3", "ms_ret_1", "ms_ret_3", "bac_ret_1", "bac_ret_3",
        "tlt_ret_1", "tlt_ret_3", "vixy_ret_1", "vixy_ret_3", "gs_minus_peers"
    ],
    "intrabar": [
        "realized_vol_1m", "up_minute_ratio", "clv", "signed_volume_ratio", "last15_ret"
    ],
    "levels": [
        "dist_prev_close", "dist_prev_high", "dist_prev_low", "dist_session_vwap",
        "ret_1d", "ret_5d", "ret_20d", "ma_dist_50h", "ma_dist_200h", "efficiency_ratio_20"
    ]
}

# Flattened list of all features
FEATURES: List[str] = [feat for grp in FEATURE_GROUPS.values() for feat in grp]


def compute_wilder_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Compute Wilder's RSI using exponential smoothing (alpha = 1/period)."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi = rsi.where(avg_loss != 0, 100.0)
    rsi = rsi.where(avg_gain != 0, 0.0)
    return rsi


def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Compute Average True Range using Wilder's smoothing."""
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    return atr


def make_features_v2(
    df_gs: pd.DataFrame,
    context_dfs: Dict[str, pd.DataFrame],
    warmup_rows: int = 200
) -> pd.DataFrame:
    """Compute all 44 stationary, causal features for GS and aligned context symbols.

    Parameters
    ----------
    df_gs : pd.DataFrame
        1-hour bars for Goldman Sachs with OHLCV and intrabar metrics.
    context_dfs : dict of pd.DataFrame
        Dictionary mapping ticker ('SPY', 'XLF', 'JPM', 'MS', 'BAC', 'TLT', 'VIXY')
        to its 1-hour OHLCV DataFrame.
    warmup_rows : int
        Number of leading rows to drop after feature calculation to eliminate rolling NaNs.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by session timestamp containing all features in FEATURES and raw OHLCV.
    """
    df = df_gs.copy()
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df = df.tz_convert(ZoneInfo("America/New_York"))

    close = df["close"]
    high = df["high"]
    low = df["low"]
    open_p = df["open"]
    vol = df["volume"]

    # 1. Standalone context returns BEFORE inner join
    ctx_returns = {}
    for sym in ["SPY", "XLF", "JPM", "MS", "BAC", "TLT", "VIXY"]:
        c = context_dfs[sym]["close"].copy()
        if c.index.tz is None:
            c.index = c.index.tz_localize("UTC")
        c.index = c.index.tz_convert(ZoneInfo("America/New_York"))
        ctx_returns[f"{sym.lower()}_ret_1"] = c.pct_change(1)
        ctx_returns[f"{sym.lower()}_ret_3"] = c.pct_change(3)

    # 2. GS Returns
    df["ret_1"] = close.pct_change(1)
    df["ret_2"] = close.pct_change(2)
    df["ret_3"] = close.pct_change(3)
    df["ret_6"] = close.pct_change(6)
    df["ret_14"] = close.pct_change(14)

    # 3. Overnight Gap (only on opening bar 09:30)
    hour_val = df.index.hour + df.index.minute / 60.0
    prev_close = close.shift(1)
    raw_gap = (open_p - prev_close) / prev_close
    df["gap"] = np.where(hour_val == 9.5, raw_gap, 0.0)

    # 4. Momentum & Oscillators
    df["rsi"] = compute_wilder_rsi(close, period=14)
    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    macd_raw = ema_12 - ema_26
    macd_signal = macd_raw.ewm(span=9, adjust=False).mean()
    df["macd_n"] = macd_raw / close
    df["macd_hist_n"] = (macd_raw - macd_signal) / close

    # 5. Volatility & Bands
    atr_series = compute_atr(high, low, close, period=14)
    df["atr_n"] = atr_series / close
    sma_20 = close.rolling(20).mean()
    std_20 = close.rolling(20).std(ddof=1)
    upper_band = sma_20 + 2.0 * std_20
    lower_band = sma_20 - 2.0 * std_20
    band_width = upper_band - lower_band
    df["bb_pos"] = (close - lower_band) / band_width.replace(0.0, np.nan)
    df["bb_width"] = band_width / sma_20
    df["vol_20"] = df["ret_1"].rolling(20).std(ddof=1)

    # 6. Volume Dynamics
    vol_sma_20 = vol.rolling(20).mean()
    df["rvol"] = vol / vol_sma_20.replace(0.0, np.nan)

    # 7. Time & Calendar Context
    df["hour"] = hour_val
    df["dow"] = df.index.dayofweek
    df["is_first_bar"] = (hour_val == 9.5).astype(int)

    # Bars since open (0 to 6) and bars to close (6 to 0)
    session_offset = ((df.index.hour * 60 + df.index.minute) - 570) // 60
    df["bars_since_open"] = session_offset
    # In regular session with 7 bars, bars_to_close is 6 - session_offset
    df["bars_to_close"] = 6 - session_offset

    # 8. Market & Sector Features
    df["spy_ret_1"] = ctx_returns["spy_ret_1"].reindex(df.index)
    df["spy_ret_3"] = ctx_returns["spy_ret_3"].reindex(df.index)
    df["xlf_ret_1"] = ctx_returns["xlf_ret_1"].reindex(df.index)
    df["xlf_ret_3"] = ctx_returns["xlf_ret_3"].reindex(df.index)
    df["rel_ret_spy"] = df["ret_1"] - df["spy_ret_1"]
    df["rel_ret_xlf"] = df["ret_1"] - df["xlf_ret_1"]

    # 9. Banking Peers, Rates & Volatility
    for sym in ["JPM", "MS", "BAC", "TLT", "VIXY"]:
        df[f"{sym.lower()}_ret_1"] = ctx_returns[f"{sym.lower()}_ret_1"].reindex(df.index)
        df[f"{sym.lower()}_ret_3"] = ctx_returns[f"{sym.lower()}_ret_3"].reindex(df.index)

    peer_mean_ret = (df["jpm_ret_1"] + df["ms_ret_1"] + df["bac_ret_1"]) / 3.0
    df["gs_minus_peers"] = df["ret_1"] - peer_mean_ret

    # 10. Intrabar Features (if not already in df_gs, fallback to safe defaults)
    for col in ["realized_vol_1m", "up_minute_ratio", "clv", "signed_volume_ratio", "last15_ret"]:
        if col not in df.columns:
            df[col] = 0.0

    # 11. Multi-Day Levels & Trend Efficiency
    # Compute daily aggregates (Close, High, Low) strictly up to previous date
    daily_df = df.groupby(df.index.date).agg({
        "close": "last",
        "high": "max",
        "low": "min"
    })
    prev_daily_close = daily_df["close"].shift(1)
    prev_daily_high = daily_df["high"].shift(1)
    prev_daily_low = daily_df["low"].shift(1)

    # Multi-day daily returns (up to previous close)
    ret_1d_daily = daily_df["close"].pct_change(1).shift(1)
    ret_5d_daily = daily_df["close"].pct_change(5).shift(1)
    ret_20d_daily = daily_df["close"].pct_change(20).shift(1)

    # Map back to hourly timestamps by date
    dates = df.index.date
    df["dist_prev_close"] = (close - pd.Series(dates, index=df.index).map(prev_daily_close)) / atr_series.replace(0.0, np.nan)
    df["dist_prev_high"] = (close - pd.Series(dates, index=df.index).map(prev_daily_high)) / atr_series.replace(0.0, np.nan)
    df["dist_prev_low"] = (close - pd.Series(dates, index=df.index).map(prev_daily_low)) / atr_series.replace(0.0, np.nan)

    df["ret_1d"] = pd.Series(dates, index=df.index).map(ret_1d_daily)
    df["ret_5d"] = pd.Series(dates, index=df.index).map(ret_5d_daily)
    df["ret_20d"] = pd.Series(dates, index=df.index).map(ret_20d_daily)

    # Cumulative intraday session VWAP
    vol_x_price = vol * close
    cum_vol = df.groupby(df.index.date)["volume"].cumsum()
    cum_vol_price = vol_x_price.groupby(df.index.date).cumsum()
    session_vwap = cum_vol_price / cum_vol.replace(0, np.nan)
    df["dist_session_vwap"] = (close - session_vwap) / session_vwap

    # Moving average distances (50h and 200h)
    sma_50h = close.rolling(50).mean()
    sma_200h = close.rolling(200).mean()
    df["ma_dist_50h"] = (close - sma_50h) / sma_50h
    df["ma_dist_200h"] = (close - sma_200h) / sma_200h

    # Kaufman Efficiency Ratio over 20 bars: |net move| / sum of absolute moves
    net_move_20 = (close - close.shift(20)).abs()
    sum_moves_20 = close.diff().abs().rolling(20).sum()
    df["efficiency_ratio_20"] = net_move_20 / sum_moves_20.replace(0.0, np.nan)

    # Drop leading warmup rows
    if warmup_rows > 0:
        df = df.iloc[warmup_rows:].copy()

    return df
