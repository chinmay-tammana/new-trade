"""Feature engineering module for Goldman Sachs and context tickers.

Implements stationary, strictly causal technical and cross-asset features
using pure pandas and numpy without external technical analysis libraries.
"""

from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd

FEATURES = [
    # Returns
    "ret_1", "ret_2", "ret_3", "ret_6", "ret_14",
    # Overnight gap
    "gap",
    # Momentum
    "rsi", "macd_n", "macd_hist_n",
    # Volatility
    "bb_pos", "bb_width", "atr_n", "vol_20",
    # Volume
    "rvol",
    # Time
    "hour", "dow", "is_first_bar",
    # Context
    "spy_ret_1", "spy_ret_3", "xlf_ret_1", "xlf_ret_3", "rel_ret_1"
]


def compute_wilder_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Compute Wilder's RSI using exponential smoothing (alpha = 1/period)."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    # Wilder's smoothing is equivalent to ewm with alpha=1/period, adjust=False
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    # Fill potential divisions by zero
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


def make_features(df_gs: pd.DataFrame, df_spy: pd.DataFrame, df_xlf: pd.DataFrame, warmup_rows: int = 30) -> pd.DataFrame:
    """Compute all causal stationary features for GS given aligned context series.

    Parameters
    ----------
    df_gs : pd.DataFrame
        1-hour bars for Goldman Sachs with columns open, high, low, close, volume.
    df_spy : pd.DataFrame
        1-hour bars for SPY with column close.
    df_xlf : pd.DataFrame
        1-hour bars for XLF with column close.
    warmup_rows : int
        Number of leading rows to drop after feature calculation to eliminate NaNs.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by session timestamp containing FEATURES and raw OHLCV.
    """
    df = df_gs.copy()
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df = df.tz_convert(ZoneInfo("America/New_York"))

    # 1. Standalone context returns BEFORE merging
    spy_c = df_spy["close"].copy()
    if spy_c.index.tz is None:
        spy_c.index = spy_c.index.tz_localize("UTC")
    spy_c.index = spy_c.index.tz_convert(ZoneInfo("America/New_York"))
    spy_ret_1 = spy_c.pct_change(1)
    spy_ret_3 = spy_c.pct_change(3)

    xlf_c = df_xlf["close"].copy()
    if xlf_c.index.tz is None:
        xlf_c.index = xlf_c.index.tz_localize("UTC")
    xlf_c.index = xlf_c.index.tz_convert(ZoneInfo("America/New_York"))
    xlf_ret_1 = xlf_c.pct_change(1)
    xlf_ret_3 = xlf_c.pct_change(3)

    # 2. Returns
    close = df["close"]
    df["ret_1"] = close.pct_change(1)
    df["ret_2"] = close.pct_change(2)
    df["ret_3"] = close.pct_change(3)
    df["ret_6"] = close.pct_change(6)
    df["ret_14"] = close.pct_change(14)

    # 3. Overnight Gap: open[t] / close[t-1] - 1
    prev_close = close.shift(1)
    df["gap"] = (df["open"] / prev_close) - 1.0

    # 4. Momentum Oscillators
    df["rsi"] = compute_wilder_rsi(close, period=14)

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal9 = macd.ewm(span=9, adjust=False).mean()
    df["macd_n"] = macd / close
    df["macd_hist_n"] = (macd - signal9) / close

    # 5. Volatility & Bands
    sma20 = close.rolling(20, min_periods=20).mean()
    std20 = close.rolling(20, min_periods=20).std()
    upper = sma20 + 2.0 * std20
    lower = sma20 - 2.0 * std20
    bandwidth = upper - lower
    df["bb_pos"] = (close - lower) / bandwidth.replace(0.0, np.nan)
    df["bb_width"] = bandwidth / sma20.replace(0.0, np.nan)

    atr14 = compute_atr(df["high"], df["low"], close, period=14)
    df["atr_n"] = atr14 / close
    df["vol_20"] = df["ret_1"].rolling(20, min_periods=20).std()

    # 6. Volume Dynamics
    vol_sma20 = df["volume"].rolling(20, min_periods=20).mean()
    df["rvol"] = df["volume"] / vol_sma20.replace(0.0, np.nan)

    # 7. Time Features (Eastern Time)
    df["hour"] = df.index.hour + (df.index.minute / 60.0)
    df["dow"] = df.index.dayofweek
    df["is_first_bar"] = (df["hour"] == 9.5).astype(int)

    # 8. Align Market Context Series
    df["spy_ret_1"] = spy_ret_1.reindex(df.index)
    df["spy_ret_3"] = spy_ret_3.reindex(df.index)
    df["xlf_ret_1"] = xlf_ret_1.reindex(df.index)
    df["xlf_ret_3"] = xlf_ret_3.reindex(df.index)
    df["rel_ret_1"] = df["ret_1"] - df["spy_ret_1"]

    # Print NaN diagnostics before dropping warmup
    print(f"make_features: initial total rows: {len(df)}")
    nan_counts = df[FEATURES].isna().sum()
    print("make_features: NaN counts per feature before warmup drop:")
    for col, cnt in nan_counts[nan_counts > 0].items():
        print(f"  {col}: {cnt} NaNs")

    if warmup_rows > 0:
        df = df.iloc[warmup_rows:].copy()

    # Verify no NaNs remain in FEATURES
    remaining_nans = df[FEATURES].isna().sum().sum()
    if remaining_nans > 0:
        print(f"WARNING: {remaining_nans} NaNs remaining in features after dropping {warmup_rows} warmup rows.")

    return df
