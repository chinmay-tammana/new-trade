"""
Shared 5-minute aggregation function for M5 pipeline.
Used identically in offline bar building and live operational engine.
Session-anchored in America/New_York (09:30 to 16:00).
"""
import datetime
import zoneinfo
import numpy as np
import pandas as pd

ET_TZ = zoneinfo.ZoneInfo("America/New_York")


def get_5min_bar_start(dt: datetime.datetime) -> datetime.datetime:
    """Map any timestamp dt to its session-anchored 5-minute bar start."""
    dt_et = dt.astimezone(ET_TZ)
    # Session starts at 09:30
    session_start = dt_et.replace(hour=9, minute=30, second=0, microsecond=0)
    delta_sec = (dt_et - session_start).total_seconds()
    if delta_sec < 0:
        return None
    bar_idx = int(delta_sec // 300) # 300 seconds = 5 min
    bar_start = session_start + datetime.timedelta(seconds=bar_idx * 300)
    return bar_start


def aggregate_1min_to_5min(df_1min: pd.DataFrame, symbol: str = "GS") -> pd.DataFrame:
    """
    Aggregate 1-minute OHLCV bars into session-anchored 5-minute bars.
    
    Parameters:
    -----------
    df_1min: DataFrame indexed by timestamp (timezone-aware UTC or ET)
             with columns ['open', 'high', 'low', 'close', 'volume']
             optional: ['trade_count', 'vwap']
    symbol: symbol string
    
    Returns:
    --------
    DataFrame indexed by bar_start (ET) with:
    ['open', 'high', 'low', 'close', 'volume', 'vwap', 'trade_count',
     'bar_end', 'n_minutes_with_trades', 'up_minute_ratio', 
     'signed_volume_ratio', 'realized_vol_intrabar', 'clv']
    """
    if df_1min.empty:
        cols = ['open', 'high', 'low', 'close', 'volume', 'vwap', 'trade_count',
                'bar_end', 'n_minutes_with_trades', 'up_minute_ratio', 
                'signed_volume_ratio', 'realized_vol_intrabar', 'clv']
        return pd.DataFrame(columns=cols)

    df = df_1min.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        if "timestamp" in df.columns:
            df = df.set_index("timestamp")
        else:
            raise ValueError("df_1min must have a DatetimeIndex or 'timestamp' column")

    # Ensure index is ET
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC").tz_convert(ET_TZ)
    else:
        df.index = df.index.tz_convert(ET_TZ)

    df = df.sort_index()

    # Filter strictly to regular trading hours: 09:30:00 <= time < 16:00:00
    times = df.index.time
    t_open = datetime.time(9, 30, 0)
    t_close = datetime.time(16, 0, 0)
    mask = (times >= t_open) & (times < t_close)
    df = df[mask]
    if df.empty:
        cols = ['open', 'high', 'low', 'close', 'volume', 'vwap', 'trade_count',
                'bar_end', 'n_minutes_with_trades', 'up_minute_ratio', 
                'signed_volume_ratio', 'realized_vol_intrabar', 'clv']
        return pd.DataFrame(columns=cols)

    # Assign each minute to its 5-minute bucket
    # Bucket starts at 09:30 + 5 min * floor((t - 09:30) / 5 min)
    dates = df.index.date
    # Calculate minute offset from 09:30
    minute_offsets = (df.index.hour - 9) * 60 + (df.index.minute - 30)
    bar_indices = minute_offsets // 5
    
    # Construct exact bar_start timestamps
    bar_starts = [
        datetime.datetime.combine(d, t_open, tzinfo=ET_TZ) + datetime.timedelta(minutes=int(b_idx * 5))
        for d, b_idx in zip(dates, bar_indices)
    ]
    df["bar_start"] = bar_starts

    # Group by bar_start
    records = []
    has_vwap = "vwap" in df.columns
    has_trades = "trade_count" in df.columns

    for b_start, grp in df.groupby("bar_start", sort=True):
        b_end = b_start + datetime.timedelta(minutes=5)
        n_mins = len(grp)
        b_open = grp["open"].iloc[0]
        b_high = grp["high"].max()
        b_low = grp["low"].min()
        b_close = grp["close"].iloc[-1]
        b_vol = grp["volume"].sum()
        
        if has_vwap and b_vol > 0:
            b_vwap = (grp["vwap"] * grp["volume"]).sum() / b_vol
        else:
            b_vwap = (b_open + b_high + b_low + b_close) / 4.0
            
        b_trades = grp["trade_count"].sum() if has_trades else n_mins

        # Intrabar metrics
        # 1-minute returns inside bar
        c_vals = grp["close"].values
        o_vals = grp["open"].values
        v_vals = grp["volume"].values
        
        # Up minute ratio
        up_mins = np.sum(c_vals > o_vals)
        up_min_ratio = float(up_mins / n_mins) if n_mins > 0 else 0.5
        
        # Signed volume ratio
        sign_dirs = np.sign(c_vals - o_vals)
        if b_vol > 0:
            signed_vol_ratio = float(np.sum(sign_dirs * v_vals) / b_vol)
        else:
            signed_vol_ratio = 0.0
            
        # Realized volatility inside bar
        if n_mins >= 2:
            min_rets = np.diff(np.log(np.maximum(c_vals, 1e-6)))
            rv_intrabar = float(np.sqrt(np.sum(min_rets ** 2)))
        else:
            rv_intrabar = 0.0
            
        # Close Location Value (CLV)
        hl_range = b_high - b_low
        if hl_range > 1e-6:
            clv = float(((b_close - b_low) - (b_high - b_close)) / hl_range)
        else:
            clv = 0.0

        records.append({
            "bar_start": b_start,
            "bar_end": b_end,
            "open": float(b_open),
            "high": float(b_high),
            "low": float(b_low),
            "close": float(b_close),
            "volume": float(b_vol),
            "vwap": float(b_vwap),
            "trade_count": float(b_trades),
            "n_minutes_with_trades": int(n_mins),
            "up_minute_ratio": up_min_ratio,
            "signed_volume_ratio": signed_vol_ratio,
            "realized_vol_intrabar": rv_intrabar,
            "clv": clv
        })

    out_df = pd.DataFrame(records).set_index("bar_start").sort_index()
    return out_df
