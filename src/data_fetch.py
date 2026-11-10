"""Data fetching and session-anchored 1h bar builder (V2 Upgrade).

Downloads 8+ years (Oct 2018 to Nov 2026) of 1-minute OHLCV bars for target (GS)
and context tickers (SPY, XLF, JPM, MS, BAC, TLT, VIXY) from Alpaca Market Data (SIP feed),
caches monthly raw parquet files, calculates intrabar metrics, aggregates session-anchored 1h bars,
and saves the frozen dataset boundaries.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from alpaca.data.enums import Adjustment, DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from configs.config import CONFIG


def get_alpaca_client():
    """Initialize Alpaca historical data client from .env."""
    load_dotenv(".env")
    key_id = os.getenv("APCA_API_KEY_ID")
    secret_key = os.getenv("APCA_API_SECRET_KEY")
    if not key_id or not secret_key:
        raise ValueError(
            "Missing Alpaca API credentials. Please set APCA_API_KEY_ID and "
            "APCA_API_SECRET_KEY in .env."
        )
    return StockHistoricalDataClient(key_id, secret_key)


def run_availability_test(client, symbol="GS"):
    """Verify market data access with recent 1-minute bars."""
    feed_name = CONFIG.get("alpaca_feed", "sip").lower()
    feed = DataFeed.SIP if feed_name == "sip" else DataFeed.IEX

    now_utc = datetime.now(timezone.utc)
    # Probe 2 days ago
    start_dt = now_utc - timedelta(days=3)
    end_dt = now_utc - timedelta(minutes=20)

    req = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Minute,
        start=start_dt,
        end=end_dt,
        adjustment=Adjustment.RAW,
        feed=feed,
    )
    try:
        bars = client.get_stock_bars(req)
        df = bars.df
        if df.empty:
            raise RuntimeError(f"Alpaca returned empty dataframe for {symbol} probe.")
        print(
            f"Availability test PASSED: retrieved {len(df)} 1-min bars for {symbol} using {feed_name.upper()} feed."
        )
    except Exception as e:
        print(f"CRITICAL: Alpaca market data availability test failed: {e}", file=sys.stderr)
        raise RuntimeError(f"Alpaca SIP data error: {e}")


def generate_month_ranges(start_dt, end_dt):
    """Generate (start, end, label) for each calendar month between start_dt and end_dt."""
    cur = datetime(start_dt.year, start_dt.month, 1, tzinfo=timezone.utc)
    while cur <= end_dt:
        if cur.month == 12:
            next_month = datetime(cur.year + 1, 1, 1, tzinfo=timezone.utc)
        else:
            next_month = datetime(cur.year, cur.month + 1, 1, tzinfo=timezone.utc)

        chunk_start = max(cur, start_dt)
        chunk_end = min(next_month - timedelta(microseconds=1), end_dt)
        label = cur.strftime("%Y-%m")
        yield chunk_start, chunk_end, label
        cur = next_month


def download_symbol_1min(client, symbol, start_dt, end_dt, raw_dir, refresh=False):
    """Download 1-minute bars month by month with caching, retry, and backoff."""
    sym_dir = Path(raw_dir) / "alpaca_1min" / symbol
    sym_dir.mkdir(parents=True, exist_ok=True)

    feed_name = CONFIG.get("alpaca_feed", "sip").lower()
    feed = DataFeed.SIP if feed_name == "sip" else DataFeed.IEX

    month_dfs = []
    month_ranges = list(generate_month_ranges(start_dt, end_dt))
    now_utc = datetime.now(timezone.utc)

    for chunk_start, chunk_end, label in month_ranges:
        cache_file = sym_dir / f"{label}.parquet"
        is_current_month = (chunk_start.year == now_utc.year and chunk_start.month == now_utc.month)

        if cache_file.exists() and not refresh and not is_current_month:
            try:
                df_chunk = pd.read_parquet(cache_file)
                month_dfs.append(df_chunk)
                continue
            except Exception:
                pass  # Re-download corrupt file

        max_retries = 5
        backoff = 1.5
        df_chunk = None

        for attempt in range(max_retries):
            try:
                req = StockBarsRequest(
                    symbol_or_symbols=symbol,
                    timeframe=TimeFrame.Minute,
                    start=chunk_start,
                    end=chunk_end,
                    adjustment=Adjustment.RAW,
                    feed=feed,
                )
                bars = client.get_stock_bars(req)
                raw_df = bars.df
                if raw_df.empty:
                    df_chunk = pd.DataFrame()
                    break

                # Extract symbol level if multiindex
                if isinstance(raw_df.index, pd.MultiIndex):
                    if symbol in raw_df.index.levels[0]:
                        raw_df = raw_df.xs(symbol, level=0)
                    else:
                        raw_df = raw_df.droplevel(0)

                df_chunk = raw_df[["open", "high", "low", "close", "volume", "vwap"]].copy()
                if "trade_count" in raw_df.columns:
                    df_chunk["trade_count"] = raw_df["trade_count"]

                df_chunk.sort_index(inplace=True)
                break
            except Exception as e:
                err_msg = str(e).lower()
                if "subscription" in err_msg or "unauthorized" in err_msg or "403" in err_msg:
                    raise RuntimeError(f"Alpaca API permission error: {e}")
                if attempt == max_retries - 1:
                    print(f"Warning: Failed to download {symbol} {label} after {max_retries} attempts: {e}")
                    df_chunk = pd.DataFrame()
                else:
                    time.sleep(backoff)
                    backoff *= 2.0

        if df_chunk is not None and not df_chunk.empty:
            df_chunk.to_parquet(cache_file)
            month_dfs.append(df_chunk)
        elif cache_file.exists():
            df_chunk = pd.read_parquet(cache_file)
            month_dfs.append(df_chunk)

        time.sleep(0.05)

    if not month_dfs:
        return pd.DataFrame()

    full_df = pd.concat(month_dfs, axis=0)
    full_df = full_df[~full_df.index.duplicated(keep="last")]
    full_df.sort_index(inplace=True)
    return full_df


def compute_intrabar_metrics(df_sub):
    """Compute causal intrabar features from 1-minute bars inside a 1h candle."""
    if len(df_sub) == 0:
        return pd.Series({
            "realized_vol_1m": 0.0,
            "up_minute_ratio": 0.5,
            "clv": 0.0,
            "signed_volume_ratio": 0.0,
            "last15_ret": 0.0
        })

    c = df_sub["close"].to_numpy()
    o = df_sub["open"].to_numpy()
    v = df_sub["volume"].to_numpy()
    
    # Realized vol: sample std of 1-min returns
    if len(c) > 1:
        rets_1m = np.diff(c) / c[:-1]
        realized_vol_1m = float(np.std(rets_1m, ddof=1)) if len(rets_1m) > 1 else 0.0
    else:
        realized_vol_1m = 0.0

    # Up minute ratio
    up_minutes = np.sum(c > o)
    up_minute_ratio = float(up_minutes / len(c)) if len(c) > 0 else 0.5

    # Close Location Value (CLV)
    high_val = float(df_sub["high"].max())
    low_val = float(df_sub["low"].min())
    close_val = float(c[-1])
    hl_range = high_val - low_val
    if hl_range > 1e-8:
        clv = float(((close_val - low_val) - (high_val - close_val)) / hl_range)
    else:
        clv = 0.0

    # Signed volume ratio
    up_vol = float(np.sum(v[c >= o]))
    down_vol = float(np.sum(v[c < o]))
    tot_vol = up_vol + down_vol
    signed_volume_ratio = float((up_vol - down_vol) / tot_vol) if tot_vol > 0 else 0.0

    # Last 15 min return: price of last minute vs price 15 minutes before the end
    if len(c) >= 15:
        last15_ret = float((c[-1] - c[-15]) / c[-15])
    else:
        last15_ret = float((c[-1] - c[0]) / c[0])

    return pd.Series({
        "realized_vol_1m": realized_vol_1m,
        "up_minute_ratio": up_minute_ratio,
        "clv": clv,
        "signed_volume_ratio": signed_volume_ratio,
        "last15_ret": last15_ret
    })


def build_session_anchored_1h_bars(df_1min, symbol, tz_str="America/New_York"):
    """Aggregate 1-minute bars into session-anchored 1h bars with intrabar metrics."""
    if df_1min.empty:
        return pd.DataFrame()

    df = df_1min.copy()
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df_ny = df.tz_convert(ZoneInfo(tz_str))

    # Keep only regular session: 09:30 <= t < 16:00
    minutes_from_midnight = df_ny.index.hour * 60 + df_ny.index.minute
    session_mask = (minutes_from_midnight >= 570) & (minutes_from_midnight < 960)
    df_session = df_ny[session_mask].copy()

    if df_session.empty:
        return pd.DataFrame()

    # Calculate session-anchored bar start
    offset = (df_session.index.hour * 60 + df_session.index.minute) - 570
    bar_num = offset // 60
    dates = df_session.index.normalize()
    bar_start_times = dates + pd.to_timedelta(570 + bar_num * 60, unit="m")
    df_session["bar_start"] = bar_start_times

    # Basic OHLCV aggregations
    has_vwap = "vwap" in df_session.columns
    if has_vwap and "volume" in df_session.columns:
        df_session["vol_x_vwap"] = df_session["volume"] * df_session["vwap"].fillna(df_session["close"])
    else:
        df_session["vol_x_vwap"] = df_session["volume"] * df_session["close"]

    agg_rules = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
        "vol_x_vwap": "sum",
    }
    if "trade_count" in df_session.columns:
        agg_rules["trade_count"] = "sum"

    grouped = df_session.groupby("bar_start").agg(agg_rules)
    grouped["vwap"] = grouped["vol_x_vwap"] / grouped["volume"].replace(0, np.nan)
    grouped["vwap"] = grouped["vwap"].fillna(grouped["close"])
    grouped.drop(columns=["vol_x_vwap"], inplace=True)

    # Compute intrabar metrics per bar
    intrabar_df = df_session.groupby("bar_start").apply(compute_intrabar_metrics)
    grouped = pd.concat([grouped, intrabar_df], axis=1)

    grouped.index.name = "timestamp"
    grouped["symbol"] = symbol
    return grouped


def main():
    parser = argparse.ArgumentParser(description="Fetch Alpaca 1-min data and build session-anchored 1h bars.")
    parser.add_argument("--refresh", action="store_true", help="Force re-download of all cached chunks.")
    args = parser.parse_args()

    # 1. Print system date and time
    now_utc = datetime.now(timezone.utc)
    now_ny = now_utc.astimezone(ZoneInfo(CONFIG["tz"]))
    print(f"System UTC Time: {now_utc.isoformat()}")
    print(f"System NY Time:  {now_ny.isoformat()}")

    client = get_alpaca_client()
    run_availability_test(client, symbol=CONFIG["ticker"])

    raw_dir = CONFIG["paths"]["raw_dir"]
    proc_dir = Path("data/processed")
    proc_dir.mkdir(parents=True, exist_ok=True)

    # Freeze date & data_end
    freeze_date_str = CONFIG["freeze_date"]  # "2026-11-09"
    freeze_dt_ny = datetime.strptime(freeze_date_str, "%Y-%m-%d").replace(hour=16, minute=0, tzinfo=ZoneInfo(CONFIG["tz"]))
    freeze_dt_utc = freeze_dt_ny.astimezone(timezone.utc)

    # 8 years + 30 days buffer before freeze_date
    history_days = CONFIG["history_years"] * 365 + CONFIG["history_buffer_days"]
    start_dt_utc = freeze_dt_utc - timedelta(days=history_days)

    all_symbols = [CONFIG["ticker"]] + CONFIG["context_tickers"]
    print(f"\n--- Ingesting 8-Year History for {all_symbols} ---")
    print(f"Start: {start_dt_utc.date()} | End (Freeze Date): {freeze_dt_utc.date()} ({freeze_date_str})")

    symbol_bars = {}

    for sym in all_symbols:
        print(f"\nFetching {sym} (8 years)...")
        df_1min = download_symbol_1min(client, sym, start_dt_utc, freeze_dt_utc, raw_dir, refresh=args.refresh)
        print(f"Total 1-min bars for {sym}: {len(df_1min):,}")

        df_1h = build_session_anchored_1h_bars(df_1min, sym, tz_str=CONFIG["tz"])
        
        # Filter strictly up to freeze_date 16:00 ET
        df_1h = df_1h[df_1h.index.strftime("%Y-%m-%d") <= freeze_date_str]
        print(f"Total session-anchored 1h bars for {sym}: {len(df_1h):,} (Range: {df_1h.index[0]} to {df_1h.index[-1]})")

        out_path = proc_dir / f"bars_{sym}_1h.parquet"
        df_1h.to_parquet(out_path)
        symbol_bars[sym] = df_1h

    # Inner join and audit rows lost
    gs_bars = symbol_bars[CONFIG["ticker"]]
    gs_count = len(gs_bars)
    gs_timestamps = set(gs_bars.index)

    print(f"\n--- Checking Context Symbols Synchronization against {CONFIG['ticker']} ({gs_count} bars) ---")
    valid_context_symbols = []

    for sym in CONFIG["context_tickers"]:
        sym_ts = set(symbol_bars[sym].index)
        intersection_count = len(gs_timestamps.intersection(sym_ts))
        lost_rows = gs_count - intersection_count
        lost_pct = (lost_rows / gs_count) * 100.0
        print(f"  - {sym}: {len(symbol_bars[sym])} bars | Shared with GS: {intersection_count} | Lost: {lost_rows} ({lost_pct:.2f}%)")

        if lost_pct > 2.0:
            print(f"  WARNING: Dropping {sym} because it loses > 2% of GS bars ({lost_pct:.2f}% > 2.0%)!")
        else:
            valid_context_symbols.append(sym)

    # Compute final synchronized timestamps
    common_ts = gs_timestamps
    for sym in valid_context_symbols:
        common_ts = common_ts.intersection(set(symbol_bars[sym].index))

    print(f"\nTotal Synchronized 1h Bars across GS and valid context tickers: {len(common_ts):,} (from {gs_count})")
    
    # Save data freeze metadata
    sorted_ts = sorted(list(common_ts))
    data_end = str(sorted_ts[-1])
    data_start = str(sorted_ts[0])

    freeze_metadata = {
        "data_start": data_start,
        "data_end": data_end,
        "freeze_date": freeze_date_str,
        "test_day": CONFIG["test_day"],
        "gs_total_bars": gs_count,
        "synchronized_bars": len(sorted_ts),
        "kept_context_symbols": valid_context_symbols,
        "frozen_at": datetime.now(timezone.utc).isoformat()
    }
    with open(CONFIG["paths"]["data_freeze"], "w") as f:
        json.dump(freeze_metadata, f, indent=2)
    print(f"\nData freeze metadata saved to {CONFIG['paths']['data_freeze']}:")
    print(json.dumps(freeze_metadata, indent=2))


if __name__ == "__main__":
    main()
