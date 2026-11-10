"""Data fetching and session-anchored 1h bar builder.

Downloads 1-minute OHLCV bars for target (GS) and context tickers (SPY, XLF)
from Alpaca Market Data (SIP feed), caches monthly raw parquet files, and aggregates
session-anchored 1h bars (09:30 to 16:00 ET).
"""

import argparse
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
    """Verify market data access with 1 day of 1-minute bars from ~3 years ago."""
    feed_name = CONFIG.get("alpaca_feed", "sip").lower()
    feed = DataFeed.SIP if feed_name == "sip" else DataFeed.IEX

    # Probe 5 weekdays ~3 years ago
    base_date = datetime.now(timezone.utc) - timedelta(days=3 * 365)
    success = False
    last_err = None

    for day_offset in range(5):
        probe_dt = base_date + timedelta(days=day_offset)
        if probe_dt.weekday() >= 5:  # Skip weekend
            continue
        start_dt = probe_dt.replace(hour=14, minute=0, second=0, microsecond=0)
        end_dt = probe_dt.replace(hour=20, minute=0, second=0, microsecond=0)

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
            if not df.empty:
                print(
                    f"Availability test PASSED: retrieved {len(df)} 1-min bars for {symbol} on {probe_dt.strftime('%Y-%m-%d')} using {feed_name.upper()} feed."
                )
                success = True
                break
        except Exception as e:
            last_err = e

    if not success:
        raise RuntimeError(
            f"Alpaca availability test FAILED for feed={feed_name}. Error: {last_err}"
        )


def generate_month_ranges(start_dt, end_dt):
    """Generate (start, end, label) for each calendar month between start_dt and end_dt."""
    cur = datetime(start_dt.year, start_dt.month, 1, tzinfo=timezone.utc)
    while cur <= end_dt:
        # End of current month
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
            df_chunk = pd.read_parquet(cache_file)
            month_dfs.append(df_chunk)
            continue

        print(f"Downloading {symbol} 1-min bars for {label} ({chunk_start.date()} to {chunk_end.date()})...")
        max_retries = 5
        backoff = 2.0
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
                if raw_df is not None and not raw_df.empty:
                    # Clean multi-index if symbol is in index
                    if isinstance(raw_df.index, pd.MultiIndex):
                        raw_df = raw_df.xs(symbol, level=0 if raw_df.index.names[0] == "symbol" else "symbol")
                    df_chunk = raw_df.copy()
                    df_chunk.index.name = "timestamp"
                else:
                    df_chunk = pd.DataFrame(columns=["open", "high", "low", "close", "volume", "trade_count", "vwap"])
                break
            except Exception as e:
                err_msg = str(e)
                if "429" in err_msg or "rate limit" in err_msg.lower():
                    print(f"Rate limited on {symbol} {label}. Backing off {backoff:.1f}s (attempt {attempt+1}/{max_retries})...")
                    time.sleep(backoff)
                    backoff *= 2.0
                else:
                    print(f"Error downloading {symbol} {label}: {e}. Retrying in {backoff:.1f}s...")
                    time.sleep(backoff)
                    backoff *= 1.5

        if df_chunk is None:
            raise RuntimeError(f"Failed to download 1-min bars for {symbol} in {label} after {max_retries} attempts.")

        # Cache chunk
        df_chunk.to_parquet(cache_file)
        month_dfs.append(df_chunk)
        time.sleep(0.1)  # small inter-request pacing

    if not month_dfs:
        return pd.DataFrame()

    full_df = pd.concat(month_dfs, axis=0)
    full_df = full_df[~full_df.index.duplicated(keep="last")]
    full_df.sort_index(inplace=True)
    return full_df


def build_session_anchored_1h_bars(df_1min, symbol, tz_str="America/New_York"):
    """Aggregate 1-minute bars into session-anchored 1h bars (09:30 to 16:00 ET).

    Produces candles anchored at:
    09:30, 10:30, 11:30, 12:30, 13:30, 14:30, and 15:30 (covering the final 30 minutes).
    Half-days naturally end early with the session close.
    """
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
    # offset from 09:30 in minutes
    offset = (df_session.index.hour * 60 + df_session.index.minute) - 570
    bar_num = offset // 60  # 0: 09:30-10:29, 1: 10:30-11:29, ..., 6: 15:30-15:59

    # Build exact bar_start timestamp
    dates = df_session.index.normalize()
    bar_start_times = dates + pd.to_timedelta(570 + bar_num * 60, unit="m")
    df_session["bar_start"] = bar_start_times

    # Aggregations
    # VWAP = sum(close * volume) / sum(volume) or volume-weighted
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

    grouped.index.name = "timestamp"
    grouped["symbol"] = symbol
    return grouped


def main():
    parser = argparse.ArgumentParser(description="Fetch Alpaca 1-min data and build session-anchored 1h bars.")
    parser.add_argument("--refresh", action="store_true", help="Force re-download of all cached chunks.")
    args = parser.parse_args()

    client = get_alpaca_client()
    run_availability_test(client, symbol=CONFIG["ticker"])

    raw_dir = CONFIG["paths"]["raw_dir"]
    proc_dir = Path("data/processed")
    proc_dir.mkdir(parents=True, exist_ok=True)

    # Time window: 4 years + 30 days buffer
    now_utc = datetime.now(timezone.utc)
    end_lag_minutes = CONFIG.get("alpaca_end_lag_minutes", 20)
    end_dt = now_utc - timedelta(minutes=end_lag_minutes)
    total_days = CONFIG["history_years"] * CONFIG["era_days"] + CONFIG["history_buffer_days"]
    start_dt = end_dt - timedelta(days=total_days)

    all_symbols = [CONFIG["ticker"]] + CONFIG["context_tickers"]
    print(f"Fetching data for {all_symbols} from {start_dt.strftime('%Y-%m-%d')} to {end_dt.strftime('%Y-%m-%d')}...")

    for sym in all_symbols:
        print(f"\n--- Processing {sym} ---")
        df_1min = download_symbol_1min(client, sym, start_dt, end_dt, raw_dir, refresh=args.refresh)
        print(f"Total 1-min bars for {sym}: {len(df_1min):,}")

        df_1h = build_session_anchored_1h_bars(df_1min, sym, tz_str=CONFIG["tz"])
        print(f"Total session-anchored 1h bars for {sym}: {len(df_1h):,}")

        out_path = proc_dir / f"bars_{sym}_1h.parquet"
        df_1h.to_parquet(out_path)
        print(f"Saved {out_path} ({len(df_1h)} rows). Range: {df_1h.index[0]} to {df_1h.index[-1]}")

    print("\nData collection complete.")


if __name__ == "__main__":
    main()
