"""
Download 1-minute historical bars from Alpaca IEX for M5 pipeline.
Cached monthly as data/raw/iex_1min/{symbol}/{YYYY-MM}.parquet
Resumable, rate-limit aware, exponential backoff.
"""
import os
import sys
import time
import argparse
import datetime
import zoneinfo
import pandas as pd
from dotenv import load_dotenv

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed, Adjustment

load_dotenv()

ET_TZ = zoneinfo.ZoneInfo("America/New_York")
UTC_TZ = datetime.timezone.utc

SYMBOLS = ["GS", "SPY", "XLF", "JPM", "MS", "BAC", "TLT", "VIXY"]
RAW_DIR = "data/raw/iex_1min"


def get_client() -> StockHistoricalDataClient:
    api_key = os.getenv("APCA_API_KEY_ID")
    secret_key = os.getenv("APCA_API_SECRET_KEY")
    if not api_key or not secret_key:
        raise ValueError("APCA_API_KEY_ID and APCA_API_SECRET_KEY must be set in .env")
    return StockHistoricalDataClient(api_key, secret_key)


def get_month_ranges(start_year: int = 2021, start_month: int = 11, end_date: datetime.date = None):
    if end_date is None:
        end_date = datetime.date(2026, 11, 13)
    
    ranges = []
    y, m = start_year, start_month
    while True:
        m_start = datetime.datetime(y, m, 1, 0, 0, tzinfo=ET_TZ)
        if m == 12:
            next_y, next_m = y + 1, 1
        else:
            next_y, next_m = y, m + 1
        
        m_end = datetime.datetime(next_y, next_m, 1, 0, 0, tzinfo=ET_TZ)
        
        # If this month goes past end_date, cap m_end
        cap_end = datetime.datetime(end_date.year, end_date.month, end_date.day, 23, 59, 59, tzinfo=ET_TZ)
        is_last = False
        if m_end > cap_end:
            m_end = cap_end
            is_last = True
            
        ranges.append((f"{y:04d}-{m:02d}", m_start, m_end))
        if is_last or (y == end_date.year and m == end_date.month):
            break
        y, m = next_y, next_m
        
    return ranges


def fetch_symbol_month(client: StockHistoricalDataClient, symbol: str, month_str: str, start_dt: datetime.datetime, end_dt: datetime.datetime, out_path: str, refresh: bool = False):
    if os.path.exists(out_path) and not refresh:
        # Check if non-empty
        try:
            df_exist = pd.read_parquet(out_path)
            if not df_exist.empty or month_str < "2026-11":
                return len(df_exist), "cached"
        except Exception:
            pass

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    req = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Minute,
        start=start_dt,
        end=end_dt,
        feed=DataFeed.IEX,
        adjustment=Adjustment.RAW
    )
    
    max_retries = 5
    for attempt in range(max_retries):
        try:
            bars = client.get_stock_bars(req)
            df = bars.df
            if df.empty:
                df_out = pd.DataFrame(columns=["open", "high", "low", "close", "volume", "trade_count", "vwap"])
            else:
                if isinstance(df.index, pd.MultiIndex):
                    df = df.xs(symbol, level="symbol")
                df = df.sort_index()
                df_out = df[["open", "high", "low", "close", "volume", "trade_count", "vwap"]].copy()
            
            df_out.to_parquet(out_path)
            return len(df_out), "downloaded"
        except Exception as e:
            err_str = str(e)
            if "rate limit" in err_str.lower() or "429" in err_str:
                sleep_s = (2 ** attempt) * 2
                print(f"[{symbol} {month_str}] Rate limited, backing off {sleep_s}s...")
                time.sleep(sleep_s)
            elif attempt == max_retries - 1:
                print(f"[{symbol} {month_str}] Failed after {max_retries} attempts: {e}")
                raise e
            else:
                time.sleep(2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true", help="Force overwrite existing cached chunks")
    parser.add_argument("--symbols", nargs="+", default=SYMBOLS)
    args = parser.parse_args()

    client = get_client()
    month_ranges = get_month_ranges()
    
    print(f"Starting Alpaca IEX 1-minute historical data sync...")
    print(f"Symbols: {args.symbols}")
    print(f"Month chunks: {len(month_ranges)} months ({month_ranges[0][0]} to {month_ranges[-1][0]})")
    
    for sym in args.symbols:
        t0 = time.time()
        total_bars = 0
        cached_count = 0
        downloaded_count = 0
        for m_str, m_start, m_end in month_ranges:
            out_file = os.path.join(RAW_DIR, sym, f"{m_str}.parquet")
            n_bars, status = fetch_symbol_month(client, sym, m_str, m_start, m_end, out_file, refresh=args.refresh)
            total_bars += n_bars
            if status == "cached":
                cached_count += 1
            else:
                downloaded_count += 1
            # Polite pause to avoid hitting Alpaca 200 req/min
            time.sleep(0.15)
        
        elapsed = time.time() - t0
        print(f"[{sym:5s}] Done in {elapsed:.1f}s: {total_bars:,} bars total ({cached_count} cached, {downloaded_count} downloaded)")


if __name__ == "__main__":
    main()
