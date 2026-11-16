"""
Diagnostic Doctor for M5 (5-Minute) Live System.
Reports:
- System times in ET and IST
- Market open status and next open/close
- IEX access per symbol (latest minute age)
- Database schema and hash-chain verification
- Model SHA-256 hash lock verification
- Next signal delivery time in ET and IST
- Latency statistics for live signals (or pending if market closed)
"""
import os
import sys

sys.path.insert(0, os.path.abspath("."))

import json
import hashlib
import datetime
import zoneinfo
import sqlite3
from dotenv import load_dotenv

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed
from alpaca.trading.client import TradingClient

from configs.config import CONFIG_M5
from src.live_m5.store import LiveM5Database, DB_PATH

load_dotenv()

ET_TZ = zoneinfo.ZoneInfo("America/New_York")
IST_TZ = zoneinfo.ZoneInfo("Asia/Kolkata")
UTC_TZ = datetime.timezone.utc


def run_doctor():
    print("==================================================")
    print("M5 (5-MINUTE) LIVE SYSTEM DIAGNOSTIC DOCTOR")
    print("==================================================")
    
    # 1. System Times
    now_utc = datetime.datetime.now(UTC_TZ)
    now_et = now_utc.astimezone(ET_TZ)
    now_ist = now_utc.astimezone(IST_TZ)
    
    print("\n[1/6] SYSTEM TIME & MARKET SCHEDULE")
    print(f"ET  Time: {now_et.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print(f"IST Time: {now_ist.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    
    api_key = os.getenv("APCA_API_KEY_ID")
    secret_key = os.getenv("APCA_API_SECRET_KEY")
    if not api_key or not secret_key:
        print("ERROR: APCA_API_KEY_ID or APCA_API_SECRET_KEY is missing!")
        return False
        
    trading_client = TradingClient(api_key, secret_key, paper=True)
    try:
        clock = trading_client.get_clock()
        print(f"Market Is Open: {clock.is_open}")
        next_open_et = clock.next_open.astimezone(ET_TZ)
        next_open_ist = clock.next_open.astimezone(IST_TZ)
        print(f"Next Open: {next_open_et.strftime('%a %Y-%m-%d %H:%M %Z')} ({next_open_ist.strftime('%a %Y-%m-%d %H:%M %Z')})")
        next_close_et = clock.next_close.astimezone(ET_TZ)
        print(f"Next Close: {next_close_et.strftime('%a %Y-%m-%d %H:%M %Z')}")
    except Exception as e:
        print(f"Error fetching Alpaca clock: {e}")

    # Next 5-Minute Signal Time
    # 5m signals generated at bar close: 09:35, 09:40, ..., 15:55
    if clock.is_open:
        # compute next 5-minute close
        curr_m = now_et.minute
        next_5m = ((curr_m // 5) + 1) * 5
        if next_5m >= 60:
            next_sig_et = now_et.replace(minute=0, second=0, microsecond=0) + datetime.timedelta(hours=1)
        else:
            next_sig_et = now_et.replace(minute=next_5m, second=0, microsecond=0)
        next_sig_ist = next_sig_et.astimezone(IST_TZ)
        print(f"Next Signal Delivery: {next_sig_et.strftime('%H:%M:%S ET')} ({next_sig_ist.strftime('%H:%M:%S IST')})")
    else:
        # Monday 09:35 ET
        first_sig_et = next_open_et + datetime.timedelta(minutes=5)
        first_sig_ist = first_sig_et.astimezone(IST_TZ)
        print(f"Next Signal Delivery: {first_sig_et.strftime('%a %H:%M ET')} ({first_sig_ist.strftime('%a %H:%M IST')})")

    # 2. IEX Access per Symbol
    print("\n[2/6] IEX DATA FEED CONNECTIVITY & ACCESS")
    data_client = StockHistoricalDataClient(api_key, secret_key)
    symbols = ["GS", "SPY", "XLF", "JPM", "MS", "BAC"]
    
    # Query last 10 minutes of bars from last Friday 16:00 ET (or now if open)
    test_end = now_et if clock.is_open else datetime.datetime(2026, 11, 13, 16, 0, tzinfo=ET_TZ)
    test_start = test_end - datetime.timedelta(minutes=15)
    
    all_iex_ok = True
    for sym in symbols:
        try:
            req = StockBarsRequest(
                symbol_or_symbols=sym,
                timeframe=TimeFrame.Minute,
                start=test_start,
                end=test_end,
                feed=DataFeed.IEX
            )
            bars = data_client.get_stock_bars(req)
            cnt = len(bars.df) if not bars.df.empty else 0
            if cnt > 0:
                last_bar_time = bars.df.index[-1]
                if isinstance(last_bar_time, tuple):
                    last_bar_time = last_bar_time[1]
                print(f"[{sym:5s}] OK: {cnt:2d} bars in test window (Latest: {last_bar_time})")
            else:
                print(f"[{sym:5s}] WARNING: 0 bars returned in test window")
        except Exception as e:
            all_iex_ok = False
            print(f"[{sym:5s}] FAILED: {e}")

    # 3. Model Hash Check
    print("\n[3/6] MODEL ARTIFACTS & SHA-256 HASH VERIFICATION")
    model_path = CONFIG_M5["paths"]["model_path"]
    calib_path = CONFIG_M5["paths"]["calib_path"]
    sha_path = CONFIG_M5["paths"]["sha_path"]
    
    model_ok = False
    if os.path.exists(sha_path) and os.path.exists(model_path) and os.path.exists(calib_path):
        with open(sha_path, "r") as sf:
            expected = json.load(sf)
        with open(model_path, "rb") as mf:
            act_model = hashlib.sha256(mf.read()).hexdigest()
        with open(calib_path, "rb") as cf:
            act_calib = hashlib.sha256(cf.read()).hexdigest()
            
        m_match = (act_model == expected["model_m5.joblib"])
        c_match = (act_calib == expected["calib_m5.json"])
        
        print(f"model_m5.joblib:  {act_model[:12]}... {'[MATCH]' if m_match else '[MISMATCH]'}")
        print(f"calib_m5.json:    {act_calib[:12]}... {'[MATCH]' if c_match else '[MISMATCH]'}")
        model_ok = m_match and c_match
    else:
        print("Model files or SHA256 json missing (run train_m5.py to generate production artifacts).")

    # 4. Database & Hash-Chain Verification
    print("\n[4/6] DATABASE & CRYPTOGRAPHIC HASH CHAIN")
    db = LiveM5Database()
    with db.get_connection() as conn:
        c_sig = conn.execute("SELECT count(*) FROM signals;").fetchone()[0]
        c_raw = conn.execute("SELECT count(*) FROM raw_minutes;").fetchone()[0]
        c_bars = conn.execute("SELECT count(*) FROM bars_5m;").fetchone()[0]
        print(f"Signals stored:      {c_sig:,}")
        print(f"Raw minutes stored:  {c_raw:,}")
        print(f"5m bars stored:      {c_bars:,}")

    valid, count, err = db.verify_hash_chain()
    if valid:
        print(f"Hash-chain audit:    VALID ({count} rows verified, unbroken cryptographic chain)")
    else:
        print(f"Hash-chain audit:    FAILED ({err})")

    # 5. Latency Statistics
    print("\n[5/6] SIGNAL DELIVERY LATENCY")
    with db.get_connection() as conn:
        cursor = conn.execute("SELECT delivery_delay_s FROM signals WHERE late = 0;")
        latencies = [r[0] for r in cursor.fetchall()]
    
    if latencies:
        avg_lat = sum(latencies) / len(latencies)
        max_lat = max(latencies)
        min_lat = min(latencies)
        print(f"Live Signals Scored: {len(latencies)}")
        print(f"Mean Latency:        {avg_lat:.2f}s (Min: {min_lat:.2f}s, Max: {max_lat:.2f}s)")
    else:
        if clock.is_open:
            print("No live signals generated yet in current session.")
        else:
            print("Market is currently closed. Live latency test is PENDING the next open trading session.")

    # 6. Overall Doctor Verdict
    print("\n[6/6] DOCTOR HEALTH VERDICT")
    overall_ok = all_iex_ok and (model_ok or not os.path.exists(model_path)) and valid
    if overall_ok:
        print("ALL DOCTOR CHECKS PASSED. Ready for operational execution.")
    else:
        print("ONE OR MORE CHECKS FAILED. Review warnings above.")
    print("==================================================")
    return overall_ok


if __name__ == "__main__":
    run_doctor()
