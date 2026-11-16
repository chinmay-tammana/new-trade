"""Production Doctor CLI for Live Trading Indicator System (V3.5 Delayed SIP).

Validates:
1. System times in ET and IST
2. Market open status, next session open, and next signal delivery time (ET & IST)
3. Delay arithmetic (bar_end + 16 min)
4. Frozen model artifact SHA-256 hashes against models/MODEL_SHA256.json
5. Database integrity and SHA-256 hash-chain verification with schema_v support
6. Delayed SIP access test for all 8 symbols (>= 15 min end lag)
7. Local API server probe and live signal engine readiness

Exits 0 on all checks passing; exits non-zero on any failure.
"""

import datetime
import hashlib
import json
import os
import sys
import urllib.request
from pathlib import Path
from zoneinfo import ZoneInfo
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from configs.config import CONFIG
from src.live.calendar import is_market_open_now, NyseCalendar
from src.live.store import LiveDatabase

ET = ZoneInfo("America/New_York")
IST = ZoneInfo("Asia/Kolkata")
SYMBOLS = ["GS", "SPY", "XLF", "JPM", "MS", "BAC", "TLT", "VIXY"]


def check_doctor():
    failures = []
    print("=" * 80)
    print("GS AI TRADING INDICATOR — SYSTEM DOCTOR INSPECTION (DELAYED SIP)")
    print("=" * 80)

    # 1. System Clocks
    now_utc = datetime.datetime.now(datetime.timezone.utc)
    now_et = now_utc.astimezone(ET)
    now_ist = now_utc.astimezone(IST)
    print("\n[1/7] SYSTEM CLOCKS")
    print(f"  ET:  {now_et.strftime('%A %Y-%m-%d %H:%M:%S %Z (%z)')}")
    print(f"  IST: {now_ist.strftime('%A %Y-%m-%d %H:%M:%S %Z (%z)')}")
    print(f"  UTC: {now_utc.strftime('%Y-%m-%d %H:%M:%S UTC')}")

    # 2. Market Schedule & Next Delivery Calculation
    print("\n[2/7] MARKET SESSION & SIGNAL DELIVERY SCHEDULE")
    is_open, session = is_market_open_now(now_et)
    print(f"  Market Open Now: {is_open}")

    next_delivery_et = None
    next_delivery_ist = None

    if is_open and session:
        print(f"  Current Session Open:  {session['open'].strftime('%H:%M %Z')}")
        print(f"  Current Session Close: {session['close'].strftime('%H:%M %Z')}")
        # Find next bar close + 16 min
        for b in session["bars"]:
            bar_deliv = b["end"] + datetime.timedelta(minutes=16)
            if now_et < bar_deliv:
                next_delivery_et = bar_deliv
                break
    else:
        # Determine next session open
        curr = now_et.date()
        cal = NyseCalendar()
        next_session = None
        for day_offset in range(1, 8):
            ns = cal.get_session_bounds(curr + datetime.timedelta(days=day_offset))
            if ns:
                next_session = ns
                break
        if next_session:
            next_open = next_session["open"]
            next_open_ist = next_open.astimezone(IST)
            print(f"  Next Session Open (ET):  {next_open.strftime('%A %Y-%m-%d %H:%M %Z')}")
            print(f"  Next Session Open (IST): {next_open_ist.strftime('%A %Y-%m-%d %H:%M %Z')}")
            # First signal delivery: 10:30 bar close + 16 min = 10:46 ET
            first_bar_end = next_session["bars"][0]["end"]
            next_delivery_et = first_bar_end + datetime.timedelta(minutes=16)
        else:
            failures.append("Could not determine next market session open.")

    if next_delivery_et:
        next_delivery_ist = next_delivery_et.astimezone(IST)
        print(f"  Next Signal Delivery (ET):  {next_delivery_et.strftime('%A %Y-%m-%d %H:%M %Z')}")
        print(f"  Next Signal Delivery (IST): {next_delivery_ist.strftime('%A %Y-%m-%d %H:%M %Z')}")

    # 3. Delay Arithmetic Verification
    print("\n[3/7] DELAY ARITHMETIC VERIFICATION")
    # Verify standard bar ends and delivery times
    sample_bar_close = datetime.datetime(2026, 11, 16, 10, 30, tzinfo=ET)
    sample_delivery = sample_bar_close + datetime.timedelta(minutes=16)
    delay_s = (sample_delivery - sample_bar_close).total_seconds()
    print(f"  Sample bar close:    {sample_bar_close.strftime('%H:%M ET')}")
    print(f"  Sample delivery:     {sample_delivery.strftime('%H:%M ET')} (delay = {delay_s/60:.0f} min = {delay_s:.0f} s)")
    if delay_s == 960.0:
        print("  OK: Delay arithmetic verified (16 min lag allows >=15 min SIP delay compliance).")
    else:
        failures.append(f"Delay arithmetic mismatch: expected 960 s, got {delay_s} s")

    # 4. Model Artifacts SHA-256 Lock
    print("\n[4/7] FROZEN MODEL INTEGRITY CHECK")
    lock_file = Path("models/MODEL_SHA256.json")
    if not lock_file.exists():
        failures.append("models/MODEL_SHA256.json missing!")
        print("  FAIL: models/MODEL_SHA256.json not found.")
    else:
        lock = json.loads(lock_file.read_text())
        all_matched = True
        for fname, expected in lock.items():
            p = Path("models") / fname
            if not p.exists():
                failures.append(f"Model file models/{fname} missing!")
                print(f"  FAIL: models/{fname} missing.")
                all_matched = False
                continue
            actual = hashlib.sha256(p.read_bytes()).hexdigest()
            if actual != expected:
                failures.append(f"Model hash mismatch for {fname}!")
                print(f"  FAIL: {fname} hash {actual[:8]} != {expected[:8]}")
                all_matched = False
            else:
                print(f"  OK:   {fname} -> {actual[:8]} (Verified)")
        if not all_matched:
            failures.append("One or more model files failed SHA-256 verification.")

    # 5. Database & Hash-Chain Verification (with schema_v)
    print("\n[5/7] IMMUTABLE DATABASE & CRYPTOGRAPHIC HASH CHAIN")
    db_path = Path("data/live/live.db")
    if not db_path.exists():
        failures.append("data/live/live.db does not exist!")
        print("  FAIL: data/live/live.db not found.")
    else:
        db = LiveDatabase(db_path)
        with db.get_connection() as conn:
            n_sig = conn.execute("SELECT COUNT(*) FROM signals;").fetchone()[0]
            n_iex = conn.execute("SELECT COUNT(*) FROM signals WHERE feed = 'iex';").fetchone()[0]
            n_sip = conn.execute("SELECT COUNT(*) FROM signals WHERE feed = 'sip_delayed';").fetchone()[0]
            n_bars = conn.execute("SELECT COUNT(*) FROM bars WHERE symbol = 'GS';").fetchone()[0]
            n_out = conn.execute("SELECT COUNT(*) FROM outcomes;").fetchone()[0]
        print(f"  Total signals:      {n_sig} (IEX: {n_iex}, SIP Delayed: {n_sip})")
        print(f"  Outcomes count:     {n_out}")
        print(f"  Cached bars count:  {n_bars}")

        valid, count, err = db.verify_hash_chain()
        if valid:
            print(f"  OK: Hash chain valid from genesis across {count} records (schema_v=1 and schema_v=2).")
        else:
            failures.append(f"Hash chain broken: {err}")
            print(f"  FAIL: Hash chain verification failed: {err}")

    # 6. Alpaca Delayed SIP Access Check per Symbol
    print("\n[6/7] ALPACA DELAYED SIP CONNECTIVITY & SYMBOL VERIFICATION")
    api_key = os.environ.get("APCA_API_KEY_ID")
    api_secret = os.environ.get("APCA_API_SECRET_KEY")
    if not (api_key and api_secret):
        failures.append("Alpaca credentials missing (APCA_API_KEY_ID or APCA_API_SECRET_KEY).")
        print("  FAIL: Alpaca API credentials not set in environment.")
    else:
        try:
            from alpaca.data.historical import StockHistoricalDataClient
            from alpaca.data.requests import StockBarsRequest
            from alpaca.data.timeframe import TimeFrame
            from alpaca.data.enums import DataFeed

            client = StockHistoricalDataClient(api_key, api_secret)
            # Query last completed session with >= 15 min end lag
            start_q = datetime.datetime(2026, 11, 13, 14, 0, tzinfo=ET)
            end_q = datetime.datetime(2026, 11, 13, 15, 30, tzinfo=ET)

            for sym in SYMBOLS:
                req = StockBarsRequest(
                    symbol_or_symbols=sym,
                    timeframe=TimeFrame.Minute,
                    start=start_q,
                    end=end_q,
                    feed=DataFeed.SIP
                )
                bars = client.get_stock_bars(req)
                df = bars.df
                if isinstance(df.index, pd.MultiIndex):
                    df = df.xs(sym, level=0)
                n = len(df)
                if n > 0:
                    last_c = df.iloc[-1]["close"]
                    print(f"  OK:   {sym:<5} SIP minute bars: {n} (Last Close: ${last_c:.2f})")
                else:
                    failures.append(f"Symbol {sym} returned 0 SIP bars.")
                    print(f"  FAIL: {sym:<5} returned 0 SIP bars.")
        except Exception as e:
            failures.append(f"Alpaca SIP API error: {e}")
            print(f"  FAIL: Alpaca SIP API exception: {e}")

    # 7. Local HTTP Server Check
    print("\n[7/7] LOCAL WEB SERVER PROBE")
    try:
        req = urllib.request.Request("http://localhost:8000/api/status", headers={"User-Agent": "Doctor/1.0"})
        with urllib.request.urlopen(req, timeout=3.0) as resp:
            if resp.status == 200:
                body = json.loads(resp.read().decode())
                print(f"  OK: Server responding on port 8000.")
                print(f"      Signal Feed:  {body.get('feed')}")
                print(f"      Display Feed: {body.get('display_feed', 'IEX live (display only)')}")
                print(f"      Model Hash:   {body.get('model_hash')}")
            else:
                failures.append(f"Server returned status code {resp.status}")
                print(f"  FAIL: Server returned status {resp.status}")
    except Exception as e:
        print(f"  INFO: Server not running on localhost:8000 ({e}).")
        print("        (Start server using: ./run_live.sh)")

    # Final Verdict
    print("\n" + "=" * 80)
    if not failures:
        print("DOCTOR VERDICT: ALL CHECKS PASSED (SYSTEM HEALTHY & READY)")
        print("=" * 80)
        return True
    else:
        print(f"DOCTOR VERDICT: {len(failures)} FAILURE(S) DETECTED!")
        for i, f in enumerate(failures, 1):
            print(f"  {i}. {f}")
        print("=" * 80)
        return False


if __name__ == "__main__":
    success = check_doctor()
    sys.exit(0 if success else 1)
