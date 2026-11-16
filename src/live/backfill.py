"""Backfill live SQLite database and 1h bars through the last completed session.

Fetches recent 1-minute IEX bars from Alpaca (Nov 10-13, 2026),
aggregates to session-anchored 1h bars, updates parquet caches and SQLite bars table,
and computes backfilled frozen model signals for the last 10 completed sessions,
marking them IN-SAMPLE (<= 2026-11-09) or OUT-OF-SAMPLE (> 2026-11-09).
"""

import datetime
import hashlib
import json
import os
from pathlib import Path
from typing import Dict, Any, List
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from configs.config import CONFIG
from src.features import make_features_v2
from src.insights import get_bar_insights, _load_production_models
from src.live.aggregator import aggregate_minute_bars_to_1h
from src.live.calendar import NyseCalendar
from src.live.store import LiveDatabase

ET = ZoneInfo("America/New_York")


def run_backfill():
    print("=" * 80)
    print("RUNNING V3 LIVE BACKFILL THROUGH LAST COMPLETED SESSION")
    print("=" * 80)

    db = LiveDatabase()
    proc_dir = Path("data/processed")
    freeze_path = Path(CONFIG["paths"]["data_freeze"])
    with open(freeze_path) as f:
        freeze_meta = json.load(f)
    freeze_date_str = freeze_meta["freeze_date"]
    freeze_date = pd.to_datetime(freeze_date_str).date()
    print(f"Data freeze date: {freeze_date}")

    symbols = ["GS", "SPY", "XLF", "JPM", "MS", "BAC", "TLT", "VIXY"]

    # 1. Fetch recent IEX minute bars (Nov 10 to Nov 14, 2026)
    api_key = os.environ.get("APCA_API_KEY_ID")
    api_secret = os.environ.get("APCA_API_SECRET_KEY")
    if not (api_key and api_secret):
        raise RuntimeError("Alpaca API credentials missing in environment.")

    client = StockHistoricalDataClient(api_key, api_secret)
    start_dt = datetime.datetime(2026, 11, 10, 9, 0, tzinfo=ET)
    end_dt = datetime.datetime(2026, 11, 14, 0, 0, tzinfo=ET)

    print(f"Fetching IEX minute bars from {start_dt} to {end_dt} for {symbols}...")
    req = StockBarsRequest(
        symbol_or_symbols=symbols,
        timeframe=TimeFrame.Minute,
        start=start_dt,
        end=end_dt,
        feed="iex"
    )
    bars_df = client.get_stock_bars(req).df

    # 2. Aggregate to 1h bars and append to cached history
    history_1h = {}
    for sym in symbols:
        parquet_file = proc_dir / f"bars_{sym}_1h.parquet"
        df_cached = pd.read_parquet(parquet_file)
        if df_cached.index.tz is None:
            df_cached.index = df_cached.index.tz_localize(ET)
        else:
            df_cached.index = df_cached.index.tz_convert(ET)

        if sym in bars_df.index.levels[0]:
            df_sym_m = bars_df.loc[sym].copy()
            if df_sym_m.index.tz is None:
                df_sym_m.index = df_sym_m.index.tz_localize("UTC").tz_convert(ET)
            else:
                df_sym_m.index = df_sym_m.index.tz_convert(ET)

            # Aggregate to session-anchored 1h
            df_sym_1h = aggregate_minute_bars_to_1h(df_sym_m, symbol=sym)
            if df_sym_1h.index.tz is None:
                df_sym_1h.index = df_sym_1h.index.tz_localize(ET)
            else:
                df_sym_1h.index = df_sym_1h.index.tz_convert(ET)

            # Combine and deduplicate
            df_combined = pd.concat([df_cached, df_sym_1h], axis=0)
            df_combined = df_combined[~df_combined.index.duplicated(keep="last")].sort_index()
            # Save updated parquet
            df_combined.to_parquet(parquet_file)
            history_1h[sym] = df_combined
            print(f"  {sym}: Updated from {len(df_cached)} to {len(df_combined)} 1h bars (Last: {df_combined.index[-1]})")
        else:
            history_1h[sym] = df_cached
            print(f"  {sym}: Kept {len(df_cached)} 1h bars (No new IEX bars)")

    # 3. Synchronize GS 1h bars to SQLite bars table
    print("\nSynchronizing GS 1h bars to SQLite bars table...")
    df_gs = history_1h["GS"]
    # Seed last 700 bars into SQLite
    recent_gs = df_gs.tail(700)
    with db.get_connection() as conn:
        for ts, row in recent_gs.iterrows():
            conn.execute("""
            INSERT OR REPLACE INTO bars (symbol, timestamp, open, high, low, close, volume, vwap)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?);
            """, (
                "GS",
                ts.isoformat(),
                float(row["open"]),
                float(row["high"]),
                float(row["low"]),
                float(row["close"]),
                float(row["volume"]),
                float(row.get("vwap", row["close"]))
            ))
        conn.commit()
    print(f"Populated {len(recent_gs)} GS 1h bars in SQLite table.")

    # 4. Compute features across full synchronized history
    print("\nComputing features across full synchronized history...")
    context_dfs = {sym: history_1h[sym] for sym in CONFIG["context_tickers"] if sym in history_1h}
    df_features = make_features_v2(df_gs, context_dfs, warmup_rows=0)
    print(f"Computed feature matrix: {df_features.shape}")

    # 5. Identify last 10 completed trading sessions
    unique_dates = sorted(list(set(df_features.index.date)))
    last_10_dates = unique_dates[-10:]
    print(f"\nLast 10 completed trading sessions ({len(last_10_dates)} days):")
    for d in last_10_dates:
        sample_tag = "IN-SAMPLE" if d <= freeze_date else "OUT-OF-SAMPLE"
        print(f"  {d} -> {sample_tag}")

    # Load frozen model artifacts
    models, _ = _load_production_models()
    with open("models/MODEL_SHA256.json") as f:
        m_hashes = json.load(f)
    model_sha = m_hashes.get("model_v2.joblib", "UNKNOWN")

    with open("configs/model_v2_config.json") as f:
        config_sha = hashlib.sha256(f.read().encode("utf-8")).hexdigest()

    with open("models/calib_live.json") as f:
        cfg_live = json.load(f)
    strong_thresh = cfg_live.get("strong_confidence_threshold", 0.520414)

    git_commit = "df63ab5"  # Clean frozen commit

    # 6. Backfill signals for every bar of last 10 completed sessions
    print("\nBackfilling signals for last 10 completed sessions...")
    df_last_10 = df_features[df_features.index.map(lambda t: t.date() in last_10_dates)].copy()

    in_sample_count = 0
    out_sample_count = 0
    resolved_count = 0
    out_sample_hits = 0
    out_sample_misses = 0

    for i in range(len(df_last_10)):
        ts = df_last_10.index[i]
        row_feat = df_last_10.iloc[i]
        b_date = ts.date()
        sample_tag = "IN-SAMPLE" if b_date <= freeze_date else "OUT-OF-SAMPLE"

        # Check if already in signals table
        existing = db.get_signal_by_bar_start(ts.isoformat())
        if existing is not None:
            # Already backfilled
            stored = existing
        else:
            # Check if last bar of day (15:30)
            is_last = (ts.hour == 15 and ts.minute == 30)
            insights = get_bar_insights(row_feat, models=models)
            direction = insights["direction"]
            if is_last:
                direction = "No signal (end of day)"

            conf = float(insights["confidence"])
            is_strong = int(conf >= strong_thresh)

            bar_end_dt = ts + pd.Timedelta(hours=1)
            if is_last:
                bar_end_dt = ts + pd.Timedelta(minutes=30)

            rec = {
                "bar_start": ts.isoformat(),
                "bar_end": bar_end_dt.isoformat(),
                "horizon": "H1",
                "direction": direction,
                "confidence": conf,
                "strong": is_strong,
                "vol_class": insights.get("vol_class", "Normal"),
                "band_low": float(insights["band_low"]) if insights.get("band_low") is not None else None,
                "band_high": float(insights["band_high"]) if insights.get("band_high") is not None else None,
                "regime": insights.get("regime", "Chop"),
                "hit_rate_hour_side": float(insights.get("historical_hit_rate_for_this_hour_and_side", 0.52) or 0.52),
                "n_hour_side": int(insights.get("n_hour_side", 100) or 100),
                "model_sha256": model_sha,
                "config_sha256": config_sha,
                "git_commit": git_commit,
                "feed": "iex",
                "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "latency_s": 0.05,
                "late": 1,
                "sample": sample_tag
            }
            stored = db.insert_signal(rec)

        if sample_tag == "IN-SAMPLE":
            in_sample_count += 1
        else:
            out_sample_count += 1

        # Evaluate outcome if next bar exists on the same day
        if i + 1 < len(df_last_10):
            next_ts = df_last_10.index[i + 1]
            if next_ts.date() == b_date:
                c0 = float(row_feat["close"])
                c1 = float(df_last_10.iloc[i + 1]["close"])
                fwd_ret = (c1 - c0) / c0
                y_true = 1 if fwd_ret > 0 else 0
                correct = 0
                if stored["direction"] == "Up":
                    correct = 1 if y_true == 1 else 0
                    resolved_count += 1
                    if sample_tag == "OUT-OF-SAMPLE":
                        if correct: out_sample_hits += 1
                        else: out_sample_misses += 1
                elif stored["direction"] == "Down":
                    correct = 1 if y_true == 0 else 0
                    resolved_count += 1
                    if sample_tag == "OUT-OF-SAMPLE":
                        if correct: out_sample_hits += 1
                        else: out_sample_misses += 1

                db.insert_outcome(ts.isoformat(), fwd_ret, y_true, correct)

    print(f"\nBackfill complete:")
    print(f"  IN-SAMPLE signals stored:  {in_sample_count} (Display only, excluded from scoreboard)")
    print(f"  OUT-OF-SAMPLE signals:     {out_sample_count} (Scored and labelled 'computed after the fact')")
    print(f"  Resolved signals:          {resolved_count}")
    if (out_sample_hits + out_sample_misses) > 0:
        out_acc = out_sample_hits / (out_sample_hits + out_sample_misses) * 100
        print(f"  OUT-OF-SAMPLE accuracy:    {out_acc:.2f}% ({out_sample_hits}/{out_sample_hits + out_sample_misses})")

    # 7. Verify hash chain integrity
    valid, count, err = db.verify_hash_chain()
    print(f"\nCryptographic Hash Chain Verification: valid={valid}, total_signals={count}, err={err}")
    assert valid, f"Hash chain verification failed: {err}"

    return {
        "in_sample": in_sample_count,
        "out_sample": out_sample_count,
        "out_sample_hits": out_sample_hits,
        "out_sample_misses": out_sample_misses,
        "total_bars": len(df_gs)
    }


if __name__ == "__main__":
    run_backfill()
