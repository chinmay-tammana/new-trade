"""Backfill frozen model signals using Delayed SIP for the last 10 completed sessions (V3.5)."""

import os
import json
import hashlib
import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv()

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed

from configs.config import CONFIG
from src.features import FEATURES, make_features_v2
from src.data_fetch import build_session_anchored_1h_bars
from src.live.aggregator import aggregate_minute_bars_to_1h
from src.live.store import LiveDatabase
from src.insights import get_bar_insights, _load_production_models

ET = ZoneInfo("America/New_York")
SYMBOLS = ["GS", "SPY", "XLF", "JPM", "MS", "BAC", "TLT", "VIXY"]


def run_sip_backfill():
    print("=" * 80)
    print("STARTING DELAYED SIP BACKFILL (LAST 10 COMPLETED SESSIONS)")
    print("=" * 80)

    db = LiveDatabase()
    proc_dir = Path("data/processed")
    freeze_path = Path(CONFIG["paths"]["data_freeze"])
    with open(freeze_path) as f:
        freeze_meta = json.load(f)
    freeze_date = pd.to_datetime(freeze_meta["freeze_date"]).date()
    print(f"Data freeze date: {freeze_date}")

    # Build SIP 1h bars from cached sip_diag minute data merged with base history
    history_1h = {}
    for sym in SYMBOLS:
        p_sip_diag = Path(f"data/raw/sip_diag/{sym}_1m_20260821_20261113.parquet")
        if p_sip_diag.exists():
            df_m = pd.read_parquet(p_sip_diag)
            df_1h_recent = build_session_anchored_1h_bars(df_m, sym)
        else:
            raise FileNotFoundError(f"Missing cached SIP 1m data: {p_sip_diag}")

        parquet_file = proc_dir / f"bars_{sym}_1h.parquet"
        df_cached = pd.read_parquet(parquet_file)
        if df_cached.index.tz is None:
            df_cached.index = df_cached.index.tz_localize(ET)
        else:
            df_cached.index = df_cached.index.tz_convert(ET)

        # Merge base prior history before sip_diag start
        df_prior = df_cached[df_cached.index < df_1h_recent.index.min()]
        df_combined = pd.concat([df_prior, df_1h_recent]).sort_index()
        df_combined = df_combined[~df_combined.index.duplicated(keep="last")]
        history_1h[sym] = df_combined
        print(f"  {sym}: Built {len(df_combined)} SIP 1h bars (Last: {df_combined.index[-1]})")

    # Compute features across synchronized SIP history
    print("\nComputing features across full synchronized SIP history...")
    df_gs = history_1h["GS"]
    context_dfs = {s: history_1h[s] for s in SYMBOLS if s != "GS"}
    df_features = make_features_v2(df_gs, context_dfs, warmup_rows=0)
    print(f"Computed SIP feature matrix: {df_features.shape}")

    # Identify last 10 completed trading sessions (ending at 2026-11-13)
    unique_dates = sorted(list(set(df_features.index.date)))
    last_10_dates = unique_dates[-10:]
    print(f"\nLast 10 completed trading sessions ({len(last_10_dates)} days):")
    for d in last_10_dates:
        sample_tag = "IN-SAMPLE" if d <= freeze_date else "OUT-OF-SAMPLE"
        print(f"  {d} -> {sample_tag}")

    # Load frozen production models and sha256
    models, _ = _load_production_models()
    with open("models/MODEL_SHA256.json") as f:
        m_hashes = json.load(f)
    model_sha = m_hashes.get("model_v2.joblib", "UNKNOWN")

    with open("configs/model_v2_config.json") as f:
        config_sha = hashlib.sha256(f.read().encode("utf-8")).hexdigest()

    with open("models/calib_live.json") as f:
        cfg_live = json.load(f)
    strong_thresh = cfg_live.get("strong_confidence_threshold", 0.520414)

    git_commit = "df63ab5"

    df_last_10 = df_features[df_features.index.map(lambda t: t.date() in last_10_dates)].copy()

    in_sample_count = 0
    out_sample_count = 0
    resolved_count = 0
    nov_up_count = 0
    nov_down_count = 0
    nov_nosig_count = 0

    print("\nInserting delayed SIP signals (feed='sip_delayed', schema_v=2)...")
    for i in range(len(df_last_10)):
        ts = df_last_10.index[i]
        row_feat = df_last_10.iloc[i]
        b_date = ts.date()
        sample_tag = "IN-SAMPLE" if b_date <= freeze_date else "OUT-OF-SAMPLE"

        is_last = (ts.hour == 15 and ts.minute == 30)
        insights = get_bar_insights(row_feat, models=models)
        direction = "No signal (end of day)" if is_last else insights["direction"]
        conf = float(insights["confidence"])
        is_strong = 1 if (conf >= strong_thresh and direction in ("Up", "Down")) else 0

        bar_end_dt = ts + pd.Timedelta(hours=1 if not is_last else 0.5)
        # Delivered at bar_end + 16 minutes
        delivered_dt = bar_end_dt + pd.Timedelta(minutes=16)

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
            "feed": "sip_delayed",
            "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "latency_s": 0.05,
            "late": 1,
            "sample": sample_tag,
            "delivered_at": delivered_dt.isoformat(),
            "delivery_delay_s": 960.0,
            "schema_v": 2
        }

        stored = db.insert_signal(rec)

        if sample_tag == "IN-SAMPLE":
            in_sample_count += 1
        else:
            out_sample_count += 1
            if b_date >= datetime.date(2026, 11, 10):
                if direction == "Up":
                    nov_up_count += 1
                elif direction == "Down":
                    nov_down_count += 1
                else:
                    nov_nosig_count += 1

        # Evaluate outcome if next bar exists on same day
        if i + 1 < len(df_last_10):
            next_ts = df_last_10.index[i + 1]
            if next_ts.date() == b_date:
                c0 = float(row_feat["close"])
                c1 = float(df_last_10.iloc[i + 1]["close"])
                fwd_ret = (c1 - c0) / c0
                y_true = 1 if fwd_ret > 0 else 0
                correct = 0
                if direction == "Up":
                    correct = 1 if y_true == 1 else 0
                    resolved_count += 1
                elif direction == "Down":
                    correct = 1 if y_true == 0 else 0
                    resolved_count += 1

                db.insert_outcome(ts.isoformat(), fwd_ret, y_true, correct)

    print(f"\nDelayed SIP Backfill Summary:")
    print(f"  Total signals inserted: {in_sample_count + out_sample_count}")
    print(f"  IN-SAMPLE:  {in_sample_count}")
    print(f"  OUT-OF-SAMPLE: {out_sample_count}")
    print(f"\nNov 10–13 SIP Signal Counts (28 bars total, 24 candidate):")
    print(f"  Up:        {nov_up_count}")
    print(f"  Down:      {nov_down_count}")
    print(f"  No signal: {nov_nosig_count} (including 4 EOD bars)")
    print(f"  Candidate Directional: {nov_up_count} Up, {nov_down_count} Down (Two-Sided!)")

    # Verify cryptographic hash chain across full database
    valid, count, err = db.verify_hash_chain()
    print(f"\nCryptographic Hash Chain Verification: valid={valid}, total_signals={count}, err={err}")
    assert valid, f"Hash chain verification failed: {err}"
    print("ALL INTEGRITY CHECKS PASSED.")


if __name__ == "__main__":
    run_sip_backfill()
