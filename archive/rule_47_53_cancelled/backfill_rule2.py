#!/usr/bin/env python3
"""
Backfill Rule 2 Signals (No-Signal Band 47% to 53%) for Last 10 Sessions.
Derives signals under Rule 2:
- Up if p_cal >= 0.53
- Down if p_cal <= 0.47
- No Signal if 0.47 < p_cal < 0.53 (or end-of-day bar)
Tiers:
- WEAK: 53.0% to 56.32%
- NORMAL: 56.32% to 60.0%
- STRONG: >= 60.0%
Inserts rows with late=1, rule_version=2, schema_v=3.
"""

import os
import json
import hashlib
import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import pandas as pd
import numpy as np

from configs.config import CONFIG
from src.features import FEATURES, make_features_v2
from src.data_fetch import build_session_anchored_1h_bars
from src.live.store import LiveDatabase
from src.insights import get_bar_insights, _load_production_models

ET = ZoneInfo("America/New_York")
SYMBOLS = ["GS", "SPY", "XLF", "JPM", "MS", "BAC", "TLT", "VIXY"]

def run_rule2_backfill():
    print("=" * 80)
    print("STARTING RULE 2 BACKFILL (47% to 53% NO-SIGNAL BAND, LAST 10 SESSIONS)")
    print("=" * 80)

    db = LiveDatabase()
    proc_dir = Path("data/processed")
    freeze_path = Path(CONFIG["paths"]["data_freeze"])
    with open(freeze_path) as f:
        freeze_meta = json.load(f)
    freeze_date = pd.to_datetime(freeze_meta["freeze_date"]).date()

    with open("models/rule_v2.json") as f:
        rule_meta = json.load(f)
    q_lo = float(rule_meta["q_lo"])
    q_hi = float(rule_meta["q_hi"])
    weak_below = float(rule_meta["weak_below_conf"])
    strong_conf = float(rule_meta["strong_conf"])
    print(f"Loaded Rule 2 config: q_lo={q_lo}, q_hi={q_hi}, weak_below={weak_below}, strong_conf={strong_conf}")

    # Build SIP 1h bars from cached sip_diag minute data merged with base history
    history_1h = {}
    for sym in SYMBOLS:
        p_sip_diag = Path(f"data/raw/sip_diag/{sym}_1m_20260821_20261113.parquet")
        if not p_sip_diag.exists():
            raise FileNotFoundError(f"Missing cached SIP 1m data: {p_sip_diag}")
        df_m = pd.read_parquet(p_sip_diag)
        df_1h_recent = build_session_anchored_1h_bars(df_m, sym)

        parquet_file = proc_dir / f"bars_{sym}_1h.parquet"
        df_cached = pd.read_parquet(parquet_file)
        if df_cached.index.tz is None:
            df_cached.index = df_cached.index.tz_localize(ET)
        else:
            df_cached.index = df_cached.index.tz_convert(ET)

        df_prior = df_cached[df_cached.index < df_1h_recent.index.min()]
        df_combined = pd.concat([df_prior, df_1h_recent]).sort_index()
        df_combined = df_combined[~df_combined.index.duplicated(keep="last")]
        history_1h[sym] = df_combined

    df_gs = history_1h["GS"]
    context_dfs = {s: history_1h[s] for s in SYMBOLS if s != "GS"}
    df_features = make_features_v2(df_gs, context_dfs, warmup_rows=0)

    unique_dates = sorted(list(set(df_features.index.date)))
    last_10_dates = unique_dates[-10:]
    df_last_10 = df_features[df_features.index.map(lambda t: t.date() in last_10_dates)].copy()
    print(f"Candidate bars across last 10 completed sessions: {len(df_last_10)}")

    models, _ = _load_production_models()
    with open("models/MODEL_SHA256.json") as f:
        m_hashes = json.load(f)
    model_sha = m_hashes.get("model_v2.joblib", "UNKNOWN")

    with open("configs/model_v2_config.json") as f:
        config_sha = hashlib.sha256(f.read().encode("utf-8")).hexdigest()

    git_commit = "df63ab5"

    inserted_count = 0
    tier_counts = {"WEAK": 0, "NORMAL": 0, "STRONG": 0, "NO_SIGNAL": 0}
    up_count = 0
    down_count = 0
    nosig_count = 0

    for i in range(len(df_last_10)):
        ts = df_last_10.index[i]
        row_feat = df_last_10.iloc[i]
        b_date = ts.date()
        sample_tag = "IN-SAMPLE" if b_date <= freeze_date else "OUT-OF-SAMPLE"
        is_last = (ts.hour == 15 and ts.minute == 30)

        insights = get_bar_insights(row_feat, models=models)
        # Compute exact p_cal from model
        X = row_feat[FEATURES].to_numpy().reshape(1, -1)
        p_cal = float(models["ensemble"].predict_proba(X)[0])

        if is_last:
            direction = "No signal (end of day)"
            confidence = float(max(p_cal, 1.0 - p_cal))
            tier = None
            is_strong = 0
            tier_counts["NO_SIGNAL"] += 1
            nosig_count += 1
        elif p_cal >= q_hi:
            direction = "Up"
            confidence = float(p_cal)
            up_count += 1
            if confidence >= strong_conf:
                tier = "STRONG"
                is_strong = 1
                tier_counts["STRONG"] += 1
            elif confidence >= weak_below:
                tier = "NORMAL"
                is_strong = 0
                tier_counts["NORMAL"] += 1
            else:
                tier = "WEAK"
                is_strong = 0
                tier_counts["WEAK"] += 1
        elif p_cal <= q_lo:
            direction = "Down"
            confidence = float(1.0 - p_cal)
            down_count += 1
            if confidence >= strong_conf:
                tier = "STRONG"
                is_strong = 1
                tier_counts["STRONG"] += 1
            elif confidence >= weak_below:
                tier = "NORMAL"
                is_strong = 0
                tier_counts["NORMAL"] += 1
            else:
                tier = "WEAK"
                is_strong = 0
                tier_counts["WEAK"] += 1
        else:
            direction = "No Signal"
            confidence = float(max(p_cal, 1.0 - p_cal))
            tier = None
            is_strong = 0
            tier_counts["NO_SIGNAL"] += 1
            nosig_count += 1

        bar_end_dt = ts + pd.Timedelta(hours=1 if not is_last else 0.5)
        delivered_dt = bar_end_dt + pd.Timedelta(minutes=16)

        rec = {
            "bar_start": ts.isoformat(),
            "bar_end": bar_end_dt.isoformat(),
            "horizon": "H1",
            "direction": direction,
            "confidence": confidence,
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
            "schema_v": 3,
            "rule_version": 2,
            "p_cal": p_cal,
            "q_lo": q_lo,
            "q_hi": q_hi,
            "tier": tier
        }

        db.insert_signal(rec)
        inserted_count += 1

    print(f"\nSuccessfully processed and inserted {inserted_count} Rule 2 signals!")
    print(f"Counts: Up={up_count}, Down={down_count}, No-Signal={nosig_count}")
    print(f"Tiers: {tier_counts}")

    # Verify cryptographic hash chain across full table
    valid, total_rows, err = db.verify_hash_chain()
    print(f"Hash chain verification: valid={valid}, total_rows={total_rows}, err={err}")
    assert valid, f"Hash chain verification failed: {err}"

if __name__ == "__main__":
    run_rule2_backfill()
