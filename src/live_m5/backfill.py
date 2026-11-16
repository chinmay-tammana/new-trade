"""
Backfill script for M5 (5-minute) live database.
Populates at least 1,000 closed 5-minute bars contiguously into data/live/live_m5.db.
Tags: IN-SAMPLE up to freeze_date, OUT-OF-SAMPLE for holdout, late=1.
"""
import os
import json
import datetime
import zoneinfo
import numpy as np
import pandas as pd
import joblib

from configs.config import CONFIG_M5
from src.live_m5.store import LiveM5Database
from src.features_m5 import ALL_FEATURE_COLS

ET_TZ = zoneinfo.ZoneInfo("America/New_York")


def run_backfill(min_bars: int = 1200):
    db = LiveM5Database()
    
    # Check current signals count
    with db.get_connection() as conn:
        cnt = conn.execute("SELECT count(*) FROM signals;").fetchone()[0]
    if cnt >= min_bars:
        print(f"Database already contains {cnt:,} signals. Skipping backfill.")
        return

    print(f"Loading feature dataset to backfill last {min_bars} bars...")
    df_feat = pd.read_parquet(CONFIG_M5["paths"]["processed_features"])
    df_bars_gs = pd.read_parquet("data/processed/m5/bars_5m_GS.parquet")
    
    # Filter to last min_bars
    df_feat_sub = df_feat.iloc[-min_bars:].copy()
    
    # Load model and calibration
    model_bundle = joblib.load(CONFIG_M5["paths"]["model_path"])
    primary_clf = model_bundle["primary_model"]
    calibrator = model_bundle["calibrator"]
    
    with open(CONFIG_M5["paths"]["calib_path"], "r") as f:
        calib_cfg = json.load(f)
        
    with open(CONFIG_M5["paths"]["sha_path"], "r") as sf:
        sha_dict = json.load(sf)
    model_hash = sha_dict["model_m5.joblib"][:8]

    q_lo = float(calib_cfg["q_lo"])
    q_hi = float(calib_cfg["q_hi"])
    strong_thresh = float(calib_cfg["strong_threshold"])
    
    # Populate bars_5m table
    print("Populating bars_5m table...")
    for idx, r in df_bars_gs.iloc[-min_bars:].iterrows():
        b_start_str = idx.isoformat() if hasattr(idx, "isoformat") else str(idx)
        b_end = idx + datetime.timedelta(minutes=5)
        b_end_str = b_end.isoformat()
        db.insert_bar_5m("GS", b_start_str, b_end_str, r.to_dict())

    # Generate signals
    print("Generating backfilled signals with immutable hash chain...")
    n_inserted = 0
    freeze_dt = datetime.date.fromisoformat(CONFIG_M5["freeze_date"])
    
    for idx, r in df_feat_sub.iterrows():
        b_start_str = idx.isoformat() if hasattr(idx, "isoformat") else str(idx)
        b_end = idx + datetime.timedelta(minutes=5)
        b_end_str = b_end.isoformat()
        
        # End of day check (15:55)
        if idx.time() >= datetime.time(15, 55):
            dir_str = "No signal (end of day)"
            conf = 0.50
            p_cal = 0.50
            is_strong = 0
        else:
            X_curr = r[ALL_FEATURE_COLS].values.reshape(1, -1)
            raw_p = primary_clf.predict_proba(X_curr)[0, 1]
            p_cal = float(calibrator.predict_proba(np.array([raw_p]))[0])
            
            if p_cal >= q_hi:
                dir_str = "Up"
                conf = p_cal
            elif p_cal <= q_lo:
                dir_str = "Down"
                conf = 1.0 - p_cal
            else:
                dir_str = "No signal"
                conf = 0.50
            is_strong = 1 if (dir_str in ["Up", "Down"] and conf >= strong_thresh) else 0

        sample_tag = "IN-SAMPLE" if idx.date() < datetime.date(2026, 5, 18) else "OUT-OF-SAMPLE"
        
        # Delivered at = bar_end + 10s
        deliv_dt = b_end + datetime.timedelta(seconds=10)
        
        sig = {
            "bar_start": b_start_str,
            "bar_end": b_end_str,
            "horizon": "5m",
            "symbol": "GS",
            "feed": "iex",
            "direction": dir_str,
            "confidence": conf,
            "p_cal": p_cal,
            "q_lo": q_lo,
            "q_hi": q_hi,
            "is_strong": is_strong,
            "vol_class": "Normal",
            "band_lo": -0.003,
            "band_hi": +0.003,
            "regime": r.get("regime_tag", "Chop"),
            "delivered_at": deliv_dt.isoformat(),
            "delivery_delay_s": 10.0,
            "sample": sample_tag,
            "late": 1,
            "model_hash": model_hash
        }
        if db.insert_signal(sig):
            n_inserted += 1

    valid, count, err = db.verify_hash_chain()
    print(f"Backfill complete: {n_inserted} signals inserted. Hash chain verified: {valid} ({count} rows, err: {err})")


if __name__ == "__main__":
    run_backfill(1200)
