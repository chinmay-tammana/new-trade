"""
Predict offline 5-minute signals for an entire trading day using the frozen M5 model.
Used for replay equivalence verification against the operational engine.
"""
import os
import json
import datetime
import zoneinfo
import numpy as np
import pandas as pd
import joblib

from configs.config import CONFIG_M5
from src.features_m5 import make_features_m5, ALL_FEATURE_COLS
from src.aggregate_m5 import ET_TZ


def predict_day_m5(day_str: str) -> pd.DataFrame:
    """
    Compute frozen M5 signals for a given session date.
    Returns DataFrame indexed by bar_start with direction, confidence, p_cal, strong.
    """
    target_date = datetime.date.fromisoformat(day_str)
    from src.features_m5 import load_and_build_features_m5
    df_feat = load_and_build_features_m5(target_date, db_path=CONFIG_M5["paths"].get("db_path", "data/live/live_m5.db"))
    if df_feat.empty:
        raise ValueError(f"No bars found for date {day_str}")
    day_feat = df_feat[df_feat.index.date == target_date].copy()
    if day_feat.empty:
        raise ValueError(f"No bars found for date {day_str}")
    
    # Load model bundle and thresholds
    bundle = joblib.load(CONFIG_M5["paths"]["model_path"])
    primary_clf = bundle["primary_model"]
    calibrator = bundle["calibrator"]
    
    # Read active rule thresholds if present, fallback to calib_m5.json
    active_ptr_path = "models/m5/active_rule.json"
    if os.path.exists(active_ptr_path):
        with open(active_ptr_path, "r") as f:
            ptr = json.load(f)
        rule_file = os.path.join("models/m5", ptr["active_rule_file"])
        with open(rule_file, "r") as rf:
            calib_cfg = json.load(rf)
    else:
        with open(CONFIG_M5["paths"]["calib_path"], "r") as f:
            calib_cfg = json.load(f)
    q_lo = float(calib_cfg.get("q_lo", 0.50))
    q_hi = float(calib_cfg.get("q_hi", 0.50))
    weak_thresh = float(calib_cfg.get("weak_threshold", 0.5208))
    strong_thresh = float(calib_cfg.get("strong_threshold", 0.5811))
    is_always_call = (calib_cfg.get("rule_type") == "always_call") or (abs(q_lo - 0.5) < 1e-6 and abs(q_hi - 0.5) < 1e-6)

    results = []
    for idx, r in day_feat.iterrows():
        b_start_str = idx.isoformat()
        if idx.time() >= datetime.time(15, 55):
            dir_str = "No signal"
            conf = 0.50
            p_cal = 0.50
            is_strong = 0
            tier = "NORMAL"
        else:
            X_curr = r[ALL_FEATURE_COLS].values.reshape(1, -1)
            nan_count = int(pd.isna(X_curr).sum())
            if nan_count > CONFIG_M5.get("max_nan_features", 0):
                dir_str = "No signal (feature gap)"
                conf = 0.50
                p_cal = 0.50
                is_strong = 0
                tier = "NORMAL"
            else:
                raw_p = primary_clf.predict_proba(X_curr)[0, 1]
                p_cal = float(calibrator.predict_proba(np.array([raw_p]))[0])
            
            if is_always_call:
                if abs(p_cal - 0.5) < 1e-9:
                    dir_str = "No signal"
                    conf = 0.50
                elif p_cal > 0.5:
                    dir_str = "Up"
                    conf = p_cal
                else:
                    dir_str = "Down"
                    conf = 1.0 - p_cal
            else:
                if p_cal >= q_hi:
                    dir_str = "Up"
                    conf = p_cal
                elif p_cal <= q_lo:
                    dir_str = "Down"
                    conf = 1.0 - p_cal
                else:
                    dir_str = "No signal"
                    conf = 0.50

            if dir_str in ["Up", "Down"]:
                if conf >= strong_thresh:
                    tier = "STRONG"
                    is_strong = 1
                elif conf <= weak_thresh:
                    tier = "WEAK"
                    is_strong = 0
                else:
                    tier = "NORMAL"
                    is_strong = 0
            else:
                tier = "NORMAL"
                is_strong = 0

        results.append({
            "bar_start": b_start_str,
            "direction": dir_str,
            "confidence": conf,
            "p_cal": p_cal,
            "is_strong": is_strong,
            "tier": tier
        })
        
    return pd.DataFrame(results).set_index("bar_start")


if __name__ == "__main__":
    import sys
    d = sys.argv[1] if len(sys.argv) > 1 else "2026-11-13"
    df = predict_day_m5(d)
    print(f"Predicted {len(df)} 5-minute bars for {d}:")
    print(df.head(10))
