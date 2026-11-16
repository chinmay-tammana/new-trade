"""
Apply Signal Rule CLI for M5 (5-Minute) System.
Allows dynamically applying coverage-targeted rules (e.g. --coverage 0.30 or 0.40)
without retraining models, modifying existing rows, or breaking database immutability.

Usage:
    python3 -m src.m5.apply_rule --coverage 0.30
    python3 -m src.m5.apply_rule --coverage 0.40
"""
import os
import sys
import json
import argparse
import hashlib
import datetime
import zoneinfo
import sqlite3
import numpy as np
import pandas as pd
import joblib

from configs.config import CONFIG_M5
from src.features_m5 import ALL_FEATURE_COLS
from src.live_m5.store import LiveM5Database, DB_PATH, ET_TZ

MODELS_DIR = "models/m5"


def get_active_rule():
    active_ptr_path = os.path.join(MODELS_DIR, "active_rule.json")
    if os.path.exists(active_ptr_path):
        with open(active_ptr_path, "r") as f:
            ptr = json.load(f)
            rule_file = os.path.join(MODELS_DIR, ptr["active_rule_file"])
            if os.path.exists(rule_file):
                with open(rule_file, "r") as rf:
                    return json.load(rf), ptr
    # Fallback to calib_m5.json
    with open(CONFIG_M5["paths"]["calib_path"], "r") as f:
        calib = json.load(f)
    return {
        "rule_version": 1,
        "q_lo": calib["q_lo"],
        "q_hi": calib["q_hi"],
        "weak_threshold": 0.56,
        "strong_threshold": calib.get("strong_threshold", 0.65),
        "coverage_target": calib.get("coverage_c", 0.10)
    }, {"active_rule_version": 1, "active_rule_file": "calib_m5.json"}


def compute_rule_thresholds(coverage_target: float, min_margin: float = 0.01):
    """
    Computes Rule B quantiles from trailing 20 completed sessions' p_cal
    with min_margin guard.
    """
    df_features = pd.read_parquet(CONFIG_M5["paths"]["processed_features"]).sort_index()
    df_valid = df_features.dropna(subset=["y"] + ALL_FEATURE_COLS).copy()
    all_dates = sorted(list(set(df_valid["date"])))
    
    # Trailing 20 completed sessions
    trailing_20_dates = all_dates[-20:]
    df_trailing = df_valid[df_valid["date"].isin(trailing_20_dates)].copy()
    
    # Load production model
    bundle = joblib.load(CONFIG_M5["paths"]["model_path"])
    prod_clf = bundle["primary_model"]
    prod_calib = bundle["calibrator"]
    
    raw_p = prod_clf.predict_proba(df_trailing[ALL_FEATURE_COLS].values)[:, 1]
    p_cal = prod_calib.predict_proba(raw_p)
    
    # Rule B Quantiles
    q_lo = float(np.quantile(p_cal, coverage_target / 2.0))
    q_hi = float(np.quantile(p_cal, 1.0 - (coverage_target / 2.0)))
    
    # Margin guard
    eff_q_lo = min(q_lo, 0.50 - min_margin)
    eff_q_hi = max(q_hi, 0.50 + min_margin)
    
    # Directional confidence quantiles for tiers
    is_up = p_cal >= eff_q_hi
    is_down = p_cal <= eff_q_lo
    confs = np.where(is_up, p_cal, np.where(is_down, 1.0 - p_cal, np.nan))
    sig_confs = confs[~np.isnan(confs)]
    
    if len(sig_confs) > 0:
        weak_thresh = float(np.quantile(sig_confs, 0.25))
        strong_thresh = float(np.quantile(sig_confs, 0.75))
    else:
        weak_thresh = eff_q_hi
        strong_thresh = eff_q_hi + 0.05
        
    return {
        "q_lo": eff_q_lo,
        "q_hi": eff_q_hi,
        "weak_threshold": weak_thresh,
        "strong_threshold": strong_thresh,
        "n_bars_eval": len(p_cal),
        "achieved_cov_eval": float(np.mean(is_up | is_down))
    }


def apply_rule(coverage_target: float, n_sessions: int = 10, min_margin: float = 0.01, rule_version: int = None):
    curr_rule, active_ptr = get_active_rule()
    new_version = rule_version if rule_version is not None else curr_rule["rule_version"] + 1
    
    print(f"Applying Signal Rule: Coverage Target = {coverage_target:.1%}, New Version = {new_version}")
    
    # 1. Compute Thresholds
    thresh = compute_rule_thresholds(coverage_target, min_margin=min_margin)
    q_lo = thresh["q_lo"]
    q_hi = thresh["q_hi"]
    weak_thresh = thresh["weak_threshold"]
    strong_thresh = thresh["strong_threshold"]
    
    print(f"Computed Thresholds (Rule B Trailing 20 Sessions):")
    print(f"  q_lo: {q_lo:.5f} | q_hi: {q_hi:.5f} | min_margin: {min_margin}")
    print(f"  Tiers: WEAK < {weak_thresh:.4f} | STRONG >= {strong_thresh:.4f}")

    # 2. Save Rule File
    rule_data = {
        "rule_version": new_version,
        "rule_type": "rule_b",
        "coverage_target": coverage_target,
        "q_lo": q_lo,
        "q_hi": q_hi,
        "min_margin": min_margin,
        "weak_threshold": weak_thresh,
        "strong_threshold": strong_thresh,
        "source": f"apply_rule CLI --coverage {coverage_target}",
        "created_at": datetime.datetime.now(ET_TZ).isoformat()
    }
    rule_filename = f"rule_m5_v{new_version}.json"
    rule_filepath = os.path.join(MODELS_DIR, rule_filename)
    with open(rule_filepath, "w") as f:
        json.dump(rule_data, f, indent=2)
    print(f"Saved rule config to {rule_filepath}")

    # 3. Update Active Pointer
    active_ptr_data = {
        "active_rule_version": new_version,
        "active_rule_file": rule_filename,
        "description": f"Target {coverage_target:.1%} coverage via Rule B",
        "updated_at": datetime.datetime.now(ET_TZ).isoformat()
    }
    active_ptr_path = os.path.join(MODELS_DIR, "active_rule.json")
    with open(active_ptr_path, "w") as f:
        json.dump(active_ptr_data, f, indent=2)
    print(f"Updated active rule pointer in {active_ptr_path}")

    # 4. Update MODEL_SHA256_m5.json
    with open(rule_filepath, "rb") as rf:
        rule_sha = hashlib.sha256(rf.read()).hexdigest()
    with open(active_ptr_path, "rb") as pf:
        ptr_sha = hashlib.sha256(pf.read()).hexdigest()

    sha_path = CONFIG_M5["paths"]["sha_path"]
    with open(sha_path, "r") as sf:
        sha_dict = json.load(sf)
    sha_dict[rule_filename] = rule_sha
    sha_dict["active_rule.json"] = ptr_sha
    with open(sha_path, "w") as sf:
        json.dump(sha_dict, sf, indent=2)
    print(f"Updated {sha_path} with rule hash {rule_sha[:16]}...")

    # 5. Database: Re-derive Signals for Last N Sessions
    db = LiveM5Database()
    db.init_db() # Ensures schema is updated
    
    with db.get_connection() as conn:
        cursor = conn.execute("SELECT DISTINCT date(bar_start) FROM signals ORDER BY date(bar_start) ASC;")
        distinct_dates = [r[0] for r in cursor.fetchall()]
        
    sessions_to_rederive = distinct_dates[-n_sessions:]
    print(f"\nRe-deriving signals for last {len(sessions_to_rederive)} sessions: {sessions_to_rederive[0]} to {sessions_to_rederive[-1]}")

    before_counts = {}
    after_counts = {}
    
    # Gather baseline counts before insertion
    with db.get_connection() as conn:
        for d in sessions_to_rederive:
            rows = conn.execute("SELECT direction FROM signals WHERE date(bar_start) = ? AND rule_version = 1;", (d,)).fetchall()
            u = sum(1 for r in rows if r["direction"] == "Up")
            dw = sum(1 for r in rows if r["direction"] == "Down")
            ns = sum(1 for r in rows if r["direction"] == "No signal")
            tot = len(rows)
            cov = (u + dw) / tot if tot > 0 else 0.0
            before_counts[d] = {"up": u, "down": dw, "no_sig": ns, "total": tot, "cov": cov}

    # Fetch rows to rederive from rule_version = 1
    with db.get_connection() as conn:
        cursor = conn.execute("""
        SELECT * FROM signals
        WHERE date(bar_start) IN ({}) AND rule_version = 1
        ORDER BY bar_start ASC;
        """.format(",".join(f"'{d}'" for d in sessions_to_rederive)))
        v1_rows = [dict(r) for r in cursor.fetchall()]

    print(f"Processing {len(v1_rows)} bars to insert rule_version = {new_version}...")
    inserted_count = 0
    
    for r in v1_rows:
        p_cal_val = float(r["p_cal"])
        if p_cal_val >= q_hi:
            new_dir = "Up"
            new_conf = p_cal_val
        elif p_cal_val <= q_lo:
            new_dir = "Down"
            new_conf = 1.0 - p_cal_val
        else:
            new_dir = "No signal"
            new_conf = 0.50
            
        if new_dir in ["Up", "Down"]:
            if new_conf >= strong_thresh:
                new_tier = "STRONG"
                is_str = 1
            elif new_conf < weak_thresh:
                new_tier = "WEAK"
                is_str = 0
            else:
                new_tier = "NORMAL"
                is_str = 0
        else:
            new_tier = "NORMAL"
            is_str = 0
            
        new_sig = {
            "bar_start": r["bar_start"],
            "bar_end": r["bar_end"],
            "horizon": r["horizon"],
            "symbol": r["symbol"],
            "feed": r["feed"],
            "direction": new_dir,
            "confidence": new_conf,
            "p_cal": p_cal_val,
            "q_lo": q_lo,
            "q_hi": q_hi,
            "is_strong": is_str,
            "vol_class": r["vol_class"],
            "band_lo": r["band_lo"],
            "band_hi": r["band_hi"],
            "regime": r["regime"],
            "delivered_at": r["delivered_at"],
            "delivery_delay_s": r["delivery_delay_s"],
            "sample": r["sample"],
            "late": 1, # Explicitly marked as late re-derived signal
            "model_hash": r["model_hash"],
            "rule_version": new_version,
            "tier": new_tier
        }
        
        ok = db.insert_signal(new_sig)
        if ok:
            inserted_count += 1

    print(f"Successfully inserted {inserted_count} new rows with rule_version = {new_version}.")
    
    # Verify hash chain
    ok_chain, n_chain, err_msg = db.verify_hash_chain()
    print(f"Hash chain verification: {'VALID' if ok_chain else 'BROKEN'} ({n_chain} total rows) - {err_msg or 'OK'}")
    assert ok_chain, f"Database hash chain failed verification after rule application: {err_msg}"

    # Gather after counts
    with db.get_connection() as conn:
        for d in sessions_to_rederive:
            rows = conn.execute("SELECT direction FROM signals WHERE date(bar_start) = ? AND rule_version = ?;", (d, new_version)).fetchall()
            u = sum(1 for r in rows if r["direction"] == "Up")
            dw = sum(1 for r in rows if r["direction"] == "Down")
            ns = sum(1 for r in rows if r["direction"] == "No signal")
            tot = len(rows)
            cov = (u + dw) / tot if tot > 0 else 0.0
            after_counts[d] = {"up": u, "down": dw, "no_sig": ns, "total": tot, "cov": cov}

    # Print Before vs After for last 5 sessions
    last_5_sessions = sessions_to_rederive[-5:]
    print("\n==========================================================================================")
    print(f"BEFORE AND AFTER SIGNAL COUNTS (LAST 5 SESSIONS) - RULE v1 vs RULE v{new_version}")
    print("==========================================================================================")
    print(f"{'Date':12s} | {'Rule 1 (Up / Down / No-Sig)':30s} {'Cov 1':7s} | {'Rule ' + str(new_version) + ' (Up / Down / No-Sig)':30s} {'Cov ' + str(new_version):7s}")
    print("-" * 90)
    for d in last_5_sessions:
        b = before_counts[d]
        a = after_counts[d]
        b_str = f"Up: {b['up']:2d}, Down: {b['down']:2d}, No-Sig: {b['no_sig']:2d}"
        a_str = f"Up: {a['up']:2d}, Down: {a['down']:2d}, No-Sig: {a['no_sig']:2d}"
        print(f"{d:12s} | {b_str:30s} {b['cov']:6.1%} | {a_str:30s} {a['cov']:6.1%}")
    print("==========================================================================================\n")


def apply_always_call(n_sessions: int = 10):
    rule_v3_path = os.path.join(MODELS_DIR, "rule_m5_v3.json")
    with open(rule_v3_path, "r") as f:
        rule_v3 = json.load(f)
        
    weak_thresh = float(rule_v3["weak_threshold"])
    strong_thresh = float(rule_v3["strong_threshold"])
    new_version = int(rule_v3.get("rule_version", 3))
    
    print(f"Applying Always-Call Rule (Rule v{new_version})")
    print(f"  Tiers: WEAK <= {weak_thresh:.4f} ('close to a coin flip') | STRONG >= {strong_thresh:.4f}")
    
    # Update active pointer
    active_ptr_data = {
        "active_rule_version": new_version,
        "active_rule_file": "rule_m5_v3.json",
        "description": "Always-Call directional signals on every eligible interval (Owner Decision 2026-11-16)",
        "updated_at": datetime.datetime.now(ET_TZ).isoformat()
    }
    active_ptr_path = os.path.join(MODELS_DIR, "active_rule.json")
    with open(active_ptr_path, "w") as f:
        json.dump(active_ptr_data, f, indent=2)
        
    # Update hashes
    with open(rule_v3_path, "rb") as rf:
        rule_sha = hashlib.sha256(rf.read()).hexdigest()
    with open(active_ptr_path, "rb") as pf:
        ptr_sha = hashlib.sha256(pf.read()).hexdigest()
        
    sha_path = CONFIG_M5["paths"]["sha_path"]
    with open(sha_path, "r") as sf:
        sha_dict = json.load(sf)
    sha_dict["rule_m5_v3.json"] = rule_sha
    sha_dict["active_rule.json"] = ptr_sha
    with open(sha_path, "w") as sf:
        json.dump(sha_dict, sf, indent=2)
        
    db = LiveM5Database()
    with db.get_connection() as conn:
        cursor = conn.execute("SELECT DISTINCT date(bar_start) FROM signals ORDER BY date(bar_start) ASC;")
        distinct_dates = [r[0] for r in cursor.fetchall()]
        
    sessions_to_rederive = distinct_dates[-n_sessions:]
    print(f"\nRe-deriving signals for last {len(sessions_to_rederive)} sessions: {sessions_to_rederive[0]} to {sessions_to_rederive[-1]}")
    
    before_counts = {}
    after_counts = {}
    
    # Gather baseline counts before insertion (rule_version 1)
    with db.get_connection() as conn:
        for d in sessions_to_rederive:
            rows = conn.execute("SELECT direction FROM signals WHERE date(bar_start) = ? AND rule_version = 1;", (d,)).fetchall()
            u = sum(1 for r in rows if r["direction"] == "Up")
            dw = sum(1 for r in rows if r["direction"] == "Down")
            ns = sum(1 for r in rows if r["direction"] == "No signal")
            tot = len(rows)
            cov = (u + dw) / tot if tot > 0 else 0.0
            before_counts[d] = {"up": u, "down": dw, "no_sig": ns, "total": tot, "cov": cov}

    # Fetch rows to rederive from rule_version = 1
    with db.get_connection() as conn:
        cursor = conn.execute("""
        SELECT * FROM signals
        WHERE date(bar_start) IN ({}) AND rule_version = 1
        ORDER BY bar_start ASC;
        """.format(",".join(f"'{d}'" for d in sessions_to_rederive)))
        v1_rows = [dict(r) for r in cursor.fetchall()]
        
    print(f"Processing {len(v1_rows)} bars to insert rule_version = {new_version}...")
    inserted_count = 0
    
    for r in v1_rows:
        # Check if already inserted
        with db.get_connection() as conn:
            existing = conn.execute(
                "SELECT id FROM signals WHERE bar_start = ? AND horizon = ? AND rule_version = ?;",
                (r["bar_start"], r["horizon"], new_version)
            ).fetchone()
            if existing:
                continue
                
        p_cal_val = float(r["p_cal"])
        bar_start_dt = datetime.datetime.fromisoformat(r["bar_start"])
        is_eod = (bar_start_dt.hour == 15 and bar_start_dt.minute >= 55)
        
        if is_eod:
            new_dir = "No signal"
            new_conf = 0.50
            new_tier = "NORMAL"
            is_str = 0
        elif abs(p_cal_val - 0.5) < 1e-9: # Exact tie
            new_dir = "No signal"
            new_conf = 0.50
            new_tier = "NORMAL"
            is_str = 0
        elif p_cal_val > 0.5:
            new_dir = "Up"
            new_conf = p_cal_val
            if new_conf >= strong_thresh:
                new_tier = "STRONG"
                is_str = 1
            elif new_conf <= weak_thresh:
                new_tier = "WEAK"
                is_str = 0
            else:
                new_tier = "NORMAL"
                is_str = 0
        else:
            new_dir = "Down"
            new_conf = 1.0 - p_cal_val
            if new_conf >= strong_thresh:
                new_tier = "STRONG"
                is_str = 1
            elif new_conf <= weak_thresh:
                new_tier = "WEAK"
                is_str = 0
            else:
                new_tier = "NORMAL"
                is_str = 0
                
        new_sig = {
            "bar_start": r["bar_start"],
            "bar_end": r["bar_end"],
            "horizon": r["horizon"],
            "symbol": r["symbol"],
            "feed": r["feed"],
            "direction": new_dir,
            "confidence": new_conf,
            "p_cal": p_cal_val,
            "q_lo": 0.50,
            "q_hi": 0.50,
            "is_strong": is_str,
            "vol_class": r["vol_class"],
            "band_lo": r["band_lo"],
            "band_hi": r["band_hi"],
            "regime": r["regime"],
            "delivered_at": r["delivered_at"],
            "delivery_delay_s": r["delivery_delay_s"],
            "sample": r["sample"],
            "late": 1,
            "model_hash": r["model_hash"],
            "rule_version": new_version,
            "tier": new_tier
        }
        ok = db.insert_signal(new_sig)
        if ok:
            inserted_count += 1
            
    print(f"Successfully inserted {inserted_count} new rows with rule_version = {new_version}.")
    
    # Verify hash chain
    ok_chain, n_chain, err_msg = db.verify_hash_chain()
    print(f"Hash chain verification: {'VALID' if ok_chain else 'BROKEN'} ({n_chain} total rows) - {err_msg or 'OK'}")
    assert ok_chain, f"Database hash chain failed verification after always-call application: {err_msg}"
    
    # Gather after counts
    with db.get_connection() as conn:
        for d in sessions_to_rederive:
            rows = conn.execute("SELECT direction FROM signals WHERE date(bar_start) = ? AND rule_version = ?;", (d, new_version)).fetchall()
            u = sum(1 for r in rows if r["direction"] == "Up")
            dw = sum(1 for r in rows if r["direction"] == "Down")
            ns = sum(1 for r in rows if r["direction"] == "No signal")
            tot = len(rows)
            cov = (u + dw) / tot if tot > 0 else 0.0
            after_counts[d] = {"up": u, "down": dw, "no_sig": ns, "total": tot, "cov": cov}

    # Print Before vs After for last 5 sessions
    last_5_sessions = sessions_to_rederive[-5:]
    print("\n==========================================================================================")
    print(f"BEFORE AND AFTER SIGNAL COUNTS (LAST 5 SESSIONS) - RULE v1 vs ALWAYS-CALL (RULE v{new_version})")
    print("==========================================================================================")
    print(f"{'Date':12s} | {'Rule 1 (Up / Down / No-Sig)':30s} {'Cov 1':7s} | {'Rule ' + str(new_version) + ' (Up / Down / No-Sig)':30s} {'Cov ' + str(new_version):7s}")
    print("-" * 90)
    for d in last_5_sessions:
        b = before_counts[d]
        a = after_counts[d]
        b_str = f"Up: {b['up']:2d}, Down: {b['down']:2d}, No-Sig: {b['no_sig']:2d}"
        a_str = f"Up: {a['up']:2d}, Down: {a['down']:2d}, No-Sig: {a['no_sig']:2d}"
        print(f"{d:12s} | {b_str:30s} {b['cov']:6.1%} | {a_str:30s} {a['cov']:6.1%}")
    print("==========================================================================================\n")


def main():
    parser = argparse.ArgumentParser(description="Apply new coverage-targeted rule or always-call rule to M5 system.")
    parser.add_argument("--mode", type=str, choices=["quantile", "always"], default="always", help="Mode: 'always' (Always-Call on every interval) or 'quantile'")
    parser.add_argument("--coverage", type=float, default=0.30, help="Target directional coverage for quantile mode (e.g. 0.30, 0.40)")
    parser.add_argument("--sessions", type=int, default=10, help="Number of trailing completed sessions to rederive")
    parser.add_argument("--min_margin", type=float, default=0.01, help="Minimum margin guard (default 0.01)")
    parser.add_argument("--rule_version", type=int, default=None, help="Explicit rule version number")
    
    args = parser.parse_args()
    if args.mode == "always":
        apply_always_call(n_sessions=args.sessions)
    else:
        apply_rule(
            coverage_target=args.coverage,
            n_sessions=args.sessions,
            min_margin=args.min_margin,
            rule_version=args.rule_version
        )


if __name__ == "__main__":
    main()
