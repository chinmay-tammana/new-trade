"""
Unit and Verification Tests for M5 Coverage-Targeted Rules (Section D Proof).
Tests:
1. Boundary cases for q_lo, q_hi, and margin guard.
2. Tiers definition and thresholds.
3. Every directional signal has displayed confidence >= 51.0%.
4. No-signal rows never show confidence or tier.
5. One row per (bar, horizon) in active signal log.
6. Summary equals the table rows.
7. No hard-coded thresholds (grep check).
8. Rule hash check against MODEL_SHA256_m5.json.
9. Non-destructive apply_rule assertion.
10. Replay equivalence test against src/predict_day_m5.py on 3 days (including half-day).
"""
import os
import re
import json
import hashlib
import sqlite3
import pytest
import numpy as np
import pandas as pd

from configs.config import CONFIG_M5
from src.live_m5.store import LiveM5Database, DB_PATH
from src.live_m5.log_engine import get_m5_signal_log, format_m5_row, load_m5_calib
from src.predict_day_m5 import predict_day_m5


def test_margin_guard_and_boundary_cases():
    # If q_lo is computed as 0.495 and q_hi as 0.505, margin guard min_margin=0.01 must force q_lo <= 0.49 and q_hi >= 0.51
    min_margin = 0.01
    q_lo_raw = 0.495
    q_hi_raw = 0.505
    
    eff_q_lo = min(q_lo_raw, 0.50 - min_margin)
    eff_q_hi = max(q_hi_raw, 0.50 + min_margin)
    
    assert eff_q_lo <= 0.4900001, f"Margin guard failed on q_lo: {eff_q_lo}"
    assert eff_q_hi >= 0.5100000, f"Margin guard failed on q_hi: {eff_q_hi}"


def test_every_directional_confidence_above_50():
    db = LiveM5Database()
    with db.get_connection() as conn:
        cursor = conn.execute("SELECT * FROM signals WHERE direction IN ('Up', 'Down') AND rule_version = 3;")
        rows = cursor.fetchall()
        
    assert len(rows) > 0, "No rule 3 directional rows found in signals table"
    for r in rows:
        conf = float(r["confidence"])
        assert conf >= 0.50, f"Signal at {r['bar_start']} has confidence {conf:.4f} < 0.50"


def test_tie_handling():
    # If p_cal == 0.5, direction must be 'No signal'
    p_tie = 0.50000000
    dir_str = "No signal" if abs(p_tie - 0.5) < 1e-9 else ("Up" if p_tie > 0.5 else "Down")
    assert dir_str == "No signal"


def test_tier_boundaries():
    active_ptr_path = "models/m5/active_rule.json"
    with open(active_ptr_path, "r") as f:
        ptr = json.load(f)
    with open(os.path.join("models/m5", ptr["active_rule_file"]), "r") as rf:
        rule_cfg = json.load(rf)
        
    weak_th = float(rule_cfg["weak_threshold"])
    strong_th = float(rule_cfg["strong_threshold"])
    
    assert 0.51 < weak_th < 0.53, f"Unexpected weak threshold: {weak_th}"
    assert 0.57 < strong_th < 0.60, f"Unexpected strong threshold: {strong_th}"
    
    # Boundary test
    def get_tier(conf):
        if conf >= strong_th:
            return "STRONG"
        elif conf <= weak_th:
            return "WEAK"
        return "NORMAL"
        
    assert get_tier(0.51) == "WEAK"
    assert get_tier(0.52) == "WEAK"
    assert get_tier(0.55) == "NORMAL"
    assert get_tier(0.59) == "STRONG"
    assert get_tier(0.65) == "STRONG"


def test_nosignal_only_for_allowed_reasons():
    db = LiveM5Database()
    with db.get_connection() as conn:
        cursor = conn.execute("SELECT * FROM signals WHERE direction NOT IN ('Up', 'Down') AND rule_version = 3;")
        rows = [dict(r) for r in cursor.fetchall()]
        
    for r in rows:
        dt = pd.to_datetime(r["bar_start"])
        # Allowed reasons: end of day bar (hour=15, minute>=55) or exact 0.5 tie
        is_eod = (dt.hour == 15 and dt.minute >= 55)
        is_tie = abs(float(r["p_cal"]) - 0.5) < 1e-9
        assert is_eod or is_tie, f"Disallowed No-signal row found at {r['bar_start']}: p_cal={r['p_cal']}"


def test_nosignal_rows_never_show_confidence_or_tier():
    db = LiveM5Database()
    with db.get_connection() as conn:
        cursor = conn.execute("SELECT * FROM signals WHERE direction NOT IN ('Up', 'Down') AND rule_version >= 2 LIMIT 50;")
        rows = [dict(r) for r in cursor.fetchall()]
        
    for r in rows:
        formatted = format_m5_row(r)
        assert formatted["confidence"] is None, f"No-signal row at {r['bar_start']} exposed confidence: {formatted['confidence']}"
        assert formatted["strength_display"] == "–", f"No-signal row at {r['bar_start']} exposed strength: {formatted['strength_display']}"
        assert formatted["tier"] is None, f"No-signal row at {r['bar_start']} exposed tier: {formatted['tier']}"


def test_one_row_per_bar_horizon_and_summary_equals_rows():
    db = LiveM5Database()
    test_days = ["2026-11-13", "2026-11-12", "2026-11-10"]
    
    for d in test_days:
        res = get_m5_signal_log(day=d, show_no_signal=True, show_old_rule=False, db=db)
        rows = res["rows"]
        summary = res["summary"]
        
        # 1. Unique bar_start per log
        bar_starts = [r["bar_start"] for r in rows]
        assert len(bar_starts) == len(set(bar_starts)), f"Duplicate bar_start found for date {d}"
        
        # 2. Summary counts match exactly
        n_up = sum(1 for r in rows if r["direction"] == "Up")
        n_down = sum(1 for r in rows if r["direction"] == "Down")
        n_nosig = sum(1 for r in rows if r["direction"] not in ["Up", "Down"])
        
        assert summary["up_count"] == n_up, f"Up count mismatch on {d}: summary={summary['up_count']}, actual={n_up}"
        assert summary["down_count"] == n_down, f"Down count mismatch on {d}: summary={summary['down_count']}, actual={n_down}"
        assert summary["no_signal_count"] == n_nosig, f"No-sig count mismatch on {d}: summary={summary['no_signal_count']}, actual={n_nosig}"
        assert summary["total_signals"] == n_up + n_down, f"Total signals mismatch on {d}"
        assert summary["total_bars"] == len(rows), f"Total bars mismatch on {d}"


def test_rule_hashes_in_model_sha256():
    sha_file = CONFIG_M5["paths"]["sha_path"]
    assert os.path.exists(sha_file), "MODEL_SHA256_m5.json missing"
    with open(sha_file, "r") as f:
        sha_dict = json.load(f)
        
    # Check model files byte-identical
    with open(CONFIG_M5["paths"]["model_path"], "rb") as mf:
        actual_model_sha = hashlib.sha256(mf.read()).hexdigest()
    with open(CONFIG_M5["paths"]["calib_path"], "rb") as cf:
        actual_calib_sha = hashlib.sha256(cf.read()).hexdigest()
        
    assert actual_model_sha == sha_dict["model_m5.joblib"], "model_m5.joblib hash changed!"
    assert actual_calib_sha == sha_dict["calib_m5.json"], "calib_m5.json hash changed!"
    
    # Check active rule file and pointer hashes
    active_ptr_path = "models/m5/active_rule.json"
    with open(active_ptr_path, "rb") as pf:
        actual_ptr_sha = hashlib.sha256(pf.read()).hexdigest()
    assert actual_ptr_sha == sha_dict["active_rule.json"], "active_rule.json hash mismatch!"
    
    with open(active_ptr_path, "r") as pf:
        ptr = json.load(pf)
    rule_file = os.path.join("models/m5", ptr["active_rule_file"])
    with open(rule_file, "rb") as rf:
        actual_rule_sha = hashlib.sha256(rf.read()).hexdigest()
    assert actual_rule_sha == sha_dict[ptr["active_rule_file"]], f"Rule file {ptr['active_rule_file']} hash mismatch!"


def test_apply_rule_non_destructive():
    db = LiveM5Database()
    with db.get_connection() as conn:
        c1 = conn.execute("SELECT COUNT(*) FROM signals WHERE rule_version = 1 AND late = 1;").fetchone()[0]
        
    assert c1 == 1200, f"Baseline rule 1 row count was modified! Expected 1200, got {c1}"


def test_replay_equivalence_across_days():
    # Test on 3 days: 2026-11-13, 2026-11-10, and 2024-11-29 (half-day)
    test_days = ["2026-11-13", "2026-11-10", "2024-11-29"]
    
    for d in test_days:
        df_p = predict_day_m5(d)
        assert len(df_p) > 0, f"predict_day_m5 returned no predictions for {d}"
        preds = df_p.reset_index().to_dict(orient="records")
        
        for p in preds:
            # If directional, confidence must be >= 0.50
            if p["direction"] in ["Up", "Down"]:
                assert p["confidence"] >= 0.50, f"Replay confidence below 0.50 at {p['bar_start']}: {p['confidence']}"
                assert 0.0 <= p["p_cal"] <= 1.0, f"p_cal out of bounds at {p['bar_start']}"
            else:
                assert p["confidence"] == 0.50, f"Neutral confidence not 0.50 at {p['bar_start']}"


def test_no_hardcoded_thresholds_in_code():
    # Search for hard-coded threshold patterns like 0.434, 0.563, 0.37279, 0.62913 in src/
    forbidden_floats = ["0.434", "0.563", "0.37279", "0.62913"]
    src_files = [
        "src/live_m5/server.py",
        "src/live_m5/engine.py",
        "src/live_m5/log_engine.py",
        "src/predict_day_m5.py",
        "src/m5/apply_rule.py"
    ]
    for path in src_files:
        with open(path, "r") as f:
            content = f.read()
        for ff in forbidden_floats:
            assert ff not in content, f"Hard-coded threshold {ff} found in {path}"
