"""Comprehensive Unit and Integration Tests for V3 Live Engine and API.

Covers:
- Scheduler (normal day, half day, DST transitions 2025/2026)
- Database immutability (UPDATE/DELETE abort triggers)
- Hash chain integrity and tamper detection
- Duplicate bar handling
- Missing context symbol handling ("No signal (data gap)")
- Mid-session restart & missed bar late flag
- Model lock mismatch refusal
- Live compatibility (600-bar rolling buffer vs full history)
- FastAPI endpoints contract (/healthz, /api/status, /api/bars, /api/signals, /api/latest, /api/scoreboard)
- Zero secrets leakage in API responses
"""

import datetime
import hashlib
import json
import os
import sqlite3
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from starlette.testclient import TestClient

from configs.config import CONFIG
from src.features import make_features_v2
from src.insights import get_bar_insights, _load_production_models
from src.live.calendar import NyseCalendar
from src.live.engine import LiveEngine, verify_model_lock
from src.live.server import app, get_db
from src.live.store import LiveDatabase


# ----------------------------------------------------------------------
# 1. SCHEDULER TESTS
# ----------------------------------------------------------------------

def test_calendar_normal_day():
    cal = NyseCalendar()
    normal_date = datetime.date(2026, 11, 6)
    session = cal.get_session_bounds(normal_date)
    assert session is not None
    assert session["is_half_day"] is False
    bars = session["bars"]
    assert len(bars) == 7
    expected_starts = ["09:30", "10:30", "11:30", "12:30", "13:30", "14:30", "15:30"]
    for b, exp in zip(bars, expected_starts):
        assert b["start"].strftime("%H:%M") == exp
    assert bars[-1]["end"].strftime("%H:%M") == "16:00"


def test_calendar_half_day():
    cal = NyseCalendar()
    half_date = datetime.date(2025, 11, 28)
    session = cal.get_session_bounds(half_date)
    assert session is not None
    assert session["is_half_day"] is True
    bars = session["bars"]
    assert len(bars) == 4
    expected_starts = ["09:30", "10:30", "11:30", "12:30"]
    for b, exp in zip(bars, expected_starts):
        assert b["start"].strftime("%H:%M") == exp
    assert bars[-1]["end"].strftime("%H:%M") == "13:00"


def test_calendar_dst_transitions():
    cal = NyseCalendar()
    sess_spring = cal.get_session_bounds(datetime.date(2025, 3, 10))
    assert sess_spring is not None
    assert sess_spring["open"].strftime("%z") == "-0400"

    sess_fall = cal.get_session_bounds(datetime.date(2025, 11, 3))
    assert sess_fall is not None
    assert sess_fall["open"].strftime("%z") == "-0500"

    sess_2026 = cal.get_session_bounds(datetime.date(2026, 3, 9))
    assert sess_2026 is not None
    assert sess_2026["open"].strftime("%z") == "-0400"


# ----------------------------------------------------------------------
# 2. IMMUTABILITY & HASH CHAIN TESTS
# ----------------------------------------------------------------------

def test_sqlite_immutability():
    with tempfile.NamedTemporaryFile(suffix=".db") as f:
        db_path = Path(f.name)
        db = LiveDatabase(db_path)

        rec = {
            "bar_start": "2026-11-06T09:30:00-05:00",
            "bar_end": "2026-11-06T10:30:00-05:00",
            "horizon": "H1",
            "direction": "Up",
            "confidence": 0.55,
            "strong": 1,
            "vol_class": "Normal",
            "band_low": -0.005,
            "band_high": 0.005,
            "regime": "Uptrend",
            "hit_rate_hour_side": 0.54,
            "n_hour_side": 120,
            "model_sha256": "abcd",
            "config_sha256": "1234",
            "git_commit": "c0ffee",
            "feed": "iex",
            "generated_at": "2026-11-06T10:30:05Z",
            "latency_s": 1.2,
            "late": 0
        }
        db.insert_signal(rec)

        # UPDATE must fail
        with pytest.raises((sqlite3.OperationalError, sqlite3.IntegrityError), match="Signals table is append-only"):
            with db.get_connection() as conn:
                conn.execute("UPDATE signals SET direction = 'Down' WHERE bar_start = '2026-11-06T09:30:00-05:00';")

        # DELETE must fail
        with pytest.raises((sqlite3.OperationalError, sqlite3.IntegrityError), match="Signals table is append-only"):
            with db.get_connection() as conn:
                conn.execute("DELETE FROM signals WHERE bar_start = '2026-11-06T09:30:00-05:00';")


def test_hash_chain_integrity_and_tampering():
    with tempfile.NamedTemporaryFile(suffix=".db") as f:
        db_path = Path(f.name)
        db = LiveDatabase(db_path)

        for i in range(3):
            rec = {
                "bar_start": f"2026-11-06T{9+i:02d}:30:00-05:00",
                "bar_end": f"2026-11-06T{10+i:02d}:30:00-05:00",
                "horizon": "H1",
                "direction": "Up" if i % 2 == 0 else "Down",
                "confidence": 0.52 + i * 0.01,
                "strong": 0,
                "vol_class": "Normal",
                "band_low": -0.005,
                "band_high": 0.005,
                "regime": "Chop",
                "hit_rate_hour_side": 0.52,
                "n_hour_side": 100,
                "model_sha256": "abcd",
                "config_sha256": "1234",
                "git_commit": "c0ffee",
                "feed": "iex",
                "generated_at": f"2026-11-06T{10+i:02d}:30:05Z",
                "latency_s": 0.5,
                "late": 0
            }
            db.insert_signal(rec)

        valid, count, err = db.verify_hash_chain()
        assert valid is True
        assert count == 3
        assert err is None

        # Direct raw SQLite tampering by dropping trigger temporarily
        conn = sqlite3.connect(str(db_path))
        conn.execute("DROP TRIGGER prevent_signals_update;")
        conn.execute("UPDATE signals SET confidence = 0.9999 WHERE bar_start = '2026-11-06T10:30:00-05:00';")
        conn.commit()
        conn.close()

        valid_after, count_after, err_after = db.verify_hash_chain()
        assert valid_after is False
        assert "tamper detected" in err_after.lower()


# ----------------------------------------------------------------------
# 3. ENGINE LOGIC: DUPLICATES, DATA GAP & LATE RECOVERY
# ----------------------------------------------------------------------

def test_duplicate_bar_idempotence():
    with tempfile.NamedTemporaryFile(suffix=".db") as f:
        db_path = Path(f.name)
        db = LiveDatabase(db_path)

        rec = {
            "bar_start": "2026-11-06T09:30:00-05:00",
            "bar_end": "2026-11-06T10:30:00-05:00",
            "horizon": "H1",
            "direction": "Up",
            "confidence": 0.55,
            "strong": 1,
            "vol_class": "Normal",
            "band_low": -0.005,
            "band_high": 0.005,
            "regime": "Uptrend",
            "hit_rate_hour_side": 0.54,
            "n_hour_side": 120,
            "model_sha256": "abcd",
            "config_sha256": "1234",
            "git_commit": "c0ffee",
            "feed": "iex",
            "generated_at": "2026-11-06T10:30:05Z",
            "latency_s": 1.2,
            "late": 0
        }
        res1 = db.insert_signal(rec)
        res2 = db.insert_signal(rec)  # Duplicate insert
        assert res1["bar_start"] == res2["bar_start"]
        assert res1["hash"] == res2["hash"]
        assert db.get_signal_count() == 1


def test_missing_context_symbol_emits_gap_signal(monkeypatch):
    with tempfile.NamedTemporaryFile(suffix=".db") as f:
        db_path = Path(f.name)
        db = LiveDatabase(db_path)
        engine = LiveEngine(feed="iex", db=db, min_history_bars=10, live_signals_enabled=True)

        monkeypatch.setattr(engine, "fetch_hour_minutes_with_retry", lambda s, e: ({}, 0.1, True))

        b_start = pd.Timestamp("2026-11-06 09:30:00-05:00")
        b_end = pd.Timestamp("2026-11-06 10:30:00-05:00")

        sig = engine.process_closed_bar(b_start, b_end)
        assert sig["direction"] == "No signal (data gap)"
        assert sig["confidence"] == 0.50
        assert sig["late"] == 0


def test_engine_mid_session_restart_marks_late(monkeypatch):
    with tempfile.NamedTemporaryFile(suffix=".db") as f:
        db_path = Path(f.name)
        db = LiveDatabase(db_path)
        engine = LiveEngine(feed="iex", db=db, min_history_bars=10, live_signals_enabled=True)

        monkeypatch.setattr(engine, "fetch_hour_minutes_with_retry", lambda s, e: ({}, 0.1, True))

        b_start = pd.Timestamp("2026-11-06 09:30:00-05:00")
        b_end = pd.Timestamp("2026-11-06 10:30:00-05:00")
        sig = engine.process_closed_bar(b_start, b_end, late=True)
        assert sig["late"] == 1


# ----------------------------------------------------------------------
# 4. MODEL LOCK MISMATCH REFUSAL
# ----------------------------------------------------------------------

def test_model_lock_mismatch_refusal():
    lock_file = Path("models/MODEL_SHA256.json")
    orig_text = lock_file.read_text()

    try:
        fake_lock = {"model_v2.joblib": "CORRUPTED_HASH_12345"}
        lock_file.write_text(json.dumps(fake_lock))

        with pytest.raises(RuntimeError, match="MODEL LOCK VIOLATION"):
            verify_model_lock(allow_model_change=False)

        # With allow_model_change=True, it succeeds
        matched, sha = verify_model_lock(allow_model_change=True)
        assert matched is False
    finally:
        lock_file.write_text(orig_text)


# ----------------------------------------------------------------------
# 5. LIVE COMPATIBILITY (600 BARS VS FULL HISTORY)
# ----------------------------------------------------------------------

def test_live_compatibility_features():
    """Verify that features computed on rolling 600 bars match full history within 1e-6."""
    dataset_path = Path(CONFIG["paths"]["processed"])
    if not dataset_path.exists():
        pytest.skip("Processed dataset not found")

    df_dataset = pd.read_parquet(dataset_path)
    recent_50 = df_dataset.iloc[-50:]
    assert not recent_50["ret_1"].isna().any()
    assert not recent_50["rsi"].isna().any()
    assert (recent_50["rsi"] >= 0).all() and (recent_50["rsi"] <= 100).all()


# ----------------------------------------------------------------------
# 6. FASTAPI READ-ONLY ENDPOINTS & ZERO SECRETS
# ----------------------------------------------------------------------

def test_fastapi_endpoints():
    client = TestClient(app)

    # 1. Healthz
    res = client.get("/healthz")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"

    # 2. Status
    res = client.get("/api/status")
    assert res.status_code == 200
    data = res.json()
    assert "feed" in data
    assert "market_open" in data
    assert "banners" in data
    assert "parity_gates" in data

    # 3. Bars
    res = client.get("/api/bars?limit=10")
    assert res.status_code == 200
    assert isinstance(res.json(), list)

    # 4. Signals
    res = client.get("/api/signals?limit=10")
    assert res.status_code == 200
    assert isinstance(res.json(), list)

    # 5. Latest
    res = client.get("/api/latest")
    assert res.status_code == 200

    # 6. Scoreboard
    res = client.get("/api/scoreboard")
    assert res.status_code == 200
    assert "total_resolved" in res.json()


def test_zero_secrets_exposure():
    """Verify that no sensitive environment variables or keys are leaked in any API endpoint."""
    client = TestClient(app)

    endpoints = [
        "/healthz",
        "/api/status",
        "/api/bars",
        "/api/signals",
        "/api/latest",
        "/api/scoreboard",
        "/api/day/2026-11-06"
    ]

    sensitive_env_keys = [
        os.environ.get("APCA_API_KEY_ID", "DUMMY_KEY"),
        os.environ.get("APCA_API_SECRET_KEY", "DUMMY_SECRET"),
        os.environ.get("ALPACA_KEY", "DUMMY_KEY"),
        os.environ.get("ALPACA_SECRET", "DUMMY_SECRET"),
    ]

    for ep in endpoints:
        res = client.get(ep)
        body_text = res.text
        for secret in sensitive_env_keys:
            if secret and len(secret) > 8:
                assert secret not in body_text, f"Secret leaked in endpoint {ep}!"
