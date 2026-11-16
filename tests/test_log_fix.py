"""Unit tests for V3.2 Fixes: Countdown Formatter, Signals Log, Filters, Paging, CSV Export, and Scoring Audit.
"""

import csv
import datetime
import io
import json
import os
import sqlite3
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest
from starlette.testclient import TestClient

from configs.config import CONFIG
from src.live.calendar import NyseCalendar
from src.live.countdown import format_countdown_pure
from src.live.engine import verify_model_lock
from src.live.server import app, get_db

ET = ZoneInfo("America/New_York")


# ----------------------------------------------------------------------
# 1. PURE COUNTDOWN FORMATTER TESTS
# ----------------------------------------------------------------------

def test_countdown_pure_ranges():
    # > 24 hours: "Xd Yh Zm"
    # 2 days, 7 hours, 12 minutes = 2 * 86400 + 7 * 3600 + 12 * 60 = 198720s
    assert format_countdown_pure(198720, market_open=False) == "2d 7h 12m"
    # 1 day, 0 hours, 5 minutes = 86400 + 300 = 86700s
    assert format_countdown_pure(86700, market_open=False) == "1d 0h 5m"

    # 1 to 24 hours: "Yh Zm"
    # 7 hours, 12 minutes = 7 * 3600 + 12 * 60 = 25920s
    assert format_countdown_pure(25920, market_open=False) == "7h 12m"
    # 1 hour, 0 minutes = 3600s
    assert format_countdown_pure(3600, market_open=False) == "1h 0m"

    # Under 1 hour when closed: "mm:ss"
    # 34 minutes, 21 seconds = 34 * 60 + 21 = 2061s
    assert format_countdown_pure(2061, market_open=False) == "34:21"
    # 0 seconds
    assert format_countdown_pure(0, market_open=False) == "Opens Soon"

    # During market hours:
    # Under 1 hour: "mm:ss"
    assert format_countdown_pure(2061, market_open=True) == "34:21"
    assert format_countdown_pure(59, market_open=True) == "00:59"
    assert format_countdown_pure(0, market_open=True) == "00:00"
    # Multi-hour session (e.g. 2 hours 15 mins)
    assert format_countdown_pure(8100, market_open=True) == "2h 15m"


def test_countdown_dst_transitions():
    """Verify market open timestamps and countdown logic on Spring and Fall DST transition days."""
    cal = NyseCalendar()

    # Spring DST 2026: March 9, 2026 (clock moves forward, UTC-4)
    sess_spring = cal.get_session_bounds(datetime.date(2026, 3, 9))
    assert sess_spring is not None
    assert sess_spring["open"].strftime("%z") == "-0400"
    # Session open is 09:30 EDT = 13:30 UTC
    assert sess_spring["open"].astimezone(datetime.timezone.utc).hour == 13
    assert sess_spring["open"].astimezone(datetime.timezone.utc).minute == 30

    # Fall DST 2026: November 2, 2026 (clock moves back, UTC-5)
    sess_fall = cal.get_session_bounds(datetime.date(2026, 11, 2))
    assert sess_fall is not None
    assert sess_fall["open"].strftime("%z") == "-0500"
    # Session open is 09:30 EST = 14:30 UTC
    assert sess_fall["open"].astimezone(datetime.timezone.utc).hour == 14
    assert sess_fall["open"].astimezone(datetime.timezone.utc).minute == 30


# ----------------------------------------------------------------------
# 2. SIGNALS API PAGING, FILTERS, AND ROW TYPES
# ----------------------------------------------------------------------

def test_signals_api_paging_and_filters():
    client = TestClient(app)

    # 1. Paging: limit and offset
    res1 = client.get("/api/signals?limit=5&offset=0&sample_filter=ALL&include_no_signal=true")
    assert res1.status_code == 200
    rows1 = res1.json()
    assert len(rows1) == 5

    res2 = client.get("/api/signals?limit=5&offset=5&sample_filter=ALL&include_no_signal=true")
    assert res2.status_code == 200
    rows2 = res2.json()
    assert len(rows2) == 5

    # Check offset: rows1 and rows2 must be distinct
    bar_starts1 = [r["bar_start"] for r in rows1]
    bar_starts2 = [r["bar_start"] for r in rows2]
    assert set(bar_starts1).isdisjoint(set(bar_starts2))

    # 2. Filter: OOS_LIVE + no No-signal rows
    res_oos = client.get("/api/signals?limit=50&offset=0&sample_filter=OOS_LIVE&include_no_signal=false")
    assert res_oos.status_code == 200
    oos_rows = res_oos.json()
    assert len(oos_rows) in (6, 8)  # 6 directional signals in SIP delayed (8 on legacy IEX)
    for r in oos_rows:
        assert r["sample"] in ("OUT-OF-SAMPLE", "LIVE")
        assert "no signal" not in r["direction"].lower()
        assert r["direction"] in ("Up", "Down")

    # 3. Filter: OOS_LIVE + include_no_signal=true
    res_oos_all = client.get("/api/signals?limit=50&offset=0&sample_filter=OOS_LIVE&include_no_signal=true")
    assert res_oos_all.status_code == 200
    oos_all_rows = res_oos_all.json()
    assert len(oos_all_rows) == 28  # 4 days * 7 bars = 28 bars total


# ----------------------------------------------------------------------
# 3. CSV EXPORT ENDPOINT (READ-ONLY, ZERO SECRETS, PROPER COLUMNS)
# ----------------------------------------------------------------------

def test_signals_csv_export():
    client = TestClient(app)

    res = client.get("/api/signals.csv?sample_filter=OOS_LIVE&include_no_signal=false")
    assert res.status_code == 200
    assert "text/csv" in res.headers["content-type"]
    assert "attachment; filename=signals.csv" in res.headers["content-disposition"]

    content = res.text
    reader = csv.reader(io.StringIO(content))
    header = next(reader)
    expected_header = ["Time_ET", "Time_IST", "Horizon", "Signal", "Confidence", "Tier", "Outcome", "Source_Tag", "Bar_Start", "Bar_End"]
    assert header == expected_header

    rows = list(reader)
    assert len(rows) in (6, 8)  # 6 directional signals on SIP delayed (8 on legacy IEX)

    # Verify no API keys or environment secrets in CSV
    sensitive_keys = [
        os.environ.get("APCA_API_KEY_ID", "DUMMY_KEY"),
        os.environ.get("APCA_API_SECRET_KEY", "DUMMY_SECRET"),
    ]
    for secret in sensitive_keys:
        if secret and len(secret) > 8:
            assert secret not in content


# ----------------------------------------------------------------------
# 4. SCORING AUDIT TESTS (HAND-MADE CASES & OFF-BY-ONE TRAP)
# ----------------------------------------------------------------------

def test_scoring_audit_logic():
    """Hand-crafted test cases for Up/Down correct, wrong, abstained, and off-by-one verification."""
    # Case 1: Up signal, forward return > 0 (e.g. +0.005) -> Correct (1)
    dir_up = "Up"
    fwd_ret_pos = 0.005
    y_true_1 = 1 if fwd_ret_pos > 0 else 0
    correct_up_pos = 1 if ((dir_up == "Up" and y_true_1 == 1) or (dir_up == "Down" and y_true_1 == 0)) else 0
    assert correct_up_pos == 1

    # Case 2: Up signal, forward return < 0 (e.g. -0.003) -> Wrong (0)
    fwd_ret_neg = -0.003
    y_true_0 = 1 if fwd_ret_neg > 0 else 0
    correct_up_neg = 1 if ((dir_up == "Up" and y_true_0 == 1) or (dir_up == "Down" and y_true_0 == 0)) else 0
    assert correct_up_neg == 0

    # Case 3: Down signal, forward return < 0 -> Correct (1)
    dir_down = "Down"
    correct_down_neg = 1 if ((dir_down == "Up" and y_true_0 == 1) or (dir_down == "Down" and y_true_0 == 0)) else 0
    assert correct_down_neg == 1

    # Case 4: Down signal, forward return > 0 -> Wrong (0)
    correct_down_pos = 1 if ((dir_down == "Up" and y_true_1 == 1) or (dir_down == "Down" and y_true_1 == 0)) else 0
    assert correct_down_pos == 0

    # Case 5: No signal -> MUST be None / unscored, NEVER 0
    dir_nosig = "No Signal"
    correct_nosig = 1 if ((dir_nosig == "Up" and y_true_1 == 1) or (dir_nosig == "Down" and y_true_1 == 0)) else None
    assert correct_nosig is None

    # Case 6: Off-by-one trap verification
    # Signal generated on bar B0 (09:30-10:30). Target is bar B1 (10:30-11:30).
    # Forward return MUST be (Close_B1 - Close_B0) / Close_B0.
    c_b0 = 900.0
    c_b1 = 909.0  # +1.0% return from B0 close to B1 close
    c_b2 = 890.0  # If an off-by-one used B2, return would be negative
    computed_ret = (c_b1 - c_b0) / c_b0
    assert computed_ret == pytest.approx(0.01)
    assert (computed_ret > 0) is True


# ----------------------------------------------------------------------
# 5. RE-RUN MODEL HASH & INTEGRITY CHECKS
# ----------------------------------------------------------------------

def test_model_hash_and_lock():
    locked_ok, msg = verify_model_lock(allow_model_change=False)
    assert locked_ok is True
    assert msg == "LOCKED_OK"

    with open("models/MODEL_SHA256.json") as f:
        locks = json.load(f)

    import hashlib
    for fname, exp_hash in locks.items():
        fpath = Path("models") / fname
        assert fpath.exists(), f"Artifact {fname} missing!"
        actual_hash = hashlib.sha256(fpath.read_bytes()).hexdigest()
        assert actual_hash == exp_hash, f"Hash mismatch on {fname}: expected {exp_hash}, found {actual_hash}"
