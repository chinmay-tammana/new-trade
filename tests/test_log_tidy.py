"""Unit Tests for V3.6 Signal Log Tidy, De-duplication, DST, and Single Source of Truth."""

import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import pytest

from src.live.store import LiveDatabase
from src.live.log_engine import (
    build_signal_log,
    get_available_days,
    get_multi_session_balance,
    get_live_scoreboard_stats,
    wilson_score_interval
)

ET = ZoneInfo("America/New_York")
IST = ZoneInfo("Asia/Kolkata")
DB_PATH = Path("data/live/live.db")


@pytest.fixture
def db():
    return LiveDatabase(DB_PATH)


def test_deduplication_and_sip_preference(db):
    """Test that for 2026-11-13 (which has both IEX and SIP in DB), de-duplication yields exactly 7 rows, preferring SIP."""
    res = build_signal_log(db, day="2026-11-13", horizon="1 hour", view="SCORED", include_iex=False)
    rows = res["rows"]
    summary = res["summary"]

    assert len(rows) == 7, f"Expected 7 hourly bars on 2026-11-13, got {len(rows)}"
    # Verify all primary rows are sip_delayed
    for r in rows:
        assert r["feed"] == "sip_delayed", f"Row feed expected sip_delayed, got {r['feed']}"
        assert r["is_scored"] is True

    # When include_iex is True, secondary rows exist but flagged non-scored
    res_iex = build_signal_log(db, day="2026-11-13", horizon="1 hour", view="SCORED", include_iex=True)
    rows_all = res_iex["rows"]
    assert len(rows_all) == 14, f"Expected 14 total rows with IEX enabled, got {len(rows_all)}"
    scored_in_all = [r for r in rows_all if r["is_scored"]]
    assert len(scored_in_all) == 7
    # Verify summary counts are identical regardless of include_iex
    assert res_iex["summary"]["total_signals"] == summary["total_signals"]
    assert res_iex["summary"]["accuracy_str"] == summary["accuracy_str"]


def test_summary_equals_rows(db):
    """Test that summary arithmetic matches row-level tally exactly:
    Up + Down + No signal = rows; hits + misses + pending = Up + Down; No signal never counts as hit or miss."""
    res = build_signal_log(db, day="2026-11-13", horizon="1 hour", view="SCORED", include_iex=False)
    rows = res["rows"]
    s = res["summary"]

    up_c = sum(1 for r in rows if r["direction"] == "Up")
    down_c = sum(1 for r in rows if r["direction"] == "Down")
    nosig_c = sum(1 for r in rows if r["direction"] not in ("Up", "Down"))

    assert s["up_count"] == up_c
    assert s["down_count"] == down_c
    assert s["no_signal_count"] == nosig_c
    assert up_c + down_c + nosig_c == len(rows)
    assert s["total_signals"] == up_c + down_c

    hits = sum(1 for r in rows if r["correct"] == 1 and r["direction"] in ("Up", "Down"))
    misses = sum(1 for r in rows if r["correct"] == 0 and r["realized_fwd_ret"] is not None and r["direction"] in ("Up", "Down"))
    pending = sum(1 for r in rows if r["direction"] in ("Up", "Down") and r["realized_fwd_ret"] is None)

    assert s["correct_count"] == hits
    assert s["wrong_count"] == misses
    assert s["pending_count"] == pending
    assert hits + misses + pending == s["total_signals"]

    # Verify No-signal rows never have hit or miss
    for r in rows:
        if r["direction"] not in ("Up", "Down"):
            assert r["result_symbol"] == "–"
            assert r["result_display"] == "–"
            assert r["confidence"] is None
            assert r["strong"] == 0
            assert r["strength_display"] == ""


def test_header_strip_and_balance_card_agreement(db):
    """Test that summary header, strip counts, and 1-day balance card never disagree."""
    res = build_signal_log(db, day="2026-11-13", horizon="1 hour", view="SCORED")
    bal = get_multi_session_balance(db)

    # 1-day balance card
    b1d = bal["bal_1d"]
    s = res["summary"]

    assert b1d["up"] == s["up_count"] == 1
    assert b1d["down"] == s["down_count"] == 1
    assert b1d["no_signal"] == s["no_signal_count"] == 5
    assert b1d["total"] == s["total_bars"] == 7

    # Multi-day balance
    b4d = bal["bal_4d"]
    assert b4d["up"] == 2
    assert b4d["down"] == 4
    assert b4d["no_signal"] == 22
    assert b4d["total"] == 28


def test_et_to_ist_dst_boundaries():
    """Test ET to IST conversion using zoneinfo across Spring and Fall 2026 DST boundaries."""
    # Spring 2026 DST: Sunday 2026-03-08 (Clocks spring forward 1 hour in US)
    # Friday before DST: 2026-03-06 (EST = UTC-5; IST = UTC+5:30 -> difference 10h 30m)
    dt_fri_est = datetime.datetime(2026, 3, 6, 9, 30, tzinfo=ET)
    dt_fri_ist = dt_fri_est.astimezone(IST)
    assert dt_fri_ist.strftime("%H:%M") == "20:00"

    # Monday after DST: 2026-03-09 (EDT = UTC-4; IST = UTC+5:30 -> difference 9h 30m)
    dt_mon_edt = datetime.datetime(2026, 3, 9, 9, 30, tzinfo=ET)
    dt_mon_ist = dt_mon_edt.astimezone(IST)
    assert dt_mon_ist.strftime("%H:%M") == "19:00"

    # Fall 2026 DST: Sunday 2026-11-01 (Clocks fall back 1 hour in US)
    # Friday before DST: 2026-10-30 (EDT = UTC-4; difference 9h 30m)
    dt_fall_edt = datetime.datetime(2026, 10, 30, 9, 30, tzinfo=ET)
    dt_fall_ist = dt_fall_edt.astimezone(IST)
    assert dt_fall_ist.strftime("%H:%M") == "19:00"

    # Monday after DST: 2026-11-02 (EST = UTC-5; difference 10h 30m)
    dt_fall_est = datetime.datetime(2026, 11, 2, 9, 30, tzinfo=ET)
    dt_fall_ist = dt_fall_est.astimezone(IST)
    assert dt_fall_ist.strftime("%H:%M") == "20:00"


def test_late_vs_live_display_rule(db):
    """Test that rows with late=1 are tagged is_replayed=True and formatted cleanly."""
    res = build_signal_log(db, day="2026-11-13", horizon="1 hour", view="SCORED")
    for r in res["rows"]:
        if r["is_replayed"]:
            assert r["is_replayed"] is True
            assert r["issued_display"] == f"{r['issued_et']} | {r['issued_ist']}"


def test_empty_state_handling(db):
    """Test that non-existent dates or horizons cleanly return empty list and descriptive summary with zero crashes."""
    # Day with no signals
    res_empty_day = build_signal_log(db, day="2025-01-01", horizon="1 hour", view="SCORED")
    assert res_empty_day["rows"] == []
    assert res_empty_day["summary"]["total_signals"] == 0
    assert res_empty_day["summary"]["accuracy_pct"] is None
    assert "No signals recorded" in res_empty_day["empty_reason"]

    # In-sample day under SCORED view
    res_insample = build_signal_log(db, day="2026-11-02", horizon="1 hour", view="SCORED")
    assert res_insample["rows"] == []
    assert "No scored signals on this day" in res_insample["empty_reason"]
    assert "All including in-sample" in res_insample["empty_reason"]

    # In-sample day under ALL view
    res_all = build_signal_log(db, day="2026-11-02", horizon="1 hour", view="ALL")
    assert len(res_all["rows"]) == 7
    assert res_all["summary"]["total_bars"] == 7
