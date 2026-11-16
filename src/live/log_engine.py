"""Single Source of Truth Log Engine for Goldman Sachs AI Trading Indicator (V3.6 / Restored).

Computes unified signal logs, day summaries, hour strips, and multi-session balances.
Every number on the page (log, day header, hour strip, signal balance card, scoreboard)
derives from this module, ensuring zero discrepancies.
Thresholds are read from frozen calib_live.json and calib_v2.joblib (q_lo ~ 0.434, q_hi ~ 0.563).
"""

import datetime
import json
import math
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from zoneinfo import ZoneInfo
import pandas as pd
import joblib

from src.live.store import LiveDatabase

ET = ZoneInfo("America/New_York")
IST = ZoneInfo("Asia/Kolkata")


def get_rule_thresholds() -> Dict[str, Any]:
    """Read decision thresholds dynamically from frozen calibration files."""
    q_lo = 0.43427959108206615
    q_hi = 0.563232433386711
    strong_conf = 0.60
    try:
        calib_v2 = joblib.load("models/calib_v2.joblib")
        if isinstance(calib_v2, dict):
            q_lo = float(calib_v2.get("q_lo", q_lo))
            q_hi = float(calib_v2.get("q_hi", q_hi))
    except Exception:
        pass
    try:
        with open("models/calib_live.json") as f:
            cfg = json.load(f)
            val = float(cfg.get("strong_confidence_threshold", 0.60))
            if val >= 0.55:
                strong_conf = val
    except Exception:
        pass
    return {
        "rule_version": 1,
        "q_lo": q_lo,
        "q_hi": q_hi,
        "strong_conf": strong_conf
    }


RULE_CFG = get_rule_thresholds()
RULE_Q_LO = float(RULE_CFG["q_lo"])
RULE_Q_HI = float(RULE_CFG["q_hi"])
RULE_STRONG_CONF = float(RULE_CFG["strong_conf"])


def wilson_score_interval(k: int, n: int, z: float = 1.95996) -> Tuple[Optional[float], Optional[float]]:
    """Compute 95% Wilson score interval for binomial proportions."""
    if n == 0:
        return None, None
    p_hat = k / n
    denom = 1.0 + (z ** 2) / n
    centre = (p_hat + (z ** 2) / (2 * n)) / denom
    margin = (z / denom) * math.sqrt((p_hat * (1.0 - p_hat) / n) + ((z ** 2) / (4 * (n ** 2))))
    low = max(0.0, round((centre - margin) * 100, 2))
    high = min(100.0, round((centre + margin) * 100, 2))
    return low, high


def get_available_days(db: LiveDatabase) -> List[Dict[str, Any]]:
    """Query database for all available trading days with counts and latest indicator."""
    query = """
    SELECT s.bar_start, s.direction, s.sample, s.feed, s.rule_version
    FROM signals s
    WHERE (s.rule_version IS NULL OR s.rule_version = 1)
    ORDER BY s.bar_start ASC, CASE WHEN s.feed = 'sip_delayed' THEN 1 ELSE 2 END ASC;
    """
    with db.get_connection() as conn:
        cursor = conn.execute(query)
        rows = cursor.fetchall()

    days_map: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        bs = r["bar_start"]
        dt_et = pd.to_datetime(bs).astimezone(ET)
        day_str = dt_et.strftime("%Y-%m-%d")

        if day_str not in days_map:
            days_map[day_str] = {
                "date": day_str,
                "day_formatted": dt_et.strftime("%a %d %b %Y"),
                "sample": r["sample"],
                "total_bars": set(),
                "up_count": 0,
                "down_count": 0,
                "no_sig_count": 0,
                "has_oos": False,
                "has_live": False,
                "feed_sip_count": 0,
                "feed_iex_count": 0,
            }

        d_entry = days_map[day_str]
        # Only tally preferred feed per bar to prevent double-counting in day list
        if r["feed"] == "sip_delayed":
            d_entry["feed_sip_count"] += 1
            if bs not in d_entry["total_bars"]:
                d_entry["total_bars"].add(bs)
                if r["direction"] == "Up":
                    d_entry["up_count"] += 1
                elif r["direction"] == "Down":
                    d_entry["down_count"] += 1
                else:
                    d_entry["no_sig_count"] += 1
        else:
            d_entry["feed_iex_count"] += 1
            if bs not in d_entry["total_bars"]:
                d_entry["total_bars"].add(bs)
                if r["direction"] == "Up":
                    d_entry["up_count"] += 1
                elif r["direction"] == "Down":
                    d_entry["down_count"] += 1
                else:
                    d_entry["no_sig_count"] += 1

        if r["sample"] in ("OUT-OF-SAMPLE", "LIVE"):
            d_entry["has_oos"] = True
        if r["sample"] == "LIVE":
            d_entry["has_live"] = True

    result = []
    latest_scored_day = None
    for d_str in sorted(days_map.keys(), reverse=True):
        if days_map[d_str]["has_oos"]:
            latest_scored_day = d_str
            break

    for d_str in sorted(days_map.keys(), reverse=True):
        info = days_map[d_str]
        result.append({
            "date": info["date"],
            "day_formatted": info["day_formatted"],
            "sample": info["sample"],
            "total_bars": len(info["total_bars"]),
            "up_count": info["up_count"],
            "down_count": info["down_count"],
            "no_sig_count": info["no_sig_count"],
            "total_signals": info["up_count"] + info["down_count"],
            "has_oos": info["has_oos"],
            "has_live": info["has_live"],
            "is_latest_scored": (info["date"] == latest_scored_day)
        })

    return result


def build_signal_log(
    db: LiveDatabase,
    day: Optional[str] = None,
    horizon: str = "1 hour",
    view: str = "SCORED",
    include_iex: bool = False,
    include_nosignal: bool = True
) -> Dict[str, Any]:
    """Build unified, de-duplicated signal log rows and matching summary strip.
    
    Parameters:
    - day: YYYY-MM-DD string in ET. If None, picks latest session with scored signals.
    - horizon: '1 hour', '3 hours', or 'Rest of day'.
    - view: 'SCORED' (default, OOS + LIVE) or 'ALL' (includes IN-SAMPLE).
    - include_iex: if True, secondary IEX rows are included for transparency, flagged non-scored.
    - include_nosignal: if True (default for day view), neutral bars are displayed.
    """
    days_list = get_available_days(db)
    if not days_list:
        return {
            "day": day or datetime.date.today().isoformat(),
            "horizon": horizon,
            "view": view,
            "rows": [],
            "summary": _empty_summary(day or datetime.date.today().isoformat()),
            "empty_reason": "No signals exist in database."
        }

    # Resolve target day
    target_day = day
    if not target_day or target_day == "latest":
        if view == "SCORED":
            scored_day = next((d["date"] for d in days_list if d["has_oos"]), None)
            target_day = scored_day or days_list[0]["date"]
        else:
            target_day = days_list[0]["date"]

    # Normalize horizon mapping
    horizon_norm = "H1"
    if "3" in horizon:
        horizon_norm = "H3"
    elif "rest" in horizon.lower() or "eod" in horizon.lower():
        horizon_norm = "EOD"

    # Query all candidate signals for this day (strictly Rule 1 visible rows)
    query = """
    SELECT s.*, 
           b1.open AS open_sig,
           b1.close AS close_sig,
           b2.close AS close_target,
           o.realized_fwd_ret, 
           o.y_true, 
           o.correct
    FROM signals s
    LEFT JOIN bars b1 ON b1.symbol = 'GS' AND b1.timestamp = s.bar_start
    LEFT JOIN bars b2 ON b2.symbol = 'GS' AND b2.timestamp = s.bar_end
    LEFT JOIN outcomes_v2 o ON s.bar_start = o.bar_start
    WHERE s.bar_start LIKE ? AND (s.rule_version IS NULL OR s.rule_version = 1)
    ORDER BY s.bar_start ASC, CASE WHEN s.feed = 'sip_delayed' THEN 1 ELSE 2 END ASC;
    """
    with db.get_connection() as conn:
        cursor = conn.execute(query, (f"{target_day}%",))
        raw_rows = [dict(r) for r in cursor.fetchall()]

    if not raw_rows:
        return {
            "day": target_day,
            "horizon": horizon,
            "view": view,
            "rows": [],
            "summary": _empty_summary(target_day),
            "empty_reason": f"No signals recorded on {target_day}."
        }

    # Filter by horizon
    raw_rows = [r for r in raw_rows if r.get("horizon", "H1") == horizon_norm]
    if not raw_rows:
        return {
            "day": target_day,
            "horizon": horizon,
            "view": view,
            "rows": [],
            "summary": _empty_summary(target_day),
            "empty_reason": f"No signals recorded for {horizon} on {target_day}."
        }

    # De-duplicate by bar_start: strictly exactly one row per bar
    by_bar: Dict[str, List[Dict[str, Any]]] = {}
    for r in raw_rows:
        bs = r["bar_start"]
        if bs not in by_bar:
            by_bar[bs] = []
        by_bar[bs].append(r)

    processed_rows: List[Dict[str, Any]] = []

    for bs, group in by_bar.items():
        # Preference: SIP delayed, then legacy IEX
        sip_row = next((r for r in group if r.get("feed") == "sip_delayed"), None)
        iex_row = next((r for r in group if r.get("feed") == "iex"), None)

        primary = sip_row or iex_row
        if not primary:
            continue

        # Check view filter on primary
        is_primary_oos = primary["sample"] in ("OUT-OF-SAMPLE", "LIVE")
        if view == "SCORED" and not is_primary_oos:
            continue

        formatted_primary = _format_row(primary, is_scored=(primary["feed"] == "sip_delayed"), is_iex=False)
        processed_rows.append(formatted_primary)

        # If user explicitly enabled IEX rows
        if include_iex and iex_row and iex_row != primary:
            formatted_iex = _format_row(iex_row, is_scored=False, is_iex=True)
            processed_rows.append(formatted_iex)

    # Sort chronological by bar_start (and secondary IEX row directly after primary)
    processed_rows.sort(key=lambda r: (r["bar_start"], 1 if r["is_iex"] else 0))

    # Calculate single-source-of-truth summary EXCLUSIVELY on scored rows
    scored_rows = [r for r in processed_rows if r["is_scored"]]
    in_sample_exist_count = sum(1 for r in raw_rows if r["sample"] == "IN-SAMPLE")

    if not scored_rows and view == "SCORED":
        return {
            "day": target_day,
            "horizon": horizon,
            "view": view,
            "rows": [],
            "summary": _empty_summary(target_day),
            "empty_reason": f"No scored signals on this day; {in_sample_exist_count} exist if you enable 'All including in-sample'."
        }

    # Tally summary metrics
    total_bars = len(scored_rows)
    up_count = sum(1 for r in scored_rows if r["direction"] == "Up")
    down_count = sum(1 for r in scored_rows if r["direction"] == "Down")
    no_sig_count = sum(1 for r in scored_rows if r["direction"] not in ("Up", "Down"))
    total_signals = up_count + down_count

    correct_count = sum(1 for r in scored_rows if r["correct"] == 1 and r["direction"] in ("Up", "Down"))
    wrong_count = sum(1 for r in scored_rows if r["correct"] == 0 and r["realized_fwd_ret"] is not None and r["direction"] in ("Up", "Down"))
    pending_count = sum(1 for r in scored_rows if r["direction"] in ("Up", "Down") and r["realized_fwd_ret"] is None)

    # Accuracy and CI
    acc_pct = None
    ci_low, ci_high = None, None
    wilson_str = "n/a"
    if total_signals > 0:
        evaluated = correct_count + wrong_count
        if evaluated > 0:
            acc_pct = round((correct_count / evaluated) * 100, 1)
            ci_low, ci_high = wilson_score_interval(correct_count, evaluated)
            if ci_low is not None and ci_high is not None:
                wilson_str = f"Wilson 95% CI [{ci_low}%, {ci_high}%]"

    # Day Open & Close price tracking directly from bars table for full accuracy
    day_open = None
    day_close = None
    with db.get_connection() as conn:
        c_open = conn.execute("SELECT open FROM bars WHERE symbol = 'GS' AND timestamp LIKE ? ORDER BY timestamp ASC LIMIT 1;", (f"{target_day}%",)).fetchone()
        c_close = conn.execute("SELECT close FROM bars WHERE symbol = 'GS' AND timestamp LIKE ? ORDER BY timestamp DESC LIMIT 1;", (f"{target_day}%",)).fetchone()
        if c_open and c_open["open"] is not None:
            day_open = float(c_open["open"])
        if c_close and c_close["close"] is not None:
            day_close = float(c_close["close"])

    day_return_pct = 0.0
    price_str = "–"
    if day_open is not None and day_close is not None:
        day_return_pct = round(((day_close - day_open) / day_open) * 100, 2)
        sign = "+" if day_return_pct >= 0 else ""
        price_str = f"GS ${day_open:.2f} → ${day_close:.2f} ({sign}{day_return_pct:.2f}%)"

    acc_display = f"{correct_count}/{correct_count + wrong_count}" if (correct_count + wrong_count) > 0 else "0/0"
    day_dt = pd.to_datetime(target_day).tz_localize(ET)
    day_formatted = day_dt.strftime("%a %d %b %Y")

    summary = {
        "date": target_day,
        "day_str": day_formatted,
        "symbol": "GS",
        "open_price": day_open,
        "close_price": day_close,
        "day_return_pct": day_return_pct,
        "price_str": price_str,
        "total_bars": total_bars,
        "total_signals": total_signals,
        "up_count": up_count,
        "down_count": down_count,
        "no_signal_count": no_sig_count,
        "correct_count": correct_count,
        "wrong_count": wrong_count,
        "pending_count": pending_count,
        "accuracy_pct": acc_pct,
        "accuracy_str": acc_display,
        "wilson_ci": [ci_low, ci_high] if ci_low is not None else None,
        "wilson_ci_str": wilson_str,
        "demonstration_note": f"demonstration, not proof (n={total_signals})",
        "summary_line": f"{day_formatted} | {price_str} | Signals {total_signals} (Up {up_count}, Down {down_count}) | Correct {correct_count} | Wrong {wrong_count} | Pending {pending_count} | No signal {no_sig_count} | Accuracy {acc_display} ({wilson_str})"
    }

    # Filter rows by include_nosignal if requested
    output_rows = processed_rows
    if not include_nosignal:
        output_rows = [r for r in processed_rows if r["direction"] in ("Up", "Down")]

    return {
        "day": target_day,
        "horizon": horizon,
        "view": view,
        "rows": output_rows,
        "summary": summary,
        "empty_reason": None
    }


def _empty_summary(date_str: str) -> Dict[str, Any]:
    """Fallback zero summary structure."""
    return {
        "date": date_str,
        "day_str": date_str,
        "symbol": "GS",
        "open_price": None,
        "close_price": None,
        "day_return_pct": 0.0,
        "price_str": "–",
        "total_bars": 0,
        "total_signals": 0,
        "up_count": 0,
        "down_count": 0,
        "no_signal_count": 0,
        "correct_count": 0,
        "wrong_count": 0,
        "pending_count": 0,
        "accuracy_pct": None,
        "accuracy_str": "0/0",
        "wilson_ci": None,
        "wilson_ci_str": "n/a",
        "demonstration_note": "demonstration, not proof (n=0)",
        "summary_line": f"{date_str} | Signals 0 | Accuracy 0/0"
    }


def _format_row(r: Dict[str, Any], is_scored: bool = True, is_iex: bool = False, is_v1_legacy: bool = False) -> Dict[str, Any]:
    """Format an individual database signal record into an unambiguous log row."""
    dt_start = pd.to_datetime(r["bar_start"])
    if dt_start.tzinfo is None:
        dt_start = dt_start.replace(tzinfo=ET)
    else:
        dt_start = dt_start.astimezone(ET)

    dt_end = pd.to_datetime(r["bar_end"])
    if dt_end.tzinfo is None:
        dt_end = dt_end.replace(tzinfo=ET)
    else:
        dt_end = dt_end.astimezone(ET)

    start_str = dt_start.strftime("%H:%M")
    end_str = dt_end.strftime("%H:%M")
    
    # Target window is bar_end to bar_end + horizon
    if start_str == "15:30" or end_str >= "16:00":
        hour_predicted_et = "16:00 (market closed)"
    elif start_str == "14:30":
        hour_predicted_et = "15:30-16:00 (close)"
    else:
        target_end_dt = dt_end + datetime.timedelta(hours=1)
        hour_predicted_et = f"{end_str}-{target_end_dt.strftime('%H:%M')}"

    # Issued (ET | IST)
    if r.get("delivered_at"):
        issued_dt = pd.to_datetime(r["delivered_at"])
        if issued_dt.tzinfo is None:
            issued_dt = issued_dt.replace(tzinfo=ET)
        else:
            issued_dt = issued_dt.astimezone(ET)
    else:
        issued_dt = dt_end + datetime.timedelta(minutes=16)

    issued_et = issued_dt.strftime("%H:%M")
    issued_ist = issued_dt.astimezone(IST).strftime("%H:%M")
    is_replayed = bool(r.get("late", 0) == 1 or r.get("sample") == "REPLAY")

    # Call formatting
    dir_val = r.get("direction", "No signal")
    is_up = dir_val == "Up"
    is_down = dir_val == "Down"

    conf = float(r["confidence"]) if r.get("confidence") is not None else None
    row_q_hi = RULE_Q_HI
    row_q_lo = RULE_Q_LO

    is_strong = bool((is_up or is_down) and (r.get("strong") == 1 or (conf is not None and conf >= RULE_STRONG_CONF)))

    if is_up:
        p_up = conf if conf is not None else float(r.get("p_cal") or 0.5)
        call_pill = f"▲ UP {conf*100:.0f}%" if conf is not None else "▲ UP"
        call_reason = None
        strength_display = "STRONG" if is_strong else ""
    elif is_down:
        p_up = (1.0 - conf) if conf is not None else float(r.get("p_cal") or 0.5)
        call_pill = f"▼ DOWN {conf*100:.0f}%" if conf is not None else "▼ DOWN"
        call_reason = None
        strength_display = "STRONG" if is_strong else ""
    else:
        p_up = float(r.get("p_cal") or (conf if conf is not None else 0.5))
        call_pill = "– NO SIGNAL"
        strength_display = ""
        conf = None
        dir_lower = dir_val.lower()
        if "end of day" in dir_lower or end_str >= "16:00":
            call_reason = "end of day"
        elif "gap" in dir_lower:
            call_reason = "data gap"
        elif "closed" in dir_lower:
            call_reason = "market closed"
        else:
            call_reason = "inside neutral band"

    threshold_str = f"P(Up) {p_up*100:.1f}%. Up needs >= {row_q_hi*100:.1f}%, Down needs <= {row_q_lo*100:.1f}%"

    vol_class = r.get("vol_class") or "Normal"
    b_low = r.get("band_low")
    b_high = r.get("band_high")
    if b_low is not None and b_high is not None:
        b_low_bps = round(float(b_low) * 10000, 1)
        b_high_bps = round(float(b_high) * 10000, 1)
        sign_h = "+" if b_high_bps >= 0 else ""
        vol_band_display = f"{vol_class} | {b_low_bps} / {sign_h}{b_high_bps} bps"
    else:
        vol_band_display = f"{vol_class} | –"

    close_sig = float(r["close_sig"]) if r.get("close_sig") is not None else None
    close_target = float(r["close_target"]) if r.get("close_target") is not None else None
    realized_ret = float(r["realized_fwd_ret"]) if r.get("realized_fwd_ret") is not None else None

    price_move_display = "pending"
    if close_sig is not None and close_target is not None:
        pct = realized_ret * 100 if realized_ret is not None else ((close_target - close_sig) / close_sig) * 100
        sign_p = "+" if pct >= 0 else ""
        price_move_display = f"${close_sig:.2f} → ${close_target:.2f} ({sign_p}{pct:.2f}%)"
    elif close_sig is not None:
        price_move_display = f"${close_sig:.2f} → pending"

    # Dynamic accuracy determination from realized forward return
    if is_up:
        is_correct = 1 if (realized_ret is not None and realized_ret > 0) else (0 if realized_ret is not None else None)
    elif is_down:
        is_correct = 1 if (realized_ret is not None and realized_ret < 0) else (0 if realized_ret is not None else None)
    else:
        is_correct = None

    if is_up or is_down:
        if is_correct == 1:
            result_display = "✔ Correct"
            result_symbol = "✔"
        elif is_correct == 0 and realized_ret is not None:
            result_display = "✘ Wrong"
            result_symbol = "✘"
        else:
            result_display = "○ Pending"
            result_symbol = "○"
    else:
        result_display = "–"
        result_symbol = "–"

    return {
        "bar_start": r["bar_start"],
        "bar_end": r["bar_end"],
        "horizon": r.get("horizon", "H1"),
        "hour_predicted_et": hour_predicted_et,
        "issued_et": issued_et,
        "issued_ist": issued_ist,
        "issued_display": f"{issued_et} | {issued_ist}",
        "is_replayed": is_replayed,
        "direction": dir_val,
        "call_display": call_pill,
        "call_reason": call_reason,
        "confidence": conf if (is_up or is_down) else None,
        "p_up": round(p_up, 4),
        "q_lo": row_q_lo,
        "q_hi": row_q_hi,
        "threshold_str": threshold_str,
        "strong": 1 if is_strong else 0,
        "strength_display": strength_display,
        "vol_class": vol_class,
        "vol_band_display": vol_band_display,
        "close_sig": close_sig,
        "close_target": close_target,
        "realized_fwd_ret": realized_ret,
        "price_move_display": price_move_display,
        "correct": is_correct,
        "result_display": result_display,
        "result_symbol": result_symbol,
        "sample": r.get("sample", "OUT-OF-SAMPLE"),
        "feed": r.get("feed", "sip_delayed"),
        "is_iex": is_iex,
        "is_scored": is_scored,
        "regime": r.get("regime", "Chop"),
        "model_hash": "75d0b0bf",
        "delivered_at": r.get("delivered_at"),
        "delivery_delay_s": float(r.get("delivery_delay_s") or 960.0),
        "row_hash": r.get("row_hash") or "genesis_hash",
        "bar_close_time_et": end_str
    }


def get_multi_session_balance(db: LiveDatabase) -> Dict[str, Any]:
    """Single source of truth for 1d, 4d, and 20d session signal balance."""
    query = """
    SELECT s.bar_start, s.direction, s.sample
    FROM signals s
    WHERE s.feed = 'sip_delayed' AND (s.rule_version IS NULL OR s.rule_version = 1)
    ORDER BY s.bar_start ASC;
    """
    with db.get_connection() as conn:
        cursor = conn.execute(query)
        rows = [dict(r) for r in cursor.fetchall()]

    by_day: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        d = r["bar_start"][:10]
        if d not in by_day:
            by_day[d] = []
        by_day[d].append(r)

    sorted_days = sorted(by_day.keys())

    def summarize_days(days_subset: List[str]) -> Dict[str, int]:
        u, d, n = 0, 0, 0
        for dy in days_subset:
            for r in by_day[dy]:
                if r["direction"] == "Up":
                    u += 1
                elif r["direction"] == "Down":
                    d += 1
                else:
                    n += 1
        return {"up": u, "down": d, "no_signal": n, "total": u + d + n}

    bal_1d = summarize_days(sorted_days[-1:] if sorted_days else [])
    bal_4d = summarize_days(sorted_days[-4:] if sorted_days else [])
    bal_20d = summarize_days(sorted_days[-20:] if sorted_days else [])

    dir_signals = [r["direction"] for r in rows if r["direction"] in ("Up", "Down")]
    consec_count = 0
    consec_side = None
    if dir_signals:
        last_side = dir_signals[-1]
        for s in reversed(dir_signals):
            if s == last_side:
                consec_count += 1
            else:
                break
        consec_side = last_side

    return {
        "bal_1d": bal_1d,
        "bal_4d": bal_4d,
        "bal_20d": bal_20d,
        "last_day_str": sorted_days[-1] if sorted_days else "n/a",
        "consec_count": consec_count,
        "consec_side": consec_side,
        "one_sided_alert": (consec_count >= 6)
    }


def get_live_scoreboard_stats(db: LiveDatabase) -> Dict[str, Any]:
    """Single source of truth for live out-of-sample scoreboard metrics."""
    query = """
    SELECT s.direction, s.strong, s.sample, o.correct, o.realized_fwd_ret
    FROM signals s
    LEFT JOIN outcomes_v2 o ON s.bar_start = o.bar_start
    WHERE s.feed = 'sip_delayed' AND (s.rule_version IS NULL OR s.rule_version = 1) AND s.sample IN ('OUT-OF-SAMPLE', 'LIVE');
    """
    with db.get_connection() as conn:
        cursor = conn.execute(query)
        rows = [dict(r) for r in cursor.fetchall()]

    candidate_bars = len(rows)
    dir_rows = [r for r in rows if r["direction"] in ("Up", "Down")]
    resolved = [r for r in dir_rows if r["realized_fwd_ret"] is not None]

    for r in resolved:
        ret = float(r["realized_fwd_ret"])
        if r["direction"] == "Up":
            r["is_hit"] = 1 if ret > 0 else 0
        elif r["direction"] == "Down":
            r["is_hit"] = 1 if ret < 0 else 0
        else:
            r["is_hit"] = 0

    hits = sum(r["is_hit"] for r in resolved)
    n_resolved = len(resolved)
    hit_rate = (hits / n_resolved) if n_resolved > 0 else 0.0

    up_signals = [r for r in resolved if r["direction"] == "Up"]
    up_hits = sum(r["is_hit"] for r in up_signals)
    up_acc = (up_hits / len(up_signals)) if up_signals else 0.0

    down_signals = [r for r in resolved if r["direction"] == "Down"]
    down_hits = sum(r["is_hit"] for r in down_signals)
    down_acc = (down_hits / len(down_signals)) if down_signals else 0.0

    strong_signals = [r for r in resolved if r.get("strong") == 1]
    strong_hits = sum(r["is_hit"] for r in strong_signals)
    strong_acc = (strong_hits / len(strong_signals)) if strong_signals else 0.0

    ci_low, ci_high = wilson_score_interval(hits, n_resolved)
    coverage_pct = round((len(dir_rows) / candidate_bars) * 100, 1) if candidate_bars > 0 else 0.0

    return {
        "label": "OUT-OF-SAMPLE (descriptive only, small n)",
        "total_resolved": n_resolved,
        "n_candidate_candles": candidate_bars,
        "hits": hits,
        "accuracy_pct": round(hit_rate * 100, 2),
        "wilson_ci": [ci_low, ci_high] if ci_low is not None else [0.0, 0.0],
        "up_signals_count": len(up_signals),
        "up_count": len(up_signals),
        "up_hits": up_hits,
        "up_accuracy": up_acc,
        "up_hit_rate": round(up_acc * 100, 2),
        "down_signals_count": len(down_signals),
        "down_count": len(down_signals),
        "down_hits": down_hits,
        "down_accuracy": down_acc,
        "down_hit_rate": round(down_acc * 100, 2),
        "strong_signals_count": len(strong_signals),
        "strong_hits": strong_hits,
        "strong_hit_rate": round(strong_acc * 100, 2),
        "coverage_count": len(dir_rows),
        "coverage_pct": coverage_pct,
        "all_candle_hits_pct": round((hits / candidate_bars) * 100, 1) if candidate_bars > 0 else 0.0,
        "note": "Evaluated strictly on delayed SIP tape. Earlier 5 of 8 result came from IEX signals and is not scored."
    }
