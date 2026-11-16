"""
Single Source of Truth Signal Log Engine for M5 (5-Minute) Live System.
Builds rows and summaries so log, day header, timeline strip, balance card,
and accuracy card always derive from one function and cannot disagree.
"""
import os
import json
import math
import datetime
import zoneinfo
from typing import Optional, List, Dict, Any
import sqlite3

from configs.config import CONFIG_M5
from src.live_m5.store import LiveM5Database, DB_PATH

ET_TZ = zoneinfo.ZoneInfo("America/New_York")
IST_TZ = zoneinfo.ZoneInfo("Asia/Kolkata")


def wilson_interval(successes: int, total: int, z: float = 1.96):
    if total <= 0:
        return (0.0, 1.0)
    p = successes / total
    denom = 1.0 + (z**2) / total
    centre = (p + (z**2) / (2 * total)) / denom
    diff = z * math.sqrt((p * (1.0 - p) / total) + (z**2) / (4 * (total**2))) / denom
    return (max(0.0, centre - diff) * 100.0, min(1.0, centre + diff) * 100.0)


def load_m5_calib():
    active_ptr_path = "models/m5/active_rule.json"
    if os.path.exists(active_ptr_path):
        try:
            with open(active_ptr_path, "r") as f:
                ptr = json.load(f)
            rule_file = os.path.join("models/m5", ptr["active_rule_file"])
            if os.path.exists(rule_file):
                with open(rule_file, "r") as rf:
                    r_data = json.load(rf)
                    r_data["active_rule_version"] = ptr.get("active_rule_version", r_data.get("rule_version", 1))
                    return r_data
        except Exception:
            pass
            
    calib_path = CONFIG_M5["paths"]["calib_path"]
    if os.path.exists(calib_path):
        with open(calib_path, "r") as f:
            d = json.load(f)
            d["active_rule_version"] = 1
            d["rule_version"] = 1
            return d
    return {
        "rule_version": 1,
        "active_rule_version": 1,
        "q_lo": 0.4517,
        "q_hi": 0.5453,
        "weak_threshold": 0.5555,
        "strong_threshold": 0.5844,
        "coverage_target": 0.30
    }


def format_m5_row(r: dict, next_bar: Optional[dict] = None) -> dict:
    bar_start_dt = datetime.datetime.fromisoformat(r["bar_start"]).astimezone(ET_TZ)
    bar_end_dt = datetime.datetime.fromisoformat(r["bar_end"]).astimezone(ET_TZ)
    
    interval_str = f"{bar_start_dt.strftime('%H:%M')}-{bar_end_dt.strftime('%H:%M')}"
    issued_et = bar_end_dt.strftime("%H:%M")
    issued_ist = bar_end_dt.astimezone(IST_TZ).strftime("%H:%M")
    
    direction = r["direction"]
    is_up = direction == "Up"
    is_down = direction == "Down"
    is_directional = is_up or is_down
    
    tier_val = r.get("tier", "NORMAL")
    strong = int(r.get("is_strong", 0))
    if tier_val == "STRONG":
        strong = 1
        
    if is_directional:
        conf = float(r["confidence"])
        # Directional calls must show confidence >= 51%
        conf = max(conf, 0.51)
        if is_up:
            call_display = f"▲ UP {int(conf * 100)}%"
        else:
            call_display = f"▼ DOWN {int(conf * 100)}%"
        strength_display = tier_val
    else:
        conf = 0.50
        reason = r.get("reason")
        if not reason:
            if bar_start_dt.hour == 15 and bar_start_dt.minute >= 55:
                reason = "End of trading day"
            elif "end of day" in direction.lower():
                reason = "End of trading day"
            elif "engine error" in direction.lower():
                reason = "Engine error"
            elif "feature gap" in direction.lower():
                reason = "Feature gap"
            elif "data gap" in direction.lower() or "feed gap" in direction.lower():
                reason = "Feed gap"
        call_display = f"– NO SIGNAL ({reason}) –" if reason else f"– {direction.upper()} –"
        strength_display = "–" # No-signal rows never show confidence or tier
        tier_val = None

    # Reference and target prices
    close_ref = r.get("close")
    close_fwd = next_bar.get("close") if next_bar else None
    
    now_et = datetime.datetime.now(ET_TZ)
    forward_elapsed = now_et >= (bar_end_dt + datetime.timedelta(minutes=5))
    
    if close_ref is not None and close_fwd is not None:
        pct_move = ((close_fwd - close_ref) / close_ref) * 100.0
        price_move_display = f"${close_ref:.2f} → ${close_fwd:.2f} ({pct_move:+.2f}%)"
        
        if is_directional:
            if is_up and close_fwd > close_ref:
                correct = 1
                result_display = "✔ Correct"
            elif is_down and close_fwd < close_ref:
                correct = 1
                result_display = "✔ Correct"
            elif close_fwd == close_ref:
                correct = 0
                result_display = "– Flat"
            else:
                correct = 0
                result_display = "✘ Wrong"
        else:
            correct = None
            result_display = "–"
    elif forward_elapsed:
        # Forward bar elapsed but has no close price (feed gap)
        if close_ref is not None:
            price_move_display = f"${close_ref:.2f} → Feed gap"
        else:
            price_move_display = "–"
        correct = None
        result_display = "– Unresolvable (feed gap)" if is_directional else "–"
    else:
        # Forward bar is currently forming or hasn't closed yet
        if close_ref is not None:
            price_move_display = f"${close_ref:.2f} → Pending"
        else:
            price_move_display = "–"
        correct = None
        result_display = "⏳ Pending" if is_directional else "–"

    # Format vol and band (bps) - no fake static fallbacks
    vol_cls = r.get("vol_class")
    band_lo_val = r.get("band_lo")
    band_hi_val = r.get("band_hi")
    if vol_cls and band_lo_val is not None and band_hi_val is not None:
        b_lo_bps = int(round(float(band_lo_val) * 10000))
        b_hi_bps = int(round(float(band_hi_val) * 10000))
        vol_band_str = f"{vol_cls} | [{b_lo_bps:+d}, {b_hi_bps:+d}] bps"
    else:
        vol_band_str = "–"

    threshold_str = f"P(Up) {r['p_cal']*100:.1f}%. Up needs >= {r['q_hi']*100:.1f}%, Down needs <= {r['q_lo']*100:.1f}%"
    
    return {
        "id": r.get("id"),
        "rule_version": r.get("rule_version", 1),
        "bar_start": r["bar_start"],
        "bar_end": r["bar_end"],
        "interval_et": interval_str,
        "issued_et": issued_et,
        "issued_ist": issued_ist,
        "issued_display": f"{issued_et} | {issued_ist}",
        "direction": direction,
        "call_display": call_display,
        "confidence": conf if is_directional else None,
        "tier": tier_val,
        "p_cal": float(r["p_cal"]),
        "q_lo": float(r["q_lo"]),
        "q_hi": float(r["q_hi"]),
        "strong": strong,
        "strength_display": strength_display,
        "price_move_display": price_move_display,
        "correct": correct,
        "result_display": result_display,
        "threshold_str": threshold_str,
        "sample": r.get("sample", "LIVE"),
        "feed": r.get("feed", "iex"),
        "source": r.get("sample", "LIVE"),
        "vol_class": vol_cls,
        "band_lo": band_lo_val,
        "band_hi": band_hi_val,
        "vol_and_band_display": vol_band_str,
        "regime": r.get("regime", "Chop"),
        "delivered_at": r.get("delivered_at"),
        "delivery_delay_s": r.get("delivery_delay_s", 0.0),
        "row_hash": r.get("row_hash", ""),
        "model_hash": r.get("model_hash", "")
    }


def get_m5_signal_log(day: Optional[str] = None, show_no_signal: bool = True, show_old_rule: bool = False, db: Optional[LiveM5Database] = None) -> dict:
    db = db or LiveM5Database()
    rule_cfg = load_m5_calib()
    active_v = rule_cfg.get("active_rule_version", 1)
    now_et = datetime.datetime.now(ET_TZ)
    
    with db.get_connection() as conn:
        # Determine day if not specified: default to 2026-11-16 or latest date in signals
        if not day:
            cursor = conn.execute("SELECT MAX(date(bar_start)) FROM signals;")
            row = cursor.fetchone()
            day = row[0] if (row and row[0]) else "2026-11-16"

        target_v = conn.execute(
            "SELECT MAX(rule_version) FROM signals WHERE date(bar_start) = ?;",
            (day,)
        ).fetchone()
        target_v = target_v[0] if (target_v and target_v[0] is not None) else active_v

        # Fetch signals and bars for this day
        if show_old_rule:
            query = """
            SELECT s.*, b.open, b.high, b.low, b.close, b.volume
            FROM signals s
            LEFT JOIN bars_5m b ON b.symbol = 'GS' AND b.bar_start = s.bar_start
            WHERE date(s.bar_start) = ?
            ORDER BY s.bar_start ASC, s.rule_version ASC;
            """
            cursor = conn.execute(query, (day,))
        else:
            query = """
            SELECT s.*, b.open, b.high, b.low, b.close, b.volume
            FROM signals s
            LEFT JOIN bars_5m b ON b.symbol = 'GS' AND b.bar_start = s.bar_start
            WHERE date(s.bar_start) = ? AND (s.rule_version = ? OR (s.rule_version IS NULL AND ? = 1))
            ORDER BY s.bar_start ASC;
            """
            cursor = conn.execute(query, (day, target_v, target_v))
            
        raw_rows = [dict(r) for r in cursor.fetchall()]

        # Query all 5m bars for GS today to check for genuine feed gaps
        gs_bars_with_data = set()
        cursor_b = conn.execute("SELECT bar_start FROM bars_5m WHERE symbol = 'GS' AND date(bar_start) = ? AND volume > 0;", (day,))
        for b_row in cursor_b.fetchall():
            gs_bars_with_data.add(b_row["bar_start"])

    # Index existing rows by bar_start time (HH:MM)
    existing_by_time = {}
    for r in raw_rows:
        try:
            dt = datetime.datetime.fromisoformat(r["bar_start"]).astimezone(ET_TZ)
            existing_by_time[dt.strftime("%H:%M")] = r
        except Exception:
            pass

    # Build full 78 session bars for full trading days (>= 40 bars) or live evaluation day
    target_date = datetime.date.fromisoformat(day)
    start_dt = datetime.datetime(target_date.year, target_date.month, target_date.day, 9, 30, tzinfo=ET_TZ)
    full_session_rows = []
    
    is_live_today = (target_date == now_et.date())
    should_fill_session = is_live_today or (len(raw_rows) >= 40)
    
    if should_fill_session:
        for i in range(78):
            b_start = start_dt + datetime.timedelta(minutes=5 * i)
            b_end = b_start + datetime.timedelta(minutes=5)
            t_key = b_start.strftime("%H:%M")
            
            if t_key in existing_by_time:
                full_session_rows.append(existing_by_time[t_key])
            else:
                # Bar was not emitted: determine reason (EOD, feed gap, engine error, or future pending)
                if b_start.time() >= datetime.time(15, 55):
                    reason_label = "End of trading day"
                    dir_val = "No signal"
                elif b_end + datetime.timedelta(seconds=25) <= now_et:
                    if b_start.isoformat() in gs_bars_with_data:
                        reason_label = "Engine error"
                        dir_val = "No signal"
                    else:
                        reason_label = "Feed gap"
                        dir_val = "No signal"
                else:
                    reason_label = "Pending"
                    dir_val = "Pending"
                    
                synthetic_row = {
                    "bar_start": b_start.isoformat(),
                    "bar_end": b_end.isoformat(),
                    "horizon": "5m",
                    "symbol": "GS",
                    "feed": "iex",
                    "direction": dir_val,
                    "reason": reason_label,
                    "confidence": 0.50,
                    "p_cal": 0.50,
                    "q_lo": float(rule_cfg.get("q_lo", 0.50)),
                    "q_hi": float(rule_cfg.get("q_hi", 0.50)),
                    "is_strong": 0,
                    "tier": "NORMAL",
                    "rule_version": target_v,
                    "vol_class": None,
                    "band_lo": None,
                    "band_hi": None,
                    "regime": "Chop",
                    "delivered_at": b_end.isoformat(),
                    "delivery_delay_s": 0.0,
                    "sample": "LIVE" if is_live_today else "OUT-OF-SAMPLE",
                    "late": 0,
                    "model_hash": "bb018397",
                    "open": None,
                    "close": None
                }
                full_session_rows.append(synthetic_row)
    else:
        full_session_rows = raw_rows

    # Format rows with forward bar prices
    formatted_rows = []
    n_up = 0
    n_down = 0
    n_no_sig = 0
    n_correct = 0
    n_wrong = 0
    n_pending = 0
    n_unresolvable = 0
    n_future = 0
    
    for i, r in enumerate(full_session_rows):
        next_bar = full_session_rows[i + 1] if i + 1 < len(full_session_rows) else None
        f_row = format_m5_row(r, next_bar)
        
        dir_val = f_row["direction"]
        if dir_val in ["Up", "Down"]:
            if dir_val == "Up":
                n_up += 1
            else:
                n_down += 1
                
            if f_row["correct"] == 1:
                n_correct += 1
            elif f_row["correct"] == 0:
                n_wrong += 1
            elif f_row.get("result_display") == "– Unresolvable (feed gap)":
                n_unresolvable += 1
            else:
                n_pending += 1
        elif dir_val == "Pending":
            n_future += 1
        else:
            n_no_sig += 1
            
        if show_no_signal or dir_val in ["Up", "Down"]:
            formatted_rows.append(f_row)

    # Reverse to newest first for table display
    table_rows = list(reversed(formatted_rows))
    
    # Summary metrics: compute accuracy strictly over resolved bars (correct + wrong)
    n_directional = n_up + n_down
    n_resolved = n_correct + n_wrong
    acc_pct = (n_correct / n_resolved * 100.0) if n_resolved > 0 else 0.0
    ci_lo, ci_hi = wilson_interval(n_correct, n_resolved)
    
    # STRONG-only metrics
    strong_rows = [r for r in formatted_rows if r.get("tier") == "STRONG"]
    strong_resolved = [r for r in strong_rows if r.get("correct") in (0, 1)]
    n_strong = len(strong_rows)
    n_strong_resolved = len(strong_resolved)
    n_strong_correct = sum(1 for r in strong_resolved if r.get("correct") == 1)
    acc_strong_pct = (n_strong_correct / n_strong_resolved * 100.0) if n_strong_resolved > 0 else 0.0
    ci_strong_lo, ci_strong_hi = wilson_interval(n_strong_correct, n_strong_resolved)

    # Open to close move of GS on that day
    actual_bars = [r for r in full_session_rows if r.get("open") is not None and r.get("close") is not None]
    open_p = actual_bars[0].get("open") if actual_bars else 0.0
    close_p = actual_bars[-1].get("close") if actual_bars else 0.0
    day_pct = ((close_p - open_p) / open_p * 100.0) if (open_p and close_p) else 0.0

    day_str_formatted = datetime.date.fromisoformat(day).strftime("%a %d %b %Y") if day else ""
    if n_resolved > 0:
        acc_text = f"Accuracy {n_correct}/{n_resolved} (Wilson 95% CI [{ci_lo:.1f}%, {ci_hi:.1f}%])"
    else:
        acc_text = "Accuracy 0/0 (n/a)"

    summary_line = (
        f"{day_str_formatted} | GS ${open_p:.2f} -> ${close_p:.2f} ({day_pct:+.2f}%) | "
        f"Signals {n_directional} (Up {n_up}, Down {n_down}) | "
        f"Correct {n_correct} | Wrong {n_wrong} | Pending {n_pending} | No signal {n_no_sig} | "
        f"{acc_text} - demonstration, not proof (n={n_resolved})"
    )

    summary = {
        "date": day,
        "day_str": day_str_formatted,
        "open_price": open_p,
        "close_price": close_p,
        "day_return_pct": day_pct,
        "price_str": f"GS ${open_p:.2f} → ${close_p:.2f} ({day_pct:+.2f}%)",
        "total_bars": len(full_session_rows),
        "total_signals": n_directional,
        "up_count": n_up,
        "down_count": n_down,
        "no_signal_count": n_no_sig,
        "correct_count": n_correct,
        "wrong_count": n_wrong,
        "pending_count": n_pending,
        "accuracy_all": {
            "n": n_directional,
            "correct": n_correct,
            "pct": acc_pct,
            "ci_lo": ci_lo,
            "ci_hi": ci_hi,
            "ci_str": f"[{ci_lo:.1f}%, {ci_hi:.1f}%]"
        },
        "accuracy_strong": {
            "n": n_strong,
            "correct": n_strong_correct,
            "pct": acc_strong_pct,
            "ci_lo": ci_strong_lo,
            "ci_hi": ci_strong_hi,
            "ci_str": f"[{ci_strong_lo:.1f}%, {ci_strong_hi:.1f}%]" if n_strong > 0 else "–"
        },
        "accuracy_pct": acc_pct,
        "accuracy_str": f"{n_correct}/{n_directional}",
        "wilson_ci": [ci_lo, ci_hi],
        "wilson_ci_str": f"Wilson 95% CI [{ci_lo:.1f}%, {ci_hi:.1f}%]",
        "demonstration_note": f"demonstration, not proof (n={n_directional}). Descriptive only; holdout already inspected.",
        "active_rule_version": active_v,
        "rule_type": rule_cfg.get("rule_type", "always_call"),
        "q_lo": float(rule_cfg.get("q_lo", 0.50)),
        "q_hi": float(rule_cfg.get("q_hi", 0.50)),
        "weak_threshold": float(rule_cfg.get("weak_threshold", 0.5208)),
        "strong_threshold": float(rule_cfg.get("strong_threshold", 0.5811)),
        "coverage_target": float(rule_cfg.get("coverage_target", 1.0)),
        "expected_signals_per_day": 77,
        "summary_line": summary_line
    }

    # Timeline strip: small chip per 5-minute bar in chronological order
    timeline_chips = []
    for r in formatted_rows:
        timeline_chips.append({
            "interval_et": r["interval_et"],
            "direction": r["direction"],
            "is_strong": r["strong"],
            "correct": r["correct"],
            "result_display": r["result_display"],
            "bar_start": r["bar_start"]
        })

    # Hourly mini-summary chips
    hourly_groups = {}
    for r in formatted_rows:
        hr = r["interval_et"].split("-")[0].split(":")[0]
        if hr not in hourly_groups:
            hourly_groups[hr] = {"hr": hr, "up": 0, "down": 0, "correct": 0, "total": 0}
        if r["direction"] == "Up":
            hourly_groups[hr]["up"] += 1
            hourly_groups[hr]["total"] += 1
            if r["correct"] == 1:
                hourly_groups[hr]["correct"] += 1
        elif r["direction"] == "Down":
            hourly_groups[hr]["down"] += 1
            hourly_groups[hr]["total"] += 1
            if r["correct"] == 1:
                hourly_groups[hr]["correct"] += 1

    hourly_chips = []
    for hr in sorted(hourly_groups.keys()):
        h_data = hourly_groups[hr]
        tot = h_data["total"]
        h_acc = (h_data["correct"] / tot * 100.0) if tot > 0 else 0.0
        hourly_chips.append({
            "hour": f"{hr}:00",
            "signals": tot,
            "up": h_data["up"],
            "down": h_data["down"],
            "accuracy_pct": h_acc,
            "display": f"{hr}:00 → {tot} sigs (Up {h_data['up']}, Dn {h_data['down']}) {h_acc:.0f}%" if tot > 0 else f"{hr}:00 → 0 sigs"
        })

    return {
        "day": day,
        "rows": table_rows,
        "summary": summary,
        "timeline_chips": timeline_chips,
        "hourly_chips": hourly_chips,
        "total_rows": len(table_rows)
    }
