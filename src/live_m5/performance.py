"""
Performance Computation Engine for M5 (5-Minute System).
Provides:
1. Dynamic determination of the previous completed market day in America/New_York.
2. High-precision per-session performance metrics and bar-by-bar call strips.
3. Multi-session and overall performance metrics (Holdout, OOF, Live).
"""
import os
import sys
import json
import sqlite3
import datetime
from zoneinfo import ZoneInfo
from typing import Optional, Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.abspath("."))
from src.live.calendar import get_trading_session_info, ET
from src.predict_day_m5 import predict_day_m5

PERF_CACHE_PATH = "results/m5/performance_cache.json"


def wilson_interval(k: int, n: int, confidence: float = 0.95) -> Tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    z = stats.norm.ppf((1 + confidence) / 2.0)
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    spread = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return max(0.0, center - spread), min(1.0, center + spread)


def get_previous_market_day(now_dt: Optional[datetime.datetime] = None) -> datetime.date:
    """
    Compute previous completed regular trading day dynamically in America/New_York.
    - If now_dt >= 16:00 ET on a trading day, today is the completed day.
    - Otherwise, step backwards to find the most recent trading day.
    """
    if now_dt is None:
        now_dt = datetime.datetime.now(ET)
    else:
        now_dt = now_dt.astimezone(ET)

    session = get_trading_session_info(now_dt.date())
    if session is not None and now_dt >= session["close"]:
        candidate = now_dt.date()
    else:
        candidate = now_dt.date() - datetime.timedelta(days=1)

    while True:
        info = get_trading_session_info(candidate)
        if info is not None:
            return candidate
        candidate -= datetime.timedelta(days=1)


def get_day_performance_details(day_str: str, flat_as_miss: bool = False) -> Dict[str, Any]:
    """
    Compute comprehensive performance stats for a single session.
    """
    target_date = datetime.date.fromisoformat(day_str)
    session_info = get_trading_session_info(target_date)
    is_half_day = session_info["is_half_day"] if session_info else False
    theoretical_bars = 48 if is_half_day else 78

    # 1. Load bars from parquet
    bars_path = "data/processed/m5/bars_5m_GS.parquet"
    df_bars_all = pd.read_parquet(bars_path)
    day_bars = df_bars_all[df_bars_all.index.date == target_date].sort_index().copy()
    
    if day_bars.empty:
        return {
            "date": day_str,
            "weekday": target_date.strftime("%A"),
            "empty": True,
            "message": f"No bar data available for {day_str}."
        }

    # Generate signals via replay/prediction logic
    df_signals = predict_day_m5(day_str)

    # Calculate actual price changes and labels
    day_bars["next_close"] = day_bars["close"].shift(-1)
    day_bars["has_next"] = ~day_bars["next_close"].isna()
    
    # Ground truth: y = 1 if close(t+1) > close(t) else 0
    day_bars["is_up"] = day_bars["next_close"] > day_bars["close"]
    day_bars["is_down"] = day_bars["next_close"] < day_bars["close"]
    day_bars["is_flat"] = (day_bars["next_close"] == day_bars["close"]) & day_bars["has_next"]
    
    # Identify missing intervals (data gaps)
    # Expected intervals: 09:30 through 15:55 in 5m steps
    expected_times = []
    curr_time = datetime.datetime.combine(target_date, datetime.time(9, 30), tzinfo=ET)
    close_time = session_info["close"] if session_info else datetime.datetime.combine(target_date, datetime.time(16, 0), tzinfo=ET)
    while curr_time < close_time:
        expected_times.append(curr_time.time())
        curr_time += datetime.timedelta(minutes=5)

    actual_times_set = set(t.time() for t in day_bars.index)
    missing_bar_times = [t.strftime("%H:%M") for t in expected_times if t not in actual_times_set]
    data_gaps_count = len(missing_bar_times)

    # Build call-by-call array
    calls_list = []
    n_up = 0
    n_down = 0
    n_nosig_eod = 0
    n_nosig_tie = 0
    
    for idx, r in day_bars.iterrows():
        b_iso = idx.isoformat()
        t_str = idx.strftime("%H:%M")
        sig_row = df_signals.loc[b_iso] if b_iso in df_signals.index else None
        
        dir_call = sig_row["direction"] if sig_row is not None else "No signal"
        conf = float(sig_row["confidence"]) if sig_row is not None else 0.50
        tier = sig_row["tier"] if sig_row is not None else "NORMAL"
        
        has_nxt = bool(r["has_next"])
        is_flat = bool(r["is_flat"])
        
        if dir_call == "Up":
            n_up += 1
        elif dir_call == "Down":
            n_down += 1
        else:
            if idx.time() >= datetime.time(15, 55):
                n_nosig_eod += 1
            else:
                n_nosig_tie += 1

        # Outcome evaluation
        if not has_nxt:
            actual_dir = "EOD"
            is_correct = None
        else:
            if r["is_up"]:
                actual_dir = "Up"
            elif r["is_down"]:
                actual_dir = "Down"
            else: # Flat
                actual_dir = "Flat"
                
            if actual_dir == "Flat":
                if flat_as_miss:
                    is_correct = False
                else:
                    # Under strict y = 1 if close(t+1) > close(t) else 0: flat is y=0, so Down is correct
                    is_correct = (dir_call == "Down")
            else:
                is_correct = (dir_call == actual_dir)

        calls_list.append({
            "time": t_str,
            "bar_start": b_iso,
            "direction": dir_call,
            "confidence": round(conf * 100, 2),
            "tier": tier,
            "close": float(r["close"]),
            "next_close": float(r["next_close"]) if has_nxt else None,
            "actual": actual_dir,
            "is_flat": is_flat,
            "is_correct": is_correct
        })

    total_recorded_bars = len(day_bars)
    labelled_calls = [c for c in calls_list if c["is_correct"] is not None]
    n_labelled = len(labelled_calls)
    n_correct = sum(1 for c in labelled_calls if c["is_correct"] is True)
    
    acc_all = (n_correct / n_labelled) if n_labelled > 0 else 0.0
    ci_lo, ci_hi = wilson_interval(n_correct, n_labelled)

    # Directional splits
    up_calls = [c for c in labelled_calls if c["direction"] == "Up"]
    n_up_corr = sum(1 for c in up_calls if c["is_correct"] is True)
    up_acc = (n_up_corr / len(up_calls)) if up_calls else 0.0
    up_ci = wilson_interval(n_up_corr, len(up_calls))

    down_calls = [c for c in labelled_calls if c["direction"] == "Down"]
    n_down_corr = sum(1 for c in down_calls if c["is_correct"] is True)
    down_acc = (n_down_corr / len(down_calls)) if down_calls else 0.0
    down_ci = wilson_interval(n_down_corr, len(down_calls))

    # Tiers
    tiers_breakdown = {}
    for t_name in ["WEAK", "NORMAL", "STRONG"]:
        t_calls = [c for c in labelled_calls if c["tier"] == t_name]
        t_corr = sum(1 for c in t_calls if c["is_correct"] is True)
        t_acc = (t_corr / len(t_calls)) if t_calls else 0.0
        t_ci = wilson_interval(t_corr, len(t_calls))
        t_u = sum(1 for c in t_calls if c["direction"] == "Up")
        t_d = sum(1 for c in t_calls if c["direction"] == "Down")
        tiers_breakdown[t_name] = {
            "n": len(t_calls),
            "correct": t_corr,
            "accuracy": round(t_acc * 100, 2),
            "ci": [round(t_ci[0] * 100, 2), round(t_ci[1] * 100, 2)],
            "up_calls": t_u,
            "down_calls": t_d
        }

    # Base rate
    actual_up_count = sum(1 for c in labelled_calls if c["actual"] == "Up")
    base_rate_up = (actual_up_count / n_labelled) if n_labelled > 0 else 0.0
    base_rate_down = 1.0 - base_rate_up
    larger_base = max(base_rate_up, base_rate_down)
    edge = acc_all - larger_base

    # GS day return open to close
    day_open = float(day_bars.iloc[0]["open"])
    day_close = float(day_bars.iloc[-1]["close"])
    gs_ret_pct = (day_close - day_open) / day_open * 100
    gs_ret_bps = (day_close - day_open) / day_open * 10000

    # Hourly accuracy
    hourly = {}
    for c in labelled_calls:
        h = c["time"].split(":")[0] + ":30"
        if h not in hourly:
            hourly[h] = {"hour": h, "total": 0, "correct": 0, "up": 0, "down": 0}
        hourly[h]["total"] += 1
        if c["is_correct"]:
            hourly[h]["correct"] += 1
        if c["direction"] == "Up":
            hourly[h]["up"] += 1
        elif c["direction"] == "Down":
            hourly[h]["down"] += 1

    hourly_list = []
    for h in sorted(hourly.keys()):
        h_data = hourly[h]
        h_acc = (h_data["correct"] / h_data["total"] * 100) if h_data["total"] > 0 else 0.0
        hourly_list.append({
            "window": f"{h} – {h.split(':')[0]}:55",
            "total": h_data["total"],
            "correct": h_data["correct"],
            "accuracy": round(h_acc, 1),
            "up": h_data["up"],
            "down": h_data["down"]
        })

    # Flat bars count
    flat_calls = [c for c in calls_list if c["is_flat"]]
    flat_times = [c["time"] for c in flat_calls]

    return {
        "date": day_str,
        "weekday": target_date.strftime("%A"),
        "total_bars": total_recorded_bars,
        "theoretical_bars": theoretical_bars,
        "data_gaps": data_gaps_count,
        "missing_bar_times": missing_bar_times,
        "n_up": n_up,
        "n_down": n_down,
        "n_nosig_eod": n_nosig_eod,
        "n_nosig_tie": n_nosig_tie,
        "n_nosig_total": n_nosig_eod + n_nosig_tie,
        "sum_check_passed": (n_up + n_down + n_nosig_eod + n_nosig_tie == total_recorded_bars),
        "n_labelled": n_labelled,
        "n_correct": n_correct,
        "accuracy_all": round(acc_all * 100, 2),
        "ci_all": [round(ci_lo * 100, 2), round(ci_hi * 100, 2)],
        "accuracy_up": round(up_acc * 100, 2),
        "ci_up": [round(up_ci[0] * 100, 2), round(up_ci[1] * 100, 2)],
        "n_up_calls": len(up_calls),
        "accuracy_down": round(down_acc * 100, 2),
        "ci_down": [round(down_ci[0] * 100, 2), round(down_ci[1] * 100, 2)],
        "n_down_calls": len(down_calls),
        "tiers": tiers_breakdown,
        "base_rate_up": round(base_rate_up * 100, 2),
        "base_rate_down": round(base_rate_down * 100, 2),
        "majority_base_rate": round(larger_base * 100, 2),
        "edge": round(edge * 100, 2),
        "gs_open": round(day_open, 2),
        "gs_close": round(day_close, 2),
        "gs_return_pct": round(gs_ret_pct, 2),
        "gs_return_bps": round(gs_ret_bps, 1),
        "calls": calls_list,
        "hourly_accuracy": hourly_list,
        "flat_bars_count": len(flat_calls),
        "flat_bar_times": flat_times,
        "flat_as_miss": flat_as_miss,
        "disclaimer": f"One-day result: demonstration, not proof (n = {n_labelled})"
    }


def get_overall_performance() -> Dict[str, Any]:
    """
    Get full multi-column overall performance metrics and last 20 sessions chart data.
    """
    if os.path.exists(PERF_CACHE_PATH):
        with open(PERF_CACHE_PATH, "r") as f:
            return json.load(f)
    else:
        from src.live_m5.build_performance_cache import build_cache
        build_cache()
        with open(PERF_CACHE_PATH, "r") as f:
            return json.load(f)
