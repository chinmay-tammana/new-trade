"""Live Scoring CLI for V3 Live Operations.

Reads all signals from SQLite database since a specified date, computes
forward 1h returns once the forward bar has closed, evaluates directional accuracy
(with Wilson 95% CIs), baselines, and simulated net return, and writes
results/tables/live_scoreboard.csv and results/tables/live_{date}.csv.
"""

import argparse
import datetime
import math
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd

from configs.config import CONFIG
from src.live.calendar import NyseCalendar
from src.live.store import LiveDatabase


def wilson_score_interval(successes: int, trials: int, confidence: float = 0.95) -> Tuple[float, float]:
    """Calculate Wilson score interval for a binomial proportion."""
    if trials == 0:
        return 0.0, 0.0
    z = 1.95996  # 95% normal quantile
    p = successes / trials
    denom = 1.0 + (z**2) / trials
    center = (p + (z**2) / (2.0 * trials)) / denom
    half_width = (z * math.sqrt((p * (1.0 - p) + (z**2) / (4.0 * trials)) / trials)) / denom
    return max(0.0, center - half_width), min(1.0, center + half_width)


def get_forward_1h_return(
    db: LiveDatabase,
    bar_start_iso: str,
    calendar: NyseCalendar
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Look up the next 1h bar close from SQLite bars table or parquet cache.
    Returns (close_t, close_t1, fwd_ret) or (None, None, None) if next bar not yet closed.
    """
    ts_start = pd.to_datetime(bar_start_iso)
    if ts_start.tzinfo is None:
        ts_start = ts_start.tz_localize("America/New_York")
    else:
        ts_start = ts_start.tz_convert("America/New_York")

    # The current bar's close and next bar's close
    with db.get_connection() as conn:
        cursor = conn.execute("""
        SELECT timestamp, close FROM bars
        WHERE symbol = 'GS'
        ORDER BY timestamp ASC;
        """)
        rows = cursor.fetchall()

    if not rows:
        # Fallback to processed dataset if running in simulation / backfill
        if Path(CONFIG["paths"]["processed"]).exists():
            df = pd.read_parquet(CONFIG["paths"]["processed"])
            if ts_start in df.index:
                idx_pos = df.index.get_loc(ts_start)
                if idx_pos + 1 < len(df):
                    c0 = float(df.iloc[idx_pos]["close"])
                    c1 = float(df.iloc[idx_pos + 1]["close"])
                    # Check if same day
                    if df.index[idx_pos].date() == df.index[idx_pos + 1].date():
                        return c0, c1, (c1 - c0) / c0
                    else:
                        return c0, c1, None  # Session close bar, unlabelled
        return None, None, None

    # Parse rows
    ts_series = []
    closes = []
    for r in rows:
        t = pd.to_datetime(r["timestamp"])
        if t.tzinfo is None:
            t = t.tz_localize("America/New_York")
        else:
            t = t.tz_convert("America/New_York")
        ts_series.append(t)
        closes.append(float(r["close"]))

    df_bars = pd.DataFrame({"close": closes}, index=pd.DatetimeIndex(ts_series)).sort_index()

    if ts_start not in df_bars.index:
        return None, None, None

    pos = df_bars.index.get_loc(ts_start)
    if pos + 1 >= len(df_bars):
        return float(df_bars.iloc[pos]["close"]), None, None

    next_ts = df_bars.index[pos + 1]
    c0 = float(df_bars.iloc[pos]["close"])
    c1 = float(df_bars.iloc[pos + 1]["close"])

    # If next bar is on next day, this was the session close bar (unlabelled)
    if next_ts.date() != ts_start.date():
        return c0, c1, None

    return c0, c1, (c1 - c0) / c0


def score_live_signals(
    since_date: str,
    db_path: Path = Path("data/live/live.db"),
    out_dir: Path = Path("results/tables")
) -> Dict[str, Any]:
    """Score all signals since since_date."""
    out_dir.mkdir(parents=True, exist_ok=True)
    db = LiveDatabase(db_path)
    calendar = NyseCalendar()

    with db.get_connection() as conn:
        cursor = conn.execute("""
        SELECT * FROM signals
        WHERE bar_start >= ?
        ORDER BY bar_start ASC;
        """, (since_date,))
        signals = [dict(r) for r in cursor.fetchall()]

    if not signals:
        print(f"No signals found in {db_path} since {since_date}.")
        return {}

    # Check number of trading days elapsed
    unique_dates = sorted(list(set(pd.to_datetime(s["bar_start"]).strftime("%Y-%m-%d") for s in signals)))
    n_days = len(unique_dates)

    if n_days < 5:
        print("\n" + "=" * 80)
        print(f"INTERIM RESULTS — {n_days} of 5 days completed; statistical power is low (see docs/live_expectations.md)")
        print("=" * 80 + "\n")

    # Evaluate each signal
    detailed_rows = []
    up_signals = 0
    down_signals = 0
    no_signals = 0
    up_hits = 0
    down_hits = 0
    evaluated_signals = 0
    late_signals = 0

    net_returns_2bps = []
    net_returns_5bps = []
    net_returns_10bps = []

    # Persistence baseline (sign of previous return)
    # RSI baseline
    for sig in signals:
        b_start = sig["bar_start"]
        direction = sig["direction"]
        late = sig.get("late", 0)

        c0, c1, fwd_ret = get_forward_1h_return(db, b_start, calendar)

        row = {
            "timestamp": b_start,
            "direction": direction,
            "confidence": sig["confidence"],
            "strong": sig["strong"],
            "vol_class": sig["vol_class"],
            "regime": sig["regime"],
            "latency_s": sig.get("latency_s", 0.0),
            "late": late,
            "close": c0,
            "next_close": c1,
            "fwd_ret_bps": round(fwd_ret * 10000, 1) if fwd_ret is not None else None,
            "outcome": "Pending"
        }

        if fwd_ret is None:
            if c0 is not None and c1 is not None:
                row["outcome"] = "Unlabelled (Session Close)"
            else:
                row["outcome"] = "Pending (Bar Open)"
            detailed_rows.append(row)
            continue

        if late == 1:
            late_signals += 1
            row["outcome"] = "Late Bar (Excluded from Official Score)"
            detailed_rows.append(row)
            continue

        # Valid closed labelled bar
        if direction == "Up":
            up_signals += 1
            hit = 1 if fwd_ret > 0 else 0
            up_hits += hit
            evaluated_signals += 1
            row["outcome"] = "Correct (Up)" if hit else "Incorrect (Down)"

            # Simulated returns (long)
            gross = fwd_ret
            net_returns_2bps.append(gross - 0.0002)
            net_returns_5bps.append(gross - 0.0005)
            net_returns_10bps.append(gross - 0.0010)

        elif direction == "Down":
            down_signals += 1
            hit = 1 if fwd_ret < 0 else 0
            down_hits += hit
            evaluated_signals += 1
            row["outcome"] = "Correct (Down)" if hit else "Incorrect (Up)"

            # Simulated returns (short)
            gross = -fwd_ret
            net_returns_2bps.append(gross - 0.0002)
            net_returns_5bps.append(gross - 0.0005)
            net_returns_10bps.append(gross - 0.0010)

        else:
            no_signals += 1
            row["outcome"] = "Abstained"

        detailed_rows.append(row)

    df_details = pd.DataFrame(detailed_rows)

    # Save per-day CSVs
    for date_str in unique_dates:
        day_mask = df_details["timestamp"].str.startswith(date_str)
        df_day = df_details[day_mask]
        day_file = out_dir / f"live_{date_str}.csv"
        df_day.to_csv(day_file, index=False)
        print(f"Saved daily live log: {day_file} ({len(df_day)} bars)")

    # Scoreboard metrics
    total_hits = up_hits + down_hits
    hit_rate = (total_hits / evaluated_signals) if evaluated_signals > 0 else 0.0
    ci_lo, ci_hi = wilson_score_interval(total_hits, evaluated_signals)

    up_acc = (up_hits / up_signals) if up_signals > 0 else 0.0
    up_ci_lo, up_ci_hi = wilson_score_interval(up_hits, up_signals)

    down_acc = (down_hits / down_signals) if down_signals > 0 else 0.0
    down_ci_lo, down_ci_hi = wilson_score_interval(down_hits, down_signals)

    # Base rates on labelled bars
    fwd_rets_valid = [r["fwd_ret_bps"] / 10000.0 for r in detailed_rows if r["fwd_ret_bps"] is not None]
    base_up = (sum(1 for r in fwd_rets_valid if r > 0) / len(fwd_rets_valid)) if fwd_rets_valid else 0.50

    scoreboard_row = {
        "as_of_date": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "since_date": since_date,
        "trading_days_completed": n_days,
        "total_signals": len(signals),
        "evaluated_active_signals": evaluated_signals,
        "up_signals": up_signals,
        "down_signals": down_signals,
        "no_signals": no_signals,
        "late_signals": late_signals,
        "total_hits": total_hits,
        "total_misses": evaluated_signals - total_hits,
        "hit_rate_pct": round(hit_rate * 100, 2),
        "hit_rate_95_ci": f"[{ci_lo*100:.2f}%, {ci_hi*100:.2f}%]",
        "up_accuracy_pct": round(up_acc * 100, 2),
        "up_95_ci": f"[{up_ci_lo*100:.2f}%, {up_ci_hi*100:.2f}%]",
        "down_accuracy_pct": round(down_acc * 100, 2),
        "down_95_ci": f"[{down_ci_lo*100:.2f}%, {down_ci_hi*100:.2f}%]",
        "base_up_rate_pct": round(base_up * 100, 2),
        "mean_net_return_2bps": round(np.mean(net_returns_2bps) * 10000, 2) if net_returns_2bps else 0.0,
        "mean_net_return_5bps": round(np.mean(net_returns_5bps) * 10000, 2) if net_returns_5bps else 0.0,
        "mean_net_return_10bps": round(np.mean(net_returns_10bps) * 10000, 2) if net_returns_10bps else 0.0
    }

    df_scoreboard = pd.DataFrame([scoreboard_row])
    scoreboard_file = out_dir / "live_scoreboard.csv"
    df_scoreboard.to_csv(scoreboard_file, index=False)
    print(f"\nSaved live scoreboard to: {scoreboard_file}")

    print("\n--- LIVE SCOREBOARD SUMMARY ---")
    for k, v in scoreboard_row.items():
        print(f"  {k}: {v}")
    print("--------------------------------\n")

    return scoreboard_row


def main():
    parser = argparse.ArgumentParser(description="Live Scoring CLI")
    parser.add_argument("--since", type=str, required=True, help="Start date for scoring (YYYY-MM-DD)")
    parser.add_argument("--db", type=str, default="data/live/live.db", help="Path to SQLite live db")
    parser.add_argument("--out", type=str, default="results/tables", help="Output directory")
    args = parser.parse_args()

    score_live_signals(args.since, db_path=Path(args.db), out_dir=Path(args.out))


if __name__ == "__main__":
    main()
