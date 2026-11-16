"""Scoring Audit and outcomes_v2 Population (V3.2 Fix).

Verifies forward returns, labels, and directional correctness on all OUT-OF-SAMPLE and LIVE signals.
Creates outcomes_v2 table in SQLite with nullable correct column.
Writes results/tables/live_scoring_audit.csv.
"""

import datetime
import math
import sqlite3
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats

def wilson_ci(k: int, n: int, z: float = 1.95996):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (round(max(0.0, center - half) * 100, 2), round(min(1.0, center + half) * 100, 2))


def run_audit():
    print("=" * 80)
    print("RUNNING V3.2 LIVE SCOREBOARD AUDIT")
    print("=" * 80)

    db_path = Path("data/live/live.db")
    con = sqlite3.connect(str(db_path))
    con.row_factory = sqlite3.Row

    # 1. Fetch bars
    bars = pd.read_sql_query('SELECT * FROM bars WHERE symbol="GS" ORDER BY timestamp ASC', con)
    bars['dt_utc'] = pd.to_datetime(bars['timestamp'], utc=True)
    bars['dt_et'] = bars['dt_utc'].dt.tz_convert('America/New_York')
    bars = bars.sort_values('dt_utc').reset_index(drop=True)

    bar_map = {}
    for i, r in bars.iterrows():
        bar_map[r['timestamp']] = (i, r)

    # 2. Fetch signals
    sigs = pd.read_sql_query('SELECT * FROM signals ORDER BY bar_start ASC', con)
    print(f"Total signals in ledger: {len(sigs)}")

    # 3. Create outcomes_v2 table (keeps original outcomes intact)
    con.execute("""
    CREATE TABLE IF NOT EXISTS outcomes_v2 (
        bar_start TEXT PRIMARY KEY,
        realized_fwd_ret REAL,
        y_true INTEGER,
        correct INTEGER,  -- NULL for No signal / abstentions / session closes
        recorded_at TEXT NOT NULL
    );
    """)

    audit_rows = []
    outcomes_v2_entries = []
    out_dates = set()

    for idx, sig in sigs.iterrows():
        b_start = sig['bar_start']
        b_end = sig['bar_end']
        direction = sig['direction']
        conf = float(sig['confidence'])
        sample = sig['sample']

        if b_start not in bar_map:
            continue

        i, bar_row = bar_map[b_start]
        c_sig = float(bar_row['close'])
        b_date = bar_row['dt_et'].date()

        if sample in ['OUT-OF-SAMPLE', 'LIVE']:
            out_dates.add(b_date)

        c_target = None
        target_end = None
        fwd_ret = None
        y_true = None
        correct = None

        # Next bar on same trading session
        if i + 1 < len(bars):
            next_bar = bars.iloc[i + 1]
            if next_bar['dt_et'].date() == b_date:
                c_target = float(next_bar['close'])
                target_end = (next_bar['dt_et'] + pd.Timedelta(hours=1 if not (next_bar['dt_et'].hour == 15 and next_bar['dt_et'].minute == 30) else 0.5)).isoformat()
                fwd_ret = (c_target - c_sig) / c_sig
                y_true = 1 if fwd_ret > 0 else 0

                # Strictly score Up and Down. No signal rows are NEVER counted.
                if direction == 'Up':
                    correct = 1 if y_true == 1 else 0
                elif direction == 'Down':
                    correct = 1 if y_true == 0 else 0
                else:
                    correct = None
            else:
                # Target is next day -> unlabelled session close
                correct = None
        else:
            correct = None

        outcomes_v2_entries.append((
            b_start,
            fwd_ret,
            y_true,
            correct,
            datetime.datetime.now(datetime.timezone.utc).isoformat()
        ))

        # One row per scored signal in audit table (OUT-OF-SAMPLE and LIVE, directional only)
        if sample in ['OUT-OF-SAMPLE', 'LIVE'] and direction in ['Up', 'Down'] and correct is not None:
            audit_rows.append({
                'bar_start': b_start,
                'bar_end': b_end,
                'horizon': sig['horizon'],
                'direction': direction,
                'confidence': round(conf, 4),
                'target_bar_end': target_end,
                'close_signal_bar': round(c_sig, 2),
                'close_target_bar': round(c_target, 2),
                'forward_return': round(fwd_ret, 6),
                'label_y': y_true,
                'outcome': 'Correct' if correct == 1 else 'Wrong',
                'correct': correct,
                'source_tag': sample
            })

    # Insert into outcomes_v2
    con.executemany("""
    INSERT OR REPLACE INTO outcomes_v2 (bar_start, realized_fwd_ret, y_true, correct, recorded_at)
    VALUES (?, ?, ?, ?, ?);
    """, outcomes_v2_entries)
    con.commit()
    print(f"Populated {len(outcomes_v2_entries)} rows in outcomes_v2 table.")

    # 4. Save audit CSV
    df_audit = pd.DataFrame(audit_rows)
    audit_csv_path = Path("results/tables/live_scoring_audit.csv")
    audit_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df_audit.to_csv(audit_csv_path, index=False)
    print(f"Saved scoring audit CSV to {audit_csv_path} ({len(df_audit)} scored rows).")

    # 5. Print Statistics
    n_scored = len(df_audit)
    n_hits = int(df_audit['correct'].sum()) if n_scored > 0 else 0
    acc = n_hits / n_scored if n_scored > 0 else 0.0
    ci_lo, ci_hi = wilson_ci(n_hits, n_scored)
    binom_p = stats.binomtest(n_hits, n_scored, 0.5, alternative='two-sided').pvalue if n_scored > 0 else 1.0

    print("\n--- AUDIT SUMMARY METRICS ---")
    print(f"Total Scored Directional Signals: {n_scored}")
    print(f"Total Hits: {n_hits}")
    print(f"Signal Accuracy: {acc*100:.2f}%")
    print(f"Wilson 95% CI: [{ci_lo:.2f}%, {ci_hi:.2f}%]")
    print(f"Exact Binomial Two-Sided p-value vs 50%: {binom_p:.4f}")

    for side in ['Up', 'Down']:
        sub = df_audit[df_audit['direction'] == side]
        s_n = len(sub)
        s_hits = int(sub['correct'].sum()) if s_n > 0 else 0
        s_acc = (s_hits / s_n * 100) if s_n > 0 else 0.0
        s_lo, s_hi = wilson_ci(s_hits, s_n)
        s_p = stats.binomtest(s_hits, s_n, 0.5, alternative='two-sided').pvalue if s_n > 0 else 1.0
        print(f"Side {side}: N={s_n}, Hits={s_hits}, Acc={s_acc:.2f}%, Wilson CI=[{s_lo:.2f}%, {s_hi:.2f}%], Binomial p={s_p:.4f}")

    # Base rates across all candidate candles in the 4 sessions
    oos_bars = bars[bars['dt_et'].map(lambda d: d.date() in out_dates)].copy()
    oos_fwd = []
    for i in range(len(oos_bars) - 1):
        b0 = oos_bars.iloc[i]
        b1 = oos_bars.iloc[i+1]
        if b0['dt_et'].date() == b1['dt_et'].date():
            ret = (b1['close'] - b0['close']) / b0['close']
            oos_fwd.append(ret)

    oos_fwd = np.array(oos_fwd)
    n_all_candles = len(oos_fwd)
    base_up = (oos_fwd > 0).mean() * 100
    base_down = (oos_fwd < 0).mean() * 100
    print(f"\nAll Candidate Out-of-Sample Candles: N={n_all_candles} across 4 trading days")
    print(f"Base Up-rate on all candles:   {base_up:.2f}% ({sum(oos_fwd > 0)}/{n_all_candles})")
    print(f"Base Down-rate on all candles: {base_down:.2f}% ({sum(oos_fwd < 0)}/{n_all_candles})")
    print(f"Abstentions (No Signal):       {n_all_candles - n_scored}/{n_all_candles} ({(n_all_candles - n_scored)/n_all_candles*100:.1f}%)")
    print("\nNOTE: This is a tiny descriptive sample (N=8 directional signals, N=24 candidate candles).")
    print("It demonstrates execution mechanics, not statistical proof.")

    con.close()


if __name__ == "__main__":
    run_audit()
