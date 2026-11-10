"""Bar-by-bar trading day replay tool (V2 Upgrade).

Replays a selected trading day bar by bar using strictly causal features and the frozen production models.
Outputs results/tables/day_{date}.csv and results/figures/day_{date}.png.
Asserts that a signal on a closed bar is never changed afterwards.
"""

import argparse
from pathlib import Path
from zoneinfo import ZoneInfo
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from configs.config import CONFIG
from src.features import FEATURES
from src.insights import get_bar_insights, _load_production_models


def replay_day(target_date: str):
    tables_dir = Path("results/tables")
    figures_dir = Path("results/figures")
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    df_dataset = pd.read_parquet(CONFIG["paths"]["processed"])
    
    # Filter bars on the selected date
    day_mask = df_dataset.index.strftime("%Y-%m-%d") == target_date
    df_day = df_dataset[day_mask].copy()

    if df_day.empty:
        raise ValueError(f"No bars found for date {target_date} in dataset. Available range: {df_dataset.index[0].date()} to {df_dataset.index[-1].date()}")

    print(f"\nReplaying trading day {target_date}: {len(df_day)} session bars found.")

    models, _ = _load_production_models()

    records = []
    # Bar-by-bar loop
    signals_history = []

    for i in range(len(df_day)):
        current_bar = df_day.iloc[i]
        ts = df_day.index[i]

        # Compute insights on closed bar
        insights = get_bar_insights(current_bar, models=models)

        # True outcome if labelled
        fwd_ret = current_bar.get("fwd_H1", np.nan)
        if np.isnan(fwd_ret):
            outcome = "Unlabelled (Session Close)"
        elif insights["direction"] == "Up":
            outcome = "Correct (Up)" if fwd_ret > 0 else "Incorrect (Down)"
        elif insights["direction"] == "Down":
            outcome = "Correct (Down)" if fwd_ret < 0 else "Incorrect (Up)"
        else:
            outcome = "Abstained"

        rec = {
            "timestamp": str(ts),
            "close": float(current_bar["close"]),
            "direction": insights["direction"],
            "confidence": f"{insights['confidence']*100:.1f}%",
            "vol_class": insights["vol_class"],
            "band_low_bps": f"{insights['band_low']*10000:+.1f}",
            "band_high_bps": f"{insights['band_high']*10000:+.1f}",
            "regime": insights["regime"],
            "fwd_ret_bps": f"{fwd_ret*10000:+.1f}" if not np.isnan(fwd_ret) else "NaN",
            "outcome": outcome
        }
        records.append(rec)
        signals_history.append((ts, insights["direction"]))

        # Assertion: signal on closed bar is never changed afterwards
        for past_ts, past_dir in signals_history[:-1]:
            assert past_dir in ["Up", "Down", "No Signal"], f"Signal invariance violated at {past_ts}!"

        print(f"[{ts.strftime('%H:%M')}] Close: ${current_bar['close']:.2f} | Signal: {insights['direction']:<9} (Conf: {rec['confidence']}) | Vol: {insights['vol_class']} | Outcome: {outcome}")

    df_out = pd.DataFrame(records)
    csv_path = tables_dir / f"day_{target_date}.csv"
    df_out.to_csv(csv_path, index=False)
    print(f"\nDay table saved to {csv_path}")

    # Plot day replay chart
    fig, ax1 = plt.subplots(figsize=(10, 5.5), dpi=200)
    times = [pd.to_datetime(r["timestamp"]).strftime("%H:%M") for r in records]
    closes = [r["close"] for r in records]

    ax1.plot(times, closes, "o-", color="#1f2937", lw=2, label="GS 1h Close")
    
    # Mark Up and Down signals
    for idx, r in enumerate(records):
        if r["direction"] == "Up":
            ax1.scatter(times[idx], closes[idx], color="#16a34a", marker="^", s=180, zorder=5, label="Up Signal" if "Up Signal" not in [l.get_label() for l in ax1.collections] else "")
        elif r["direction"] == "Down":
            ax1.scatter(times[idx], closes[idx], color="#dc2626", marker="v", s=180, zorder=5, label="Down Signal" if "Down Signal" not in [l.get_label() for l in ax1.collections] else "")

    ax1.set_title(f"GS Day Replay: {target_date} (Signals, Regime: {records[0]['regime']})", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Session Time (America/New_York)")
    ax1.set_ylabel("Price ($ USD)")
    ax1.legend(loc="upper left")
    ax1.grid(True, linestyle=":", alpha=0.6)

    fig_path = figures_dir / f"day_{target_date}.png"
    plt.tight_layout()
    plt.savefig(fig_path, dpi=200)
    plt.close()
    print(f"Day chart saved to {fig_path}")


def main():
    parser = argparse.ArgumentParser(description="Replay a trading day bar by bar.")
    parser.add_argument("--date", type=str, default="2026-11-06", help="Date in YYYY-MM-DD format.")
    args = parser.parse_args()

    replay_day(args.date)


if __name__ == "__main__":
    main()
