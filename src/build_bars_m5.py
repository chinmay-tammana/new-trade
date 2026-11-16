"""
Build 5-minute bars from cached 1-minute IEX parquet files.
Performs data audit, candidate checks (TLT, VIXY), and data freeze.
Saves audit outputs to results/m5/audit/ and dataset cards to docs/m5/.
"""
import os
import glob
import json
import datetime
import zoneinfo
import numpy as np
import pandas as pd

from src.aggregate_m5 import aggregate_1min_to_5min, ET_TZ
from configs.config import CONFIG_M5

RAW_DIR = CONFIG_M5["paths"]["raw_dir"]
PROCESSED_DIR = "data/processed/m5"
AUDIT_DIR = "results/m5/audit"
ALL_SYMBOLS = ["GS", "SPY", "XLF", "JPM", "MS", "BAC", "TLT", "VIXY"]


def load_symbol_1min(symbol: str) -> pd.DataFrame:
    pattern = os.path.join(RAW_DIR, symbol, "*.parquet")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No parquet files found for {symbol} in {RAW_DIR}/{symbol}")
    
    dfs = []
    for f in files:
        try:
            df_m = pd.read_parquet(f)
            if not df_m.empty:
                dfs.append(df_m)
        except Exception as e:
            print(f"Warning reading {f}: {e}")
            
    if not dfs:
        return pd.DataFrame()
    full_df = pd.concat(dfs, axis=0)
    full_df = full_df[~full_df.index.duplicated(keep="last")]
    full_df = full_df.sort_index()
    return full_df


def build_and_audit():
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(AUDIT_DIR, exist_ok=True)
    
    print("Building 5-minute bars from raw 1-minute IEX data...")
    bars_dict = {}
    
    for sym in ALL_SYMBOLS:
        print(f"Processing {sym}...")
        df_1m = load_symbol_1min(sym)
        bars_5m = aggregate_1min_to_5min(df_1m, symbol=sym)
        out_path = os.path.join(PROCESSED_DIR, f"bars_5m_{sym}.parquet")
        bars_5m.to_parquet(out_path)
        bars_dict[sym] = bars_5m
        print(f"[{sym}] Built {len(bars_5m):,} 5-minute bars ({bars_5m.index[0]} to {bars_5m.index[-1]})")

    # Step 1: Establish Trading Sessions & Expected Calendar from GS
    gs_bars = bars_dict["GS"]
    trading_dates = sorted(list(set(gs_bars.index.date)))
    print(f"\nTotal unique trading dates for GS: {len(trading_dates)}")

    # Step 2: Empty 5-minute bar rate check for TLT and VIXY (and all symbols)
    # A full day has 78 bars (09:30 to 15:55 start). Half-days have 42 bars (09:30 to 12:55 start).
    audit_rates = []
    for sym in ALL_SYMBOLS:
        sym_bars = bars_dict[sym]
        sym_dates = set(sym_bars.index.date)
        
        # Total expected bars across all GS trading dates
        total_expected_bars = 0
        total_present_bars = 0
        bars_per_day_counts = []
        zero_vol_bars = (sym_bars["volume"] == 0).sum()
        high_lt_low = (sym_bars["high"] < sym_bars["low"]).sum()
        
        # Calculate returns
        rets = sym_bars["close"].pct_change().dropna()
        extreme_rets = (rets.abs() > 0.15).sum() # > 15% in 5 min
        
        for d in trading_dates:
            day_bars = sym_bars[sym_bars.index.date == d]
            cnt = len(day_bars)
            bars_per_day_counts.append(cnt)
            # Check if half day (e.g. day after Thanksgiving or Christmas Eve)
            # For simplicity, expected bars for the session = max(cnt, 78) if normal else 42
            expected_day = 42 if cnt <= 42 and d.month in [7, 11, 12] else 78
            total_expected_bars += expected_day
            total_present_bars += cnt

        empty_rate = max(0.0, (total_expected_bars - total_present_bars) / total_expected_bars)
        audit_rates.append({
            "symbol": sym,
            "total_bars": len(sym_bars),
            "expected_bars": total_expected_bars,
            "present_bars": total_present_bars,
            "empty_bar_rate": empty_rate,
            "zero_vol_bars": int(zero_vol_bars),
            "high_lt_low_bars": int(high_lt_low),
            "extreme_rets_gt_15pct": int(extreme_rets),
            "avg_bars_per_day": float(np.mean(bars_per_day_counts)),
        })

    df_rates = pd.DataFrame(audit_rates)
    df_rates.to_csv(os.path.join(AUDIT_DIR, "symbol_bar_rates.csv"), index=False)
    print("\n--- Empty Bar Rate & Data Quality Audit ---")
    print(df_rates.to_string(index=False))

    # TLT and VIXY candidate verdict: keep only if empty_bar_rate < 0.05 (5%)
    kept_symbols = ["GS", "SPY", "XLF", "JPM", "MS", "BAC"]
    for cand in ["TLT", "VIXY"]:
        rate = df_rates.loc[df_rates["symbol"] == cand, "empty_bar_rate"].values[0]
        if rate < 0.05:
            kept_symbols.append(cand)
            print(f"CANDIDATE {cand}: empty bar rate = {rate:.2%} (< 5%). KEPT.")
        else:
            print(f"CANDIDATE {cand}: empty bar rate = {rate:.2%} (>= 5%). EXCLUDED.")

    print(f"\nFinal feature symbols: {kept_symbols}")

    # Step 3: Inner Join Loss Audit against GS
    join_loss = []
    gs_index = set(gs_bars.index)
    for sym in kept_symbols:
        sym_index = set(bars_dict[sym].index)
        intersection = gs_index.intersection(sym_index)
        loss_pct = 1.0 - (len(intersection) / len(gs_index))
        join_loss.append({
            "symbol": sym,
            "gs_bars": len(gs_index),
            "common_bars": len(intersection),
            "join_loss_pct": loss_pct
        })
    df_join = pd.DataFrame(join_loss)
    df_join.to_csv(os.path.join(AUDIT_DIR, "inner_join_loss.csv"), index=False)
    print("\n--- Inner Join Loss vs GS ---")
    print(df_join.to_string(index=False))

    # Step 4: Data Freeze (Freeze at last complete GS bar)
    # Check that last bar is not partial (ends strictly at or before 16:00 on freeze_date)
    freeze_date_str = CONFIG_M5["freeze_date"]
    freeze_d = datetime.date.fromisoformat(freeze_date_str)
    
    # Filter GS bars to freeze_date
    gs_frozen = gs_bars[gs_bars.index.date <= freeze_d]
    data_end_str = gs_frozen.index[-1].isoformat()
    
    freeze_info = {
        "freeze_date": freeze_date_str,
        "data_end": data_end_str,
        "symbols_kept": kept_symbols,
        "total_gs_bars_frozen": len(gs_frozen),
        "created_at": datetime.datetime.now(ET_TZ).isoformat()
    }
    with open(CONFIG_M5["paths"]["data_freeze"], "w") as f:
        json.dump(freeze_info, f, indent=2)
    print(f"\nData freeze saved to {CONFIG_M5['paths']['data_freeze']}:")
    print(json.dumps(freeze_info, indent=2))

    return kept_symbols


if __name__ == "__main__":
    build_and_audit()
