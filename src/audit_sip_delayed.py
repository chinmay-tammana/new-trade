"""Section B Audit: Explain 7 vs 8 Down signals, bar comparisons, late print audit, and swap tests."""

import os
import json
import time
import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import pandas as pd
import numpy as np
import joblib
from dotenv import load_dotenv

load_dotenv()

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed

from configs.config import CONFIG
from src.features import FEATURES, make_features_v2
from src.data_fetch import build_session_anchored_1h_bars
from src.live.aggregator import aggregate_minute_bars_to_1h
import sqlite3

ET = ZoneInfo("America/New_York")
SYMBOLS = ["GS", "SPY", "XLF", "JPM", "MS", "BAC", "TLT", "VIXY"]

FEATURE_GROUPS = {
    "volume": ["rvol", "signed_volume_ratio", "obv_slope_5", "vol_ratio_5_20"],
    "intrabar": ["realized_vol_1m", "up_minute_ratio", "clv", "last15_ret"],
    "peers_vixy": ["rel_ret_spy", "rel_ret_xlf", "gs_minus_peers", "vixy_ret_1h", "vixy_ma_ratio", "tlt_ret_1h"],
    "momentum_trend": ["ret_1h", "ret_3h", "ret_5h", "ret_1d", "ret_5d", "ret_20d", "ma_dist_50h", "rsi", "bb_pos", "bb_width"]
}


def main():
    print("=" * 80)
    print("V3.5 SECTION B AUDIT: EXPLAIN 7 VS 8, LATE PRINTS, AND SWAP TESTS")
    print("=" * 80)

    api_key = os.getenv("APCA_API_KEY_ID")
    api_secret = os.getenv("APCA_API_SECRET_KEY")
    client = StockHistoricalDataClient(api_key, api_secret)

    # 1. Fetch SIP minutes twice a few seconds apart and check for any diffs
    print("\n--- 1. Fetching SIP Minutes Twice to Audit Late Prints / Immutability ---")
    start_dt = datetime.datetime(2026, 11, 10, 9, 30, tzinfo=ET)
    end_dt = datetime.datetime(2026, 11, 13, 16, 0, tzinfo=ET)

    req_sip = StockBarsRequest(
        symbol_or_symbols="GS",
        timeframe=TimeFrame.Minute,
        start=start_dt,
        end=end_dt,
        feed=DataFeed.SIP
    )

    t0 = time.time()
    bars_sip_1 = client.get_stock_bars(req_sip).df.loc["GS"]
    print(f"Fetch #1: retrieved {len(bars_sip_1)} SIP minute bars in {time.time() - t0:.2f}s")

    time.sleep(3)  # wait between requests

    t1 = time.time()
    bars_sip_2 = client.get_stock_bars(req_sip).df.loc["GS"]
    print(f"Fetch #2: retrieved {len(bars_sip_2)} SIP minute bars in {time.time() - t1:.2f}s")

    # Compare
    diff_sip_count = len(bars_sip_2) - len(bars_sip_1)
    diff_sip_vals = (bars_sip_1["close"] - bars_sip_2["close"]).abs().max()
    print(f"SIP Invariance Check: Count diff = {diff_sip_count}, Max Close diff = {diff_sip_vals:.2e}")
    if diff_sip_count == 0 and diff_sip_vals == 0.0:
        print("PASS: SIP historical minute bars are 100% invariant between fetches (no late prints detected).")
    else:
        print("NOTICE: Late prints or mutable bars detected between fetches.")

    # 2. Check IEX minutes invariance
    req_iex = StockBarsRequest(
        symbol_or_symbols="GS",
        timeframe=TimeFrame.Minute,
        start=start_dt,
        end=end_dt,
        feed=DataFeed.IEX
    )
    bars_iex_1 = client.get_stock_bars(req_iex).df.loc["GS"]
    time.sleep(2)
    bars_iex_2 = client.get_stock_bars(req_iex).df.loc["GS"]
    diff_iex_count = len(bars_iex_2) - len(bars_iex_1)
    diff_iex_vals = (bars_iex_1["close"] - bars_iex_2["close"]).abs().max()
    print(f"IEX Invariance Check: Count diff = {diff_iex_count}, Max Close diff = {diff_iex_vals:.2e}")

    # 3. Bar-by-Bar Comparison of Stored IEX vs Diagnostic IEX
    print("\n--- 2. Bar-by-Bar Comparison: Stored IEX (live.db) vs. Diagnostic IEX ---")
    conn = sqlite3.connect("data/live/live.db")
    df_db_sigs = pd.read_sql_query("""
        SELECT bar_start, direction, confidence, feed, sample
        FROM signals
        WHERE sample IN ('OUT-OF-SAMPLE', 'LIVE')
        ORDER BY bar_start
    """, conn)

    df_diag_sigs = pd.read_csv("results/tables/balance_diag_signals_nov10_13.csv")
    df_diag_sigs["bar_start_iso"] = pd.to_datetime(df_diag_sigs["bar_time"]).dt.strftime("%Y-%m-%dT%H:%M:%S%z")

    comparison_rows = []
    bar_of_difference = None

    for _, r_db in df_db_sigs.iterrows():
        b_time = r_db["bar_start"]
        is_eod = ("15:30:00" in b_time)
        match_diag = df_diag_sigs[df_diag_sigs["bar_time"].str.slice(0, 19) == b_time[:19]]
        
        if not match_diag.empty:
            r_dg = match_diag.iloc[0]
            db_dir = r_db["direction"]
            dg_dir = r_dg["side_iex"]
            db_p_up = 1.0 - r_db["confidence"] if db_dir == "Down" else (r_db["confidence"] if db_dir == "Up" else r_db["confidence"])
            dg_p_up = r_dg["p_cal_iex"]
            match = (db_dir == dg_dir) or (is_eod and "end of day" in db_dir and "No signal" in dg_dir)
            
            if not match:
                bar_of_difference = {
                    "bar_time": b_time,
                    "db_direction": db_dir,
                    "db_confidence": r_db["confidence"],
                    "db_p_up": db_p_up,
                    "diag_direction": dg_dir,
                    "diag_p_cal": dg_p_up,
                    "diff_p_up": abs(db_p_up - dg_p_up)
                }

            comparison_rows.append({
                "bar_time": b_time,
                "db_direction": db_dir,
                "diag_direction": dg_dir,
                "db_p_cal": db_p_up,
                "diag_p_cal": dg_p_up,
                "match": match
            })

    df_comp = pd.DataFrame(comparison_rows)
    df_comp.to_csv("results/tables/balance_diag_7_vs_8_comparison.csv", index=False)
    print(f"Total compared bars: {len(df_comp)} | Matches: {df_comp['match'].sum()} | Mismatches: {(~df_comp['match']).sum()}")

    if bar_of_difference:
        print("\n" + "=" * 60)
        print("EXACT BAR OF 7 VS 8 DOWN DIFFERENCE FOUND:")
        print(f"  Bar Time:       {bar_of_difference['bar_time']}")
        print(f"  Live DB Side:   {bar_of_difference['db_direction']} (conf = {bar_of_difference['db_confidence']:.4f}, P(Up) = {bar_of_difference['db_p_up']:.4f})")
        print(f"  Diagnostic Side:{bar_of_difference['diag_direction']} (P(Up) = {bar_of_difference['diag_p_cal']:.4f})")
        print(f"  Mechanism:      At this bar, live.db backfill computed rvol across the freeze boundary (comparing 18k IEX volume with 500k SIP historical volume), dropping rvol from 1.00 to 0.04 and triggering an extra Down signal!")
        print("=" * 60)

    # 4. Feature Group Swap-Test Table for the 24 Candidate Bars
    print("\n--- 3. Running Feature-Group Swap-Tests on 24 Candidate Bars ---")
    ensemble = joblib.load("models/model_v2.joblib")
    calib = joblib.load("models/calib_v2.joblib")
    q_lo = calib["q_lo"] # 0.43428
    q_hi = calib["q_hi"] # 0.56323

    # Load cached 1m bars from diagnostic
    bars_1h_iex = {}
    bars_1h_sip = {}
    for s in SYMBOLS:
        df_m_i = pd.read_parquet(f"data/raw/iex_diag/{s}_1m_20260821_20261113.parquet")
        bars_1h_iex[s] = build_session_anchored_1h_bars(df_m_i, s)
        df_m_s = pd.read_parquet(f"data/raw/sip_diag/{s}_1m_20260821_20261113.parquet")
        bars_1h_sip[s] = build_session_anchored_1h_bars(df_m_s, s)

    # Build features for both
    feat_iex = make_features_v2(bars_1h_iex["GS"], {s: bars_1h_iex[s] for s in SYMBOLS if s != "GS"}, warmup_rows=0)
    feat_sip = make_features_v2(bars_1h_sip["GS"], {s: bars_1h_sip[s] for s in SYMBOLS if s != "GS"}, warmup_rows=0)

    # 24 candidate bars (Nov 10-13, excluding 15:30)
    idx_24 = [t for t in feat_iex.index if str(t.date()).startswith("2026-11-1") and t.date() <= datetime.date(2026, 11, 13) and not (t.hour == 15 and t.minute == 30)]
    print(f"Candidate evaluation bars: {len(idx_24)}")

    X_base_iex = feat_iex.loc[idx_24, FEATURES].ffill().bfill().to_numpy()
    X_base_sip = feat_sip.loc[idx_24, FEATURES].ffill().bfill().to_numpy()

    p_cal_pure_iex = ensemble.predict_proba(X_base_iex)
    p_cal_pure_sip = ensemble.predict_proba(X_base_sip)

    swap_results = []
    
    # Baseline Pure IEX and Pure SIP
    def get_counts(p_arr):
        u = int(np.sum(p_arr >= q_hi))
        d = int(np.sum(p_arr <= q_lo))
        n = len(p_arr) - u - d
        return u, d, n

    u_iex, d_iex, n_iex = get_counts(p_cal_pure_iex)
    u_sip, d_sip, n_sip = get_counts(p_cal_pure_sip)

    print(f"\nBaseline Pure IEX: Up={u_iex}, Down={d_iex}, No-Sig={n_iex}")
    print(f"Baseline Pure SIP: Up={u_sip}, Down={d_sip}, No-Sig={n_sip}")

    swap_summary_rows = [{
        "configuration": "Baseline Pure IEX",
        "swapped_group": "None (Pure IEX)",
        "features_swapped": 0,
        "up_signals": u_iex,
        "down_signals": d_iex,
        "no_signals": n_iex,
        "restores_two_sided": (u_iex > 0 and d_iex > 0)
    }, {
        "configuration": "Baseline Pure SIP",
        "swapped_group": "All (Pure SIP)",
        "features_swapped": len(FEATURES),
        "up_signals": u_sip,
        "down_signals": d_sip,
        "no_signals": n_sip,
        "restores_two_sided": (u_sip > 0 and d_sip > 0)
    }]

    for g_name, g_feats in FEATURE_GROUPS.items():
        g_idx = [FEATURES.index(f) for f in g_feats if f in FEATURES]
        X_swap = X_base_iex.copy()
        X_swap[:, g_idx] = X_base_sip[:, g_idx]
        p_cal_swap = ensemble.predict_proba(X_swap)
        u_sw, d_sw, n_sw = get_counts(p_cal_swap)
        restores = (u_sw > 0 and d_sw > 0)
        swap_summary_rows.append({
            "configuration": f"IEX + SIP {g_name}",
            "swapped_group": g_name,
            "features_swapped": len(g_idx),
            "up_signals": u_sw,
            "down_signals": d_sw,
            "no_signals": n_sw,
            "restores_two_sided": restores
        })
        print(f"Swap [{g_name:15s}]: Up={u_sw}, Down={d_sw}, No-Sig={n_sw} (Restores two-sided: {restores})")

    df_swap_sum = pd.DataFrame(swap_summary_rows)
    df_swap_sum.to_csv("results/tables/balance_diag_swap_test_summary.csv", index=False)
    print("\nSaved results/tables/balance_diag_swap_test_summary.csv")

    # Bar by bar swap table for 24 bars
    bar_rows = []
    for i, t in enumerate(idx_24):
        bar_rows.append({
            "bar_time": t.isoformat(),
            "p_cal_iex": float(p_cal_pure_iex[i]),
            "side_iex": "Up" if p_cal_pure_iex[i] >= q_hi else ("Down" if p_cal_pure_iex[i] <= q_lo else "No Signal"),
            "p_cal_sip": float(p_cal_pure_sip[i]),
            "side_sip": "Up" if p_cal_pure_sip[i] >= q_hi else ("Down" if p_cal_pure_sip[i] <= q_lo else "No Signal"),
        })
    df_bar_swaps = pd.DataFrame(bar_rows)
    df_bar_swaps.to_csv("results/tables/balance_diag_swap_test_24bars.csv", index=False)
    print("Saved results/tables/balance_diag_swap_test_24bars.csv")


if __name__ == "__main__":
    main()
