#!/usr/bin/env python3
"""
Measurement Script for V3.7 Rule (47% to 53% No-Signal Band)
Computes coverage, descriptive accuracy by tier, and expected signals per day
under both Old Rule (0.43428, 0.56323) and New Rule (0.47, 0.53).
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd

from src.features import FEATURES
from src.live.log_engine import wilson_score_interval

def get_wilson(hits, n):
    if n == 0:
        return 0.0, [0.0, 0.0]
    lo, hi = wilson_score_interval(hits, n, z=1.95996)
    acc = round(hits / n * 100.0, 2)
    return acc, [round(lo, 2), round(hi, 2)]

def assign_tier(conf, weak_below=0.563232433386711, strong_conf=0.60):
    if conf < weak_below:
        return "WEAK"
    elif conf < strong_conf:
        return "NORMAL"
    else:
        return "STRONG"

def main():
    tables_dir = Path("results/tables")
    tables_dir.mkdir(parents=True, exist_ok=True)

    # Thresholds
    OLD_Q_LO = 0.43427959108206615
    OLD_Q_HI = 0.563232433386711
    NEW_Q_LO = 0.47
    NEW_Q_HI = 0.53
    STRONG_CONF = 0.60

    print("=== Step 1: Load Data Sources ===")
    # 1. Nov 10-13 (24 candidate bars, SIP)
    df_nov = pd.read_csv("results/tables/balance_diag_signals_nov10_13.csv")
    print(f"Loaded Nov 10-13 candidate bars: {len(df_nov)}")

    # 2. Last 60 completed sessions (SIP feed)
    df_60_sip = pd.read_csv("results/tables/balance_diag_signals_60d_sip.csv")
    print(f"Loaded 60 sessions SIP: {len(df_60_sip)}")

    # 3. V2 Holdout predictions (Pre-refit model)
    df_holdout = pd.read_csv("results/tables/holdout_predictions_v2.csv")
    print(f"Loaded V2 holdout predictions: {len(df_holdout)}")

    # 4. Pre-holdout Base Training Reference / OOF
    # We load dataset_v2.parquet to get pre-holdout labelled bars and model predictions
    p_dataset = Path("data/processed/dataset_v2.parquet")
    import joblib
    models_dir = Path("models")
    ensemble = joblib.load(models_dir / "model_v2.joblib")

    df_dataset = pd.read_parquet(p_dataset)
    holdout_start = pd.to_datetime("2025-10-16 09:30:00-04:00")
    df_pre = df_dataset[df_dataset.index < holdout_start].dropna(subset=FEATURES + ["y_H1"])
    # Non-EOD bars
    df_pre = df_pre[~((df_pre.index.hour == 15) & (df_pre.index.minute == 30))]
    X_pre = df_pre[FEATURES].to_numpy()
    p_cal_pre = ensemble.predict_proba(X_pre)
    df_pre["p_cal"] = p_cal_pre
    print(f"Pre-holdout Base Reference bars: {len(df_pre)}")

    # =========================================================================
    # Part 1: Coverage under Old Rule vs New Rule
    # =========================================================================
    print("\n=== Computing Coverage Comparison ===")
    coverage_rows = []

    periods = [
        ("Last 60 Completed Sessions (SIP Feed)", df_60_sip, "p_cal", True, "Deployed model & calibrator"),
        ("Nov 10-13 Out-of-Sample (24 Bars, SIP Feed)", df_nov, "p_cal_sip", False, "Deployed model & calibrator"),
        ("V2 Research Holdout (Pre-Refit Model)", df_holdout, "p_cal", True, "different calibrator, indicative only"),
        ("Pre-Holdout Walk-Forward / Base Ref", df_pre, "p_cal", False, "different calibrator, indicative only")
    ]

    for label, df_data, p_col, filter_eod, note in periods:
        data = df_data.copy()
        if filter_eod and "bar_time" in data.columns:
            ts = pd.to_datetime(data["bar_time"], utc=True).dt.tz_convert("America/New_York")
            data = data[~((ts.dt.hour == 15) & (ts.dt.minute == 30))]
        elif filter_eod and "timestamp" in data.columns:
            ts = pd.to_datetime(data["timestamp"], utc=True).dt.tz_convert("America/New_York")
            data = data[~((ts.dt.hour == 15) & (ts.dt.minute == 30))]

        p_vals = data[p_col].dropna().values
        n_bars = len(p_vals)

        # Old Rule (0.434, 0.563)
        up_old = int(np.sum(p_vals >= OLD_Q_HI))
        down_old = int(np.sum(p_vals <= OLD_Q_LO))
        nosig_old = n_bars - up_old - down_old
        cov_old = (up_old + down_old) / n_bars if n_bars > 0 else 0.0

        # New Rule (0.47, 0.53)
        up_new = int(np.sum(p_vals >= NEW_Q_HI))
        down_new = int(np.sum(p_vals <= NEW_Q_LO))
        nosig_new = n_bars - up_new - down_new
        cov_new = (up_new + down_new) / n_bars if n_bars > 0 else 0.0

        coverage_rows.append({
            "period": label,
            "calibrator_status": note,
            "total_candidate_bars": n_bars,
            "old_rule_up": up_old,
            "old_rule_down": down_old,
            "old_rule_no_signal": nosig_old,
            "old_rule_coverage_pct": round(cov_old * 100, 2),
            "new_rule_up": up_new,
            "new_rule_down": down_new,
            "new_rule_no_signal": nosig_new,
            "new_rule_coverage_pct": round(cov_new * 100, 2)
        })

    df_cov = pd.DataFrame(coverage_rows)
    df_cov.to_csv(tables_dir / "rule_47_53_coverage.csv", index=False)
    print("Saved rule_47_53_coverage.csv:")
    print(df_cov.to_string())

    # =========================================================================
    # Part 2: Descriptive Accuracy by Confidence Tier Under New Rule
    # =========================================================================
    print("\n=== Computing Descriptive Accuracy by Confidence Tier ===")
    tier_rows = []

    # Helper to evaluate tiers on a dataframe
    def evaluate_tiers(df_sub, p_col, ret_col, sample_tag, dataset_name):
        df_eval = df_sub.dropna(subset=[p_col, ret_col]).copy()
        for idx, row in df_eval.iterrows():
            p_val = float(row[p_col])
            ret = float(row[ret_col])
            if p_val >= NEW_Q_HI:
                side = "Up"
                conf = p_val
                hit = 1 if ret > 0 else 0
            elif p_val <= NEW_Q_LO:
                side = "Down"
                conf = 1.0 - p_val
                hit = 1 if ret < 0 else 0
            else:
                side = "No Signal"
                conf = None
                hit = None

            df_eval.at[idx, "side"] = side
            df_eval.at[idx, "conf"] = conf
            df_eval.at[idx, "hit"] = hit
            df_eval.at[idx, "tier"] = assign_tier(conf, OLD_Q_HI, STRONG_CONF) if conf is not None else None

        # Tally per tier and side
        for tier_name in ["WEAK", "NORMAL", "STRONG", "ALL_DIRECTIONAL"]:
            for side_target in ["Up", "Down", "Total"]:
                if tier_name == "ALL_DIRECTIONAL":
                    sub = df_eval[df_eval["side"].isin(["Up", "Down"])]
                else:
                    sub = df_eval[(df_eval["tier"] == tier_name) & (df_eval["side"].isin(["Up", "Down"]))]

                if side_target in ("Up", "Down"):
                    sub = sub[sub["side"] == side_target]

                n_tier = len(sub)
                hits = int(sub["hit"].sum()) if n_tier > 0 else 0
                acc, ci = get_wilson(hits, n_tier)

                tier_rows.append({
                    "dataset": dataset_name,
                    "sample_tag": sample_tag,
                    "tier": tier_name,
                    "side": side_target,
                    "n": n_tier,
                    "hits": hits,
                    "accuracy_pct": acc,
                    "wilson_ci_95": f"[{ci[0]}%, {ci[1]}%]",
                    "wilson_low": ci[0],
                    "wilson_high": ci[1]
                })

    # (a) Pre-holdout Base / OOF Reference
    evaluate_tiers(
        df_pre,
        p_col="p_cal",
        ret_col="fwd_H1",
        sample_tag="IN-SAMPLE (optimistic)",
        dataset_name="Pre-Holdout Base Reference (IN-SAMPLE)"
    )

    # (b) Nov 10-13 Out-of-Sample Bars
    evaluate_tiers(
        df_nov,
        p_col="p_cal_sip",
        ret_col="fwd_return_sip",
        sample_tag="OUT-OF-SAMPLE",
        dataset_name="Nov 10-13 Out-of-Sample (24 Bars)"
    )

    df_tier_acc = pd.DataFrame(tier_rows)
    df_tier_acc.to_csv(tables_dir / "rule_47_53_tier_accuracy.csv", index=False)
    print("Saved rule_47_53_tier_accuracy.csv:")
    print(df_tier_acc.to_string())

    # =========================================================================
    # Part 3: Expected Signals Per Day (Last 60 Sessions)
    # =========================================================================
    print("\n=== Computing Expected Signals Per Day (Last 60 Sessions) ===")
    df_60 = df_60_sip.copy()
    ts_60 = pd.to_datetime(df_60["bar_time"], utc=True).dt.tz_convert("America/New_York")
    df_60["date"] = ts_60.dt.date
    # Exclude EOD 15:30 bar from directional eligibility
    df_60_cand = df_60[~((ts_60.dt.hour == 15) & (ts_60.dt.minute == 30))].copy()

    df_60_cand["sig_old"] = df_60_cand["p_cal"].apply(lambda p: 1 if (p >= OLD_Q_HI or p <= OLD_Q_LO) else 0)
    df_60_cand["sig_new"] = df_60_cand["p_cal"].apply(lambda p: 1 if (p >= NEW_Q_HI or p <= NEW_Q_LO) else 0)

    daily_old = df_60_cand.groupby("date")["sig_old"].sum()
    daily_new = df_60_cand.groupby("date")["sig_new"].sum()

    per_day_summary = [
        {
            "rule": "Old Rule (43.4% to 56.3% neutral band)",
            "q_lo": round(OLD_Q_LO, 4),
            "q_hi": round(OLD_Q_HI, 4),
            "sessions_evaluated": len(daily_old),
            "mean_signals_per_day": round(float(daily_old.mean()), 2),
            "median_signals_per_day": float(daily_old.median()),
            "min_signals_per_day": int(daily_old.min()),
            "max_signals_per_day": int(daily_old.max()),
            "std_signals_per_day": round(float(daily_old.std()), 2)
        },
        {
            "rule": "New Rule (47.0% to 53.0% neutral band)",
            "q_lo": NEW_Q_LO,
            "q_hi": NEW_Q_HI,
            "sessions_evaluated": len(daily_new),
            "mean_signals_per_day": round(float(daily_new.mean()), 2),
            "median_signals_per_day": float(daily_new.median()),
            "min_signals_per_day": int(daily_new.min()),
            "max_signals_per_day": int(daily_new.max()),
            "std_signals_per_day": round(float(daily_new.std()), 2)
        }
    ]

    df_daily = pd.DataFrame(per_day_summary)
    df_daily.to_csv(tables_dir / "rule_47_53_signals_per_day.csv", index=False)
    print("Saved rule_47_53_signals_per_day.csv:")
    print(df_daily.to_string())

if __name__ == "__main__":
    main()
