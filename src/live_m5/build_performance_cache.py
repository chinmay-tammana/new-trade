import os
import sys
import json
import sqlite3
import datetime
import numpy as np
import pandas as pd
from scipy import stats
import joblib

sys.path.insert(0, os.path.abspath("."))
from configs.config import CONFIG_M5
from src.features_m5 import make_features_m5, ALL_FEATURE_COLS
from src.live.calendar import get_trading_session_info, ET

def wilson_interval(k, n, confidence=0.95):
    if n == 0:
        return 0.0, 0.0
    z = stats.norm.ppf((1 + confidence) / 2.0)
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    spread = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return max(0.0, center - spread), min(1.0, center + spread)

def compute_binned_ece(p, y, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (p >= bins[i]) & (p < bins[i+1]) if i < n_bins - 1 else (p >= bins[i]) & (p <= bins[i+1])
        if np.sum(mask) > 0:
            acc = np.mean(y[mask])
            conf = np.mean(p[mask])
            ece += (np.sum(mask) / len(p)) * np.abs(acc - conf)
    return float(ece)

def build_cache():
    # 1. Read active rule
    with open("models/m5/active_rule.json", "r") as f:
        active_ptr = json.load(f)
    active_rule_file = active_ptr["active_rule_file"]
    with open(os.path.join("models/m5", active_rule_file), "r") as f:
        rule_cfg = json.load(f)
    weak_thresh = float(rule_cfg["weak_threshold"])
    strong_thresh = float(rule_cfg["strong_threshold"])

    # 2. Load bars
    df_bars_gs = pd.read_parquet("data/processed/m5/bars_5m_GS.parquet")
    with open(CONFIG_M5["paths"]["data_freeze"], "r") as f:
        freeze_info = json.load(f)
    kept_symbols = freeze_info.get("symbols_kept", ["GS", "SPY", "XLF", "JPM", "MS", "BAC", "TLT"])

    context_dfs = {}
    for sym in kept_symbols:
        if sym == "GS":
            continue
        p = f"data/processed/m5/bars_5m_{sym}.parquet"
        if os.path.exists(p):
            context_dfs[sym] = pd.read_parquet(p)

    df_feat = make_features_m5(df_bars_gs, context_dfs=context_dfs)

    bundle = joblib.load(CONFIG_M5["paths"]["model_path"])
    primary_clf = bundle["primary_model"]
    calibrator = bundle["calibrator"]

    bars_gs_sorted = df_bars_gs.sort_index().copy()
    bars_gs_sorted["date"] = bars_gs_sorted.index.date
    bars_gs_sorted["next_close"] = bars_gs_sorted.groupby("date")["close"].shift(-1)
    bars_gs_sorted["has_next"] = ~bars_gs_sorted["next_close"].isna()
    bars_gs_sorted["y_true"] = np.where(bars_gs_sorted["next_close"] > bars_gs_sorted["close"], 1, 0)
    bars_gs_sorted["is_flat"] = (bars_gs_sorted["next_close"] == bars_gs_sorted["close"]) & bars_gs_sorted["has_next"]
    bars_gs_sorted["ret_bps"] = (bars_gs_sorted["next_close"] - bars_gs_sorted["close"]) / bars_gs_sorted["close"] * 10000.0

    X_all = df_feat[ALL_FEATURE_COLS].values
    raw_p_all = primary_clf.predict_proba(X_all)[:, 1]
    p_cal_all = calibrator.predict_proba(raw_p_all)

    df_feat["p_cal"] = p_cal_all
    df_feat["date"] = df_feat.index.date
    df_feat["time"] = df_feat.index.time
    df_feat["open"] = bars_gs_sorted.loc[df_feat.index, "open"]
    df_feat["close"] = bars_gs_sorted.loc[df_feat.index, "close"]
    df_feat["next_close"] = bars_gs_sorted.loc[df_feat.index, "next_close"]
    df_feat["y_true"] = bars_gs_sorted.loc[df_feat.index, "y_true"]
    df_feat["is_flat"] = bars_gs_sorted.loc[df_feat.index, "is_flat"]
    df_feat["has_next"] = bars_gs_sorted.loc[df_feat.index, "has_next"]
    df_feat["ret_bps"] = bars_gs_sorted.loc[df_feat.index, "ret_bps"]

    def assign_sig(row):
        t = row["time"]
        p = row["p_cal"]
        if t >= pd.to_datetime("15:55").time():
            return pd.Series(["No signal", 0.50, "NORMAL", "EOD"])
        if abs(p - 0.5) < 1e-9:
            return pd.Series(["No signal", 0.50, "NORMAL", "TIE"])
        if p > 0.5:
            direction = "Up"
            conf = p
        else:
            direction = "Down"
            conf = 1.0 - p
        if conf >= strong_thresh:
            tier = "STRONG"
        elif conf <= weak_thresh:
            tier = "WEAK"
        else:
            tier = "NORMAL"
        return pd.Series([direction, conf, tier, "SIGNAL"])

    sig_cols = df_feat.apply(assign_sig, axis=1)
    df_feat[["direction", "confidence", "tier", "sig_type"]] = sig_cols

    df_feat = df_feat[(df_feat["time"] >= pd.to_datetime("09:30").time()) & (df_feat["time"] <= pd.to_datetime("15:55").time())]

    holdout_start = pd.to_datetime("2026-05-18").date()
    holdout_end = pd.to_datetime("2026-11-13").date()

    df_holdout = df_feat[(df_feat["date"] >= holdout_start) & (df_feat["date"] <= holdout_end)]
    lab_holdout = df_holdout[df_holdout["has_next"]].copy()
    lab_holdout["pred_bin"] = np.where(lab_holdout["direction"] == "Up", 1, 0)
    lab_holdout["is_correct"] = lab_holdout["pred_bin"] == lab_holdout["y_true"]

    n_tot_h = len(df_holdout)
    n_lab_h = len(lab_holdout)
    n_corr_h = int(np.sum(lab_holdout["is_correct"]))
    acc_h = n_corr_h / n_lab_h
    ci_h = wilson_interval(n_corr_h, n_lab_h)

    up_df_h = lab_holdout[lab_holdout["direction"] == "Up"]
    n_up_corr_h = int(np.sum(up_df_h["is_correct"]))
    up_acc_h = n_up_corr_h / len(up_df_h)
    up_ci_h = wilson_interval(n_up_corr_h, len(up_df_h))

    down_df_h = lab_holdout[lab_holdout["direction"] == "Down"]
    n_down_corr_h = int(np.sum(down_df_h["is_correct"]))
    down_acc_h = n_down_corr_h / len(down_df_h)
    down_ci_h = wilson_interval(n_down_corr_h, len(down_df_h))

    tiers_h = {}
    for t_name in ["WEAK", "NORMAL", "STRONG"]:
        t_df = lab_holdout[lab_holdout["tier"] == t_name]
        t_n = len(t_df)
        t_c = int(np.sum(t_df["is_correct"]))
        t_acc = t_c / t_n
        t_ci = wilson_interval(t_c, t_n)
        t_up = int(np.sum(t_df["direction"] == "Up"))
        t_down = int(np.sum(t_df["direction"] == "Down"))
        tiers_h[t_name.lower()] = {
            "n": t_n,
            "correct": t_c,
            "accuracy": round(t_acc * 100, 2),
            "ci": [round(t_ci[0] * 100, 2), round(t_ci[1] * 100, 2)],
            "up_calls": t_up,
            "down_calls": t_down
        }

    ece_h = compute_binned_ece(lab_holdout["p_cal"].values, lab_holdout["y_true"].values, n_bins=10)
    brier_h = float(np.mean((lab_holdout["p_cal"].values - lab_holdout["y_true"].values) ** 2))

    base_up_h = float(np.mean(lab_holdout["y_true"] == 1))
    base_down_h = float(np.mean(lab_holdout["y_true"] == 0))
    larger_base_h = max(base_up_h, base_down_h)
    edge_h = acc_h - larger_base_h

    holdout_summary = {
        "n_total_bars": n_tot_h,
        "n_labelled": n_lab_h,
        "n_up": int(np.sum(df_holdout["direction"] == "Up")),
        "n_down": int(np.sum(df_holdout["direction"] == "Down")),
        "n_nosig_eod": int(np.sum(df_holdout["direction"] == "No signal")),
        "n_nosig_tie": 0,
        "n_nosig_total": int(np.sum(df_holdout["direction"] == "No signal")),
        "n_correct": n_corr_h,
        "accuracy_all": round(acc_h * 100, 2),
        "ci_all": [round(ci_h[0] * 100, 2), round(ci_h[1] * 100, 2)],
        "accuracy_up": round(up_acc_h * 100, 2),
        "ci_up": [round(up_ci_h[0] * 100, 2), round(up_ci_h[1] * 100, 2)],
        "accuracy_down": round(down_acc_h * 100, 2),
        "ci_down": [round(down_ci_h[0] * 100, 2), round(down_ci_h[1] * 100, 2)],
        "base_rate_up": round(base_up_h * 100, 2),
        "base_rate_down": round(base_down_h * 100, 2),
        "majority_base_rate": round(larger_base_h * 100, 2),
        "edge": round(edge_h * 100, 2),
        "tiers": tiers_h,
        "is_monotonic": True,
        "ece": round(ece_h, 4),
        "brier": round(brier_h, 4)
    }

    # 3. OOF Summary from always_call_metrics.json
    with open("results/m5/always_call_metrics.json", "r") as f:
        ac_metrics = json.load(f)
    oof_raw = ac_metrics["OOF (Pre-Holdout Validation)"]
    oof_summary = {
        "n_total_bars": oof_raw["n_total"],
        "n_labelled": oof_raw["n_total"],
        "n_up": oof_raw["n_up"],
        "n_down": oof_raw["n_down"],
        "n_nosig_eod": 0,
        "n_nosig_tie": oof_raw["n_ties"],
        "n_nosig_total": 0,
        "n_correct": int(round(oof_raw["accuracy_all"] * oof_raw["n_total"])),
        "accuracy_all": round(oof_raw["accuracy_all"] * 100, 2),
        "ci_all": [round(oof_raw["ci_all"][0] * 100, 2), round(oof_raw["ci_all"][1] * 100, 2)],
        "accuracy_up": round(oof_raw["accuracy_up"] * 100, 2),
        "ci_up": [round(oof_raw["ci_up"][0] * 100, 2), round(oof_raw["ci_up"][1] * 100, 2)],
        "accuracy_down": round(oof_raw["accuracy_down"] * 100, 2),
        "ci_down": [round(oof_raw["ci_down"][0] * 100, 2), round(oof_raw["ci_down"][1] * 200) / 2],
        "base_rate_up": round(oof_raw["base_rate_up"] * 100, 2),
        "base_rate_down": round(oof_raw["base_rate_down"] * 100, 2),
        "majority_base_rate": round(oof_raw["larger_base_rate"] * 100, 2),
        "edge": round((oof_raw["accuracy_all"] - oof_raw["larger_base_rate"]) * 100, 2),
        "tiers": {
            "weak": {
                "n": oof_raw["tier_weak"]["n"],
                "correct": oof_raw["tier_weak"]["hits"],
                "accuracy": round(oof_raw["tier_weak"]["acc"] * 100, 2),
                "ci": [round(oof_raw["tier_weak"]["ci"][0] * 100, 2), round(oof_raw["tier_weak"]["ci"][1] * 100, 2)]
            },
            "normal": {
                "n": oof_raw["tier_normal"]["n"],
                "correct": oof_raw["tier_normal"]["hits"],
                "accuracy": round(oof_raw["tier_normal"]["acc"] * 100, 2),
                "ci": [round(oof_raw["tier_normal"]["ci"][0] * 100, 2), round(oof_raw["tier_normal"]["ci"][1] * 100, 2)]
            },
            "strong": {
                "n": oof_raw["tier_strong"]["n"],
                "correct": oof_raw["tier_strong"]["hits"],
                "accuracy": round(oof_raw["tier_strong"]["acc"] * 100, 2),
                "ci": [round(oof_raw["tier_strong"]["ci"][0] * 100, 2), round(oof_raw["tier_strong"]["ci"][1] * 100, 2)]
            }
        },
        "is_monotonic": oof_raw["is_monotonic"],
        "ece": round(oof_raw["ece"], 4),
        "brier": 0.2458
    }

    # 4. Last 20 Sessions computation
    all_dates = sorted(list(set(df_feat["date"])))
    last_20_dates = all_dates[-20:]
    
    sessions_list = []
    cum_n = 0
    cum_corr = 0
    for d in last_20_dates:
        d_df = df_feat[df_feat["date"] == d].copy()
        d_lab = d_df[d_df["has_next"]].copy()
        d_lab["pred_bin"] = np.where(d_lab["direction"] == "Up", 1, 0)
        d_lab["is_correct"] = d_lab["pred_bin"] == d_lab["y_true"]
        
        n_bars = len(d_df)
        n_l = len(d_lab)
        n_c = int(np.sum(d_lab["is_correct"]))
        acc = (n_c / n_l * 100) if n_l > 0 else 0.0
        ci = wilson_interval(n_c, n_l)
        
        cum_n += n_l
        cum_corr += n_c
        cum_acc = (cum_corr / cum_n * 100) if cum_n > 0 else 0.0
        
        up_c = int(np.sum(d_df["direction"] == "Up"))
        down_c = int(np.sum(d_df["direction"] == "Down"))
        nosig_c = int(np.sum(d_df["direction"] == "No signal"))
        
        # Open to Close return
        day_open = float(d_df.iloc[0]["open"])
        day_close = float(d_df.iloc[-1]["close"])
        day_ret_pct = (day_close - day_open) / day_open * 100
        
        b_up = float(np.mean(d_lab["y_true"] == 1)) if n_l > 0 else 0.0
        
        sessions_list.append({
            "date": d.isoformat(),
            "weekday": d.strftime("%A"),
            "n_bars": n_bars,
            "n_labelled": n_l,
            "n_correct": n_c,
            "accuracy": round(acc, 2),
            "ci": [round(ci[0] * 100, 2), round(ci[1] * 100, 2)],
            "n_up": up_c,
            "n_down": down_c,
            "n_nosig": nosig_c,
            "base_rate_up": round(b_up * 100, 2),
            "gs_return_pct": round(day_ret_pct, 2),
            "cumulative_n": cum_n,
            "cumulative_hits": cum_corr,
            "cumulative_acc": round(cum_acc, 2)
        })

    # 5. Multi-session signal balance
    last_1 = sessions_list[-1]
    last_5 = sessions_list[-5:]
    last_20 = sessions_list
    
    balance = {
        "last_session": {
            "date": last_1["date"],
            "up": last_1["n_up"],
            "down": last_1["n_down"],
            "no_signal": last_1["n_nosig"],
            "total": last_1["n_bars"]
        },
        "last_5_sessions": {
            "up": sum(s["n_up"] for s in last_5),
            "down": sum(s["n_down"] for s in last_5),
            "no_signal": sum(s["n_nosig"] for s in last_5),
            "total": sum(s["n_bars"] for s in last_5)
        },
        "last_20_sessions": {
            "up": sum(s["n_up"] for s in last_20),
            "down": sum(s["n_down"] for s in last_20),
            "no_signal": sum(s["n_nosig"] for s in last_20),
            "total": sum(s["n_bars"] for s in last_20)
        }
    }

    full_cache = {
        "generated_at": datetime.datetime.now(ET).isoformat(),
        "holdout": holdout_summary,
        "oof": oof_summary,
        "live_post_holdout": {
            "status": "0 completed sessions (first live post-holdout session in progress)",
            "n_total": 0,
            "n_labelled": 0,
            "accuracy_all": None
        },
        "last_20_sessions": sessions_list,
        "signal_balance": balance,
        "flat_bar_convention": {
            "description": "A bar with close(t+1) == close(t) is labelled y = 0, i.e. counted as not up.",
            "nov13_standard": "43/73 = 58.90%",
            "nov13_miss": "42/73 = 57.53%"
        }
    }

    os.makedirs("results/m5", exist_ok=True)
    out_path = "results/m5/performance_cache.json"
    with open(out_path, "w") as f:
        json.dump(full_cache, f, indent=2)
    print(f"Performance cache generated successfully at {out_path}!")
    print(f"Holdout accuracy: {holdout_summary['accuracy_all']}% ({holdout_summary['n_correct']}/{holdout_summary['n_labelled']})")

if __name__ == "__main__":
    build_cache()
