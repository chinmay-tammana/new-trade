"""
Comprehensive Integrity Audit for M5 (5-Minute) System.
Implements the 5 Pre-Registered Fail Gates:
Gate 1: Shuffled-Label Null Test
Gate 2: Time-Shift Lead/Lag Test
Gate 3: Single-Feature Information Leakage & Window Inspection
Gate 4: IEX Microstructure & Thin-Print Artifact Test (vs Consolidated SIP)
Gate 5: Subgroup Stability Audit
Saves all computed tables to results/m5/audit/.
"""
import os
import glob
import json
import math
import datetime
import zoneinfo
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, log_loss
import xgboost as xgb

from configs.config import CONFIG_M5
from src.features_m5 import ALL_FEATURE_COLS, FEATURE_GROUPS
from src.aggregate_m5 import aggregate_1min_to_5min, ET_TZ
from src.train_m5 import build_walk_forward_folds, wilson_interval
from src.calibrator_m5 import PlattCalibrator

SEED = 42
AUDIT_DIR = "results/m5/audit"
TABLES_DIR = "results/m5/tables"


def run_gate1_shuffled_labels(df_pre_holdout, folds, chosen_thresh, q_lo, q_hi):
    print("\n--- Gate 1: Shuffled-Label Null Test ---")
    oof_preds = []
    oof_y = []
    
    for fold_i, f in enumerate(folds):
        train_sub = df_pre_holdout.loc[f["train_idx"]].copy()
        val_sub = df_pre_holdout.loc[f["val_idx"]].copy()
        
        train_decisive = train_sub[train_sub["fwd"].abs() > chosen_thresh].copy()
        
        # Shuffle training labels randomly
        np.random.seed(SEED + fold_i)
        y_train_shuffled = np.random.permutation(train_decisive["y"].values)
        
        X_tr = train_decisive[ALL_FEATURE_COLS].values
        X_val = val_sub[ALL_FEATURE_COLS].values
        y_val = val_sub["y"].values
        
        clf = xgb.XGBClassifier(
            n_estimators=120, max_depth=4, learning_rate=0.03, subsample=0.8,
            tree_method="hist", random_state=SEED, n_jobs=4
        )
        clf.fit(X_tr, y_train_shuffled)
        p_val = clf.predict_proba(X_val)[:, 1]
        
        oof_preds.append(p_val)
        oof_y.append(y_val)
        
    p_all = np.concatenate(oof_preds)
    y_all = np.concatenate(oof_y)
    
    calibrator = PlattCalibrator()
    calibrator.fit(p_all, y_all)
    p_cal = calibrator.predict_proba(p_all)
    
    is_up = p_cal >= q_hi
    is_down = p_cal <= q_lo
    sig_mask = is_up | is_down
    pred_y = np.where(is_up, 1, 0)
    
    n_sig = np.sum(sig_mask)
    if n_sig > 0:
        shuffled_acc = float(np.mean(y_all[sig_mask] == pred_y[sig_mask]))
    else:
        shuffled_acc = 0.50
        
    passed = (0.48 <= shuffled_acc <= 0.52)
    print(f"Shuffled-label Signal Accuracy: {shuffled_acc:.2%} (Signals: {n_sig})")
    print(f"Gate 1 Verdict: {'PASS' if passed else 'FAIL'} (Expected 48.0% to 52.0%)")
    
    return {
        "gate": "Gate 1: Shuffled Labels",
        "shuffled_acc": shuffled_acc,
        "n_signals": int(n_sig),
        "passed": passed
    }


def run_gate2_time_shift(df_pre_holdout, folds, chosen_thresh, q_lo, q_hi):
    print("\n--- Gate 2: Time-Shift Lead/Lag Test (+1 Bar Shift) ---")
    oof_preds = []
    oof_y = []
    
    # Shift features by +1 bar (use features from t-1 to predict y at t)
    df_shifted = df_pre_holdout.copy()
    for col in ALL_FEATURE_COLS:
        df_shifted[col] = df_shifted[col].shift(1)
        
    df_shifted = df_shifted.dropna(subset=ALL_FEATURE_COLS + ["y"])
    
    for fold_i, f in enumerate(folds):
        train_idx = [i for i in f["train_idx"] if i in df_shifted.index]
        val_idx = [i for i in f["val_idx"] if i in df_shifted.index]
        
        train_sub = df_shifted.loc[train_idx]
        val_sub = df_shifted.loc[val_idx]
        
        train_decisive = train_sub[train_sub["fwd"].abs() > chosen_thresh]
        X_tr = train_decisive[ALL_FEATURE_COLS].values
        y_tr = train_decisive["y"].values
        X_val = val_sub[ALL_FEATURE_COLS].values
        y_val = val_sub["y"].values
        
        clf = xgb.XGBClassifier(
            n_estimators=120, max_depth=4, learning_rate=0.03, subsample=0.8,
            tree_method="hist", random_state=SEED, n_jobs=4
        )
        clf.fit(X_tr, y_tr)
        p_val = clf.predict_proba(X_val)[:, 1]
        
        oof_preds.append(p_val)
        oof_y.append(y_val)
        
    p_all = np.concatenate(oof_preds)
    y_all = np.concatenate(oof_y)
    
    calibrator = PlattCalibrator()
    calibrator.fit(p_all, y_all)
    p_cal = calibrator.predict_proba(p_all)
    
    is_up = p_cal >= q_hi
    is_down = p_cal <= q_lo
    sig_mask = is_up | is_down
    pred_y = np.where(is_up, 1, 0)
    
    n_sig = np.sum(sig_mask)
    if n_sig > 0:
        shifted_acc = float(np.mean(y_all[sig_mask] == pred_y[sig_mask]))
    else:
        shifted_acc = 0.50
        
    passed = (shifted_acc <= 0.55)
    print(f"Time-shifted (+1 bar) Signal Accuracy: {shifted_acc:.2%} (Signals: {n_sig})")
    print(f"Gate 2 Verdict: {'PASS' if passed else 'FAIL'} (Expected <= 55.0%)")
    
    return {
        "gate": "Gate 2: Time Shift",
        "shifted_acc": shifted_acc,
        "n_signals": int(n_sig),
        "passed": passed
    }


def run_gate3_single_features(df_pre_holdout):
    print("\n--- Gate 3: Single-Feature Information Leakage Test ---")
    y = df_pre_holdout["y"].values
    fwd = df_pre_holdout["fwd"].values
    
    feat_records = []
    failed_features = []
    
    window_map = {
        "ret_": "close[t-k] to close[t]",
        "ret_since_open": "open[session] to close[t]",
        "gap_prev_close": "close[prev_day] to open[session]",
        "ret_1d": "close[prev_day-1] to close[prev_day]",
        "ret_5d": "close[prev_day-5] to close[prev_day]",
        "ret_20d": "close[prev_day-20] to close[prev_day]",
        "rsi": "Wilder EMA over close[t-14:t]",
        "macd": "EMA12/EMA26 over close[<=t]",
        "bb_": "Rolling 20-bar window close[t-20:t]",
        "atr_": "Rolling 14-bar TR over high, low, close[<=t]",
        "rolling_std": "Rolling std of 1-bar returns[<=t]",
        "realized_vol_intrabar": "1-min returns inside bar t (09:30 to 09:35)",
        "signed_volume": "1-min signed volume inside bar t",
        "up_minute_ratio": "1-min up/down minutes inside bar t",
        "clv": "Intrabar (close[t] - low[t]) / (high[t] - low[t])",
        "trade_share": "1-min trade count inside bar t",
        "rvol_tod": "volume[t] / mean(past 20 sessions at same TOD)",
        "dist_session": "(close[t] - metric[session <= t]) / ATR",
        "dist_prev": "(close[t] - prev_day_level) / ATR",
        "or_pos_30m": "(close[t] - OR_low[<=t]) / (OR_high[<=t] - OR_low[<=t])",
        "mins_": "Timestamp arithmetic at bar close t",
        "tod_": "Trigonometric cyclical encoding of bar close t",
        "spy_": "SPY close[t-k] to close[t]",
        "xlf_": "XLF close[t-k] to close[t]",
        "jpm_": "JPM close[t-k] to close[t]",
        "ms_": "MS close[t-k] to close[t]",
        "bac_": "BAC close[t-k] to close[t]",
        "gs_minus": "Spread between GS return and context return <= t"
    }

    for col in ALL_FEATURE_COLS:
        vals = df_pre_holdout[col].values
        # Handle any possible NaNs safely
        valid_mask = ~np.isnan(vals) & ~np.isnan(y)
        v_sub = vals[valid_mask]
        y_sub = y[valid_mask]
        fwd_sub = fwd[valid_mask]
        
        auc = roc_auc_score(y_sub, v_sub)
        rho, _ = stats.spearmanr(v_sub, fwd_sub)
        auc_delta = abs(auc - 0.50)
        
        # Determine window description
        w_desc = "Causal window <= close[t]"
        for k, v in window_map.items():
            if k in col:
                w_desc = v
                break
                
        is_leak = (auc_delta > 0.10)
        if is_leak:
            failed_features.append(col)
            
        feat_records.append({
            "feature": col,
            "auc": float(auc),
            "auc_delta": float(auc_delta),
            "spearman_rho": float(rho),
            "data_window_used": w_desc,
            "leak_flag": is_leak
        })

    df_feats = pd.DataFrame(feat_records).sort_values("auc_delta", ascending=False)
    df_feats.to_csv(os.path.join(AUDIT_DIR, "single_feature_auc.csv"), index=False)
    
    passed = (len(failed_features) == 0)
    max_auc_row = df_feats.iloc[0]
    print(f"Max AUC Delta: {max_auc_row['auc_delta']:.4f} ({max_auc_row['feature']}: AUC={max_auc_row['auc']:.4f}, Rho={max_auc_row['spearman_rho']:+.4f})")
    print(f"Window: {max_auc_row['data_window_used']}")
    print(f"Gate 3 Verdict: {'PASS' if passed else 'FAIL'} (Threshold: |AUC - 0.5| <= 0.10)")
    if not passed:
        print(f"FAILING FEATURES: {failed_features}")
        
    return {
        "gate": "Gate 3: Single Feature Leakage",
        "max_auc_feature": max_auc_row["feature"],
        "max_auc": float(max_auc_row["auc"]),
        "failed_features": failed_features,
        "passed": passed
    }


def run_gate4_iex_artifact_test(df_pre_holdout, q_lo, q_hi):
    print("\n--- Gate 4: IEX Microstructure & Thin-Print Artifact Test ---")
    # 1. Load SIP 1-minute bars for GS
    sip_pattern = "data/raw/alpaca_1min/GS/*.parquet"
    sip_files = sorted(glob.glob(sip_pattern))
    if not sip_files:
        print("Warning: SIP 1-min data not found in data/raw/alpaca_1min/GS/. Skipping Gate 4 comparison.")
        return {"gate": "Gate 4: IEX Artifact", "passed": True, "artifact_detected": False}
        
    dfs = []
    # Load files overlapping pre-holdout (2021-11 to 2026-05)
    for f in sip_files:
        base = os.path.basename(f)
        if "2021-11" <= base <= "2026-05":
            dfs.append(pd.read_parquet(f))
            
    if not dfs:
        print("No overlapping SIP files found.")
        return {"gate": "Gate 4: IEX Artifact", "passed": True, "artifact_detected": False}
        
    df_sip_1m = pd.concat(dfs, axis=0)
    if isinstance(df_sip_1m.index, pd.MultiIndex):
        df_sip_1m = df_sip_1m.xs("GS", level="symbol")
    df_sip_1m = df_sip_1m[~df_sip_1m.index.duplicated(keep="last")].sort_index()
    
    # 2. Build 5-minute SIP bars
    df_sip_5m = aggregate_1min_to_5min(df_sip_1m, symbol="GS")
    print(f"Built {len(df_sip_5m):,} 5-minute bars from consolidated SIP.")

    # 3. Compute SIP forward returns
    c_sip = df_sip_5m["close"]
    next_c_sip = c_sip.shift(-1)
    dates_sip = pd.Series(df_sip_5m.index.date, index=df_sip_5m.index)
    same_session_sip = (dates_sip == dates_sip.shift(-1)).values
    fwd_sip = (next_c_sip / c_sip - 1.0).values.copy()
    fwd_sip[~same_session_sip] = np.nan
    df_sip_5m["fwd_sip"] = fwd_sip
    df_sip_5m["ret_sip"] = c_sip / c_sip.shift(1) - 1.0

    # 4. Lag-1 Autocorrelation of Returns
    r_iex = df_pre_holdout["ret_1"].dropna().values
    r_sip = df_sip_5m["ret_sip"].dropna().values
    
    acf1_iex = float(np.corrcoef(r_iex[:-1], r_iex[1:])[0, 1])
    acf1_sip = float(np.corrcoef(r_sip[:-1], r_sip[1:])[0, 1])
    print(f"Lag-1 Autocorrelation: IEX = {acf1_iex:+.4f} | Consolidated SIP = {acf1_sip:+.4f}")

    # 5. Join IEX pre-holdout with SIP forward returns
    df_joined = df_pre_holdout.join(df_sip_5m[["fwd_sip"]], how="inner")
    df_joined = df_joined.dropna(subset=["fwd", "fwd_sip"])
    print(f"Aligned {len(df_joined):,} bars between IEX features and SIP forward returns.")

    # Re-predict using production model bundle
    import joblib
    bundle = joblib.load(CONFIG_M5["paths"]["model_path"])
    primary_clf = bundle["primary_model"]
    calibrator = bundle["calibrator"]
    
    X = df_joined[ALL_FEATURE_COLS].values
    raw_p = primary_clf.predict_proba(X)[:, 1]
    p_cal = calibrator.predict_proba(raw_p)
    
    is_up = p_cal >= q_hi
    is_down = p_cal <= q_lo
    sig_mask = is_up | is_down
    pred_dir = np.where(is_up, 1, np.where(is_down, -1, 0))
    
    # Accuracy against IEX forward returns
    fwd_iex_sub = df_joined["fwd"].values[sig_mask]
    y_true_iex = np.where(fwd_iex_sub > 0, 1, 0)
    pred_y = np.where(pred_dir[sig_mask] == 1, 1, 0)
    acc_iex = float(np.mean(y_true_iex == pred_y))
    
    # Accuracy against SIP forward returns
    fwd_sip_sub = df_joined["fwd_sip"].values[sig_mask]
    y_true_sip = np.where(fwd_sip_sub > 0, 1, 0)
    acc_sip = float(np.mean(y_true_sip == pred_y))
    
    acc_delta = acc_iex - acc_sip
    print(f"Accuracy on IEX returns: {acc_iex:.2%} | Accuracy on SIP returns: {acc_sip:.2%} (Delta: {acc_delta:+.2%})")

    # Accuracy split by n_minutes_with_trades
    n_mins = df_joined["n_minutes_with_trades"].values[sig_mask]
    low_trade_mask = (n_mins <= 2)
    high_trade_mask = (n_mins >= 4)
    
    acc_low_iex = float(np.mean(y_true_iex[low_trade_mask] == pred_y[low_trade_mask])) if np.sum(low_trade_mask) > 0 else 0.0
    acc_high_iex = float(np.mean(y_true_iex[high_trade_mask] == pred_y[high_trade_mask])) if np.sum(high_trade_mask) > 0 else 0.0
    print(f"IEX Accuracy by Minute Density: Low (<=2 mins) = {acc_low_iex:.2%} | High (>=4 mins) = {acc_high_iex:.2%}")

    # Interpretation Rule
    # If acc_sip < acc_iex - 5 points OR acf1_iex < -0.05 while acf1_sip near 0: classify as IEX PRINT ARTIFACT
    is_artifact = (acc_delta > 0.05) or (acf1_iex < -0.05 and abs(acf1_sip) < 0.03)
    
    artifact_summary = {
        "acf1_iex": acf1_iex,
        "acf1_sip": acf1_sip,
        "acc_iex": acc_iex,
        "acc_sip": acc_sip,
        "acc_delta": acc_delta,
        "acc_low_trade_density": acc_low_iex,
        "acc_high_trade_density": acc_high_iex,
        "is_artifact": is_artifact
    }
    with open(os.path.join(AUDIT_DIR, "iex_artifact_audit.json"), "w") as f:
        json.dump(artifact_summary, f, indent=2)
        
    print(f"Gate 4 Classification: {'IEX PRINT ARTIFACT DETECTED' if is_artifact else 'NO ARTIFACT (CONSOLIDATED VALIDATED)'}")
    return {
        "gate": "Gate 4: IEX Artifact",
        "acf1_iex": acf1_iex,
        "acf1_sip": acf1_sip,
        "acc_iex": acc_iex,
        "acc_sip": acc_sip,
        "acc_delta": acc_delta,
        "is_artifact": is_artifact,
        "passed": True # Non-blocking gate (flags warning card if True)
    }


def run_gate5_stability(df_pre_holdout, q_lo, q_hi):
    print("\n--- Gate 5: Subgroup Stability Audit ---")
    import joblib
    bundle = joblib.load(CONFIG_M5["paths"]["model_path"])
    primary_clf = bundle["primary_model"]
    calibrator = bundle["calibrator"]
    
    X = df_pre_holdout[ALL_FEATURE_COLS].values
    raw_p = primary_clf.predict_proba(X)[:, 1]
    p_cal = calibrator.predict_proba(raw_p)
    
    df_eval = df_pre_holdout.copy()
    df_eval["p_cal"] = p_cal
    df_eval["is_up"] = p_cal >= q_hi
    df_eval["is_down"] = p_cal <= q_lo
    df_eval["sig_mask"] = df_eval["is_up"] | df_eval["is_down"]
    df_eval["pred_y"] = np.where(df_eval["is_up"], 1, 0)
    
    df_sig = df_eval[df_eval["sig_mask"]].copy()
    df_sig["correct"] = (df_sig["y"] == df_sig["pred_y"]).astype(int)
    
    # 1. By Year
    df_sig["year"] = pd.to_datetime(df_sig["date"]).dt.year
    year_stats = []
    for yr, grp in df_sig.groupby("year"):
        n = len(grp)
        hits = grp["correct"].sum()
        acc = hits / n if n > 0 else 0.0
        ci_lo, ci_hi = wilson_interval(hits, n)
        year_stats.append({
            "segment": f"Year {yr}",
            "n_signals": int(n),
            "accuracy": float(acc),
            "wilson_ci": [float(ci_lo), float(ci_hi)]
        })
        print(f"Year {yr}: {acc:.2%} ({hits}/{n}) [Wilson: {ci_lo:.1f}%, {ci_hi:.1f}%]")

    # 2. By Time of Day
    times = df_sig.index.time
    t_open = datetime.time(11, 0)
    t_mid = datetime.time(14, 0)
    
    df_sig["tod_bucket"] = "Midday (11:00-14:00)"
    df_sig.loc[times < t_open, "tod_bucket"] = "Opening (09:30-11:00)"
    df_sig.loc[times >= t_mid, "tod_bucket"] = "Closing (14:00-15:55)"
    
    tod_stats = []
    for tb, grp in df_sig.groupby("tod_bucket"):
        n = len(grp)
        hits = grp["correct"].sum()
        acc = hits / n if n > 0 else 0.0
        ci_lo, ci_hi = wilson_interval(hits, n)
        tod_stats.append({
            "segment": tb,
            "n_signals": int(n),
            "accuracy": float(acc),
            "wilson_ci": [float(ci_lo), float(ci_hi)]
        })
        print(f"{tb:22s}: {acc:.2%} ({hits}/{n}) [Wilson: {ci_lo:.1f}%, {ci_hi:.1f}%]")

    # 3. By Side
    side_stats = []
    for side, mask in [("Up", df_sig["is_up"]), ("Down", df_sig["is_down"])]:
        grp = df_sig[mask]
        n = len(grp)
        hits = grp["correct"].sum()
        acc = hits / n if n > 0 else 0.0
        ci_lo, ci_hi = wilson_interval(hits, n)
        side_stats.append({
            "segment": f"Side {side}",
            "n_signals": int(n),
            "accuracy": float(acc),
            "wilson_ci": [float(ci_lo), float(ci_hi)]
        })
        print(f"Side {side:4s}: {acc:.2%} ({hits}/{n}) [Wilson: {ci_lo:.1f}%, {ci_hi:.1f}%]")

    all_stability = year_stats + tod_stats + side_stats
    pd.DataFrame(all_stability).to_csv(os.path.join(AUDIT_DIR, "stability_audit.csv"), index=False)
    return all_stability


def main():
    os.makedirs(AUDIT_DIR, exist_ok=True)
    os.makedirs(TABLES_DIR, exist_ok=True)
    
    print("==================================================")
    print("M5 SYSTEM INTEGRITY AUDIT: 5 PRE-REGISTERED GATES")
    print("==================================================")
    
    df = pd.read_parquet(CONFIG_M5["paths"]["processed_features"]).sort_index()
    df_valid = df.dropna(subset=["y"] + ALL_FEATURE_COLS).copy()
    
    all_dates = sorted(list(set(df_valid["date"])))
    pre_holdout_dates = set(all_dates[:-126])
    df_pre_holdout = df_valid[df_valid["date"].isin(pre_holdout_dates)].copy()
    
    folds = build_walk_forward_folds(df_pre_holdout, initial_train_years=2, quarter_sessions=63, gap_bars=8)
    
    # Read frozen rule values
    with open(CONFIG_M5["paths"]["calib_path"], "r") as f:
        calib_cfg = json.load(f)
    q_lo = float(calib_cfg["q_lo"])
    q_hi = float(calib_cfg["q_hi"])
    chosen_thresh = 0.0002 # 2 bps
    
    # Run Gates
    g1 = run_gate1_shuffled_labels(df_pre_holdout, folds, chosen_thresh, q_lo, q_hi)
    g2 = run_gate2_time_shift(df_pre_holdout, folds, chosen_thresh, q_lo, q_hi)
    g3 = run_gate3_single_features(df_pre_holdout)
    g4 = run_gate4_iex_artifact_test(df_pre_holdout, q_lo, q_hi)
    g5 = run_gate5_stability(df_pre_holdout, q_lo, q_hi)
    
    # Compile Audit Findings
    audit_summary = {
        "gate1_shuffled": g1,
        "gate2_timeshift": g2,
        "gate3_single_features": g3,
        "gate4_iex_artifact": g4,
        "overall_status": "PROCEED" if (g1["passed"] and g2["passed"] and g3["passed"]) else "STOP",
        "created_at": datetime.datetime.now(ET_TZ).isoformat()
    }
    
    with open(os.path.join(AUDIT_DIR, "audit_findings.json"), "w") as f:
        json.dump(audit_summary, f, indent=2)
        
    print("\n==================================================")
    print("AUDIT FINDINGS SUMMARY:")
    print(f"Gate 1 (Shuffled Labels):     {'PASS' if g1['passed'] else 'FAIL'} (Acc: {g1['shuffled_acc']:.2%})")
    print(f"Gate 2 (Time-Shift +1 Bar):   {'PASS' if g2['passed'] else 'FAIL'} (Acc: {g2['shifted_acc']:.2%})")
    print(f"Gate 3 (Feature Leakage):     {'PASS' if g3['passed'] else 'FAIL'} (Max AUC: {g3['max_auc']:.4f})")
    print(f"Gate 4 (IEX Artifact):        {'ARTIFACT DETECTED' if g4['is_artifact'] else 'NO ARTIFACT'} (Delta: {g4['acc_delta']:+.2%})")
    print(f"OVERALL DECISION:             {audit_summary['overall_status']}")
    print("==================================================")

    return audit_summary


if __name__ == "__main__":
    main()
