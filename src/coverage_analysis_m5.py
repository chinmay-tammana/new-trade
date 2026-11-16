"""
Coverage Trade-Off Analysis and Rule Selection (Rule A vs Rule B) for M5 System.
Computes:
1. Trade-off table across c in {0.05, 0.10, 0.20, 0.30, 0.40, 0.50} on OOF and Holdout.
2. Rule A (symmetric quantiles with min_margin=0.01) on OOF, Holdout, and last 20 completed sessions.
3. Fallback decision to Rule B if achieved coverage < 15% on last 20 sessions.
4. Confidence Tiers (STRONG >= 75th percentile, NORMAL, WEAK < 25th percentile).
5. Writes outputs to results/m5/tables/ and models/m5/.
"""
import os
import sys
import json
import hashlib
import numpy as np
import pandas as pd
from scipy import stats
import xgboost as xgb
import joblib

from configs.config import CONFIG_M5
from src.features_m5 import ALL_FEATURE_COLS
from src.train_m5 import build_walk_forward_folds, wilson_interval, block_bootstrap_net_returns
from src.calibrator_m5 import PlattCalibrator

SEED = 42
TABLES_DIR = "results/m5/tables"
MODELS_DIR = "models/m5"
DOCS_DIR = "docs/m5"


def compute_metrics_for_rule(p_cal, y, fwd, q_lo, q_hi, min_margin=0.01):
    # Enforce margin guard: q_lo <= 0.50 - min_margin, q_hi >= 0.50 + min_margin
    eff_q_lo = min(q_lo, 0.50 - min_margin)
    eff_q_hi = max(q_hi, 0.50 + min_margin)
    
    is_up = p_cal >= eff_q_hi
    is_down = p_cal <= eff_q_lo
    n_up = int(np.sum(is_up))
    n_down = int(np.sum(is_down))
    n_sig = n_up + n_down
    total_bars = len(p_cal)
    cov = float(n_sig / total_bars) if total_bars > 0 else 0.0
    
    if n_sig == 0:
        return {
            "q_lo": eff_q_lo, "q_hi": eff_q_hi, "n_up": 0, "n_down": 0, "n_signals": 0,
            "coverage": 0.0, "accuracy": 0.0, "ci_lo": 0.0, "ci_hi": 0.0,
            "acc_up": 0.0, "acc_down": 0.0, "balanced_acc": 0.0,
            "net_0bps": 0.0, "net_1bps": 0.0, "net_2bps": 0.0, "net_5bps": 0.0
        }
        
    hits_up = int(np.sum(y[is_up] == 1))
    hits_down = int(np.sum(y[is_down] == 0))
    total_hits = hits_up + hits_down
    
    acc = float(total_hits / n_sig)
    ci_lo, ci_hi = wilson_interval(total_hits, n_sig)
    acc_up = float(hits_up / n_up) if n_up > 0 else 0.0
    acc_down = float(hits_down / n_down) if n_down > 0 else 0.0
    bal_acc = 0.5 * (acc_up + acc_down)
    
    sig_mask = is_up | is_down
    sig_dirs = np.where(is_up, 1, np.where(is_down, -1, 0))
    
    net_rets = block_bootstrap_net_returns(fwd[sig_mask], sig_dirs[sig_mask], costs_bps=[0, 1, 2, 5])
    
    return {
        "q_lo": eff_q_lo,
        "q_hi": eff_q_hi,
        "n_up": n_up,
        "n_down": n_down,
        "n_signals": n_sig,
        "coverage": cov,
        "accuracy": acc,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "acc_up": acc_up,
        "acc_down": acc_down,
        "balanced_acc": bal_acc,
        "net_0bps": float(net_rets["net_return_0bps"]["mean_bps"]),
        "net_1bps": float(net_rets["net_return_1bps"]["mean_bps"]),
        "net_2bps": float(net_rets["net_return_2bps"]["mean_bps"]),
        "net_5bps": float(net_rets["net_return_5bps"]["mean_bps"])
    }


def main():
    os.makedirs(TABLES_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(DOCS_DIR, exist_ok=True)
    
    print("==================================================")
    print("SECTION B: COVERAGE-TARGETED RULE OPTIMIZATION")
    print("==================================================")
    
    # 1. Load Data
    df = pd.read_parquet(CONFIG_M5["paths"]["processed_features"]).sort_index()
    df_valid = df.dropna(subset=["y"] + ALL_FEATURE_COLS).copy()
    all_dates = sorted(list(set(df_valid["date"])))
    
    # Pre-holdout is all dates except last 126 sessions
    holdout_dates = set(all_dates[-126:])
    pre_holdout_dates = set(all_dates[:-126])
    
    df_pre_holdout = df_valid[df_valid["date"].isin(pre_holdout_dates)].copy()
    df_holdout = df_valid[df_valid["date"].isin(holdout_dates)].copy()
    
    print(f"Pre-Holdout bars: {len(df_pre_holdout):,} across {len(pre_holdout_dates)} sessions")
    print(f"Holdout bars:     {len(df_holdout):,} across {len(holdout_dates)} sessions")
    
    # 2. Walk-Forward OOF Predictions on Pre-Holdout
    folds = build_walk_forward_folds(df_pre_holdout, initial_train_years=2, quarter_sessions=63, gap_bars=8)
    chosen_thresh = 0.0002 # 2 bps
    
    oof_preds = []
    oof_y = []
    oof_fwd = []
    oof_indices = []
    
    print(f"Generating OOF predictions across {len(folds)} folds...")
    for fold_i, f in enumerate(folds):
        train_sub = df_pre_holdout.loc[f["train_idx"]]
        val_sub = df_pre_holdout.loc[f["val_idx"]]
        
        train_decisive = train_sub[train_sub["fwd"].abs() > chosen_thresh]
        
        clf = xgb.XGBClassifier(
            n_estimators=120, max_depth=4, learning_rate=0.03, subsample=0.8,
            tree_method="hist", random_state=SEED, n_jobs=4
        )
        clf.fit(train_decisive[ALL_FEATURE_COLS].values, train_decisive["y"].values)
        p_val = clf.predict_proba(val_sub[ALL_FEATURE_COLS].values)[:, 1]
        
        oof_preds.append(p_val)
        oof_y.append(val_sub["y"].values)
        oof_fwd.append(val_sub["fwd"].values)
        oof_indices.append(val_sub.index)
        
    p_all_oof = np.concatenate(oof_preds)
    y_all_oof = np.concatenate(oof_y)
    fwd_all_oof = np.concatenate(oof_fwd)
    
    # Calibrate OOF
    calibrator = PlattCalibrator()
    calibrator.fit(p_all_oof, y_all_oof)
    p_cal_oof = calibrator.predict_proba(p_all_oof)
    
    # 3. Train Pre-Refit Model for Research Holdout
    train_pre_all = df_pre_holdout[df_pre_holdout["fwd"].abs() > chosen_thresh]
    pre_refit_clf = xgb.XGBClassifier(
        n_estimators=120, max_depth=4, learning_rate=0.03, subsample=0.8,
        tree_method="hist", random_state=SEED, n_jobs=4
    )
    pre_refit_clf.fit(train_pre_all[ALL_FEATURE_COLS].values, train_pre_all["y"].values)
    
    raw_p_holdout = pre_refit_clf.predict_proba(df_holdout[ALL_FEATURE_COLS].values)[:, 1]
    p_cal_holdout = calibrator.predict_proba(raw_p_holdout)
    y_holdout = df_holdout["y"].values
    fwd_holdout = df_holdout["fwd"].values
    
    # 4. Coverage Trade-off Table (5%, 10%, 20%, 30%, 40%, 50%)
    c_targets = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50]
    tradeoff_rows = []
    
    print("\n--- Coverage Trade-Off Sweep (OOF & Holdout) ---")
    for c in c_targets:
        q_lo_oof = float(np.quantile(p_cal_oof, c / 2.0))
        q_hi_oof = float(np.quantile(p_cal_oof, 1.0 - (c / 2.0)))
        
        m_oof = compute_metrics_for_rule(p_cal_oof, y_all_oof, fwd_all_oof, q_lo_oof, q_hi_oof, min_margin=0.01)
        m_hold = compute_metrics_for_rule(p_cal_holdout, y_holdout, fwd_holdout, q_lo_oof, q_hi_oof, min_margin=0.01)
        
        tradeoff_rows.append({
            "target_coverage": c,
            "sample": "OOF (Pre-Holdout)",
            "q_lo": m_oof["q_lo"],
            "q_hi": m_oof["q_hi"],
            "n_up": m_oof["n_up"],
            "n_down": m_oof["n_down"],
            "n_signals": m_oof["n_signals"],
            "achieved_coverage": m_oof["coverage"],
            "accuracy": m_oof["accuracy"],
            "wilson_ci_lo": m_oof["ci_lo"],
            "wilson_ci_hi": m_oof["ci_hi"],
            "acc_up": m_oof["acc_up"],
            "acc_down": m_oof["acc_down"],
            "balanced_accuracy": m_oof["balanced_acc"],
            "net_ret_0bps": m_oof["net_0bps"],
            "net_ret_1bps": m_oof["net_1bps"],
            "net_ret_2bps": m_oof["net_2bps"],
            "net_ret_5bps": m_oof["net_5bps"]
        })
        
        tradeoff_rows.append({
            "target_coverage": c,
            "sample": "Holdout (Pre-Refit Model, Diff Calibrator)",
            "q_lo": m_oof["q_lo"],
            "q_hi": m_oof["q_hi"],
            "n_up": m_hold["n_up"],
            "n_down": m_hold["n_down"],
            "n_signals": m_hold["n_signals"],
            "achieved_coverage": m_hold["coverage"],
            "accuracy": m_hold["accuracy"],
            "wilson_ci_lo": m_hold["ci_lo"],
            "wilson_ci_hi": m_hold["ci_hi"],
            "acc_up": m_hold["acc_up"],
            "acc_down": m_hold["acc_down"],
            "balanced_accuracy": m_hold["balanced_acc"],
            "net_ret_0bps": m_hold["net_0bps"],
            "net_ret_1bps": m_hold["net_1bps"],
            "net_ret_2bps": m_hold["net_2bps"],
            "net_ret_5bps": m_hold["net_5bps"]
        })
        
        print(f"Target c={c:.2f} | OOF Acc: {m_oof['accuracy']:.2%} (Signals: {m_oof['n_signals']}, Cov: {m_oof['coverage']:.1%}) | Holdout Acc: {m_hold['accuracy']:.2%} (Signals: {m_hold['n_signals']}, Cov: {m_hold['coverage']:.1%})")

    df_tradeoff = pd.DataFrame(tradeoff_rows)
    df_tradeoff.to_csv(os.path.join(TABLES_DIR, "coverage_tradeoff.csv"), index=False)

    # 5. Evaluate Rule A on Target c = 0.30
    c_target = 0.30
    q_lo_rule_a = float(np.quantile(p_cal_oof, c_target / 2.0))
    q_hi_rule_a = float(np.quantile(p_cal_oof, 1.0 - (c_target / 2.0)))
    
    # Margin guard: min_margin = 0.01 -> q_lo <= 0.49, q_hi >= 0.51
    eff_q_lo_a = min(q_lo_rule_a, 0.49)
    eff_q_hi_a = max(q_hi_rule_a, 0.51)
    
    print("\n--- Testing Rule A (Primary) on Last 20 Completed Sessions ---")
    # Load production model bundle
    bundle = joblib.load(CONFIG_M5["paths"]["model_path"])
    prod_clf = bundle["primary_model"]
    prod_calib = bundle["calibrator"]
    
    # Last 20 completed sessions in dataset
    last_20_dates = all_dates[-20:]
    df_last_20 = df_valid[df_valid["date"].isin(last_20_dates)].copy()
    
    raw_p_20 = prod_clf.predict_proba(df_last_20[ALL_FEATURE_COLS].values)[:, 1]
    p_cal_20 = prod_calib.predict_proba(raw_p_20)
    
    is_up_20 = p_cal_20 >= eff_q_hi_a
    is_down_20 = p_cal_20 <= eff_q_lo_a
    n_up_20 = int(np.sum(is_up_20))
    n_down_20 = int(np.sum(is_down_20))
    n_sig_20 = n_up_20 + n_down_20
    total_bars_20 = len(p_cal_20)
    cov_last_20 = float(n_sig_20 / total_bars_20) if total_bars_20 > 0 else 0.0
    
    print(f"Rule A Thresholds: q_lo = {eff_q_lo_a:.5f}, q_hi = {eff_q_hi_a:.5f}")
    print(f"Last 20 Sessions Bars: {total_bars_20:,} | Up: {n_up_20}, Down: {n_down_20}, Total Signals: {n_sig_20}")
    print(f"Achieved Coverage on Last 20 Sessions: {cov_last_20:.2%} (Target: {c_target:.1%}, Fallback threshold: 15.0%)")
    
    # 6. Fallback Decision (Rule A vs Rule B)
    use_rule_b = (cov_last_20 < 0.15)
    active_rule_name = "Rule B (Dynamic Trailing Quantiles)" if use_rule_b else "Rule A (Fixed OOF Quantiles)"
    print(f"\nDECISION ON ACTIVE RULE: {active_rule_name}")
    print(f"Reason: Achieved coverage {cov_last_20:.2%} is {'BELOW' if use_rule_b else 'ABOVE'} 15.0% fallback threshold.")

    # 7. Compute Rule B if triggered
    # Rule B: thresholds are the (c/2) and (1 - c/2) quantiles of the final model's own p_cal over trailing 20 completed sessions
    # with min_margin guard.
    if use_rule_b:
        print("\nComputing Rule B Trailing Quantiles...")
        q_lo_rule_b = float(np.quantile(p_cal_20, c_target / 2.0))
        q_hi_rule_b = float(np.quantile(p_cal_20, 1.0 - (c_target / 2.0)))
        eff_q_lo_b = min(q_lo_rule_b, 0.49)
        eff_q_hi_b = max(q_hi_rule_b, 0.51)
        active_q_lo = eff_q_lo_b
        active_q_hi = eff_q_hi_b
        active_rule_code = "rule_b"
        print(f"Rule B Thresholds from Trailing 20 Sessions: q_lo = {eff_q_lo_b:.5f}, q_hi = {eff_q_hi_b:.5f}")
    else:
        active_q_lo = eff_q_lo_a
        active_q_hi = eff_q_hi_a
        active_rule_code = "rule_a"

    # 8. Confidence Tiers
    # STRONG: confidence >= 75th percentile of OOF signal confidence
    # WEAK: confidence < 25th percentile of OOF signal confidence
    # NORMAL: between 25th and 75th percentile
    is_up_oof = p_cal_oof >= active_q_hi
    is_down_oof = p_cal_oof <= active_q_lo
    oof_confs = np.where(is_up_oof, p_cal_oof, np.where(is_down_oof, 1.0 - p_cal_oof, np.nan))
    valid_sig_confs = oof_confs[~np.isnan(oof_confs)]
    
    tier_weak_thresh = float(np.quantile(valid_sig_confs, 0.25))
    tier_strong_thresh = float(np.quantile(valid_sig_confs, 0.75))
    
    print(f"\nCONFIDENCE TIERS (OOF Signal Confidence Quantiles):")
    print(f"WEAK:   Confidence < {tier_weak_thresh:.4f} (Bottom 25%)")
    print(f"NORMAL: {tier_weak_thresh:.4f} <= Confidence < {tier_strong_thresh:.4f} (Middle 50%)")
    print(f"STRONG: Confidence >= {tier_strong_thresh:.4f} (Top 25%)")

    # Tier Accuracies on OOF and Holdout
    tier_records = []
    for sample_name, p_vals, y_vals, fwd_vals in [("OOF (Pre-Holdout)", p_cal_oof, y_all_oof, fwd_all_oof),
                                                  ("Holdout (Pre-Refit)", p_cal_holdout, y_holdout, fwd_holdout)]:
        up_mask = p_vals >= active_q_hi
        down_mask = p_vals <= active_q_lo
        confs = np.where(up_mask, p_vals, np.where(down_mask, 1.0 - p_vals, 0.0))
        dirs = np.where(up_mask, 1, np.where(down_mask, 0, -1))
        
        for t_name, t_cond in [
            ("STRONG", confs >= tier_strong_thresh),
            ("NORMAL", (confs >= tier_weak_thresh) & (confs < tier_strong_thresh)),
            ("WEAK", (confs < tier_weak_thresh) & (confs >= 0.51))
        ]:
            for s_name, s_cond in [("Pooled", up_mask | down_mask), ("Up", up_mask), ("Down", down_mask)]:
                sub_mask = t_cond & s_cond
                n_t = int(np.sum(sub_mask))
                if n_t > 0:
                    hits = int(np.sum(y_vals[sub_mask] == dirs[sub_mask]))
                    acc = float(hits / n_t)
                    ci_lo, ci_hi = wilson_interval(hits, n_t)
                else:
                    hits, acc, ci_lo, ci_hi = 0, 0.0, 0.0, 0.0
                    
                tier_records.append({
                    "sample": sample_name,
                    "tier": t_name,
                    "side": s_name,
                    "n_signals": n_t,
                    "hits": hits,
                    "accuracy": acc,
                    "wilson_ci_lo": ci_lo,
                    "wilson_ci_hi": ci_hi
                })
                print(f"[{sample_name}] Tier {t_name:6s} | {s_name:6s}: {acc:.2%} ({hits}/{n_t}) [Wilson: {ci_lo:.1f}%, {ci_hi:.1f}%]")

    df_tiers = pd.DataFrame(tier_records)
    df_tiers.to_csv(os.path.join(TABLES_DIR, "confidence_tiers.csv"), index=False)

    # 9. Write Active Rule File (models/m5/rule_m5_v2.json)
    rule_v2_data = {
        "rule_version": 2,
        "rule_type": active_rule_code,
        "coverage_target": c_target,
        "achieved_coverage_oof": float(np.mean(is_up_oof | is_down_oof)),
        "achieved_coverage_last_20": cov_last_20,
        "q_lo": active_q_lo,
        "q_hi": active_q_hi,
        "min_margin": 0.01,
        "weak_threshold": tier_weak_thresh,
        "strong_threshold": tier_strong_thresh,
        "source": "owner decision 2026-11-16 (V4.1-COVERAGE)",
        "created_at": pd.Timestamp.now(tz="America/New_York").isoformat()
    }
    rule_v2_path = os.path.join(MODELS_DIR, "rule_m5_v2.json")
    with open(rule_v2_path, "w") as f:
        json.dump(rule_v2_data, f, indent=2)
    print(f"\nSaved new rule to {rule_v2_path}")

    # Active pointer
    active_rule_ptr = {
        "active_rule_version": 2,
        "active_rule_file": "rule_m5_v2.json",
        "description": f"Target 30% coverage via {active_rule_name}",
        "updated_at": pd.Timestamp.now(tz="America/New_York").isoformat()
    }
    active_ptr_path = os.path.join(MODELS_DIR, "active_rule.json")
    with open(active_ptr_path, "w") as f:
        json.dump(active_rule_ptr, f, indent=2)
    print(f"Saved active rule pointer to {active_ptr_path}")

    # 10. Cryptographic Hashes in MODEL_SHA256_m5.json
    with open(rule_v2_path, "rb") as rf:
        rule_v2_sha = hashlib.sha256(rf.read()).hexdigest()
    with open(active_ptr_path, "rb") as pf:
        active_ptr_sha = hashlib.sha256(pf.read()).hexdigest()

    sha_file = CONFIG_M5["paths"]["sha_path"]
    with open(sha_file, "r") as sf:
        sha_dict = json.load(sf)
        
    sha_dict["rule_m5_v2.json"] = rule_v2_sha
    sha_dict["active_rule.json"] = active_ptr_sha
    
    with open(sha_file, "w") as sf:
        json.dump(sha_dict, sf, indent=2)
    print(f"Updated {sha_file} with rule hashes:")
    print(json.dumps(sha_dict, indent=2))

    return {
        "active_rule": active_rule_name,
        "rule_v2": rule_v2_data,
        "df_tradeoff": df_tradeoff,
        "df_tiers": df_tiers
    }


if __name__ == "__main__":
    main()
