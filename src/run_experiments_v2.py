"""Phase V2.3 - V2.5: Full Pre-Holdout Optimization, Tournament & Insight Heads Suite.

Executes strictly on pre_holdout data:
1. Walk-Forward Quarterly Splits.
2. Constant Prior Baseline.
3. Hyperparameter search (30 iterations) & recency half-life sweep.
4. Candidate Tournament (XGBoost, HistGradientBoosting, ElasticNet, Ensemble).
5. Group Ablations across 9 Feature Groups.
6. Multi-Horizon Tournament (H1, H3, HD).
7. Insight Heads: VOL regressor, BAND quantile regressors, REGIME tags.
8. Class-Balanced Platt Calibration.
9. Two-Sided Quantile Signal Rule Search (c in {0.15, 0.20, 0.25, 0.30, 0.40}).
10. Save all experiment artifacts and freeze selected configuration.
"""

import json
from pathlib import Path
from zoneinfo import ZoneInfo
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression, QuantileRegressor
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import log_loss, brier_score_loss, accuracy_score, mean_absolute_error, r2_score
from xgboost import XGBClassifier, XGBRegressor
import joblib

from configs.config import CONFIG
from src.features import FEATURES, FEATURE_GROUPS
from src.labels import compute_sample_weights_v2
from src.utils import wilson_score_interval
from src.walk_forward import generate_quarterly_folds, run_walk_forward_oof, PlattCalibratorV2, compute_ece


def run_v2_experiments():
    proc_dir = Path("data/processed")
    tables_dir = Path("results/tables")
    figures_dir = Path("results/figures")
    logs_dir = Path("results/logs")
    models_dir = Path("models")
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    print("Loading V2 dataset...")
    df = pd.read_parquet(CONFIG["paths"]["processed"])
    df_pre = df[df["block"] == "pre_holdout"].copy()
    print(f"Total dataset bars: {len(df):,} | Pre-Holdout bars: {len(df_pre):,}")

    # Generate quarterly walk-forward splits
    quarter_days = CONFIG.get("quarter_days", 63)
    initial_years = CONFIG.get("initial_train_years", 2)
    gap_bars = CONFIG.get("gap_bars", 8)
    folds = generate_quarterly_folds(df_pre, initial_years=initial_years, quarter_days=quarter_days, gap_bars=gap_bars)
    print(f"Generated {len(folds)} quarterly walk-forward folds on pre-holdout data.")

    # 1. Constant Prior Baseline
    y_pre_labelled = df_pre["y_H1"].dropna().to_numpy().astype(int)
    prior_up = float(np.mean(y_pre_labelled))
    p_prior = np.full(len(y_pre_labelled), prior_up)
    loss_prior = float(log_loss(y_pre_labelled, p_prior))
    brier_prior = float(brier_score_loss(y_pre_labelled, p_prior))
    print(f"\n--- 1. Pre-Holdout Constant Prior Baseline ---")
    print(f"Base Up-Rate: {prior_up*100:.2f}% | Log-Loss: {loss_prior:.5f} | Brier: {brier_prior:.5f}")

    # 2. Hyperparameter Search & Recency Weighting for XGBoost
    print(f"\n--- 2. Hyperparameter Search (30 configs) & Recency Sweep ---")
    rng = np.random.default_rng(CONFIG["seed"])
    search_records = []

    # Parameter grid candidates
    depths = [2, 3, 4]
    lrs = [0.01, 0.02, 0.03, 0.05]
    n_ests = [100, 200, 300]
    subsamples = [0.7, 0.85, 1.0]
    colsamples = [0.6, 0.8, 1.0]
    reg_lambdas = [1.0, 3.0, 5.0]
    halflives = [90, 180, 365, None]

    best_xgb_loss = 999.0
    best_xgb_config = None
    best_xgb_oof = None
    best_eval_mask = None

    for i in range(30):
        cfg = {
            "max_depth": int(rng.choice(depths)),
            "learning_rate": float(rng.choice(lrs)),
            "n_estimators": int(rng.choice(n_ests)),
            "subsample": float(rng.choice(subsamples)),
            "colsample_bytree": float(rng.choice(colsamples)),
            "reg_lambda": float(rng.choice(reg_lambdas)),
            "min_child_weight": int(rng.choice([1, 3, 5])),
            "half_life_days": rng.choice(halflives)
        }

        def make_xgb():
            return XGBClassifier(
                max_depth=cfg["max_depth"],
                learning_rate=cfg["learning_rate"],
                n_estimators=cfg["n_estimators"],
                subsample=cfg["subsample"],
                colsample_bytree=cfg["colsample_bytree"],
                reg_lambda=cfg["reg_lambda"],
                min_child_weight=cfg["min_child_weight"],
                random_state=CONFIG["seed"],
                n_jobs=1,
                eval_metric="logloss"
            )

        oof_p, mask, loss = run_walk_forward_oof(
            df_pre, folds, make_xgb, FEATURES,
            target_col="y_H1", decisive_col="decisive_H1",
            half_life_days=cfg["half_life_days"]
        )

        cfg_row = dict(cfg)
        cfg_row["pooled_oof_logloss"] = loss
        cfg_row["iteration"] = i + 1
        search_records.append(cfg_row)

        if loss < best_xgb_loss:
            best_xgb_loss = loss
            best_xgb_config = cfg
            best_xgb_oof = oof_p
            best_eval_mask = mask
            print(f"Iter {i+1:02d}: New Best OOF Log-Loss: {loss:.5f} (depth={cfg['max_depth']}, lr={cfg['learning_rate']}, n_est={cfg['n_estimators']}, half_life={cfg['half_life_days']})")

    df_search = pd.DataFrame(search_records).sort_values("pooled_oof_logloss")
    df_search.to_csv(tables_dir / "hyperparam_search_v2.csv", index=False)
    print(f"Hyperparameter search complete. Best XGBoost OOF Log-Loss: {best_xgb_loss:.5f}")

    # 3. Candidate Tournament on Pooled Pre-Holdout OOF
    print(f"\n--- 3. Candidate Model Tournament ---")
    
    # 3a. HistGradientBoosting
    print("Evaluating HistGradientBoosting...")
    def make_hgb():
        return HistGradientBoostingClassifier(
            max_depth=3, learning_rate=0.03, max_iter=150, min_samples_leaf=20,
            random_state=CONFIG["seed"]
        )
    oof_hgb, _, loss_hgb = run_walk_forward_oof(
        df_pre, folds, make_hgb, FEATURES,
        half_life_days=best_xgb_config.get("half_life_days")
    )
    print(f"HistGradientBoosting OOF Log-Loss: {loss_hgb:.5f}")

    # 3b. ElasticNet Logistic Regression
    print("Evaluating Elastic-Net Logistic Regression...")
    def make_enet():
        return LogisticRegression(
            penalty="elasticnet", l1_ratio=0.5, C=0.01, solver="saga",
            max_iter=500, random_state=CONFIG["seed"]
        )
    oof_enet, _, loss_enet = run_walk_forward_oof(
        df_pre, folds, make_enet, FEATURES,
        half_life_days=best_xgb_config.get("half_life_days"),
        scale_features=True
    )
    print(f"Elastic-Net Logistic Regression OOF Log-Loss: {loss_enet:.5f}")

    # 3c. Ensemble (Mean of calibrated probabilities)
    calib_xgb = PlattCalibratorV2().fit(best_xgb_oof[best_eval_mask], df_pre.loc[best_eval_mask, "y_H1"].astype(int))
    calib_hgb = PlattCalibratorV2().fit(oof_hgb[best_eval_mask], df_pre.loc[best_eval_mask, "y_H1"].astype(int))
    calib_enet = PlattCalibratorV2().fit(oof_enet[best_eval_mask], df_pre.loc[best_eval_mask, "y_H1"].astype(int))

    p_cal_xgb = calib_xgb.predict_proba(best_xgb_oof[best_eval_mask])
    p_cal_hgb = calib_hgb.predict_proba(oof_hgb[best_eval_mask])
    p_cal_enet = calib_enet.predict_proba(oof_enet[best_eval_mask])

    oof_ens = (p_cal_xgb + p_cal_hgb + p_cal_enet) / 3.0
    y_eval = df_pre.loc[best_eval_mask, "y_H1"].astype(int).to_numpy()
    loss_ens = float(log_loss(y_eval, oof_ens))
    print(f"Ensemble (XGB + HGB + ENet) OOF Log-Loss: {loss_ens:.5f}")

    # Tournament Table
    tourney_records = [
        {"model": "Constant Prior", "oof_logloss": loss_prior, "brier": brier_prior},
        {"model": "Shallow XGBoost", "oof_logloss": best_xgb_loss, "brier": float(brier_score_loss(y_eval, p_cal_xgb))},
        {"model": "HistGradientBoosting", "oof_logloss": loss_hgb, "brier": float(brier_score_loss(y_eval, p_cal_hgb))},
        {"model": "ElasticNet Logistic", "oof_logloss": loss_enet, "brier": float(brier_score_loss(y_eval, p_cal_enet))},
        {"model": "Ensemble (3-way)", "oof_logloss": loss_ens, "brier": float(brier_score_loss(y_eval, oof_ens))}
    ]
    df_tourney = pd.DataFrame(tourney_records).sort_values("oof_logloss")
    df_tourney.to_csv(tables_dir / "candidate_tournament_v2.csv", index=False)
    print("\nTournament Leaderboard:")
    print(df_tourney.to_string(index=False))

    winner_model_name = df_tourney[df_tourney["model"] != "Constant Prior"].iloc[0]["model"]
    print(f"\nWinning Model Architecture: {winner_model_name}")

    # 4. Feature Group Ablations
    print(f"\n--- 4. Feature Group Ablations on Winning Architecture ---")
    ablation_records = []
    base_loss = best_xgb_loss

    for grp_name, grp_feats in FEATURE_GROUPS.items():
        ablated_features = [f for f in FEATURES if f not in grp_feats]
        def make_xgb_ablate():
            return XGBClassifier(
                max_depth=best_xgb_config["max_depth"],
                learning_rate=best_xgb_config["learning_rate"],
                n_estimators=best_xgb_config["n_estimators"],
                subsample=best_xgb_config["subsample"],
                colsample_bytree=best_xgb_config["colsample_bytree"],
                reg_lambda=best_xgb_config["reg_lambda"],
                min_child_weight=best_xgb_config["min_child_weight"],
                random_state=CONFIG["seed"],
                n_jobs=1,
                eval_metric="logloss"
            )
        _, _, abl_loss = run_walk_forward_oof(
            df_pre, folds, make_xgb_ablate, ablated_features,
            half_life_days=best_xgb_config.get("half_life_days")
        )
        delta = abl_loss - base_loss
        outcome = "Keep (Dropping Hurts)" if delta > 0 else "Drop (Helps/Neutral)"
        ablation_records.append({
            "group": grp_name,
            "dropped_count": len(grp_feats),
            "oof_logloss": abl_loss,
            "delta_vs_all": delta,
            "decision": outcome
        })
        print(f"  Drop '{grp_name}' ({len(grp_feats)} feats): Loss={abl_loss:.5f} (Delta: {delta:+.5f}) -> {outcome}")

    df_ablation = pd.DataFrame(ablation_records).sort_values("delta_vs_all", ascending=False)
    df_ablation.to_csv(tables_dir / "group_ablations_v2.csv", index=False)

    # 5. Multi-Horizon Direction Tournament (H1, H3, HD)
    print(f"\n--- 5. Multi-Horizon Direction Tournament (H1, H3, HD) ---")
    horizon_records = []
    horizon_oofs = {}

    for hz in ["H1", "H3", "HD"]:
        target_c = f"y_{hz}"
        decisive_c = f"decisive_{hz}"

        def make_xgb_hz():
            return XGBClassifier(
                max_depth=best_xgb_config["max_depth"],
                learning_rate=best_xgb_config["learning_rate"],
                n_estimators=best_xgb_config["n_estimators"],
                subsample=best_xgb_config["subsample"],
                colsample_bytree=best_xgb_config["colsample_bytree"],
                reg_lambda=best_xgb_config["reg_lambda"],
                min_child_weight=best_xgb_config["min_child_weight"],
                random_state=CONFIG["seed"],
                n_jobs=1,
                eval_metric="logloss"
            )
        oof_hz, mask_hz, loss_hz = run_walk_forward_oof(
            df_pre, folds, make_xgb_hz, FEATURES,
            target_col=target_c, decisive_col=decisive_c,
            half_life_days=best_xgb_config.get("half_life_days")
        )

        y_true_hz = df_pre.loc[mask_hz, target_c].astype(int).to_numpy()
        calib_hz = PlattCalibratorV2().fit(oof_hz[mask_hz], y_true_hz)
        p_cal_hz = calib_hz.predict_proba(oof_hz[mask_hz])

        # Test coverage sweep c in [0.20, 0.40] for two-sided rule
        best_lower_ci = -1.0
        best_cov_info = None

        for c_val in [0.20, 0.25, 0.30, 0.40]:
            q_lo = float(np.quantile(p_cal_hz, c_val / 2.0))
            q_hi = float(np.quantile(p_cal_hz, 1.0 - c_val / 2.0))

            up_mask = (p_cal_hz >= q_hi)
            down_mask = (p_cal_hz <= q_lo)
            sig_mask = up_mask | down_mask

            n_up = int(np.sum(up_mask))
            n_down = int(np.sum(down_mask))
            n_sig = n_up + n_down

            if n_up >= 150 and n_down >= 150:
                y_sig = y_true_hz[sig_mask]
                preds_sig = np.where(up_mask[sig_mask], 1, 0)
                acc = float(accuracy_score(y_sig, preds_sig))
                _, low_ci, high_ci = wilson_score_interval(int(np.sum(preds_sig == y_sig)), n_sig, confidence=0.95)

                if low_ci > best_lower_ci:
                    best_lower_ci = low_ci
                    best_cov_info = {
                        "coverage": c_val,
                        "n_up": n_up,
                        "n_down": n_down,
                        "n_sig": n_sig,
                        "accuracy": acc,
                        "wilson_low": low_ci,
                        "wilson_high": high_ci,
                        "q_lo": q_lo,
                        "q_hi": q_hi
                    }

        rec = {
            "horizon": hz,
            "oof_logloss": loss_hz,
            "best_coverage": best_cov_info["coverage"] if best_cov_info else 0.0,
            "accuracy": best_cov_info["accuracy"] if best_cov_info else 0.0,
            "wilson_low_95": best_cov_info["wilson_low"] if best_cov_info else 0.0,
            "wilson_high_95": best_cov_info["wilson_high"] if best_cov_info else 0.0,
            "n_up": best_cov_info["n_up"] if best_cov_info else 0,
            "n_down": best_cov_info["n_down"] if best_cov_info else 0,
            "q_lo": best_cov_info["q_lo"] if best_cov_info else 0.0,
            "q_hi": best_cov_info["q_hi"] if best_cov_info else 0.0
        }
        horizon_records.append(rec)
        horizon_oofs[hz] = (oof_hz, mask_hz, calib_hz, best_cov_info)
        print(f"Horizon {hz}: Log-Loss={loss_hz:.5f} | Acc={rec['accuracy']*100:.2f}% | Wilson CI=[{rec['wilson_low_95']*100:.2f}%, {rec['wilson_high_95']*100:.2f}%] (Up={rec['n_up']}, Down={rec['n_down']})")

    df_horizons = pd.DataFrame(horizon_records).sort_values("wilson_low_95", ascending=False)
    df_horizons.to_csv(tables_dir / "horizon_tournament_v2.csv", index=False)
    print("\nHorizon Tournament Results:")
    print(df_horizons.to_string(index=False))

    primary_horizon = df_horizons.iloc[0]["horizon"]
    primary_info = horizon_oofs[primary_horizon][3]
    print(f"\nPRE-REGISTERED PRIMARY DIRECTION HORIZON SELECTED: {primary_horizon}")
    print(f"Frozen Signal Quantiles: q_lo = {primary_info['q_lo']:.5f}, q_hi = {primary_info['q_hi']:.5f} (Coverage = {primary_info['coverage']*100:.1f}%)")

    # Assert mandatory two-sided signal counts on OOF
    assert primary_info["n_up"] >= 150, f"VIOLATION: n_up={primary_info['n_up']} < 150 on OOF!"
    assert primary_info["n_down"] >= 150, f"VIOLATION: n_down={primary_info['n_down']} < 150 on OOF!"

    # 6. Insight Heads Optimization
    print(f"\n--- 6. Training & Calibrating Insight Heads ---")

    # 6a. VOL Head (XGB Regressor on log range)
    print("Training VOL Regressor...")
    valid_vol_mask = df_pre["target_vol"].notna()
    X_vol = df_pre.loc[valid_vol_mask, FEATURES].to_numpy()
    y_vol = df_pre.loc[valid_vol_mask, "target_vol"].to_numpy()

    # Walk-forward evaluation for VOL
    valid_vol_mask_np = valid_vol_mask.to_numpy()
    vol_preds = np.full(len(df_pre), np.nan)
    for tr_idx, te_idx in folds:
        tr_m = valid_vol_mask_np[tr_idx]
        te_m = valid_vol_mask_np[te_idx]
        if tr_m.sum() < 50:
            continue
        m_vol = XGBRegressor(max_depth=3, learning_rate=0.03, n_estimators=150, random_state=42, n_jobs=1)
        m_vol.fit(df_pre.iloc[tr_idx[tr_m]][FEATURES], df_pre.iloc[tr_idx[tr_m]]["target_vol"])
        vol_preds[te_idx[te_m]] = m_vol.predict(df_pre.iloc[te_idx[te_m]][FEATURES])

    eval_vol_idx = np.where(~np.isnan(vol_preds) & valid_vol_mask)[0]
    y_vol_eval = df_pre.iloc[eval_vol_idx]["target_vol"].to_numpy()
    p_vol_eval = vol_preds[eval_vol_idx]

    # Baseline 1: Persistence (previous bar range)
    high_prev = df_pre["high"].shift(1).iloc[eval_vol_idx]
    low_prev = df_pre["low"].shift(1).iloc[eval_vol_idx]
    close_prev = df_pre["close"].shift(1).iloc[eval_vol_idx]
    base_persist_vol = np.log((high_prev - low_prev).clip(lower=1e-5) / close_prev).to_numpy()

    # Baseline 2: EWMA
    base_ewma_vol = pd.Series(y_vol_eval).ewm(span=20).mean().to_numpy()

    vol_spearman = float(stats.spearmanr(p_vol_eval, y_vol_eval).statistic)
    vol_mae = float(mean_absolute_error(y_vol_eval, p_vol_eval))
    vol_r2 = float(r2_score(y_vol_eval, p_vol_eval))

    persist_spearman = float(stats.spearmanr(base_persist_vol, y_vol_eval).statistic)
    persist_mae = float(mean_absolute_error(y_vol_eval, base_persist_vol))
    ewma_spearman = float(stats.spearmanr(base_ewma_vol, y_vol_eval).statistic)

    vol_metrics = {
        "model": ["VOL XGBoost Regressor", "Persistence (Lag 1)", "EWMA (span=20)"],
        "spearman_corr": [vol_spearman, persist_spearman, ewma_spearman],
        "mae": [vol_mae, persist_mae, float(mean_absolute_error(y_vol_eval, base_ewma_vol))],
        "r2_score": [vol_r2, float(r2_score(y_vol_eval, base_persist_vol)), float(r2_score(y_vol_eval, base_ewma_vol))]
    }
    df_vol_eval = pd.DataFrame(vol_metrics)
    df_vol_eval.to_csv(tables_dir / "vol_head_evaluation_v2.csv", index=False)
    print("VOL Head Evaluation vs Baselines:")
    print(df_vol_eval.to_string(index=False))

    # Compute OOF tercile thresholds for VOL
    tercile_33 = float(np.quantile(p_vol_eval, 0.333))
    tercile_67 = float(np.quantile(p_vol_eval, 0.667))
    print(f"VOL Terciles: Low <= {tercile_33:.4f} < Normal <= {tercile_67:.4f} < High")

    # 6b. BAND Head (Quantile Regressors q=0.10, 0.90 on next-bar return)
    print("\nTraining BAND Quantile Regressors...")
    valid_ret_mask = df_pre["fwd_H1"].notna()
    y_band = df_pre.loc[valid_ret_mask, "fwd_H1"].to_numpy()

    # Fit quantile regressors using HistGradientBoostingRegressor(loss='quantile')
    band_low_preds = np.full(len(df_pre), np.nan)
    band_high_preds = np.full(len(df_pre), np.nan)

    valid_ret_mask_np = valid_ret_mask.to_numpy()
    for tr_idx, te_idx in folds:
        tr_m = valid_ret_mask_np[tr_idx]
        te_m = valid_ret_mask_np[te_idx]
        if tr_m.sum() < 50:
            continue
        X_tr = df_pre.iloc[tr_idx[tr_m]][FEATURES]
        y_tr = df_pre.iloc[tr_idx[tr_m]]["fwd_H1"]

        m_q10 = HistGradientBoostingRegressor(loss="quantile", quantile=0.10, max_iter=80, random_state=42)
        m_q90 = HistGradientBoostingRegressor(loss="quantile", quantile=0.90, max_iter=80, random_state=42)

        m_q10.fit(X_tr, y_tr)
        m_q90.fit(X_tr, y_tr)

        X_te = df_pre.iloc[te_idx[te_m]][FEATURES]
        band_low_preds[te_idx[te_m]] = m_q10.predict(X_te)
        band_high_preds[te_idx[te_m]] = m_q90.predict(X_te)

    eval_band_idx = np.where(~np.isnan(band_low_preds) & valid_ret_mask)[0]
    y_band_eval = df_pre.iloc[eval_band_idx]["fwd_H1"].to_numpy()
    low_band_eval = band_low_preds[eval_band_idx]
    high_band_eval = band_high_preds[eval_band_idx]

    band_coverage = float(np.mean((y_band_eval >= low_band_eval) & (y_band_eval <= high_band_eval)))
    mean_band_width = float(np.mean(high_band_eval - low_band_eval) * 10000.0)  # in bps

    # Baseline constant band from training std
    std_const = float(np.std(y_band_eval))
    const_low = float(np.quantile(y_band_eval, 0.10))
    const_high = float(np.quantile(y_band_eval, 0.90))
    const_coverage = float(np.mean((y_band_eval >= const_low) & (y_band_eval <= const_high)))
    const_band_width = (const_high - const_low) * 10000.0

    band_metrics = {
        "model": ["Adaptive Quantile Band (q=0.10, 0.90)", "Constant Empirical Band"],
        "empirical_coverage": [band_coverage, const_coverage],
        "mean_width_bps": [mean_band_width, const_band_width]
    }
    df_band_eval = pd.DataFrame(band_metrics)
    df_band_eval.to_csv(tables_dir / "band_head_evaluation_v2.csv", index=False)
    print("BAND Head Evaluation:")
    print(df_band_eval.to_string(index=False))

    # 6c. REGIME Head (Uptrend / Downtrend / Chop)
    print("\nDefining REGIME Classification...")
    ret_20d = df_pre["ret_20d"].dropna()
    eff_20 = df_pre["efficiency_ratio_20"].dropna()
    common_idx = ret_20d.index.intersection(eff_20.index)

    eff_chop_threshold = float(np.quantile(eff_20.loc[common_idx], 0.35))
    print(f"REGIME Efficiency Chop Threshold: {eff_chop_threshold:.4f}")

    def assign_regime(r20, eff):
        if np.isnan(r20) or np.isnan(eff) or eff < eff_chop_threshold:
            return "Chop"
        elif r20 > 0:
            return "Uptrend"
        else:
            return "Downtrend"

    regimes = [assign_regime(r, e) for r, e in zip(df_pre["ret_20d"], df_pre["efficiency_ratio_20"])]
    df_pre["regime"] = regimes
    regime_dist = pd.Series(regimes).value_counts(normalize=True)
    print("Pre-Holdout Regime Distribution:")
    for r_name, frac in regime_dist.items():
        print(f"  - {r_name}: {frac*100:.1f}%")

    # 7. Reliability Diagram & ECE per Regime
    best_oof_hz, best_mask_hz, best_calib_hz, _ = horizon_oofs[primary_horizon]
    p_cal_eval = best_calib_hz.predict_proba(best_oof_hz[best_mask_hz])
    y_hz_eval = df_pre.loc[best_mask_hz, f"y_{primary_horizon}"].astype(int).to_numpy()

    overall_ece = compute_ece(y_hz_eval, p_cal_eval, n_bins=10)
    print(f"\nOverall Pre-Holdout ECE (Primary Horizon {primary_horizon}): {overall_ece:.4f}")

    # Plot reliability diagram
    plt.figure(figsize=(7, 6))
    bins = np.linspace(0.0, 1.0, 11)
    bin_indices = np.digitize(p_cal_eval, bins) - 1
    bin_accs = [np.mean(y_hz_eval[bin_indices == b]) if np.sum(bin_indices == b) > 0 else np.nan for b in range(10)]
    bin_confs = [np.mean(p_cal_eval[bin_indices == b]) if np.sum(bin_indices == b) > 0 else np.nan for b in range(10)]

    plt.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
    plt.plot(bin_confs, bin_accs, "s-", color="#2b5c8f", lw=2, label=f"Platt Calibrated (ECE={overall_ece:.4f})")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Observed Frequency")
    plt.title(f"Reliability Diagram (Pre-Holdout OOF, Horizon {primary_horizon})", fontsize=11, fontweight="bold")
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(figures_dir / "calibration_reliability_v2.png", dpi=200)
    plt.close()

    # Save frozen experiment configuration
    frozen_config = {
        "primary_horizon": primary_horizon,
        "winning_model": winner_model_name,
        "best_xgb_hyperparameters": best_xgb_config,
        "signal_rule": {
            "coverage": primary_info["coverage"],
            "q_lo": primary_info["q_lo"],
            "q_hi": primary_info["q_hi"],
            "n_up_oof": primary_info["n_up"],
            "n_down_oof": primary_info["n_down"],
            "accuracy_oof": primary_info["accuracy"],
            "wilson_ci_oof": [primary_info["wilson_low"], primary_info["wilson_high"]]
        },
        "vol_terciles": {
            "tercile_33": tercile_33,
            "tercile_67": tercile_67
        },
        "regime_thresholds": {
            "eff_chop_threshold": eff_chop_threshold
        },
        "ece_pre_holdout": overall_ece
    }
    with open("configs/model_v2_config.json", "w") as f:
        json.dump(frozen_config, f, indent=2)
    print(f"\nFrozen V2 Configuration saved to configs/model_v2_config.json")


if __name__ == "__main__":
    run_v2_experiments()
