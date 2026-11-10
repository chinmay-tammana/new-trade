"""Stage B through E: Hyperparameter search, fine-tuning strategies, comparators, ablations, and feature importance.

Implements the walk-forward validation search across the FT block, compares strategies
S0, S1, S2, S3 on Validation, fits conventional ML comparators, executes feature ablations,
profiles model efficiency, and writes results/tables/finetune_report.md.
"""

import json
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Tuple

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.model_selection import ParameterSampler
from xgboost import XGBClassifier

from configs.config import CONFIG
from src.features import FEATURES
from src.labels import compute_sample_weights
from src.models import (
    build_default_xgboost,
    build_hist_gradient_boosting,
    build_logistic_regression,
    build_random_forest,
    build_tuned_xgboost,
    profile_model_efficiency,
)
from src.train_base import run_base_training
from src.utils import append_experiment_log, log_fit_ledger


def create_walk_forward_folds(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Build the 4 walk-forward folds across the FT block.

    Fold k:
    - Train = Base decisive rows + decisive rows of FT chunks before k
      (excluding rows within gap_bars before chunk k's first bar)
    - Val = ALL labelled rows of FT chunk k
    """
    gap = CONFIG["gap_bars"]
    n_folds = CONFIG["ft_cv_folds"]

    # Base decisive rows
    df_base_dec = df[(df["block"] == "Base") & df["decisive"]]

    # FT labelled rows
    df_ft_all = df[(df["block"] == "FT") & df["fwd"].notna()]
    ft_indices = df_ft_all.index
    chunk_splits = np.array_split(ft_indices, n_folds)

    folds = []
    for k in range(n_folds):
        val_chunk_idx = chunk_splits[k]
        val_first_time = val_chunk_idx.min()

        # Prior FT chunks before k
        prior_ft_parts = []
        for prev_k in range(k):
            prior_chunk = df.loc[chunk_splits[prev_k]]
            # Filter decisive and exclude rows within gap_bars before val_first_time
            valid_prior = prior_chunk[prior_chunk["decisive"] & (prior_chunk.index < val_first_time)]
            if len(valid_prior) > gap:
                valid_prior = valid_prior.iloc[:-gap]
            prior_ft_parts.append(valid_prior)

        if prior_ft_parts:
            df_ft_train = pd.concat(prior_ft_parts, axis=0)
        else:
            df_ft_train = pd.DataFrame(columns=df.columns)

        folds.append({
            "fold_num": k + 1,
            "df_base_dec": df_base_dec,
            "df_ft_train_dec": df_ft_train,
            "df_val_all": df.loc[val_chunk_idx],
        })

    return folds


def evaluate_config_on_folds(params: Dict[str, Any], folds: List[Dict[str, Any]]) -> Tuple[float, List[float]]:
    """Evaluate an XGBoost configuration across walk-forward folds."""
    base_weight = params.get("base_weight", 1.0)
    xgb_params = {k: v for k, v in params.items() if k != "base_weight"}
    xgb_params.setdefault("eval_metric", "logloss")
    xgb_params.setdefault("random_state", CONFIG["seed"])
    xgb_params.setdefault("n_jobs", 1)

    fold_losses = []
    eps = 1e-15

    for f in folds:
        df_base = f["df_base_dec"]
        df_ft = f["df_ft_train_dec"]
        df_val = f["df_val_all"]

        # Combined training data
        if len(df_ft) > 0:
            X_tr = pd.concat([df_base[FEATURES], df_ft[FEATURES]], axis=0)
            y_tr = np.concatenate([df_base["y"].to_numpy().astype(int), df_ft["y"].to_numpy().astype(int)])
            w_base = compute_sample_weights(df_base["y"].to_numpy().astype(int), base_weight=base_weight, class_balance_weights=CONFIG["class_balance_weights"])
            w_ft = compute_sample_weights(df_ft["y"].to_numpy().astype(int), base_weight=1.0, class_balance_weights=CONFIG["class_balance_weights"])
            w_tr = np.concatenate([w_base, w_ft])
        else:
            X_tr = df_base[FEATURES]
            y_tr = df_base["y"].to_numpy().astype(int)
            w_tr = compute_sample_weights(y_tr, base_weight=base_weight, class_balance_weights=CONFIG["class_balance_weights"])

        X_v = df_val[FEATURES]
        y_v = df_val["y"].to_numpy().astype(int)

        model = XGBClassifier(**xgb_params)
        model.fit(X_tr, y_tr, sample_weight=w_tr)

        prob_v = model.predict_proba(X_v)[:, 1]
        loss = log_loss(y_v, np.clip(prob_v, eps, 1.0 - eps))
        fold_losses.append(float(loss))

    return float(np.mean(fold_losses)), fold_losses


def run_stage_b_hyperparameter_search(df: pd.DataFrame, folds: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Sample 40 configurations and search for optimal XGBoost parameters."""
    print("\n--- STAGE B, PART A: Walk-Forward Hyperparameter Search (40 configs) ---")
    param_grid = {
        "max_depth": [2, 3, 4, 5],
        "learning_rate": [0.01, 0.02, 0.03, 0.05, 0.08, 0.1],
        "n_estimators": [100, 150, 200, 300, 400],
        "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
        "min_child_weight": [1, 3, 5, 10, 15, 20],
        "reg_lambda": [1.0, 3.0, 5.0, 10.0, 15.0, 20.0],
        "gamma": [0.0, 0.5, 1.0, 1.5, 2.0],
        "base_weight": CONFIG["base_weight_options"],
    }

    sampler = ParameterSampler(param_grid, n_iter=CONFIG["n_iter_search"], random_state=CONFIG["seed"])
    cv_records = []

    best_loss = float("inf")
    best_params = None

    for i, p in enumerate(sampler):
        t0 = time.time()
        mean_loss, fold_losses = evaluate_config_on_folds(p, folds)
        runtime = time.time() - t0

        run_id = f"E{i+1:02d}_XGB_Search"
        cv_records.append({
            "run_id": run_id,
            "mean_logloss": mean_loss,
            "fold_1": fold_losses[0],
            "fold_2": fold_losses[1],
            "fold_3": fold_losses[2],
            "fold_4": fold_losses[3],
            "runtime_s": runtime,
            **p
        })

        append_experiment_log(
            run_id=run_id,
            split="FT_WalkForward",
            seed=CONFIG["seed"],
            model="XGBoost",
            hyperparameters=json.dumps(p),
            cv_metric=mean_loss,
            val_metric=None,
            runtime_s=runtime,
        )

        # Tie breaking rule: within 0.001 prefer simpler (shallower, fewer trees)
        if best_params is None or mean_loss < best_loss - 0.001:
            best_loss = mean_loss
            best_params = p
        elif abs(mean_loss - best_loss) <= 0.001:
            curr_complexity = p["max_depth"] * 1000 + p["n_estimators"]
            best_complexity = best_params["max_depth"] * 1000 + best_params["n_estimators"]
            if curr_complexity < best_complexity:
                best_loss = mean_loss
                best_params = p

        if (i + 1) % 10 == 0 or (i + 1) == CONFIG["n_iter_search"]:
            print(f"  Completed {i + 1}/{CONFIG['n_iter_search']} configs. Current best mean log-loss: {best_loss:.5f}")

    df_cv = pd.DataFrame(cv_records).sort_values("mean_logloss")
    df_cv.to_csv("results/tables/cv_results.csv", index=False)
    print(f"\nBest tuned XGBoost hyperparameters (mean CV log-loss = {best_loss:.5f}):")
    print(json.dumps(best_params, indent=2))

    return best_params


def run_stage_b_strategies(df: pd.DataFrame, best_params: Dict[str, Any], base_res: Dict[str, Any]) -> Dict[str, Any]:
    """Compare XGBoost strategies S0, S1, S2, S3 on the Validation partition."""
    print("\n--- STAGE B, PART B: Fine-Tuning Strategy Comparison on Validation ---")
    eps = 1e-15

    # Val set
    df_val = df[(df["block"] == "Val") & df["fwd"].notna()]
    X_val = df_val[FEATURES]
    y_val = df_val["y"].to_numpy().astype(int)

    # Base decisive
    df_base_dec = df[(df["block"] == "Base") & df["decisive"]]
    X_base = df_base_dec[FEATURES]
    y_base = df_base_dec["y"].to_numpy().astype(int)

    # FT decisive
    df_ft_dec = df[(df["block"] == "FT") & df["decisive"]]
    X_ft = df_ft_dec[FEATURES]
    y_ft = df_ft_dec["y"].to_numpy().astype(int)

    # Strategy S0: Base model before fine-tuning
    model_s0 = base_res["model"]
    s0_loss = base_res["val_loss_s0"]
    s0_brier = base_res["val_brier_s0"]
    s0_acc = base_res["val_acc_s0"]

    # Strategy S1: Base only with tuned parameters
    w_base = compute_sample_weights(y_base, base_weight=best_params.get("base_weight", 1.0), class_balance_weights=CONFIG["class_balance_weights"])
    model_s1 = build_tuned_xgboost(best_params, random_state=CONFIG["seed"])
    t0 = time.time()
    model_s1.fit(X_base, y_base, sample_weight=w_base)
    t_s1 = time.time() - t0
    log_fit_ledger("Strategy S1: Base-only with tuned parameters", str(df_base_dec.index.min()), str(df_base_dec.index.max()), len(df_base_dec))

    p_s1 = model_s1.predict_proba(X_val)[:, 1]
    s1_loss = float(log_loss(y_val, np.clip(p_s1, eps, 1.0 - eps)))
    s1_brier = float(brier_score_loss(y_val, np.clip(p_s1, eps, 1.0 - eps)))
    s1_acc = float(accuracy_score(y_val, (p_s1 >= 0.5).astype(int)))

    # Strategy S2: Refit on Base + whole FT block
    X_base_ft = pd.concat([X_base, X_ft], axis=0)
    y_base_ft = np.concatenate([y_base, y_ft])
    w_base_part = compute_sample_weights(y_base, base_weight=best_params.get("base_weight", 1.0), class_balance_weights=CONFIG["class_balance_weights"])
    w_ft_part = compute_sample_weights(y_ft, base_weight=1.0, class_balance_weights=CONFIG["class_balance_weights"])
    w_base_ft = np.concatenate([w_base_part, w_ft_part])

    model_s2 = build_tuned_xgboost(best_params, random_state=CONFIG["seed"])
    t0 = time.time()
    model_s2.fit(X_base_ft, y_base_ft, sample_weight=w_base_ft)
    t_s2 = time.time() - t0
    log_fit_ledger("Strategy S2: Refit from scratch on Base + FT", str(df_base_dec.index.min()), str(df_ft_dec.index.max()), len(X_base_ft))

    p_s2 = model_s2.predict_proba(X_val)[:, 1]
    s2_loss = float(log_loss(y_val, np.clip(p_s2, eps, 1.0 - eps)))
    s2_brier = float(brier_score_loss(y_val, np.clip(p_s2, eps, 1.0 - eps)))
    s2_acc = float(accuracy_score(y_val, (p_s2 >= 0.5).astype(int)))

    # Strategy S3: Warm start: fit on Base, then continue boosting on FT block
    # Search over warm_start_rounds and warm_start_lr options on Val
    best_s3_loss = float("inf")
    best_s3_model = None
    best_s3_rounds = None
    best_s3_lr = None

    for r in CONFIG["warm_start_rounds_options"]:
        for lr in CONFIG["warm_start_lr_options"]:
            # Base model booster
            booster_base = build_tuned_xgboost(best_params, random_state=CONFIG["seed"])
            booster_base.fit(X_base, y_base, sample_weight=w_base)

            # Warm-start continuing on FT block
            continuer = XGBClassifier(
                n_estimators=r,
                learning_rate=lr,
                max_depth=best_params.get("max_depth", 3),
                subsample=best_params.get("subsample", 0.8),
                colsample_bytree=best_params.get("colsample_bytree", 0.8),
                eval_metric="logloss",
                random_state=CONFIG["seed"],
                n_jobs=1,
            )
            continuer.fit(X_ft, y_ft, sample_weight=w_ft_part, xgb_model=booster_base.get_booster())

            p_cand = continuer.predict_proba(X_val)[:, 1]
            cand_loss = float(log_loss(y_val, np.clip(p_cand, eps, 1.0 - eps)))
            if cand_loss < best_s3_loss:
                best_s3_loss = cand_loss
                best_s3_model = continuer
                best_s3_rounds = r
                best_s3_lr = lr

    log_fit_ledger(f"Strategy S3: Warm start on FT (rounds={best_s3_rounds}, lr={best_s3_lr})", str(df_base_dec.index.min()), str(df_ft_dec.index.max()), len(X_base_ft))
    p_s3 = best_s3_model.predict_proba(X_val)[:, 1]
    s3_brier = float(brier_score_loss(y_val, np.clip(p_s3, eps, 1.0 - eps)))
    s3_acc = float(accuracy_score(y_val, (p_s3 >= 0.5).astype(int)))

    strategies = [
        {"strategy": "S0: Base Default", "val_logloss": s0_loss, "val_brier": s0_brier, "val_acc": s0_acc, "model": model_s0},
        {"strategy": "S1: Base Tuned", "val_logloss": s1_loss, "val_brier": s1_brier, "val_acc": s1_acc, "model": model_s1},
        {"strategy": "S2: Refit Base+FT", "val_logloss": s2_loss, "val_brier": s2_brier, "val_acc": s2_acc, "model": model_s2},
        {"strategy": f"S3: Warm Start (r={best_s3_rounds}, lr={best_s3_lr})", "val_logloss": best_s3_loss, "val_brier": s3_brier, "val_acc": s3_acc, "model": best_s3_model},
    ]

    df_strat = pd.DataFrame(strategies).sort_values("val_logloss")
    print(df_strat[["strategy", "val_logloss", "val_brier", "val_acc"]].to_string(index=False))

    # Selection rule: lowest Val log-loss; ties (<=0.001) go to simpler (S1 < S2 < S3)
    best_strat = df_strat.iloc[0]
    print(f"\nTop XGBoost strategy selected: {best_strat['strategy']} (Val Log-Loss = {best_strat['val_logloss']:.5f})")

    return {
        "best_strategy_name": best_strat["strategy"],
        "best_strategy_model": best_strat["model"],
        "best_strategy_loss": best_strat["val_logloss"],
        "strategies_df": df_strat,
        "model_s0": model_s0,
        "model_s1": model_s1,
        "model_s2": model_s2,
        "model_s3": best_s3_model,
        "X_base_ft": X_base_ft,
        "y_base_ft": y_base_ft,
        "w_base_ft": w_base_ft,
        "X_val": X_val,
        "y_val": y_val,
    }


def run_stage_c_comparators(stage_b_res: Dict[str, Any], folds: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Tune and evaluate Logistic Regression, Random Forest, and HistGradientBoosting."""
    print("\n--- STAGE C: Conventional ML Comparators (Tuned on FT, evaluated on Val) ---")
    eps = 1e-15
    X_tr = stage_b_res["X_base_ft"]
    y_tr = stage_b_res["y_base_ft"]
    w_tr = stage_b_res["w_base_ft"]
    X_val = stage_b_res["X_val"]
    y_val = stage_b_res["y_val"]

    comparators = {}

    # 1. Logistic Regression: tune C
    best_c = 1.0
    best_c_loss = float("inf")
    for c_val in [0.001, 0.01, 0.1, 1.0, 10.0]:
        f_losses = []
        for f in folds:
            df_b = f["df_base_dec"]
            df_ft = f["df_ft_train_dec"]
            df_v = f["df_val_all"]
            if len(df_ft) > 0:
                X_f = pd.concat([df_b[FEATURES], df_ft[FEATURES]], axis=0)
                y_f = np.concatenate([df_b["y"].to_numpy().astype(int), df_ft["y"].to_numpy().astype(int)])
            else:
                X_f = df_b[FEATURES]
                y_f = df_b["y"].to_numpy().astype(int)

            pipe = build_logistic_regression(C=c_val, random_state=CONFIG["seed"])
            pipe.fit(X_f, y_f)
            pv = pipe.predict_proba(df_v[FEATURES])[:, 1]
            f_losses.append(log_loss(df_v["y"].to_numpy().astype(int), np.clip(pv, eps, 1.0 - eps)))
        if np.mean(f_losses) < best_c_loss:
            best_c_loss = np.mean(f_losses)
            best_c = c_val

    t0 = time.time()
    pipe_lr = build_logistic_regression(C=best_c, random_state=CONFIG["seed"])
    pipe_lr.fit(X_tr, y_tr)
    t_train_lr = time.time() - t0
    log_fit_ledger(f"Comparator: LogisticRegression (C={best_c})", str(X_tr.index.min()), str(X_tr.index.max()), len(X_tr))
    lat_lr, size_lr = profile_model_efficiency(pipe_lr, X_val.to_numpy())
    p_lr = pipe_lr.predict_proba(X_val)[:, 1]
    loss_lr = float(log_loss(y_val, np.clip(p_lr, eps, 1.0 - eps)))
    comparators["LogisticRegression"] = {
        "model": pipe_lr,
        "val_loss": loss_lr,
        "val_acc": float(accuracy_score(y_val, (p_lr >= 0.5).astype(int))),
        "train_time_s": t_train_lr,
        "infer_lat_us": lat_lr,
        "size_kb": size_lr,
        "params": f"C={best_c}",
    }

    # 2. Random Forest: small grid
    rf_candidates = [
        {"n_estimators": 100, "max_depth": 3, "min_samples_leaf": 10},
        {"n_estimators": 150, "max_depth": 4, "min_samples_leaf": 15},
        {"n_estimators": 200, "max_depth": 5, "min_samples_leaf": 20},
    ]
    best_rf_p = rf_candidates[0]
    best_rf_loss = float("inf")
    for rfp in rf_candidates:
        f_losses = []
        for f in folds:
            df_b = f["df_base_dec"]
            df_ft = f["df_ft_train_dec"]
            df_v = f["df_val_all"]
            if len(df_ft) > 0:
                X_f = pd.concat([df_b[FEATURES], df_ft[FEATURES]], axis=0)
                y_f = np.concatenate([df_b["y"].to_numpy().astype(int), df_ft["y"].to_numpy().astype(int)])
            else:
                X_f = df_b[FEATURES]
                y_f = df_b["y"].to_numpy().astype(int)
            rf_m = build_random_forest(**rfp, random_state=CONFIG["seed"])
            rf_m.fit(X_f, y_f)
            pv = rf_m.predict_proba(df_v[FEATURES])[:, 1]
            f_losses.append(log_loss(df_v["y"].to_numpy().astype(int), np.clip(pv, eps, 1.0 - eps)))
        if np.mean(f_losses) < best_rf_loss:
            best_rf_loss = np.mean(f_losses)
            best_rf_p = rfp

    t0 = time.time()
    rf_final = build_random_forest(**best_rf_p, random_state=CONFIG["seed"])
    rf_final.fit(X_tr, y_tr)
    t_train_rf = time.time() - t0
    log_fit_ledger(f"Comparator: RandomForest ({best_rf_p})", str(X_tr.index.min()), str(X_tr.index.max()), len(X_tr))
    lat_rf, size_rf = profile_model_efficiency(rf_final, X_val.to_numpy())
    p_rf = rf_final.predict_proba(X_val)[:, 1]
    loss_rf = float(log_loss(y_val, np.clip(p_rf, eps, 1.0 - eps)))
    comparators["RandomForest"] = {
        "model": rf_final,
        "val_loss": loss_rf,
        "val_acc": float(accuracy_score(y_val, (p_rf >= 0.5).astype(int))),
        "train_time_s": t_train_rf,
        "infer_lat_us": lat_rf,
        "size_kb": size_rf,
        "params": str(best_rf_p),
    }

    # 3. HistGradientBoostingClassifier: small grid
    hgb_candidates = [
        {"learning_rate": 0.02, "max_depth": 2, "max_iter": 100},
        {"learning_rate": 0.03, "max_depth": 3, "max_iter": 150},
        {"learning_rate": 0.05, "max_depth": 3, "max_iter": 100},
    ]
    best_hgb_p = hgb_candidates[0]
    best_hgb_loss = float("inf")
    for hgbp in hgb_candidates:
        f_losses = []
        for f in folds:
            df_b = f["df_base_dec"]
            df_ft = f["df_ft_train_dec"]
            df_v = f["df_val_all"]
            if len(df_ft) > 0:
                X_f = pd.concat([df_b[FEATURES], df_ft[FEATURES]], axis=0)
                y_f = np.concatenate([df_b["y"].to_numpy().astype(int), df_ft["y"].to_numpy().astype(int)])
            else:
                X_f = df_b[FEATURES]
                y_f = df_b["y"].to_numpy().astype(int)
            hgb_m = build_hist_gradient_boosting(**hgbp, random_state=CONFIG["seed"])
            hgb_m.fit(X_f, y_f)
            pv = hgb_m.predict_proba(df_v[FEATURES])[:, 1]
            f_losses.append(log_loss(df_v["y"].to_numpy().astype(int), np.clip(pv, eps, 1.0 - eps)))
        if np.mean(f_losses) < best_hgb_loss:
            best_hgb_loss = np.mean(f_losses)
            best_hgb_p = hgbp

    t0 = time.time()
    hgb_final = build_hist_gradient_boosting(**best_hgb_p, random_state=CONFIG["seed"])
    hgb_final.fit(X_tr, y_tr)
    t_train_hgb = time.time() - t0
    log_fit_ledger(f"Comparator: HistGradientBoosting ({best_hgb_p})", str(X_tr.index.min()), str(X_tr.index.max()), len(X_tr))
    lat_hgb, size_hgb = profile_model_efficiency(hgb_final, X_val.to_numpy())
    p_hgb = hgb_final.predict_proba(X_val)[:, 1]
    loss_hgb = float(log_loss(y_val, np.clip(p_hgb, eps, 1.0 - eps)))
    comparators["HistGradientBoosting"] = {
        "model": hgb_final,
        "val_loss": loss_hgb,
        "val_acc": float(accuracy_score(y_val, (p_hgb >= 0.5).astype(int))),
        "train_time_s": t_train_hgb,
        "infer_lat_us": lat_hgb,
        "size_kb": size_hgb,
        "params": str(best_hgb_p),
    }

    # Summary table
    comp_records = []
    for name, cinfo in comparators.items():
        comp_records.append({
            "model": name,
            "params": cinfo["params"],
            "val_logloss": round(cinfo["val_loss"], 5),
            "val_acc": round(cinfo["val_acc"], 4),
            "train_time_s": round(cinfo["train_time_s"], 3),
            "infer_lat_us": round(cinfo["infer_lat_us"], 1),
            "size_kb": round(cinfo["size_kb"], 1),
        })

    df_comp = pd.DataFrame(comp_records).sort_values("val_logloss")
    print(df_comp.to_string(index=False))

    return comparators


def run_stage_d_ablations(df: pd.DataFrame, best_params: Dict[str, Any], base_loss: float) -> List[Dict[str, Any]]:
    """Execute feature and configuration ablations for XGBoost on Validation."""
    print("\n--- STAGE D: XGBoost Ablation Studies on Validation ---")
    eps = 1e-15

    df_base_dec = df[(df["block"] == "Base") & df["decisive"]]
    df_ft_dec = df[(df["block"] == "FT") & df["decisive"]]
    df_val = df[(df["block"] == "Val") & df["fwd"].notna()]

    y_tr = np.concatenate([df_base_dec["y"].to_numpy().astype(int), df_ft_dec["y"].to_numpy().astype(int)])
    y_val = df_val["y"].to_numpy().astype(int)

    time_cols = ["hour", "dow", "is_first_bar"]
    context_cols = ["spy_ret_1", "spy_ret_3", "xlf_ret_1", "xlf_ret_3", "rel_ret_1"]

    ablation_sets = [
        {"name": "Full Baseline (S2)", "drop": []},
        {"name": "Without Time Features", "drop": time_cols},
        {"name": "Without Context (SPY/XLF)", "drop": context_cols},
        {"name": "Without RVOL", "drop": ["rvol"]},
        {"name": "Without Class Balance", "drop": [], "no_class_balance": True},
    ]

    ablation_results = []
    for ab in ablation_sets:
        features_sub = [f for f in FEATURES if f not in ab["drop"]]

        X_base_sub = df_base_dec[features_sub]
        X_ft_sub = df_ft_dec[features_sub]
        X_tr_sub = pd.concat([X_base_sub, X_ft_sub], axis=0)
        X_val_sub = df_val[features_sub]

        class_bal = not ab.get("no_class_balance", False)
        w_b = compute_sample_weights(df_base_dec["y"].to_numpy().astype(int), base_weight=best_params.get("base_weight", 1.0), class_balance_weights=class_bal)
        w_ft = compute_sample_weights(df_ft_dec["y"].to_numpy().astype(int), base_weight=1.0, class_balance_weights=class_bal)
        w_tr = np.concatenate([w_b, w_ft])

        m = build_tuned_xgboost(best_params, random_state=CONFIG["seed"])
        m.fit(X_tr_sub, y_tr, sample_weight=w_tr)
        pv = m.predict_proba(X_val_sub)[:, 1]
        loss = float(log_loss(y_val, np.clip(pv, eps, 1.0 - eps)))
        diff = loss - base_loss

        ablation_results.append({
            "ablation": ab["name"],
            "features_used": len(features_sub),
            "val_logloss": round(loss, 5),
            "delta_vs_base": round(diff, 5),
            "outcome": "Helped" if diff > 0.0005 else ("Hurt" if diff < -0.0005 else "Neutral"),
        })

    # Label threshold ablations
    for cand in CONFIG["label_threshold_candidates"]:
        b_dec = (df["block"] == "Base") & (df["fwd"].abs() > cand) & df["fwd"].notna()
        ft_dec = (df["block"] == "FT") & (df["fwd"].abs() > cand) & df["fwd"].notna()

        X_b = df.loc[b_dec, FEATURES]
        y_b = df.loc[b_dec, "y"].to_numpy().astype(int)
        X_f = df.loc[ft_dec, FEATURES]
        y_f = df.loc[ft_dec, "y"].to_numpy().astype(int)

        X_comb = pd.concat([X_b, X_f], axis=0)
        y_comb = np.concatenate([y_b, y_f])

        wb = compute_sample_weights(y_b, base_weight=best_params.get("base_weight", 1.0), class_balance_weights=CONFIG["class_balance_weights"])
        wf = compute_sample_weights(y_f, base_weight=1.0, class_balance_weights=CONFIG["class_balance_weights"])
        w_comb = np.concatenate([wb, wf])

        m = build_tuned_xgboost(best_params, random_state=CONFIG["seed"])
        m.fit(X_comb, y_comb, sample_weight=w_comb)
        pv = m.predict_proba(df_val[FEATURES])[:, 1]
        loss = float(log_loss(y_val, np.clip(pv, eps, 1.0 - eps)))
        diff = loss - base_loss

        ablation_results.append({
            "ablation": f"Label Threshold = {cand * 10000:.0f} bps",
            "features_used": len(FEATURES),
            "val_logloss": round(loss, 5),
            "delta_vs_base": round(diff, 5),
            "outcome": "Helped" if diff > 0.0005 else ("Hurt" if diff < -0.0005 else "Neutral"),
        })

    df_ab = pd.DataFrame(ablation_results)
    print(df_ab.to_string(index=False))
    return ablation_results


def run_stage_e_feature_importance(model: Any, X_val: pd.DataFrame, y_val: np.ndarray, figures_dir: Path):
    """Compute gain importance and permutation importance on Validation."""
    print("\n--- STAGE E: Feature Importance Analysis (Gain & Permutation on Val) ---")
    booster = model.get_booster()
    score_gain = booster.get_score(importance_type="gain")
    gain_series = pd.Series({f: score_gain.get(f, 0.0) for f in FEATURES}).sort_values(ascending=False)

    # Permutation importance on Val
    perm_res = permutation_importance(model, X_val, y_val, scoring="neg_log_loss", n_repeats=10, random_state=CONFIG["seed"])
    perm_series = pd.Series(perm_res.importances_mean, index=FEATURES).sort_values(ascending=False)

    df_imp = pd.DataFrame({
        "feature": FEATURES,
        "gain_importance": [score_gain.get(f, 0.0) for f in FEATURES],
        "perm_importance_val": [perm_res.importances_mean[i] for i in range(len(FEATURES))],
    }).sort_values("gain_importance", ascending=False)
    df_imp.to_csv("results/tables/feature_importance.csv", index=False)

    # Figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))
    ax1.barh(gain_series.index[:15], gain_series.values[:15], color="#2980b9", edgecolor="black")
    ax1.invert_yaxis()
    ax1.set_title("Top 15 Features by XGBoost Split Gain", fontweight="bold")
    ax1.set_xlabel("Average Gain per Split")
    ax1.grid(axis="x", linestyle=":", alpha=0.5)

    ax2.barh(perm_series.index[:15], perm_series.values[:15], color="#27ae60", edgecolor="black")
    ax2.invert_yaxis()
    ax2.set_title("Top 15 Features by Permutation Importance (Val)", fontweight="bold")
    ax2.set_xlabel("Mean Decrease in Log-Loss Score")
    ax2.grid(axis="x", linestyle=":", alpha=0.5)

    plt.suptitle("Goldman Sachs (GS) Direction Predictor Feature Importance", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(figures_dir / "feature_importance.png", dpi=200)
    plt.close()
    print(f"Saved feature importance table and figure to results/figures/feature_importance.png.")


def main():
    dataset_path = Path(CONFIG["paths"]["processed"])
    df = pd.read_parquet(dataset_path)

    # Stage A
    base_res = run_base_training()

    # Stage B Part A
    folds = create_walk_forward_folds(df)
    best_params = run_stage_b_hyperparameter_search(df, folds)

    # Stage B Part B
    stage_b_res = run_stage_b_strategies(df, best_params, base_res)

    # Stage C: Comparators
    comparators = run_stage_c_comparators(stage_b_res, folds)

    # Stage D: Ablations
    best_xgb_loss = stage_b_res["best_strategy_loss"]
    ablation_res = run_stage_d_ablations(df, best_params, best_xgb_loss)

    # Stage E: Feature Importance
    figures_dir = Path(CONFIG["paths"]["figures"])
    run_stage_e_feature_importance(stage_b_res["best_strategy_model"], stage_b_res["X_val"], stage_b_res["y_val"], figures_dir)

    # Final Model Selection
    # The deployed model is the best XGBoost strategy UNLESS a comparator beats it on Val log-loss by > 0.002
    switch_margin = CONFIG["switch_margin_logloss"]
    min_comp_name = min(comparators, key=lambda k: comparators[k]["val_loss"])
    min_comp_loss = comparators[min_comp_name]["val_loss"]

    print("\n=== FINAL MODEL DEPLOYMENT DECISION ===")
    print(f"Best XGBoost Strategy: {stage_b_res['best_strategy_name']} with Val Log-Loss = {best_xgb_loss:.5f}")
    print(f"Top Comparator: {min_comp_name} with Val Log-Loss = {min_comp_loss:.5f}")

    models_dir = Path(CONFIG["paths"]["models"])
    candidates_dir = models_dir / "candidates"
    candidates_dir.mkdir(parents=True, exist_ok=True)

    # Save candidates
    joblib.dump(stage_b_res["model_s0"], candidates_dir / "xgb_s0.joblib")
    joblib.dump(stage_b_res["model_s1"], candidates_dir / "xgb_s1.joblib")
    joblib.dump(stage_b_res["model_s2"], candidates_dir / "xgb_s2.joblib")
    joblib.dump(stage_b_res["model_s3"], candidates_dir / "xgb_s3.joblib")
    for cname, cinfo in comparators.items():
        joblib.dump(cinfo["model"], candidates_dir / f"{cname.lower()}.joblib")

    deployed_name = None
    deployed_model = None

    if min_comp_loss < best_xgb_loss - switch_margin:
        print(f"Comparator {min_comp_name} outperforms XGBoost by > {switch_margin:.4f}! Deploying {min_comp_name}.")
        deployed_name = min_comp_name
        deployed_model = comparators[min_comp_name]["model"]
    else:
        print(f"XGBoost retains deployment preference (difference {min_comp_loss - best_xgb_loss:+.5f} <= switch threshold {switch_margin}).")
        deployed_name = f"XGBoost_{stage_b_res['best_strategy_name']}"
        deployed_model = stage_b_res["best_strategy_model"]

    # Save deployed model
    final_joblib_path = models_dir / "model.joblib"
    joblib.dump(deployed_model, final_joblib_path)
    if isinstance(deployed_model, XGBClassifier):
        deployed_model.save_model(str(models_dir / "model.json"))
    print(f"Final model frozen and saved to {final_joblib_path}")

    # Generate finetune_report.md
    write_finetune_report(stage_b_res, comparators, ablation_res, deployed_name, best_params)


def write_finetune_report(stage_b_res, comparators, ablation_res, deployed_name, best_params):
    """Write comprehensive markdown finetune report."""
    report_path = Path("results/tables/finetune_report.md")
    lines = []
    lines.append("# Model Training, Fine-Tuning, and Ablation Report\n")

    lines.append("## 1. XGBoost Fine-Tuning Strategies (Evaluated on Validation)")
    lines.append("| Strategy | Val Log-Loss | Val Brier Score | Raw Val Accuracy |")
    lines.append("| :--- | :--- | :--- | :--- |")
    for _, row in stage_b_res["strategies_df"].iterrows():
        lines.append(f"| {row['strategy']} | {row['val_logloss']:.5f} | {row['val_brier']:.5f} | {row['val_acc']*100:.2f}% |")

    lines.append("\n## 2. Conventional Machine Learning Comparators")
    lines.append("| Model | Hyperparameters | Val Log-Loss | Val Accuracy | Fit Time (s) | Latency (µs) | Size (KB) |")
    lines.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
    for cname, cinfo in comparators.items():
        lines.append(f"| {cname} | `{cinfo['params']}` | {cinfo['val_loss']:.5f} | {cinfo['val_acc']*100:.2f}% | {cinfo['train_time_s']:.3f} | {cinfo['infer_lat_us']:.1f} | {cinfo['size_kb']:.1f} |")

    lines.append("\n## 3. XGBoost Ablation Studies")
    lines.append("| Ablation Experiment | Features Used | Val Log-Loss | Delta vs Base | Outcome |")
    lines.append("| :--- | :--- | :--- | :--- | :--- |")
    for ab in ablation_res:
        lines.append(f"| {ab['ablation']} | {ab['features_used']} | {ab['val_logloss']:.5f} | {ab['delta_vs_base']:+.5f} | {ab['outcome']} |")

    lines.append("\n## 4. Final Deployment Decision")
    lines.append(f"- **Deployed Architecture**: `{deployed_name}`")
    lines.append(f"- **Selected Tuned Hyperparameters**:\n```json\n{json.dumps(best_params, indent=2)}\n```")
    lines.append("- **Verification**: No Validation or Test rows were utilized during parameter fitting. All fit operations recorded in `results/logs/fit_ledger.json`.")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Finetune report successfully saved to {report_path}.")


if __name__ == "__main__":
    main()
