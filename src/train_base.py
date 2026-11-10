"""Stage A: Base model training and cross-validation sanity check.

Fits default XGBClassifier on Base decisive rows, validates stability via
TimeSeriesSplit (n_splits=5, gap=8) within Base, logs fit to fit_ledger.json,
and evaluates strategy S0 on the Validation block.
"""

from pathlib import Path
import time
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.model_selection import TimeSeriesSplit

from configs.config import CONFIG
from src.features import FEATURES
from src.labels import compute_sample_weights
from src.models import build_default_xgboost
from src.utils import append_experiment_log, log_fit_ledger


def run_base_training():
    dataset_path = Path(CONFIG["paths"]["processed"])
    df = pd.read_parquet(dataset_path)

    # Base decisive rows for training
    df_base_dec = df[(df["block"] == "Base") & df["decisive"]].copy()
    X_base = df_base_dec[FEATURES]
    y_base = df_base_dec["y"].to_numpy().astype(int)

    # Sample weights with class balancing
    weights_base = compute_sample_weights(
        y_base,
        base_weight=1.0,
        class_balance_weights=CONFIG.get("class_balance_weights", True)
    )

    # Validation set: ALL labelled rows
    df_val = df[(df["block"] == "Val") & df["fwd"].notna()].copy()
    X_val = df_val[FEATURES]
    y_val = df_val["y"].to_numpy().astype(int)

    print(f"=== STAGE A: BASE MODEL TRAINING ===")
    print(f"Base decisive training samples: {len(X_base)} (First: {df_base_dec.index.min()}, Last: {df_base_dec.index.max()})")
    print(f"Validation evaluation samples:  {len(X_val)}")

    # 1. Sanity Check: TimeSeriesSplit CV inside Base
    print("\nRunning TimeSeriesSplit(n_splits=5, gap=8) sanity check inside Base block...")
    tscv = TimeSeriesSplit(n_splits=5, gap=CONFIG["gap_bars"])
    cv_train_losses = []
    cv_val_losses = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_base)):
        fold_X_tr, fold_y_tr = X_base.iloc[train_idx], y_base[train_idx]
        fold_w_tr = weights_base[train_idx]
        fold_X_val, fold_y_val = X_base.iloc[val_idx], y_base[val_idx]

        model_cv = build_default_xgboost(random_state=CONFIG["seed"])
        model_cv.fit(fold_X_tr, fold_y_tr, sample_weight=fold_w_tr)

        tr_prob = model_cv.predict_proba(fold_X_tr)[:, 1]
        val_prob = model_cv.predict_proba(fold_X_val)[:, 1]

        eps = 1e-15
        tr_loss = log_loss(fold_y_tr, np.clip(tr_prob, eps, 1.0 - eps))
        v_loss = log_loss(fold_y_val, np.clip(val_prob, eps, 1.0 - eps))

        cv_train_losses.append(tr_loss)
        cv_val_losses.append(v_loss)
        print(f"  Fold {fold + 1}: Train Log-Loss = {tr_loss:.4f}, CV Val Log-Loss = {v_loss:.4f}")

    mean_cv_val_loss = float(np.mean(cv_val_losses))
    print(f"Mean CV Log-Loss: {mean_cv_val_loss:.4f} (Train Mean: {np.mean(cv_train_losses):.4f})")

    # 2. Fit full Base model
    print("\nFitting full base model on all Base decisive rows...")
    t0 = time.time()
    base_model = build_default_xgboost(random_state=CONFIG["seed"])
    base_model.fit(X_base, y_base, sample_weight=weights_base)
    fit_duration = time.time() - t0

    # 3. Score on Validation (strategy S0)
    val_probs = base_model.predict_proba(X_val)[:, 1]
    eps = 1e-15
    val_loss_s0 = float(log_loss(y_val, np.clip(val_probs, eps, 1.0 - eps)))
    val_brier_s0 = float(brier_score_loss(y_val, np.clip(val_probs, eps, 1.0 - eps)))
    val_acc_s0 = float(accuracy_score(y_val, (val_probs >= 0.5).astype(int)))

    print(f"Strategy S0 (Base Model before Fine-Tuning) on Validation:")
    print(f"  Log-Loss: {val_loss_s0:.5f}")
    print(f"  Brier Score: {val_brier_s0:.5f}")
    print(f"  Raw Accuracy (p >= 0.5): {val_acc_s0 * 100:.2f}%")

    # 4. Save base model
    models_dir = Path(CONFIG["paths"]["models"])
    models_dir.mkdir(parents=True, exist_ok=True)
    base_model_path = models_dir / "base_model.json"
    base_model.save_model(str(base_model_path))
    print(f"Saved base model to {base_model_path}")

    # 5. Ledger & Experiment Log
    log_fit_ledger(
        purpose="Base model training (Stage A / Strategy S0)",
        first_timestamp=str(df_base_dec.index.min()),
        last_timestamp=str(df_base_dec.index.max()),
        row_count=len(df_base_dec),
    )

    append_experiment_log(
        run_id="E00_S0_BaseDefault",
        split="Base_only",
        seed=CONFIG["seed"],
        model="XGBoost_Default",
        hyperparameters="max_depth=3, lr=0.05, n_est=200, sub=0.8, col=0.8",
        cv_metric=mean_cv_val_loss,
        val_metric=val_loss_s0,
        runtime_s=fit_duration,
    )

    return {
        "model": base_model,
        "val_loss_s0": val_loss_s0,
        "val_brier_s0": val_brier_s0,
        "val_acc_s0": val_acc_s0,
        "mean_cv_loss": mean_cv_val_loss,
    }


if __name__ == "__main__":
    run_base_training()
