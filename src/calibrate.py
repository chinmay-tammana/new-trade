"""Calibration and asymmetric signal threshold optimization.

Fits Platt scaling and Isotonic calibrators on the Validation block, evaluates
reliability diagrams and ECE, runs directional diagnostics to address zero-Down
issues, optimizes asymmetric confidence margins (d_up, d_down), and freezes models/calib.joblib.
"""

from pathlib import Path
import json
from typing import Any, Dict, Optional, Tuple

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss

from configs.config import CONFIG
from src.features import FEATURES
from src.utils import expected_calibration_error, wilson_score_interval


class PlattCalibrator:
    """Platt scaling: univariate logistic regression on clipped logit(p)."""

    def __init__(self, clip_val: float = 10.0):
        self.clip_val = clip_val
        self.lr = LogisticRegression(C=1.0, solver="lbfgs", max_iter=500, random_state=CONFIG["seed"])

    def fit(self, p_raw: np.ndarray, y: np.ndarray):
        eps = 1e-7
        p_c = np.clip(p_raw, eps, 1.0 - eps)
        log_odds = np.clip(logit(p_c), -self.clip_val, self.clip_val).reshape(-1, 1)
        self.lr.fit(log_odds, y)
        return self

    def predict_proba(self, p_raw: np.ndarray) -> np.ndarray:
        eps = 1e-7
        p_c = np.clip(p_raw, eps, 1.0 - eps)
        log_odds = np.clip(logit(p_c), -self.clip_val, self.clip_val).reshape(-1, 1)
        return self.lr.predict_proba(log_odds)[:, 1]


def run_calibration():
    dataset_path = Path(CONFIG["paths"]["processed"])
    df = pd.read_parquet(dataset_path)

    models_dir = Path(CONFIG["paths"]["models"])
    figures_dir = Path(CONFIG["paths"]["figures"])
    tables_dir = Path(CONFIG["paths"]["tables"])
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    # Load deployed model
    model_path = models_dir / "model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing deployed model at {model_path}")
    model = joblib.load(model_path)

    # Validation set: ALL labelled rows (no decisive filter per spec)
    df_val = df[(df["block"] == "Val") & df["fwd"].notna()].copy()
    X_val = df_val[FEATURES]
    y_val = df_val["y"].to_numpy().astype(int)
    fwd_val = df_val["fwd"].to_numpy()
    N_val = len(y_val)

    print(f"=== PHASE 6: CALIBRATION & SIGNAL RULE OPTIMIZATION ===")
    print(f"Validation calibration candles: {N_val}")

    # 1. Raw probabilities on Val
    p_raw_val = model.predict_proba(X_val)[:, 1]

    # Fit Platt scaling
    platt = PlattCalibrator()
    platt.fit(p_raw_val, y_val)
    p_platt_val = platt.predict_proba(p_raw_val)

    # Fit Isotonic regression
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(p_raw_val, y_val)
    p_iso_val = iso.predict(p_raw_val)

    eps = 1e-15
    # Metrics comparison
    brier_raw = brier_score_loss(y_val, p_raw_val)
    brier_platt = brier_score_loss(y_val, p_platt_val)
    brier_iso = brier_score_loss(y_val, p_iso_val)

    ece_raw = expected_calibration_error(y_val, p_raw_val, n_bins=10)
    ece_platt = expected_calibration_error(y_val, p_platt_val, n_bins=10)
    ece_iso = expected_calibration_error(y_val, p_iso_val, n_bins=10)

    loss_raw = log_loss(y_val, np.clip(p_raw_val, eps, 1.0 - eps))
    loss_platt = log_loss(y_val, np.clip(p_platt_val, eps, 1.0 - eps))
    loss_iso = log_loss(y_val, np.clip(p_iso_val, eps, 1.0 - eps))

    print("\nCalibration Results on Validation:")
    print(f"  Raw:      Log-Loss = {loss_raw:.5f}, Brier = {brier_raw:.5f}, ECE = {ece_raw:.4f}")
    print(f"  Platt:    Log-Loss = {loss_platt:.5f}, Brier = {brier_platt:.5f}, ECE = {ece_platt:.4f}")
    print(f"  Isotonic: Log-Loss = {loss_iso:.5f}, Brier = {brier_iso:.5f}, ECE = {ece_iso:.4f}")

    # Choice: Choose Platt unless Isotonic is clearly better (Val is small ~602 bars)
    # Platt is smooth and prevents step-function overfitting
    chosen_calibrator_type = "Platt"
    chosen_calibrator = platt
    p_cal_val = p_platt_val
    print("Selected Calibrator: Platt Scaling (smooth logistic mapping on clipped logit).")

    # Reliability diagram
    generate_reliability_diagram(y_val, p_raw_val, p_platt_val, p_iso_val, figures_dir)

    # 2. Directional Diagnostics
    run_directional_diagnostics(df, model, chosen_calibrator, p_cal_val, tables_dir)

    # 3. Asymmetric Signal Rule Search
    rule_info = search_optimal_signal_rule(y_val, fwd_val, p_cal_val)

    # 4. Print Up / Down / No-signal counts on Val
    print("\nValidation Signal Rule Counts:")
    print(f"  Up signals:        {rule_info['n_up']}")
    print(f"  Down signals:      {rule_info['n_down']}")
    print(f"  No-signal candles: {rule_info['n_no_signal']}")
    print(f"  Total active:      {rule_info['n_signals']} (Coverage: {rule_info['coverage']*100:.2f}%)")
    print(f"  Active accuracy:   {rule_info['accuracy']*100:.2f}% (Wilson 95% CI: [{rule_info['wilson_low']*100:.2f}%, {rule_info['wilson_high']*100:.2f}%])")

    # 5. Save calib.joblib
    calib_payload = {
        "calibrator": chosen_calibrator,
        "calibrator_type": chosen_calibrator_type,
        "d_up": rule_info["d_up"],
        "d_down": rule_info["d_down"],
        "enabled_sides": rule_info["enabled_sides"],
        "features": FEATURES,
        "label_threshold": CONFIG["label_threshold"],
        "horizon_bars": CONFIG["horizon_bars"],
        "val_accuracy": rule_info["accuracy"],
        "val_coverage": rule_info["coverage"],
        "val_wilson_low": rule_info["wilson_low"],
        "val_signals": rule_info["n_signals"],
    }
    calib_out = models_dir / "calib.joblib"
    joblib.dump(calib_payload, calib_out)
    print(f"Saved frozen calibrator and signal rule to {calib_out}")

    return calib_payload


def generate_reliability_diagram(y_val, p_raw, p_platt, p_iso, figures_dir: Path):
    """Plot reliability curve (calibration curve) before and after calibration."""
    prob_true_raw, prob_pred_raw = calibration_curve(y_val, p_raw, n_bins=10)
    prob_true_platt, prob_pred_platt = calibration_curve(y_val, p_platt, n_bins=10)
    prob_true_iso, prob_pred_iso = calibration_curve(y_val, p_iso, n_bins=10)

    plt.figure(figsize=(8, 7))
    plt.plot([0, 1], [0, 1], "k--", label="Perfect Calibration (y = x)", alpha=0.8)
    plt.plot(prob_pred_raw, prob_true_raw, "s-", color="#e74c3c", label=f"Uncalibrated Raw (ECE={expected_calibration_error(y_val, p_raw):.3f})")
    plt.plot(prob_pred_platt, prob_true_platt, "o-", color="#2980b9", lw=2, label=f"Platt Scaling (ECE={expected_calibration_error(y_val, p_platt):.3f})")
    plt.plot(prob_pred_iso, prob_true_iso, "^-", color="#27ae60", alpha=0.7, label=f"Isotonic (ECE={expected_calibration_error(y_val, p_iso):.3f})")

    plt.title("Reliability Diagram: Predicted Probability vs Observed Up-Rate (Val)", fontsize=11, fontweight="bold")
    plt.xlabel("Mean Predicted Probability P(Up)", fontsize=10)
    plt.ylabel("Observed Fraction of Up Moves", fontsize=10)
    plt.xlim(0.35, 0.65)
    plt.ylim(0.35, 0.65)
    plt.legend(loc="upper left")
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig(figures_dir / "calibration_reliability.png", dpi=200)
    plt.close()
    print("Saved reliability diagram to results/figures/calibration_reliability.png")


def run_directional_diagnostics(df: pd.DataFrame, model: Any, calibrator: Any, p_cal_val: np.ndarray, tables_dir: Path):
    """Compute directional diagnostics across Base, FT, and Val blocks."""
    # Label up rates
    df_base_lab = df[(df["block"] == "Base") & df["fwd"].notna()]
    df_ft_lab = df[(df["block"] == "FT") & df["fwd"].notna()]
    df_val_lab = df[(df["block"] == "Val") & df["fwd"].notna()]

    base_uprate = float((df_base_lab["y"] == 1).mean())
    ft_uprate = float((df_ft_lab["y"] == 1).mean())
    val_uprate = float((df_val_lab["y"] == 1).mean())

    # Mean predicted p
    p_raw_base = model.predict_proba(df_base_lab[FEATURES])[:, 1]
    p_raw_ft = model.predict_proba(df_ft_lab[FEATURES])[:, 1]
    p_raw_val = model.predict_proba(df_val_lab[FEATURES])[:, 1]

    p_cal_base = calibrator.predict_proba(p_raw_base)
    p_cal_ft = calibrator.predict_proba(p_raw_ft)

    # Calibrated p percentiles on Val
    percentiles = [5, 25, 50, 75, 95]
    pct_vals = np.percentile(p_cal_val, percentiles)

    diag_data = [
        {"metric": "Base Block Label Up-rate", "value": f"{base_uprate:.4f}"},
        {"metric": "FT Block Label Up-rate", "value": f"{ft_uprate:.4f}"},
        {"metric": "Val Block Label Up-rate", "value": f"{val_uprate:.4f}"},
        {"metric": "Base Mean Calibrated P(Up)", "value": f"{np.mean(p_cal_base):.4f}"},
        {"metric": "FT Mean Calibrated P(Up)", "value": f"{np.mean(p_cal_ft):.4f}"},
        {"metric": "Val Mean Calibrated P(Up)", "value": f"{np.mean(p_cal_val):.4f}"},
        {"metric": "Val Calibrated P Std", "value": f"{np.std(p_cal_val):.4f}"},
        {"metric": "Val Calibrated P Min", "value": f"{np.min(p_cal_val):.4f}"},
        {"metric": "Val Calibrated P 5th percentile", "value": f"{pct_vals[0]:.4f}"},
        {"metric": "Val Calibrated P 25th percentile", "value": f"{pct_vals[1]:.4f}"},
        {"metric": "Val Calibrated P 50th percentile (Median)", "value": f"{pct_vals[2]:.4f}"},
        {"metric": "Val Calibrated P 75th percentile", "value": f"{pct_vals[3]:.4f}"},
        {"metric": "Val Calibrated P 95th percentile", "value": f"{pct_vals[4]:.4f}"},
        {"metric": "Val Calibrated P Max", "value": f"{np.max(p_cal_val):.4f}"},
    ]

    df_diag = pd.DataFrame(diag_data)
    out_csv = tables_dir / "directional_diagnostics.csv"
    df_diag.to_csv(out_csv, index=False)
    print(f"\nDirectional Diagnostics saved to {out_csv}:")
    print(df_diag.to_string(index=False))


def search_optimal_signal_rule(y_val: np.ndarray, fwd_val: np.ndarray, p_cal_val: np.ndarray) -> Dict[str, Any]:
    """Execute joint grid search for asymmetric confidence margins (d_up, d_down)."""
    sweep_cfg = CONFIG["margin_sweep"]  # [start, stop, step]
    margins = np.arange(sweep_cfg[0], sweep_cfg[1] + 1e-5, sweep_cfg[2])

    min_cov = CONFIG["min_coverage"]
    min_sig = CONFIG["min_signals"]
    min_side = CONFIG["min_signals_per_side"]
    N_val = len(y_val)

    best_score = -1.0
    best_rule = None

    # Track symmetric rule separately
    best_sym_score = -1.0
    best_sym_rule = None

    # Evaluate candidate rules
    # 1. Both sides enabled: (d_up, d_down)
    # 2. Up only: d_up in margins, d_down = None
    # 3. Down only: d_up = None, d_down in margins
    candidates = []
    for d_u in margins:
        for d_d in margins:
            candidates.append((d_u, d_d, "both"))
    for d_u in margins:
        candidates.append((d_u, None, "up_only"))
    for d_d in margins:
        candidates.append((None, d_d, "down_only"))

    def evaluate_rule(d_u, d_d, mode):
        up_mask = (p_cal_val >= (0.5 + d_u)) if d_u is not None else np.zeros(N_val, dtype=bool)
        down_mask = (p_cal_val <= (0.5 - d_d)) if d_d is not None else np.zeros(N_val, dtype=bool)

        n_up = int(np.sum(up_mask))
        n_down = int(np.sum(down_mask))
        n_tot = n_up + n_down
        cov = n_tot / N_val

        # Constraints
        if n_tot < min_sig:
            return None
        if cov < min_cov:
            return None
        if mode == "both" and (n_up < min_side or n_down < min_side):
            return None
        if mode == "up_only" and n_up < min_side:
            return None
        if mode == "down_only" and n_down < min_side:
            return None

        # Accuracy
        correct_up = np.sum((y_val == 1) & up_mask)
        correct_down = np.sum((y_val == 0) & down_mask)
        n_correct = int(correct_up + correct_down)

        acc, w_low, w_high = wilson_score_interval(n_correct, n_tot, confidence=0.95)
        return {
            "d_up": d_u,
            "d_down": d_d,
            "enabled_sides": mode,
            "n_up": n_up,
            "n_down": n_down,
            "n_no_signal": N_val - n_tot,
            "n_signals": n_tot,
            "coverage": cov,
            "accuracy": acc,
            "wilson_low": w_low,
            "wilson_high": w_high,
        }

    valid_rules = []
    for d_u, d_d, mode in candidates:
        res = evaluate_rule(d_u, d_d, mode)
        if res is not None:
            valid_rules.append(res)
            # Check symmetric
            if mode == "both" and abs(d_u - d_d) < 1e-6:
                if res["wilson_low"] > best_sym_score:
                    best_sym_score = res["wilson_low"]
                    best_sym_rule = res

    # If no rule satisfied constraints, relax min_coverage first, then min_signals_per_side
    if not valid_rules:
        print("WARNING: No margin combination satisfied all constraints. Relaxing min_coverage to 0.10...")
        min_cov = 0.10
        for d_u, d_d, mode in candidates:
            res = evaluate_rule(d_u, d_d, mode)
            if res is not None:
                valid_rules.append(res)

    if not valid_rules:
        print("WARNING: Relaxing min_signals_per_side to 15...")
        min_side = 15
        for d_u, d_d, mode in candidates:
            res = evaluate_rule(d_u, d_d, mode)
            if res is not None:
                valid_rules.append(res)

    # Pick rule with highest Wilson lower bound
    best_rule = max(valid_rules, key=lambda r: (r["wilson_low"], r["accuracy"], r["n_signals"]))

    print("\n--- OPTIMAL SIGNAL RULE SELECTED ON VALIDATION ---")
    print(f"  Mode:              {best_rule['enabled_sides'].upper()}")
    print(f"  Margin d_up:       {best_rule['d_up']} (Threshold p >= {0.5 + (best_rule['d_up'] or 0):.4f})")
    print(f"  Margin d_down:     {best_rule['d_down']} (Threshold p <= {0.5 - (best_rule['d_down'] or 0):.4f})")
    print(f"  Wilson 95% Bound:  {best_rule['wilson_low']*100:.2f}% to {best_rule['wilson_high']*100:.2f}%")
    print(f"  Point Accuracy:    {best_rule['accuracy']*100:.2f}% across {best_rule['n_signals']} signals")
    if best_sym_rule:
        print(f"  Best Symmetric:    d_up=d_down={best_sym_rule['d_up']} -> Acc {best_sym_rule['accuracy']*100:.2f}% (Wilson Low: {best_sym_rule['wilson_low']*100:.2f}%)")

    return best_rule


if __name__ == "__main__":
    run_calibration()
