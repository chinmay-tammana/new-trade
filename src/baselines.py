"""Simple baseline models evaluated on the Validation partition.

Implements and scores 5 simple baselines on all labelled Validation rows:
1. Constant Prior (Base Up-rate)
2. Always-Up
3. Majority Class of Base
4. RSI Heuristic (Up if RSI < 40, Down if RSI > 60)
5. Previous-Bar Persistence (Up if ret_1 > 0 else Down)
"""

from pathlib import Path
from typing import Dict, Any
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, f1_score, log_loss, precision_score, recall_score

from configs.config import CONFIG
from src.utils import wilson_score_interval


def evaluate_signal_strategy(
    y_true: np.ndarray,
    fwd_returns: np.ndarray,
    signals: np.ndarray,  # 1: Up, 0: Down, np.nan: No signal
    probs: np.ndarray = None,
    costs_bps: list = [2, 5, 10]
) -> Dict[str, Any]:
    """Calculate comprehensive performance metrics for a discrete or probabilistic signal."""
    total_candles = len(y_true)
    active_mask = ~np.isnan(signals)
    n_signals = int(np.sum(active_mask))
    coverage = float(n_signals / total_candles) if total_candles > 0 else 0.0

    if n_signals == 0:
        return {
            "n_signals": 0, "coverage": 0.0, "accuracy": 0.0, "wilson_low": 0.0, "wilson_high": 0.0,
            "prec_up": 0.0, "rec_up": 0.0, "f1_up": 0.0, "prec_down": 0.0, "rec_down": 0.0, "f1_down": 0.0,
            "net_ret_2bps": 0.0, "net_ret_5bps": 0.0, "net_ret_10bps": 0.0,
            "log_loss": "-", "brier": "-",
        }

    y_active = y_true[active_mask]
    sig_active = signals[active_mask]
    fwd_active = fwd_returns[active_mask]

    n_correct = int(np.sum(sig_active == y_active))
    acc, w_low, w_high = wilson_score_interval(n_correct, n_signals, confidence=0.95)

    prec_up = float(precision_score(y_active, sig_active, pos_label=1, zero_division=0))
    rec_up = float(recall_score(y_active, sig_active, pos_label=1, zero_division=0))
    f1_up = float(f1_score(y_active, sig_active, pos_label=1, zero_division=0))

    prec_dn = float(precision_score(y_active, sig_active, pos_label=0, zero_division=0))
    rec_dn = float(recall_score(y_active, sig_active, pos_label=0, zero_division=0))
    f1_dn = float(f1_score(y_active, sig_active, pos_label=0, zero_division=0))

    # Trading returns: +1 for Up, -1 for Down
    direction = np.where(sig_active == 1, 1.0, -1.0)
    raw_rets = direction * fwd_active

    net_2 = float(np.mean(raw_rets - (2 / 10000.0)) * 10000.0)
    net_5 = float(np.mean(raw_rets - (5 / 10000.0)) * 10000.0)
    net_10 = float(np.mean(raw_rets - (10 / 10000.0)) * 10000.0)

    # Log loss and Brier score if probabilities exist across all rows
    ll_str = "-"
    br_str = "-"
    if probs is not None:
        eps = 1e-15
        p_clipped = np.clip(probs, eps, 1.0 - eps)
        ll = float(log_loss(y_true, p_clipped))
        br = float(brier_score_loss(y_true, p_clipped))
        ll_str = f"{ll:.5f}"
        br_str = f"{br:.5f}"

    return {
        "n_signals": n_signals,
        "coverage": coverage,
        "accuracy": acc,
        "wilson_low": w_low,
        "wilson_high": w_high,
        "prec_up": prec_up,
        "rec_up": rec_up,
        "f1_up": f1_up,
        "prec_down": prec_dn,
        "rec_down": rec_dn,
        "f1_down": f1_dn,
        "net_ret_2bps": net_2,
        "net_ret_5bps": net_5,
        "net_ret_10bps": net_10,
        "log_loss": ll_str,
        "brier": br_str,
    }


def run_baselines():
    dataset_path = Path(CONFIG["paths"]["processed"])
    df = pd.read_parquet(dataset_path)

    # Base labelled
    df_base = df[(df["block"] == "Base") & df["fwd"].notna()]
    base_up_rate = float((df_base["y"] == 1).mean())
    majority_class = 1 if base_up_rate >= 0.5 else 0

    # Validation labelled
    df_val = df[(df["block"] == "Val") & df["fwd"].notna()].copy()
    y_val = df_val["y"].to_numpy().astype(int)
    fwd_val = df_val["fwd"].to_numpy()
    N_val = len(y_val)

    print(f"\nEvaluating simple baselines on Validation set (N = {N_val} candles)...")
    print(f"Base Up-rate (prior): {base_up_rate:.4f} (Majority class: {majority_class})")

    results = []

    # 1. Constant Prior Baseline
    p_const = np.full(N_val, base_up_rate)
    sig_const = np.full(N_val, majority_class)
    m1 = evaluate_signal_strategy(y_val, fwd_val, sig_const, probs=p_const)
    results.append({"baseline": "Constant Prior", **m1})

    # 2. Always-Up Baseline
    sig_up = np.ones(N_val)
    m2 = evaluate_signal_strategy(y_val, fwd_val, sig_up)
    results.append({"baseline": "Always-Up", **m2})

    # 3. Majority Class of Base
    sig_maj = np.full(N_val, majority_class)
    m3 = evaluate_signal_strategy(y_val, fwd_val, sig_maj)
    results.append({"baseline": "Majority Class (Base)", **m3})

    # 4. RSI Heuristic Rule
    # Up if rsi < 40, Down if rsi > 60, No signal otherwise
    rsi_vals = df_val["rsi"].to_numpy()
    sig_rsi = np.full(N_val, np.nan)
    sig_rsi[rsi_vals < CONFIG["rsi_buy_below"]] = 1.0
    sig_rsi[rsi_vals > CONFIG["rsi_sell_above"]] = 0.0
    m4 = evaluate_signal_strategy(y_val, fwd_val, sig_rsi)
    results.append({"baseline": "RSI Rule (<40 Up, >60 Down)", **m4})

    # 5. Previous-Bar Persistence
    # Up if ret_1 > 0 else Down
    ret1_vals = df_val["ret_1"].to_numpy()
    sig_pers = np.where(ret1_vals > 0, 1.0, 0.0)
    m5 = evaluate_signal_strategy(y_val, fwd_val, sig_pers)
    results.append({"baseline": "Previous-Bar Persistence", **m5})

    df_results = pd.DataFrame(results)
    out_path = Path("results/tables/baselines_val.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_results.to_csv(out_path, index=False)
    print(f"\nBaseline results on Validation saved to {out_path}:")
    print(df_results[["baseline", "accuracy", "wilson_low", "wilson_high", "coverage", "n_signals", "net_ret_5bps", "log_loss", "brier"]].to_string(index=False))

    return df_results


if __name__ == "__main__":
    run_baselines()
