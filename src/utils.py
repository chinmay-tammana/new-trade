"""Shared statistical, evaluation, and ledger utility functions.

Implements Wilson 95% score confidence intervals, Expected Calibration Error (ECE),
block-bootstrap return intervals, fit ledger tracking, and experiment log management.
"""

import json
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import brier_score_loss, log_loss


def wilson_score_interval(successes: int, total: int, confidence: float = 0.95) -> Tuple[float, float, float]:
    """Calculate the Wilson score confidence interval for a binomial proportion.

    Parameters
    ----------
    successes : int
        Number of positive outcomes / correct predictions.
    total : int
        Total number of trials / active signals.
    confidence : float
        Confidence level (default 0.95 for 95% CI).

    Returns
    -------
    Tuple[float, float, float]
        (point_estimate, lower_bound, upper_bound)
    """
    if total == 0:
        return 0.0, 0.0, 0.0

    p_hat = successes / total
    z = stats.norm.ppf(1.0 - (1.0 - confidence) / 2.0)
    z2 = z * z
    denom = 1.0 + (z2 / total)
    center = (p_hat + (z2 / (2.0 * total))) / denom
    spread = (z * math.sqrt((p_hat * (1.0 - p_hat) / total) + (z2 / (4.0 * total * total)))) / denom

    lower = max(0.0, center - spread)
    upper = min(1.0, center + spread)
    return p_hat, lower, upper


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Compute Expected Calibration Error (ECE) across equal-width probability bins."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n_total = len(y_true)

    for i in range(n_bins):
        bin_lower = bin_edges[i]
        bin_upper = bin_edges[i + 1]

        if i == n_bins - 1:
            in_bin = (y_prob >= bin_lower) & (y_prob <= bin_upper)
        else:
            in_bin = (y_prob >= bin_lower) & (y_prob < bin_upper)

        bin_count = np.sum(in_bin)
        if bin_count > 0:
            bin_acc = np.mean(y_true[in_bin])
            bin_conf = np.mean(y_prob[in_bin])
            ece += (bin_count / n_total) * abs(bin_acc - bin_conf)

    return float(ece)


def block_bootstrap_returns(returns: np.ndarray, block_size: int = 5, n_boot: int = 2000, seed: int = 42) -> Tuple[float, float, float]:
    """Calculate moving block bootstrap 95% confidence interval for mean return per signal."""
    if len(returns) == 0:
        return 0.0, 0.0, 0.0

    n = len(returns)
    rng = np.random.default_rng(seed)
    mean_obs = float(np.mean(returns))

    if n < block_size:
        return mean_obs, mean_obs, mean_obs

    n_blocks = int(math.ceil(n / block_size))
    boot_means = np.empty(n_boot)

    for b in range(n_boot):
        start_indices = rng.integers(0, n - block_size + 1, size=n_blocks)
        sampled = []
        for s in start_indices:
            sampled.extend(returns[s : s + block_size])
        sampled = np.asarray(sampled[:n])
        boot_means[b] = np.mean(sampled)

    lower = float(np.percentile(boot_means, 2.5))
    upper = float(np.percentile(boot_means, 97.5))
    return mean_obs, lower, upper


def log_fit_ledger(
    purpose: str,
    first_timestamp: str,
    last_timestamp: str,
    row_count: int,
    ledger_path: str = "results/logs/fit_ledger.json"
):
    """Append model fit metadata to the centralized fit ledger."""
    path = Path(ledger_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    entries = []
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                entries = json.load(f)
        except Exception:
            entries = []

    new_entry = {
        "entry_id": len(entries) + 1,
        "purpose": purpose,
        "first_timestamp": str(first_timestamp),
        "last_timestamp": str(last_timestamp),
        "row_count": int(row_count),
        "logged_at": datetime.now().isoformat(),
    }
    entries.append(new_entry)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2)


def append_experiment_log(
    run_id: str,
    split: str,
    seed: int,
    model: str,
    hyperparameters: str,
    cv_metric: float,
    val_metric: float,
    runtime_s: float,
    log_path: str = "results/tables/experiment_log.csv"
):
    """Append a row to results/tables/experiment_log.csv."""
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    row = {
        "run_id": run_id,
        "split": split,
        "seed": seed,
        "model": model,
        "hyperparameters": hyperparameters,
        "cv_metric": round(cv_metric, 6) if cv_metric is not None else "",
        "val_metric": round(val_metric, 6) if val_metric is not None else "",
        "runtime_s": round(runtime_s, 2),
    }

    df_row = pd.DataFrame([row])
    if not path.exists():
        df_row.to_csv(path, index=False)
    else:
        df_row.to_csv(path, mode="a", header=False, index=False)
