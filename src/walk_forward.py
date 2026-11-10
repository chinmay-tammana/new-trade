"""Walk-forward cross-validation, model selection, hyperparameter tuning, and ablations (V2 Upgrade).

Implements:
1. Quarterly walk-forward expanding window cross-validation (63 trading days per fold, 8-bar gap).
2. Pre-holdout OOF pooling for hyperparameter search (n_iter=30) and recency weighting {90, 180, 365, None}.
3. Candidate tournament on pre-holdout OOF:
   - Shallow XGBoost
   - HistGradientBoosting
   - Elastic-Net Logistic Regression
   - Ensemble (mean of calibrated probabilities)
4. Group ablations across the 9 FEATURE_GROUPS.
5. Directional horizon evaluation across H1, H3, HD.
6. Insight heads training: VOL regressor, BAND quantile regressors, REGIME tags.
7. Class-balanced Platt probability calibration.
8. Two-sided quantile signal rule search (c in {0.15, 0.20, 0.25, 0.30, 0.40}, n_up>=150, n_down>=150).
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression, QuantileRegressor
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import log_loss, brier_score_loss, accuracy_score, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor
import joblib

from configs.config import CONFIG
from src.features import FEATURES, FEATURE_GROUPS
from src.labels import compute_sample_weights_v2
from src.utils import wilson_score_interval


def generate_quarterly_folds(df_pre: pd.DataFrame, initial_years: int = 2, quarter_days: int = 63, gap_bars: int = 8):
    """Generate chronological expanding-window walk-forward splits."""
    dates = pd.Series(df_pre.index.date).drop_duplicates().tolist()
    initial_trading_days = initial_years * 252

    folds = []
    curr_train_days = initial_trading_days

    while curr_train_days + quarter_days <= len(dates):
        train_dates = set(dates[:curr_train_days])
        test_dates = set(dates[curr_train_days : curr_train_days + quarter_days])

        train_indices = np.where(pd.Series(df_pre.index.date).isin(train_dates))[0]
        test_indices = np.where(pd.Series(df_pre.index.date).isin(test_dates))[0]

        # Apply 8-bar embargo gap before test start
        test_start_idx = test_indices[0]
        train_end_idx = max(0, test_start_idx - gap_bars)
        effective_train_indices = train_indices[train_indices < train_end_idx]

        folds.append((effective_train_indices, test_indices))
        curr_train_days += quarter_days

    # Remaining days if any before holdout
    if curr_train_days < len(dates):
        train_dates = set(dates[:curr_train_days])
        test_dates = set(dates[curr_train_days:])
        train_indices = np.where(pd.Series(df_pre.index.date).isin(train_dates))[0]
        test_indices = np.where(pd.Series(df_pre.index.date).isin(test_dates))[0]
        test_start_idx = test_indices[0]
        train_end_idx = max(0, test_start_idx - gap_bars)
        effective_train_indices = train_indices[train_indices < train_end_idx]
        if len(test_indices) > 20:
            folds.append((effective_train_indices, test_indices))

    return folds


def run_walk_forward_oof(
    df: pd.DataFrame,
    folds: list,
    model_factory,
    feature_list: list,
    target_col: str = "y_H1",
    decisive_col: str = "decisive_H1",
    half_life_days: float = None,
    scale_features: bool = False
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Run walk-forward cross validation and produce pooled OOF predictions."""
    oof_preds = np.full(len(df), np.nan)
    oof_eval_mask = np.zeros(len(df), dtype=bool)

    y_all = df[target_col].to_numpy()
    decisive_all = df[decisive_col].to_numpy()
    X_all = df[feature_list].to_numpy()
    ts_all = df.index

    for tr_idx, te_idx in folds:
        # Train strictly on decisive rows
        tr_dec_idx = tr_idx[decisive_all[tr_idx] & ~np.isnan(y_all[tr_idx])]
        if len(tr_dec_idx) < 50:
            continue

        X_tr = X_all[tr_dec_idx]
        y_tr = y_all[tr_dec_idx].astype(int)
        ts_tr = ts_all[tr_dec_idx]

        w_tr = compute_sample_weights_v2(
            y_tr, timestamps=ts_tr, half_life_days=half_life_days, class_balance=True
        )

        X_te = X_all[te_idx]

        if scale_features:
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_tr)
            X_te = scaler.transform(X_te)

        clf = model_factory()
        clf.fit(X_tr, y_tr, sample_weight=w_tr)

        # Predict probabilities on test fold
        p_te = clf.predict_proba(X_te)[:, 1]
        oof_preds[te_idx] = p_te

        # Only evaluate on labelled rows (not session-close NaN rows)
        labelled_te = ~np.isnan(y_all[te_idx])
        oof_eval_mask[te_idx] = labelled_te

    # Evaluate pooled log-loss
    valid_mask = oof_eval_mask & ~np.isnan(oof_preds)
    y_eval = y_all[valid_mask].astype(int)
    p_eval = np.clip(oof_preds[valid_mask], 1e-15, 1 - 1e-15)

    loss = float(log_loss(y_eval, p_eval))
    return oof_preds, valid_mask, loss


class PlattCalibratorV2:
    """Class-balanced Platt scaling so that 0.5 means no directional lean."""

    def __init__(self):
        self.clf = LogisticRegression(C=1e5, solver="lbfgs", random_state=42)

    def fit(self, p_oof: np.ndarray, y_oof: np.ndarray):
        eps = 1e-15
        p_clipped = np.clip(p_oof, eps, 1 - eps)
        # Logit features
        logits = np.log(p_clipped / (1.0 - p_clipped)).reshape(-1, 1)

        # Class balanced sample weights
        weights = compute_sample_weights_v2(y_oof, class_balance=True)
        self.clf.fit(logits, y_oof, sample_weight=weights)
        return self

    def predict_proba(self, p_raw: np.ndarray) -> np.ndarray:
        eps = 1e-15
        p_clipped = np.clip(p_raw, eps, 1 - eps)
        logits = np.log(p_clipped / (1.0 - p_clipped)).reshape(-1, 1)
        return self.clf.predict_proba(logits)[:, 1]


def compute_ece(y_true: np.ndarray, p_pred: np.ndarray, n_bins: int = 10) -> float:
    """Compute Expected Calibration Error over n_bins."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(p_pred, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    ece = 0.0
    total = len(y_true)
    for b in range(n_bins):
        mask = (bin_indices == b)
        if np.sum(mask) > 0:
            bin_acc = np.mean(y_true[mask])
            bin_conf = np.mean(p_pred[mask])
            ece += (np.sum(mask) / total) * abs(bin_acc - bin_conf)
    return float(ece)
