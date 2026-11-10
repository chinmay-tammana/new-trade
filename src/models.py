"""Model factory and utility functions for model instantiation and profiling."""

import os
import tempfile
import time
from typing import Any, Dict, Tuple
import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from configs.config import CONFIG
from src.preprocessing import create_linear_pipeline


class EnsembleV2:
    """Trained 3-way ensemble combining XGBoost, HistGradientBoosting, and ElasticNet."""

    def __init__(self, m_xgb, m_hgb, m_enet, scaler, calib_xgb, calib_hgb, calib_enet):
        self.m_xgb = m_xgb
        self.m_hgb = m_hgb
        self.m_enet = m_enet
        self.scaler = scaler
        self.calib_xgb = calib_xgb
        self.calib_hgb = calib_hgb
        self.calib_enet = calib_enet

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        p_raw_xgb = self.m_xgb.predict_proba(X)[:, 1]
        p_raw_hgb = self.m_hgb.predict_proba(X)[:, 1]
        p_raw_enet = self.m_enet.predict_proba(X_scaled)[:, 1]

        p_cal_xgb = self.calib_xgb.predict_proba(p_raw_xgb)
        p_cal_hgb = self.calib_hgb.predict_proba(p_raw_hgb)
        p_cal_enet = self.calib_enet.predict_proba(p_raw_enet)

        return (p_cal_xgb + p_cal_hgb + p_cal_enet) / 3.0


def build_default_xgboost(random_state: int = 42) -> XGBClassifier:
    """Build default-ish shallow, regularized XGBoost model for Stage A (S0)."""
    return XGBClassifier(
        max_depth=3,
        learning_rate=0.05,
        n_estimators=200,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=random_state,
        n_jobs=1,
    )


def build_tuned_xgboost(params: Dict[str, Any], random_state: int = 42) -> XGBClassifier:
    """Instantiate XGBClassifier with tuned hyperparameters."""
    xgb_params = {k: v for k, v in params.items() if k != "base_weight"}
    xgb_params.setdefault("eval_metric", "logloss")
    xgb_params.setdefault("random_state", random_state)
    xgb_params.setdefault("n_jobs", 1)
    return XGBClassifier(**xgb_params)


def build_logistic_regression(C: float = 1.0, random_state: int = 42):
    """Instantiate linear pipeline with in-fold Winsorizer, StandardScaler, and LogisticRegression."""
    estimator = LogisticRegression(
        C=C,
        penalty="l2",
        solver="lbfgs",
        class_weight="balanced" if CONFIG.get("class_balance_weights", True) else None,
        random_state=random_state,
        max_iter=1000,
    )
    return create_linear_pipeline(estimator)


def build_random_forest(n_estimators: int = 150, max_depth: int = 4, min_samples_leaf: int = 10, random_state: int = 42):
    """Instantiate regularized Random Forest classifier."""
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        class_weight="balanced" if CONFIG.get("class_balance_weights", True) else None,
        random_state=random_state,
        n_jobs=1,
    )


def build_hist_gradient_boosting(learning_rate: float = 0.03, max_depth: int = 3, max_iter: int = 150, random_state: int = 42):
    """Instantiate HistGradientBoostingClassifier."""
    return HistGradientBoostingClassifier(
        learning_rate=learning_rate,
        max_depth=max_depth,
        max_iter=max_iter,
        class_weight="balanced" if CONFIG.get("class_balance_weights", True) else None,
        random_state=random_state,
    )


def profile_model_efficiency(model: Any, X_sample: np.ndarray, n_reps: int = 100) -> Tuple[float, float, float]:
    """Measure training time, single-sample inference latency, and serialized file size in KB."""
    # 1. Inference latency (time per single sample in microseconds)
    single_x = X_sample[:1]
    # Warmup
    for _ in range(10):
        _ = model.predict_proba(single_x)

    t0 = time.perf_counter()
    for _ in range(n_reps):
        _ = model.predict_proba(single_x)
    t1 = time.perf_counter()
    lat_us = ((t1 - t0) / n_reps) * 1e6

    # 2. Model serialized size in KB
    with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as tmp:
        tmp_name = tmp.name
    try:
        joblib.dump(model, tmp_name)
        size_kb = os.path.getsize(tmp_name) / 1024.0
    finally:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)

    return lat_us, size_kb
