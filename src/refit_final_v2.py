"""Phase V2.7: Final Model Refit on All Data up to freeze_date.

Refits the winning Ensemble configuration (XGBoost + HistGradientBoosting + ElasticNet)
and insight heads on ALL 13,884 bars up to freeze_date (including the Research Holdout).
Saves models/model_v2.joblib and models/calib_v2.joblib.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor
import joblib

from configs.config import CONFIG
from src.features import FEATURES
from src.labels import compute_sample_weights_v2
from src.models import EnsembleV2
from src.walk_forward import PlattCalibratorV2


def refit_final_v2():
    with open("configs/model_v2_config.json") as f:
        cfg = json.load(f)

    df = pd.read_parquet(CONFIG["paths"]["processed"])
    print(f"Refitting final production models on all {len(df):,} bars up to freeze_date...")

    primary_horizon = cfg["primary_horizon"]
    target_col = f"y_{primary_horizon}"
    decisive_col = f"decisive_{primary_horizon}"
    xgb_cfg = cfg["best_xgb_hyperparameters"]

    # Filter decisive rows
    dec_mask = df[decisive_col] & df[target_col].notna()
    X_dec = df.loc[dec_mask, FEATURES].to_numpy()
    y_dec = df.loc[dec_mask, target_col].astype(int).to_numpy()
    ts_dec = df.loc[dec_mask].index

    weights = compute_sample_weights_v2(
        y_dec, timestamps=ts_dec, half_life_days=xgb_cfg.get("half_life_days"), class_balance=True
    )

    # 1. Fit Component Models
    print("Fitting production XGBoost...")
    m_xgb = XGBClassifier(
        max_depth=xgb_cfg["max_depth"],
        learning_rate=xgb_cfg["learning_rate"],
        n_estimators=xgb_cfg["n_estimators"],
        subsample=xgb_cfg["subsample"],
        colsample_bytree=xgb_cfg["colsample_bytree"],
        reg_lambda=xgb_cfg["reg_lambda"],
        min_child_weight=xgb_cfg["min_child_weight"],
        random_state=CONFIG["seed"],
        n_jobs=1,
        eval_metric="logloss"
    )
    m_xgb.fit(X_dec, y_dec, sample_weight=weights)

    print("Fitting production HistGradientBoosting...")
    m_hgb = HistGradientBoostingClassifier(
        max_depth=3, learning_rate=0.03, max_iter=150, min_samples_leaf=20, random_state=CONFIG["seed"]
    )
    m_hgb.fit(X_dec, y_dec, sample_weight=weights)

    print("Fitting production ElasticNet...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_dec)
    m_enet = LogisticRegression(
        penalty="elasticnet", l1_ratio=0.5, C=0.01, solver="saga", max_iter=500, random_state=CONFIG["seed"]
    )
    m_enet.fit(X_scaled, y_dec, sample_weight=weights)

    # 2. Fit Calibrators on full dataset predictions
    X_all = df[FEATURES].to_numpy()
    valid_mask = df[target_col].notna()
    y_all = df.loc[valid_mask, target_col].astype(int).to_numpy()

    p_raw_xgb = m_xgb.predict_proba(X_all[valid_mask])[:, 1]
    p_raw_hgb = m_hgb.predict_proba(X_all[valid_mask])[:, 1]
    p_raw_enet = m_enet.predict_proba(scaler.transform(X_all[valid_mask]))[:, 1]

    calib_xgb = PlattCalibratorV2().fit(p_raw_xgb, y_all)
    calib_hgb = PlattCalibratorV2().fit(p_raw_hgb, y_all)
    calib_enet = PlattCalibratorV2().fit(p_raw_enet, y_all)

    # Construct final Ensemble
    ensemble = EnsembleV2(m_xgb, m_hgb, m_enet, scaler, calib_xgb, calib_hgb, calib_enet)

    # Recompute global quantiles q_lo and q_hi on all data
    p_cal_all = ensemble.predict_proba(X_all[valid_mask])
    c_val = cfg["signal_rule"]["coverage"]
    q_lo_final = float(np.quantile(p_cal_all, c_val / 2.0))
    q_hi_final = float(np.quantile(p_cal_all, 1.0 - c_val / 2.0))

    # 3. Fit VOL and BAND insight models on all data
    print("Fitting production VOL regressor...")
    vol_mask = df["target_vol"].notna()
    m_vol = XGBRegressor(max_depth=3, learning_rate=0.03, n_estimators=150, random_state=42, n_jobs=1)
    m_vol.fit(df.loc[vol_mask, FEATURES], df.loc[vol_mask, "target_vol"])

    print("Fitting production BAND quantile regressors...")
    ret_mask = df["fwd_H1"].notna()
    m_q10 = HistGradientBoostingRegressor(loss="quantile", quantile=0.10, max_iter=80, random_state=42)
    m_q90 = HistGradientBoostingRegressor(loss="quantile", quantile=0.90, max_iter=80, random_state=42)
    m_q10.fit(df.loc[ret_mask, FEATURES], df.loc[ret_mask, "fwd_H1"])
    m_q90.fit(df.loc[ret_mask, FEATURES], df.loc[ret_mask, "fwd_H1"])

    # Save artifacts
    models_dir = Path("models")
    joblib.dump(ensemble, models_dir / "model_v2.joblib")
    joblib.dump({
        "q_lo": q_lo_final,
        "q_hi": q_hi_final,
        "coverage": c_val,
        "primary_horizon": primary_horizon
    }, models_dir / "calib_v2.joblib")
    joblib.dump(m_vol, models_dir / "vol_v2.joblib")
    joblib.dump({"m_q10": m_q10, "m_q90": m_q90}, models_dir / "band_v2.joblib")

    # Update metadata
    cfg["final_refit"] = {
        "data_end": CONFIG["freeze_date"],
        "total_bars": len(df),
        "q_lo": q_lo_final,
        "q_hi": q_hi_final
    }
    with open("configs/model_v2_config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    print(f"\nFinal V2 models successfully saved to models/model_v2.joblib and models/calib_v2.joblib.")
    print(f"Final Quantiles: q_lo = {q_lo_final:.5f}, q_hi = {q_hi_final:.5f}")


if __name__ == "__main__":
    refit_final_v2()
