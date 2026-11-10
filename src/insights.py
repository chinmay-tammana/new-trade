"""Reusable per-bar insight inference module for live and backtest execution (V2 Upgrade).

Provides get_bar_insights(row, model=None, calib=None, vol_model=None, band_model=None)
returning:
- direction: 'Up', 'Down', or 'No Signal'
- confidence: probability in [0.5, 1.0]
- horizon: 'H1'
- vol_class: 'High', 'Normal', or 'Low'
- band_low: predicted 10th percentile forward return
- band_high: predicted 90th percentile forward return
- regime: 'Uptrend', 'Downtrend', or 'Chop'
- historical_hit_rate_for_this_hour_and_side: historical accuracy for (hour, side)
"""

import json
from pathlib import Path
from typing import Dict, Any
import numpy as np
import pandas as pd
import joblib

from configs.config import CONFIG
from src.features import FEATURES
from src.models import EnsembleV2


# Cache pre-loaded models in memory
_CACHED_MODELS = None
_CACHED_HIST_HIT_RATES = None


def _load_production_models():
    global _CACHED_MODELS, _CACHED_HIST_HIT_RATES
    if _CACHED_MODELS is not None:
        return _CACHED_MODELS, _CACHED_HIST_HIT_RATES

    models_dir = Path("models")
    ensemble = joblib.load(models_dir / "model_v2.joblib")
    calib_meta = joblib.load(models_dir / "calib_v2.joblib")
    vol_model = joblib.load(models_dir / "vol_v2.joblib")
    band_models = joblib.load(models_dir / "band_v2.joblib")

    with open("configs/model_v2_config.json") as f:
        cfg = json.load(f)

    # Precompute historical hit rates per (hour, side) from holdout predictions if available
    preds_file = Path(CONFIG["paths"]["holdout_predictions"])
    hit_rates = {("Up", 9.5): 0.51, ("Down", 9.5): 0.52}
    if preds_file.exists():
        df_p = pd.read_csv(preds_file)
        ts_ny = pd.to_datetime(df_p["timestamp"], utc=True).dt.tz_convert("America/New_York")
        df_p["hour"] = ts_ny.dt.hour + ts_ny.dt.minute / 60.0
        for h, grp in df_p.groupby("hour"):
            for sig, side in [(1, "Up"), (-1, "Down")]:
                sub = grp[grp["signal"] == sig]
                if len(sub) > 0:
                    hit_rates[(side, float(h))] = float(sub["outcome"].mean())

    _CACHED_MODELS = {
        "ensemble": ensemble,
        "calib": calib_meta,
        "vol": vol_model,
        "band": band_models,
        "cfg": cfg
    }
    _CACHED_HIST_HIT_RATES = hit_rates
    return _CACHED_MODELS, _CACHED_HIST_HIT_RATES


def get_bar_insights(
    row: pd.Series,
    models: Dict[str, Any] = None
) -> Dict[str, Any]:
    """Compute directional and insight signals for a single closed 1-hour bar.

    Parameters
    ----------
    row : pd.Series
        Feature values containing all 51 features in FEATURES, plus hour, ret_20d, efficiency_ratio_20.
    models : dict, optional
        Preloaded models dictionary. If None, loads from disk.

    Returns
    -------
    dict
        Dictionary of direction, confidence, horizon, vol_class, band_low, band_high, regime,
        and historical_hit_rate_for_this_hour_and_side.
    """
    if models is None:
        models, hit_rates = _load_production_models()
    else:
        _, hit_rates = _load_production_models()

    ensemble = models["ensemble"]
    calib_meta = models["calib"]
    vol_model = models["vol"]
    band_models = models["band"]
    cfg = models["cfg"]

    X = row[FEATURES].to_numpy().reshape(1, -1)

    # 1. Calibrated Directional Posterior
    p_cal = float(ensemble.predict_proba(X)[0])
    q_lo = calib_meta["q_lo"]
    q_hi = calib_meta["q_hi"]

    if p_cal >= q_hi:
        direction = "Up"
        confidence = p_cal
    elif p_cal <= q_lo:
        direction = "Down"
        confidence = 1.0 - p_cal
    else:
        direction = "No Signal"
        confidence = max(p_cal, 1.0 - p_cal)

    # 2. VOL Class (High, Normal, Low)
    pred_log_vol = float(vol_model.predict(X)[0])
    terc_33 = cfg["vol_terciles"]["tercile_33"]
    terc_67 = cfg["vol_terciles"]["tercile_67"]
    if pred_log_vol <= terc_33:
        vol_class = "Low"
    elif pred_log_vol <= terc_67:
        vol_class = "Normal"
    else:
        vol_class = "High"

    # 3. Return Bands (Quantile q=0.10, 0.90)
    band_low = float(band_models["m_q10"].predict(X)[0])
    band_high = float(band_models["m_q90"].predict(X)[0])

    # 4. REGIME
    r20 = row.get("ret_20d", np.nan)
    eff = row.get("efficiency_ratio_20", np.nan)
    eff_thresh = cfg["regime_thresholds"]["eff_chop_threshold"]
    if np.isnan(r20) or np.isnan(eff) or eff < eff_thresh:
        regime = "Chop"
    elif r20 > 0:
        regime = "Uptrend"
    else:
        regime = "Downtrend"

    # 5. Historical Hit Rate for this hour and side
    hour_val = float(row.get("hour", 9.5))
    hist_hit_rate = hit_rates.get((direction, hour_val), 0.52)

    return {
        "direction": direction,
        "confidence": float(confidence),
        "horizon": "H1",
        "vol_class": vol_class,
        "band_low": float(band_low),
        "band_high": float(band_high),
        "regime": regime,
        "historical_hit_rate_for_this_hour_and_side": float(hist_hit_rate)
    }
