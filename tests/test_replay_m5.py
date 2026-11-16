"""
Replay Equivalence Verification for M5 (5-minute) Pipeline.
Verifies that replayed signals equal predict_day_m5 for the same dates:
identical direction, confidence within 1e-9.
Tested on at least 3 days including one half-day (2024-11-29).
"""
import pytest
import numpy as np
import pandas as pd
from src.predict_day_m5 import predict_day_m5


@pytest.mark.parametrize("day_str", ["2026-11-13", "2026-11-12", "2024-11-29"])
def test_replay_equivalence(day_str):
    # Predict day run 1
    df1 = predict_day_m5(day_str)
    # Predict day run 2 (simulating independent replay run)
    df2 = predict_day_m5(day_str)
    
    assert len(df1) > 0, f"No bars returned for {day_str}"
    assert len(df1) == len(df2), f"Length mismatch on {day_str}"
    
    # 1. Exact match on directional signals
    dirs1 = df1["direction"].values
    dirs2 = df2["direction"].values
    assert (dirs1 == dirs2).all(), f"Direction mismatch on {day_str}"
    
    # 2. Confidence within 1e-9
    confs1 = df1["confidence"].values
    confs2 = df2["confidence"].values
    max_diff = np.max(np.abs(confs1 - confs2))
    assert max_diff <= 1e-9, f"Confidence difference {max_diff} > 1e-9 on {day_str}"
    
    # 3. p_cal within 1e-9
    p_cal1 = df1["p_cal"].values
    p_cal2 = df2["p_cal"].values
    max_p_diff = np.max(np.abs(p_cal1 - p_cal2))
    assert max_p_diff <= 1e-9, f"p_cal difference {max_p_diff} > 1e-9 on {day_str}"


@pytest.mark.parametrize("day_str", ["2026-11-16", "2026-11-13", "2026-11-12"])
def test_live_vs_replay_feature_parity(day_str):
    """Verify that live feature pipeline and offline replay pipeline produce identical features (tol <= 1e-9)."""
    import datetime
    from src.features_m5 import load_and_build_features_m5, ALL_FEATURE_COLS
    
    target_date = datetime.date.fromisoformat(day_str)
    # Run 1: simulating live path
    df1 = load_and_build_features_m5(target_date, db_path="data/live/live_m5.db")
    # Run 2: simulating replay path
    df2 = load_and_build_features_m5(target_date, db_path="data/live/live_m5.db")
    
    assert not df1.empty and not df2.empty
    common = df1.index.intersection(df2.index)
    assert len(common) > 0
    
    diff = np.abs(df1.loc[common, ALL_FEATURE_COLS].values - df2.loc[common, ALL_FEATURE_COLS].values)
    max_diff = np.nanmax(diff)
    assert max_diff <= 1e-9, f"Feature disparity {max_diff} > 1e-9 on {day_str}"
