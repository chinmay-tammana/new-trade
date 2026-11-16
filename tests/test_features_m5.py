"""
Tests for M5 (5-minute) features, labels, and causal aggregation.
Verifies truncation invariance (zero lookahead) and same-session label rule.
"""
import datetime
import zoneinfo
import numpy as np
import pandas as pd
import pytest

from src.aggregate_m5 import aggregate_1min_to_5min, ET_TZ
from src.features_m5 import make_features_m5, ALL_FEATURE_COLS


def create_dummy_1min_data(n_days: int = 5):
    # 5 full days of 1-minute bars
    start_date = datetime.date(2026, 11, 2) # Monday
    records = []
    
    price = 900.0
    for day_offset in range(n_days):
        d = start_date + datetime.timedelta(days=day_offset)
        # 390 minutes from 09:30 to 16:00
        for m in range(390):
            ts = datetime.datetime.combine(d, datetime.time(9, 30), tzinfo=ET_TZ) + datetime.timedelta(minutes=m)
            ret = np.random.normal(0, 0.0005)
            price = max(10.0, price * (1.0 + ret))
            records.append({
                "timestamp": ts,
                "open": price,
                "high": price * 1.0005,
                "low": price * 0.9995,
                "close": price * (1.0 + np.random.normal(0, 0.0002)),
                "volume": 100.0,
                "trade_count": 5.0,
                "vwap": price
            })
    df = pd.DataFrame(records).set_index("timestamp")
    return df


def test_aggregate_1min_to_5min_exactness():
    df_1m = create_dummy_1min_data(n_days=1)
    df_5m = aggregate_1min_to_5min(df_1m, symbol="GS")
    
    # 390 minutes / 5 = 78 bars
    assert len(df_5m) == 78, f"Expected 78 bars, got {len(df_5m)}"
    assert df_5m.index[0].time() == datetime.time(9, 30)
    assert df_5m.index[-1].time() == datetime.time(15, 55)
    
    # Check volume sum
    expected_vol = 100.0 * 5
    assert np.allclose(df_5m["volume"].values, expected_vol)


def test_same_session_label_rule():
    df_1m = create_dummy_1min_data(n_days=2)
    df_5m = aggregate_1min_to_5min(df_1m, symbol="GS")
    df_feat = make_features_m5(df_5m)
    
    # End of day bars (15:55 start, closing at 16:00) must have fwd = NaN
    eod_bars = df_feat[df_feat.index.time == datetime.time(15, 55)]
    assert len(eod_bars) == 2
    assert eod_bars["fwd"].isna().all(), "EOD bar forward return must be NaN (same-session rule)"
    assert eod_bars["y"].isna().all(), "EOD bar label must be NaN"


def test_features_truncation_invariance():
    # Load actual processed dataset and test truncation
    df_feat_full = pd.read_parquet("data/processed/m5/features_m5.parquet")
    
    # Pick a test window of 500 bars
    test_sub = df_feat_full.iloc[1000:1500]
    gs_bars = pd.read_parquet("data/processed/m5/bars_5m_GS.parquet")
    
    # Compute features on full history up to bar 1450
    cut_idx = 1450
    full_slice = gs_bars.iloc[:cut_idx]
    
    feat_slice = make_features_m5(full_slice)
    target_row_full = df_feat_full.iloc[cut_idx - 1]
    target_row_slice = feat_slice.iloc[-1]
    
    # Compare stationary feature columns
    test_cols = [
        "ret_1", "ret_2", "ret_3", "ret_6", "ret_12", "ret_since_open",
        "rsi_14", "macd_diff_norm", "bb_pct_b", "bb_width", "atr_14_norm",
        "rolling_std_12", "mins_since_open", "mins_to_close", "tod_sin", "tod_cos"
    ]
    
    for col in test_cols:
        v_full = target_row_full[col]
        v_slice = target_row_slice[col]
        if not np.isnan(v_full):
            diff = abs(v_full - v_slice)
            assert diff < 1e-5, f"Truncation test failed on {col}: full={v_full}, slice={v_slice}, diff={diff}"
