"""Unit tests for feature engineering."""

import numpy as np
import pandas as pd
import pytest

from src.features import FEATURES, compute_atr, compute_wilder_rsi, make_features


def create_dummy_ohlcv(n_bars=100, seed=42):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-02 09:30", periods=n_bars, freq="1h", tz="America/New_York")
    price = 100.0 + np.cumsum(rng.normal(0, 0.5, size=n_bars))
    df = pd.DataFrame({
        "open": price + rng.normal(0, 0.1, size=n_bars),
        "high": price + np.abs(rng.normal(0.5, 0.1, size=n_bars)),
        "low": price - np.abs(rng.normal(0.5, 0.1, size=n_bars)),
        "close": price,
        "volume": rng.integers(1000, 50000, size=n_bars),
    }, index=dates)
    return df


def test_wilder_rsi_bounds():
    df = create_dummy_ohlcv(60)
    rsi = compute_wilder_rsi(df["close"], period=14)
    valid_rsi = rsi.dropna()
    assert len(valid_rsi) > 0
    assert (valid_rsi >= 0.0).all()
    assert (valid_rsi <= 100.0).all()


def test_atr_positivity():
    df = create_dummy_ohlcv(60)
    atr = compute_atr(df["high"], df["low"], df["close"], period=14)
    valid_atr = atr.dropna()
    assert len(valid_atr) > 0
    assert (valid_atr > 0.0).all()


def test_make_features_shape_and_columns():
    df_gs = create_dummy_ohlcv(80, seed=1)
    df_spy = create_dummy_ohlcv(80, seed=2)
    df_xlf = create_dummy_ohlcv(80, seed=3)

    feat_df = make_features(df_gs, df_spy, df_xlf, warmup_rows=30)
    assert len(feat_df) == 50
    for col in FEATURES:
        assert col in feat_df.columns
        assert not feat_df[col].isna().any(), f"Unexpected NaN in feature {col}"
