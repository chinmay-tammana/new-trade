"""Unit tests for statistical utils and evaluation helpers."""

import numpy as np
import pytest

from src.utils import block_bootstrap_returns, expected_calibration_error, wilson_score_interval


def test_wilson_score_interval():
    # 60 successes out of 100 trials
    p, low, high = wilson_score_interval(60, 100, confidence=0.95)
    assert p == 0.60
    assert 0.49 < low < 0.51
    assert 0.68 < high < 0.70

    # Edge cases
    p0, low0, high0 = wilson_score_interval(0, 50, confidence=0.95)
    assert p0 == 0.0
    assert low0 == 0.0
    assert high0 > 0.0

    p_empty, low_e, high_e = wilson_score_interval(0, 0)
    assert p_empty == 0.0 and low_e == 0.0 and high_e == 0.0


def test_expected_calibration_error():
    # Perfect calibration
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.05, 0.15, 0.85, 0.95])
    ece = expected_calibration_error(y_true, y_prob, n_bins=10)
    assert ece < 0.20

    # Totally uncalibrated
    y_true_bad = np.array([0, 0, 0, 0])
    y_prob_bad = np.array([0.9, 0.9, 0.9, 0.9])
    ece_bad = expected_calibration_error(y_true_bad, y_prob_bad, n_bins=10)
    assert pytest.approx(ece_bad, rel=1e-3) == 0.90


def test_block_bootstrap_returns():
    rng = np.random.default_rng(42)
    returns = rng.normal(0.001, 0.005, size=200)
    mean, low, high = block_bootstrap_returns(returns, block_size=5, n_boot=500, seed=42)
    assert low < mean < high
