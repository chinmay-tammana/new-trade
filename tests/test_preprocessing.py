"""Unit tests for preprocessing transformers."""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src.preprocessing import InFoldWinsorizer, create_linear_pipeline


def test_in_fold_winsorizer():
    X = np.array([[-100.0], [1.0], [2.0], [3.0], [100.0]])
    winsorizer = InFoldWinsorizer(lower_percentile=10.0, upper_percentile=90.0)
    winsorizer.fit(X)
    X_trans = winsorizer.transform(X)

    assert X_trans[0, 0] > -100.0
    assert X_trans[-1, 0] < 100.0
    assert X_trans[1, 0] == 1.0


def test_linear_pipeline_execution():
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, size=(50, 4))
    y = rng.integers(0, 2, size=50)

    pipe = create_linear_pipeline(LogisticRegression(random_state=42))
    pipe.fit(X, y)
    probs = pipe.predict_proba(X)
    assert probs.shape == (50, 2)
    assert np.allclose(probs.sum(axis=1), 1.0)
