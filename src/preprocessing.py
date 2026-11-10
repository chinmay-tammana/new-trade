"""Preprocessing pipelines and in-fold transformers.

Implements leak-free preprocessing transformers (Winsorizer, StandardScaler)
fitted exclusively within training folds.
"""

from typing import Optional
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class InFoldWinsorizer(BaseEstimator, TransformerMixin):
    """Clips features to empirical percentiles fitted strictly on training data.

    Default: 0.5th and 99.5th percentiles.
    """

    def __init__(self, lower_percentile: float = 0.5, upper_percentile: float = 99.5):
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile
        self.lower_bounds_ = None
        self.upper_bounds_ = None
        self.feature_names_in_ = None

    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = list(X.columns)
            X_arr = X.to_numpy()
        else:
            X_arr = np.asarray(X)

        self.lower_bounds_ = np.nanpercentile(X_arr, self.lower_percentile, axis=0)
        self.upper_bounds_ = np.nanpercentile(X_arr, self.upper_percentile, axis=0)
        return self

    def transform(self, X):
        if self.lower_bounds_ is None or self.upper_bounds_ is None:
            raise RuntimeError("InFoldWinsorizer must be fitted before transforming.")

        if isinstance(X, pd.DataFrame):
            clipped = np.clip(X.to_numpy(), self.lower_bounds_, self.upper_bounds_)
            return pd.DataFrame(clipped, index=X.index, columns=X.columns)
        else:
            return np.clip(np.asarray(X), self.lower_bounds_, self.upper_bounds_)


def create_linear_pipeline(model_estimator) -> Pipeline:
    """Build a leak-free pipeline for linear/distance models.

    Order: InFoldWinsorizer (0.5/99.5) -> StandardScaler -> model_estimator.
    """
    pipeline = Pipeline([
        ("winsorizer", InFoldWinsorizer(lower_percentile=0.5, upper_percentile=99.5)),
        ("scaler", StandardScaler()),
        ("model", model_estimator),
    ])
    return pipeline
