"""
Platt Calibrator for M5 (5-minute) Pipeline.
Fits logistic sigmoid on raw log-odds margins with class-balanced sample weights.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression

SEED = 42

class PlattCalibrator:
    def __init__(self):
        self.lr = LogisticRegression(C=1.0, solver="lbfgs", random_state=SEED)
        
    def fit(self, raw_probs: np.ndarray, y: np.ndarray):
        p = np.clip(raw_probs, 1e-6, 1.0 - 1e-6)
        logits = np.log(p / (1.0 - p)).reshape(-1, 1)
        n0 = np.sum(y == 0)
        n1 = np.sum(y == 1)
        w0 = 0.5 * len(y) / max(1, n0)
        w1 = 0.5 * len(y) / max(1, n1)
        sample_weight = np.where(y == 1, w1, w0)
        self.lr.fit(logits, y, sample_weight=sample_weight)
        return self

    def predict_proba(self, raw_probs: np.ndarray) -> np.ndarray:
        p = np.clip(raw_probs, 1e-6, 1.0 - 1e-6)
        logits = np.log(p / (1.0 - p)).reshape(-1, 1)
        return self.lr.predict_proba(logits)[:, 1]
