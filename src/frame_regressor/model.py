"""
src/frame_regressor/model.py
Defines the ClippedRegressor wrapper to enforce biologically and physically plausible
bee frame count boundaries [min_val, max_val].
"""

from typing import Any
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin


class ClippedRegressor(BaseEstimator, RegressorMixin):
    """
    Wraps an underlying scikit-learn regressor to enforce realistic physical frame count bounds [min_val, max_val].
    """

    def __init__(self, base_estimator: Any, min_val: float = 0.0, max_val: float = 32.0):
        self.base_estimator = base_estimator
        self.min_val = min_val
        self.max_val = max_val

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ClippedRegressor":
        self.base_estimator.fit(X, y)
        self.is_fitted_ = True
        self.n_features_in_ = X.shape[1] if hasattr(X, "shape") else len(X[0])
        if hasattr(self.base_estimator, "feature_importances_"):
            self.feature_importances_ = self.base_estimator.feature_importances_
        elif hasattr(self.base_estimator, "coef_"):
            self.coef_ = self.base_estimator.coef_
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        raw_pred = self.base_estimator.predict(X)
        return np.clip(raw_pred, self.min_val, self.max_val)
