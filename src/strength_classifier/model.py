"""
src/strength_classifier/model.py

Classifier architectures and a threshold-aware wrapper for binary honey-bee
colony strength classification.

Label convention:
    0 = Weak
    1 = Strong
"""

from typing import Any, Optional

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    HistGradientBoostingClassifier,
    GradientBoostingClassifier,
)

WEAK = 0
STRONG = 1

# Optional XGBoost import with graceful fallback.
HAVE_XGB = False
try:
    from xgboost import XGBClassifier

    HAVE_XGB = True
except Exception:
    XGBClassifier = None


class ColonyStrengthClassifier(ClassifierMixin, BaseEstimator):
    """
    Scikit-learn-compatible wrapper that predicts Weak (0) when P(Weak)
    reaches the configured weak_threshold.

    Parameters
    ----------
    base_estimator:
        A fitted or unfitted sklearn classifier/pipeline.
    threshold:
        Probability threshold for Strong (1) when weak_threshold is None.
        Defaults to 0.5.
    weak_threshold:
        If set, predict Weak (0) when P(Weak) >= weak_threshold.
        For example, 0.425 is the validation-selected balanced threshold.
    """

    def __init__(
        self,
        base_estimator: Any = None,
        threshold: float = 0.5,
        weak_threshold: Optional[float] = None,
    ):
        # Keep constructor parameters unchanged for sklearn compatibility.
        self.base_estimator = base_estimator
        self.threshold = threshold
        self.weak_threshold = weak_threshold

    def _validate_thresholds(self) -> None:
        if not 0.0 <= float(self.threshold) <= 1.0:
            raise ValueError("threshold must be between 0.0 and 1.0")
        if self.weak_threshold is not None:
            if not 0.0 <= float(self.weak_threshold) <= 1.0:
                raise ValueError("weak_threshold must be between 0.0 and 1.0")

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        **fit_params: Any,
    ) -> "ColonyStrengthClassifier":
        self._validate_thresholds()

        if self.base_estimator is None:
            self.base_estimator = build_classifier_pipeline("rf")

        self.base_estimator.fit(X, y, **fit_params)

        if not hasattr(self.base_estimator, "classes_"):
            raise AttributeError(
                "base_estimator must expose classes_ after fitting."
            )

        self.classes_ = np.asarray(self.base_estimator.classes_)
        if set(self.classes_.tolist()) != {WEAK, STRONG}:
            raise ValueError(
                "Expected binary labels {0, 1}, where 0=Weak and 1=Strong; "
                f"got classes {self.classes_.tolist()}."
            )

        self.n_features_in_ = getattr(
            self.base_estimator, "n_features_in_", np.asarray(X).shape[1]
        )
        if hasattr(self.base_estimator, "feature_names_in_"):
            self.feature_names_in_ = self.base_estimator.feature_names_in_

        # Expose feature importance/coefficients when available.
        final_estimator = self.base_estimator
        if hasattr(final_estimator, "named_steps"):
            final_estimator = final_estimator.steps[-1][1]

        if hasattr(final_estimator, "feature_importances_"):
            self.feature_importances_ = final_estimator.feature_importances_
        if hasattr(final_estimator, "coef_"):
            self.coef_ = final_estimator.coef_

        return self

    def _check_is_fitted(self) -> None:
        if not hasattr(self, "classes_") or not hasattr(self, "base_estimator"):
            raise AttributeError(
                "ColonyStrengthClassifier is not fitted. Load a trained model "
                "or call fit(X, y) before prediction."
            )

    def _class_index(self, label: int) -> int:
        self._check_is_fitted()
        classes = np.asarray(self.classes_)
        matches = np.flatnonzero(classes == label)
        if len(matches) != 1:
            raise ValueError(
                f"Expected class {label} exactly once in classes_={classes.tolist()}."
            )
        return int(matches[0])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return probabilities in the base estimator's classes_ column order."""
        self._check_is_fitted()

        if not hasattr(self.base_estimator, "predict_proba"):
            raise AttributeError(
                "The base estimator does not support predict_proba; "
                "calibrated threshold predictions cannot be computed."
            )

        probabilities = np.asarray(self.base_estimator.predict_proba(X), dtype=float)

        if probabilities.ndim != 2:
            raise ValueError(
                f"Expected a 2D probability array, got shape {probabilities.shape}."
            )
        if probabilities.shape[1] != len(self.classes_):
            raise ValueError(
                "Probability column count does not match classes_: "
                f"{probabilities.shape[1]} columns for {len(self.classes_)} classes."
            )
        if not np.isfinite(probabilities).all():
            raise ValueError("Model returned NaN or infinite probabilities.")
        if (probabilities < -1e-8).any() or (probabilities > 1.0 + 1e-8).any():
            raise ValueError("Model returned probabilities outside [0, 1].")

        return probabilities

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict 0=Weak or 1=Strong using the configured threshold."""
        self._validate_thresholds()
        probabilities = self.predict_proba(X)

        weak_index = self._class_index(WEAK)
        strong_index = self._class_index(STRONG)

        if self.weak_threshold is not None:
            is_weak = probabilities[:, weak_index] >= self.weak_threshold
            return np.where(is_weak, WEAK, STRONG).astype(int)

        is_strong = probabilities[:, strong_index] >= self.threshold
        return np.where(is_strong, STRONG, WEAK).astype(int)


def build_classifier_pipeline(
    model_type: str = "rf",
    random_state: int = 42,
    scale_pos_weight: Optional[float] = None,
) -> Pipeline:
    """
    Build a feature-imputation/preprocessing/classifier pipeline.

    Supported model types:
      - 'logistic', 'lr', 'baseline'
      - 'rf', 'random_forest'
      - 'hist_gbdt', 'hgb'
      - 'gbdt', 'gradient_boosting'
      - 'xgboost', 'xgb' (requires xgboost)
    """
    model_type = model_type.lower()

    if model_type in ["logistic", "lr", "baseline"]:
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        C=0.1,
                        class_weight="balanced",
                        max_iter=1000,
                        random_state=random_state,
                    ),
                ),
            ]
        )

    if model_type in ["rf", "random_forest"]:
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=200,
                        max_depth=6,
                        min_samples_leaf=2,
                        class_weight="balanced",
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        )

    if model_type in ["hist_gbdt", "hgb"]:
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        max_iter=150,
                        max_depth=4,
                        learning_rate=0.05,
                        class_weight="balanced",
                        random_state=random_state,
                    ),
                ),
            ]
        )

    if model_type in ["gbdt", "gradient_boosting"]:
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    GradientBoostingClassifier(
                        n_estimators=150,
                        max_depth=3,
                        learning_rate=0.05,
                        subsample=0.8,
                        random_state=random_state,
                    ),
                ),
            ]
        )

    if model_type in ["xgboost", "xgb"]:
        if not HAVE_XGB or XGBClassifier is None:
            raise ImportError("xgboost is not installed")

        spw = scale_pos_weight if scale_pos_weight is not None else 1.0
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    XGBClassifier(
                        n_estimators=200,
                        max_depth=4,
                        learning_rate=0.03,
                        scale_pos_weight=spw,
                        random_state=random_state,
                        eval_metric="logloss",
                    ),
                ),
            ]
        )

    raise ValueError(f"Unknown classifier model_type: '{model_type}'")
