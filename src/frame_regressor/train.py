"""
src/frame_regressor/train.py
Trains and benchmarks Machine Learning regression models for Exact Honey Bee Colony Size & Frame Count Regressor.
Predicts continuous `frames_count` (1.0 to 30.0 frames) using the identical 10 environmental features.
Evaluates MAE, RMSE, R2, Pearson correlation, and tolerance accuracy (+-1, +-2, +-3 frames).
Serializes the champion regressor to artifacts/frame_count_regressor.joblib and
benchmark report to artifacts/frame_count_regression_metrics.json.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
import joblib
from scipy.stats import pearsonr

# Add project root to sys.path so the script can be executed directly
BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import (
    RandomForestRegressor,
    HistGradientBoostingRegressor,
    GradientBoostingRegressor,
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.shared.config import settings
from src.shared.dataset import load_datasets, ENVIRONMENTAL_FEATURES
from src.frame_regressor.model import ClippedRegressor

# Optional XGBoost import with graceful fallback
HAVE_XGB = False
try:
    from xgboost import XGBRegressor

    HAVE_XGB = True
except Exception:
    HAVE_XGB = False

ARTIFACTS_DIR = settings.MSPB2_ARTIFACTS_DIR
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


def evaluate_regressor(
    pipeline: Any, X: np.ndarray, y: np.ndarray, split_name: str = "Val"
) -> Dict[str, Any]:
    y_pred = pipeline.predict(X)

    mae = float(mean_absolute_error(y, y_pred))
    mse = float(mean_squared_error(y, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y, y_pred))

    try:
        corr, pval = pearsonr(y, y_pred)
        r_val = float(corr)
    except Exception:
        r_val = 0.0

    errors = np.abs(y - y_pred)
    within_1 = float(np.mean(errors <= 1.0) * 100)
    within_2 = float(np.mean(errors <= 2.0) * 100)
    within_3 = float(np.mean(errors <= 3.0) * 100)

    return {
        "split": split_name,
        "sample_count": len(y),
        "mae": round(mae, 3),
        "rmse": round(rmse, 3),
        "r2_score": round(r2, 4),
        "pearson_r": round(r_val, 4),
        "within_1_frame_pct": round(within_1, 1),
        "within_2_frames_pct": round(within_2, 1),
        "within_3_frames_pct": round(within_3, 1),
    }


def select_champion_regressor(
    val_results: Dict[str, Any],
    priority_metric: str = "mae",
    explicit_choice: str = "auto",
) -> str:
    """
    Selects the production champion regressor based on quantitative evaluation criteria:
    - 'mae':       Primary: Lowest Mean Absolute Error (MAE), Secondary: Lowest RMSE, Tertiary: Highest R².
    - 'rmse':      Primary: Lowest RMSE, Secondary: Lowest MAE, Tertiary: Highest R².
    - 'r2':        Primary: Highest R² score, Secondary: Lowest MAE.
    - 'pearson_r': Primary: Highest Pearson correlation (r), Secondary: Lowest MAE.
    - 'tolerance': Primary: Highest accuracy within +-2 frames, Secondary: Lowest MAE.
    """
    alias_map = {
        "rf": "Random Forest Regressor",
        "random_forest": "Random Forest Regressor",
        "ridge": "Baseline (Ridge Regression)",
        "baseline": "Baseline (Ridge Regression)",
        "gbdt": "Gradient Boosting Regressor",
        "gradient_boosting": "Gradient Boosting Regressor",
        "hgb": "HistGradientBoosting Regressor",
        "hist_gbdt": "HistGradientBoosting Regressor",
        "xgb": "XGBoost Regressor",
        "xgboost": "XGBoost Regressor",
    }
    if explicit_choice != "auto":
        resolved = alias_map.get(explicit_choice.lower(), explicit_choice)
        if resolved in val_results:
            return resolved
        raise ValueError(
            f"Selected champion '{explicit_choice}' not found among candidates: {list(val_results.keys())}"
        )

    def score_key(item: Tuple[str, Dict[str, Any]]) -> Tuple:
        _, m = item
        mae = m["mae"]
        rmse = m["rmse"]
        r2 = m["r2_score"]
        pr = m["pearson_r"]
        tol = m["within_2_frames_pct"]

        if priority_metric == "mae":
            return (-mae, -rmse, r2, pr)
        elif priority_metric == "rmse":
            return (-rmse, -mae, r2, pr)
        elif priority_metric == "r2":
            return (r2, -mae, pr, -rmse)
        elif priority_metric == "pearson_r":
            return (pr, r2, -mae, -rmse)
        elif priority_metric == "tolerance":
            return (tol, -mae, -rmse, r2)
        return (-mae, -rmse, r2)

    ranked = sorted(val_results.items(), key=score_key, reverse=True)
    return ranked[0][0]


def main(priority_metric: str = "mae", champion_choice: str = "auto"):
    print("=" * 80)
    print("TRAINING & BENCHMARKING COLONY SIZE & FRAME COUNT REGRESSORS")
    print(f"Selection Policy: Priority Metric='{priority_metric}', Explicit Choice='{champion_choice}'")
    print("=" * 80)

    train_df, val_df, test_df, feature_cols = load_datasets()

    X_train = train_df[feature_cols].values
    y_train = train_df["frames_count"].values

    X_val = val_df[feature_cols].values
    y_val = val_df["frames_count"].values

    X_test = test_df[feature_cols].values
    y_test = test_df["frames_count"].values

    # Model 1: Baseline Ridge Regression
    ridge_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                ClippedRegressor(
                    Ridge(alpha=1.0, random_state=42),
                    min_val=0.0,
                    max_val=32.0,
                ),
            ),
        ]
    )

    # Model 2: Random Forest Regressor
    rf_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                ClippedRegressor(
                    RandomForestRegressor(
                        n_estimators=250,
                        max_depth=6,
                        min_samples_leaf=3,
                        random_state=42,
                        n_jobs=-1,
                    ),
                    min_val=0.0,
                    max_val=32.0,
                ),
            ),
        ]
    )

    # Model 3: Gradient Boosting Regressor
    gb_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                ClippedRegressor(
                    GradientBoostingRegressor(
                        n_estimators=150,
                        max_depth=3,
                        learning_rate=0.04,
                        subsample=0.8,
                        random_state=42,
                    ),
                    min_val=0.0,
                    max_val=32.0,
                ),
            ),
        ]
    )

    # Model 4: HistGradientBoosting Regressor
    hgb_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                ClippedRegressor(
                    HistGradientBoostingRegressor(
                        max_iter=150,
                        max_depth=4,
                        learning_rate=0.04,
                        random_state=42,
                    ),
                    min_val=0.0,
                    max_val=32.0,
                ),
            ),
        ]
    )

    candidates = {
        "Baseline (Ridge Regression)": ridge_pipe,
        "Random Forest Regressor": rf_pipe,
        "Gradient Boosting Regressor": gb_pipe,
        "HistGradientBoosting Regressor": hgb_pipe,
    }

    if HAVE_XGB:
        xgb_pipe = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    ClippedRegressor(
                        XGBRegressor(
                            n_estimators=200,
                            max_depth=4,
                            learning_rate=0.03,
                            random_state=42,
                        ),
                        min_val=0.0,
                        max_val=32.0,
                    ),
                ),
            ]
        )
        candidates["XGBoost Regressor"] = xgb_pipe

    val_results = {}
    print("\n--- Training Candidate Regressors ---")
    for name, pipe in candidates.items():
        print(f"Training {name}...")
        pipe.fit(X_train, y_train)
        metrics = evaluate_regressor(pipe, X_val, y_val, split_name="Validation")
        val_results[name] = metrics

    print("\n" + "=" * 80)
    print("VALIDATION REGRESSION BENCHMARK COMPARISON (14 UNSEEN HIVES)")
    print("=" * 80)
    header = f"{'Model':<32} | {'MAE (frames)':<12} | {'RMSE':<8} | {'R²':<8} | {'Pearson r':<10} | {'±2 frames':<10}"
    print(header)
    print("-" * len(header))
    for name, m in val_results.items():
        print(
            f"{name:<32} | {m['mae']:<12.3f} | {m['rmse']:<8.3f} | {m['r2_score']:<8.3f} | {m['pearson_r']:<10.3f} | {m['within_2_frames_pct']:5.1f}%"
        )

    champion_name = select_champion_regressor(
        val_results, priority_metric=priority_metric, explicit_choice=champion_choice
    )
    champion_pipe = candidates[champion_name]
    print(f"\n>>> Deployed Production Frame Count Regressor: {champion_name} (Policy: metric='{priority_metric}')")

    print("\n" + "=" * 80)
    print(f"FINAL TEST SET EVALUATION ({champion_name.upper()} on 14 HELD-OUT HIVES)")
    print("=" * 80)
    test_metrics = evaluate_regressor(champion_pipe, X_test, y_test, split_name="Test")
    print(f"Test Samples:              {test_metrics['sample_count']}")
    print(f"Test MAE (Mean Abs Error): {test_metrics['mae']:.3f} frames")
    print(f"Test RMSE:                 {test_metrics['rmse']:.3f} frames")
    print(f"Test R² Score:             {test_metrics['r2_score']:.4f}")
    print(f"Test Pearson Correlation:  {test_metrics['pearson_r']:.4f}")
    print(f"Accuracy within ±1 frame:  {test_metrics['within_1_frame_pct']:.1f}%")
    print(f"Accuracy within ±2 frames: {test_metrics['within_2_frames_pct']:.1f}%")
    print(f"Accuracy within ±3 frames: {test_metrics['within_3_frames_pct']:.1f}%")

    model_wrapper = champion_pipe.named_steps["model"]
    base_model = getattr(model_wrapper, "base_estimator", model_wrapper)

    top_features = []
    if hasattr(base_model, "feature_importances_"):
        importances = base_model.feature_importances_
        indices = np.argsort(importances)[::-1]
        for rank, idx in enumerate(indices, 1):
            fname = feature_cols[idx]
            imp = float(importances[idx])
            top_features.append({"rank": rank, "feature": fname, "importance": imp})
    elif hasattr(base_model, "coef_"):
        coefs = np.abs(base_model.coef_)
        norm_imp = coefs / np.sum(coefs)
        indices = np.argsort(norm_imp)[::-1]
        for rank, idx in enumerate(indices, 1):
            fname = feature_cols[idx]
            imp = float(norm_imp[idx])
            top_features.append({"rank": rank, "feature": fname, "importance": imp})

    model_out_path = ARTIFACTS_DIR / "frame_count_regressor.joblib"
    joblib.dump(champion_pipe, model_out_path)
    print(f"\nSaved Champion Regressor Pipeline to: {model_out_path.resolve()}")

    report = {
        "champion_model": champion_name,
        "feature_count": len(feature_cols),
        "validation_benchmarks": val_results,
        "test_evaluation": test_metrics,
        "top_features": top_features,
    }
    report_json_path = ARTIFACTS_DIR / "frame_count_regression_metrics.json"
    with open(report_json_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Saved Regression Benchmark Report to: {report_json_path.resolve()}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train & Benchmark MSPB2 Colony Size & Frame Count Regressor"
    )
    parser.add_argument(
        "--priority-metric",
        choices=["mae", "rmse", "r2", "pearson_r", "tolerance"],
        default="mae",
        help="Metric used to select the champion regressor model (default: mae)",
    )
    parser.add_argument(
        "--champion",
        choices=["auto", "rf", "ridge", "gbdt", "hgb", "xgboost"],
        default="auto",
        help="Explicit champion choice override (default: auto)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = parse_args()
    main(priority_metric=cli_args.priority_metric, champion_choice=cli_args.champion)
