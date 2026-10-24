"""
run_ablation_experiments.py
Runs comparative ablation experiments across feature groups using identical hive splits:
1. Temperature Only (15 features: full 24h stats + 6h/12h sub-windows + diurnal drop + linear trend + brood adherence)
2. Humidity Only (13 features: full 24h stats + 6h/12h sub-windows + linear trend + cv)
3. Time-Window Dynamics (12 features: 6h/12h/24h changes and ranges for temp & humidity)
4. All Environmental Features (All 40 features)

Evaluates BOTH tasks:
- TASK 1: Binary Colony Strength Classification (Strong >= 16 frames vs Weak < 16 frames)
- TASK 2: Continuous Colony Size & Frame Count Regression (frames_count: 1.0 to 30.0 frames)

Outputs detailed validation benchmarks, held-out test evaluations across 14 unseen hives,
per-hive comparative breakdown, and exports full results to artifacts/ablation_benchmark_results.json.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import (
    RandomForestClassifier,
    HistGradientBoostingClassifier,
    GradientBoostingClassifier,
    RandomForestRegressor,
    HistGradientBoostingRegressor,
    GradientBoostingRegressor,
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
import sys

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.shared.config import settings
from src.shared.dataset import METADATA_COLS

DATA_DIR = settings.MSPB2_BASE_DIR / "data"
ARTIFACTS_DIR = settings.MSPB2_ARTIFACTS_DIR
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


def load_datasets() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str]]:
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    val_df = pd.read_csv(DATA_DIR / "val.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")

    all_features = [c for c in train_df.columns if c not in METADATA_COLS]

    print(f"Loaded datasets with identical hive-grouped splits:")
    print(f"  Train: {len(train_df)} samples, {train_df['hive_id'].nunique()} hives")
    print(f"  Val:   {len(val_df)} samples, {val_df['hive_id'].nunique()} hives")
    print(f"  Test:  {len(test_df)} samples, {test_df['hive_id'].nunique()} hives")
    print(f"  Available feature count: {len(all_features)}")
    return train_df, val_df, test_df, all_features


def get_feature_groups(all_features: List[str]) -> Dict[str, List[str]]:
    temp_features = [
        "temperature_mean",
        "temperature_std",
        "temperature_min",
        "temperature_max",
        "temperature_range",
        "temp_cv",
        "temp_diurnal_drop",
        "temp_linear_trend",
        "temperature_mean_6h",
        "temperature_mean_12h",
        "temperature_std_6h",
        "temperature_std_12h",
        "temperature_min_6h",
        "temperature_max_6h",
        "brood_temp_adherence",
    ]
    humidity_features = [
        "humidity_mean",
        "humidity_std",
        "humidity_min",
        "humidity_max",
        "humidity_range",
        "humidity_cv",
        "humidity_linear_trend",
        "humidity_mean_6h",
        "humidity_mean_12h",
        "humidity_std_6h",
        "humidity_std_12h",
        "humidity_min_6h",
        "humidity_max_6h",
    ]
    time_window_features = [
        "temperature_change_6h",
        "temperature_change_12h",
        "temperature_change_24h",
        "temperature_range_6h",
        "temperature_range_12h",
        "temperature_range_24h",
        "humidity_change_6h",
        "humidity_change_12h",
        "humidity_change_24h",
        "humidity_range_6h",
        "humidity_range_12h",
        "humidity_range_24h",
    ]
    all_env_features = [f for f in (temp_features + humidity_features + time_window_features) if f in all_features]
    available_temp = [f for f in temp_features if f in all_features]
    available_hum = [f for f in humidity_features if f in all_features]
    available_tw = [f for f in time_window_features if f in all_features]

    groups = {}
    if available_temp:
        groups["Temperature Only (15)"] = available_temp
    if available_hum:
        groups["Humidity Only (13)"] = available_hum
    if available_tw:
        groups["Time-Window Dynamics (12)"] = available_tw
    groups["All Environmental (40)"] = all_env_features

    return groups


# =========================================================================
# CLASSIFICATION EVALUATION & MODELS
# =========================================================================

def evaluate_classification(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, Any]:
    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro"))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted"))

    p_weak = float(precision_score(y_true, y_pred, pos_label=0, zero_division=0))
    r_weak = float(recall_score(y_true, y_pred, pos_label=0, zero_division=0))
    f1_weak = float(f1_score(y_true, y_pred, pos_label=0, zero_division=0))

    p_strong = float(precision_score(y_true, y_pred, pos_label=1, zero_division=0))
    r_strong = float(recall_score(y_true, y_pred, pos_label=1, zero_division=0))
    f1_strong = float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))

    try:
        roc_auc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        roc_auc = 0.5

    try:
        pr_auc = float(average_precision_score(y_true, y_prob))
    except Exception:
        pr_auc = 0.5

    cm = confusion_matrix(y_true, y_pred).tolist()

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "weak_recall": r_weak,
        "weak_precision": p_weak,
        "weak_f1": f1_weak,
        "strong_recall": r_strong,
        "strong_precision": p_strong,
        "strong_f1": f1_strong,
        "confusion_matrix": cm,
    }


def create_classifier_suite() -> Dict[str, Pipeline]:
    return {
        "Baseline (Logistic Regression)": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        C=0.1,
                        class_weight="balanced",
                        max_iter=1000,
                        random_state=42,
                    ),
                ),
            ]
        ),
        "Random Forest": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=200,
                        max_depth=6,
                        min_samples_leaf=2,
                        class_weight="balanced",
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "HistGradientBoosting": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        max_iter=150,
                        max_depth=4,
                        learning_rate=0.05,
                        class_weight="balanced",
                        random_state=42,
                    ),
                ),
            ]
        ),
        "Gradient Boosting (GBDT)": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    GradientBoostingClassifier(
                        n_estimators=150,
                        max_depth=3,
                        learning_rate=0.05,
                        subsample=0.8,
                        random_state=42,
                    ),
                ),
            ]
        ),
    }


def run_classification_experiment(
    exp_name: str,
    feature_cols: List[str],
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> Dict[str, Any]:
    print("\n" + "=" * 80)
    print(f"[CLASSIFICATION] RUNNING: {exp_name.upper()} ({len(feature_cols)} FEATURES)")
    print("=" * 80)

    X_train = train_df[feature_cols].values
    y_train = train_df["label"].values

    X_val = val_df[feature_cols].values
    y_val = val_df["label"].values

    X_test = test_df[feature_cols].values
    y_test = test_df["label"].values

    suite = create_classifier_suite()
    val_metrics = {}
    fitted_pipes = {}

    for m_name, pipe in suite.items():
        pipe.fit(X_train, y_train)
        fitted_pipes[m_name] = pipe

        y_val_pred = pipe.predict(X_val)
        y_val_prob = pipe.predict_proba(X_val)[:, 1] if hasattr(pipe, "predict_proba") else y_val_pred
        val_metrics[m_name] = evaluate_classification(y_val, y_val_pred, y_val_prob)

    champion_name = max(val_metrics.keys(), key=lambda k: val_metrics[k]["macro_f1"])
    champion_pipe = fitted_pipes[champion_name]

    test_metrics_by_model = {}
    test_predictions_by_model = {}

    for m_name, pipe in fitted_pipes.items():
        y_test_pred = pipe.predict(X_test)
        y_test_prob = pipe.predict_proba(X_test)[:, 1] if hasattr(pipe, "predict_proba") else y_test_pred
        test_metrics_by_model[m_name] = evaluate_classification(y_test, y_test_pred, y_test_prob)
        test_predictions_by_model[m_name] = {
            "pred": y_test_pred,
            "prob": y_test_prob,
        }

    rf_test = test_metrics_by_model["Random Forest"]
    champ_test = test_metrics_by_model[champion_name]

    print(f"Validation Champion: {champion_name} (Val Macro F1: {val_metrics[champion_name]['macro_f1']:.3f})")
    print(f"Held-out Test Performance ({champion_name}):")
    print(f"  Test Accuracy:    {champ_test['accuracy']*100:5.1f}%")
    print(f"  Test Macro F1:    {champ_test['macro_f1']:7.3f}")
    print(f"  Test Weak Recall: {champ_test['weak_recall']*100:5.1f}%")
    print(f"  Test Weak F1:     {champ_test['weak_f1']:7.3f}")
    print(f"  Test ROC-AUC:     {champ_test['roc_auc']:7.3f}")

    print(f"Held-out Test Performance (Random Forest comparator):")
    print(f"  RF Test Accuracy:    {rf_test['accuracy']*100:5.1f}%")
    print(f"  RF Test Macro F1:    {rf_test['macro_f1']:7.3f}")
    print(f"  RF Test Weak Recall: {rf_test['weak_recall']*100:5.1f}%")
    print(f"  RF Test ROC-AUC:     {rf_test['roc_auc']:7.3f}")

    return {
        "experiment_name": exp_name,
        "feature_count": len(feature_cols),
        "feature_names": feature_cols,
        "champion_model": champion_name,
        "validation_metrics": val_metrics,
        "test_metrics_by_model": test_metrics_by_model,
        "champion_test_metrics": champ_test,
        "rf_test_metrics": rf_test,
        "test_predictions": test_predictions_by_model,
    }


# =========================================================================
# REGRESSION EVALUATION & MODELS
# =========================================================================

def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    y_pred_clipped = np.clip(y_pred, 0.0, 32.0)
    mae = float(mean_absolute_error(y_true, y_pred_clipped))
    mse = float(mean_squared_error(y_true, y_pred_clipped))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_true, y_pred_clipped))

    try:
        corr, _ = pearsonr(y_true, y_pred_clipped)
        r_val = float(corr)
    except Exception:
        r_val = 0.0

    errors = np.abs(y_true - y_pred_clipped)
    within_1 = float(np.mean(errors <= 1.0) * 100)
    within_2 = float(np.mean(errors <= 2.0) * 100)
    within_3 = float(np.mean(errors <= 3.0) * 100)

    return {
        "mae": round(mae, 3),
        "rmse": round(rmse, 3),
        "r2_score": round(r2, 4),
        "pearson_r": round(r_val, 4),
        "within_1_frame_pct": round(within_1, 1),
        "within_2_frames_pct": round(within_2, 1),
        "within_3_frames_pct": round(within_3, 1),
    }


def create_regressor_suite() -> Dict[str, Pipeline]:
    return {
        "Baseline (Ridge)": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=1.0, random_state=42)),
            ]
        ),
        "Random Forest Regressor": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=250,
                        max_depth=6,
                        min_samples_leaf=3,
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "HistGradientBoosting Regressor": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    HistGradientBoostingRegressor(
                        max_iter=150,
                        max_depth=4,
                        learning_rate=0.04,
                        random_state=42,
                    ),
                ),
            ]
        ),
        "Gradient Boosting Regressor": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    GradientBoostingRegressor(
                        n_estimators=150,
                        max_depth=3,
                        learning_rate=0.04,
                        subsample=0.8,
                        random_state=42,
                    ),
                ),
            ]
        ),
    }


def run_regression_experiment(
    exp_name: str,
    feature_cols: List[str],
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> Dict[str, Any]:
    print("\n" + "=" * 80)
    print(f"[REGRESSION] RUNNING: {exp_name.upper()} ({len(feature_cols)} FEATURES)")
    print("=" * 80)

    X_train = train_df[feature_cols].values
    y_train = train_df["frames_count"].values

    X_val = val_df[feature_cols].values
    y_val = val_df["frames_count"].values

    X_test = test_df[feature_cols].values
    y_test = test_df["frames_count"].values

    suite = create_regressor_suite()
    val_metrics = {}
    fitted_pipes = {}

    for m_name, pipe in suite.items():
        pipe.fit(X_train, y_train)
        fitted_pipes[m_name] = pipe

        y_val_pred = pipe.predict(X_val)
        val_metrics[m_name] = evaluate_regression(y_val, y_val_pred)

    champion_name = min(val_metrics.keys(), key=lambda k: val_metrics[k]["mae"])
    if "Random Forest Regressor" in val_metrics:
        if val_metrics["Random Forest Regressor"]["mae"] <= val_metrics[champion_name]["mae"] + 0.15:
            champion_name = "Random Forest Regressor"

    champion_pipe = fitted_pipes[champion_name]

    test_metrics_by_model = {}
    test_predictions_by_model = {}

    for m_name, pipe in fitted_pipes.items():
        y_test_pred = np.clip(pipe.predict(X_test), 0.0, 32.0)
        test_metrics_by_model[m_name] = evaluate_regression(y_test, y_test_pred)
        test_predictions_by_model[m_name] = y_test_pred

    rf_test = test_metrics_by_model["Random Forest Regressor"]
    champ_test = test_metrics_by_model[champion_name]

    print(f"Validation Champion: {champion_name} (Val MAE: {val_metrics[champion_name]['mae']:.3f} frames)")
    print(f"Held-out Test Performance ({champion_name}):")
    print(f"  Test MAE:       {champ_test['mae']:.3f} frames")
    print(f"  Test RMSE:      {champ_test['rmse']:.3f} frames")
    print(f"  Test R²:        {champ_test['r2_score']:.4f}")
    print(f"  Test Pearson r: {champ_test['pearson_r']:.4f}")
    print(f"  Within ±2 f:    {champ_test['within_2_frames_pct']:.1f}%")

    print(f"Held-out Test Performance (Random Forest Regressor comparator):")
    print(f"  RF Test MAE:       {rf_test['mae']:.3f} frames")
    print(f"  RF Test RMSE:      {rf_test['rmse']:.3f} frames")
    print(f"  RF Test R²:        {rf_test['r2_score']:.4f}")
    print(f"  RF Test Pearson r: {rf_test['pearson_r']:.4f}")

    return {
        "experiment_name": exp_name,
        "feature_count": len(feature_cols),
        "feature_names": feature_cols,
        "champion_model": champion_name,
        "validation_metrics": val_metrics,
        "test_metrics_by_model": test_metrics_by_model,
        "champion_test_metrics": champ_test,
        "rf_test_metrics": rf_test,
        "test_predictions": test_predictions_by_model,
    }


# =========================================================================
# PER-HIVE BREAKDOWNS
# =========================================================================

def compute_per_hive_classification(
    test_df: pd.DataFrame,
    clf_results: Dict[str, Dict[str, Any]],
    model_key: str = "Random Forest",
) -> pd.DataFrame:
    hive_records = []
    unique_hives = sorted(test_df["hive_id"].unique())

    for hive_id in unique_hives:
        hive_mask = (test_df["hive_id"] == hive_id).values
        n_samples = int(hive_mask.sum())
        y_true = test_df.loc[hive_mask, "label"].values

        strong_cnt = int((y_true == 1).sum())
        weak_cnt = int((y_true == 0).sum())

        row = {
            "hive_id": hive_id,
            "samples": n_samples,
            "strong_samples": strong_cnt,
            "weak_samples": weak_cnt,
            "majority_label": "Strong" if strong_cnt >= weak_cnt else "Weak",
        }

        for exp_key in clf_results.keys():
            preds = clf_results[exp_key]["test_predictions"][model_key]["pred"][hive_mask]
            acc = float(accuracy_score(y_true, preds))
            row[f"{exp_key}_acc"] = acc

        hive_records.append(row)

    return pd.DataFrame(hive_records)


def compute_per_hive_regression(
    test_df: pd.DataFrame,
    reg_results: Dict[str, Dict[str, Any]],
    model_key: str = "Random Forest Regressor",
) -> pd.DataFrame:
    hive_records = []
    unique_hives = sorted(test_df["hive_id"].unique())

    for hive_id in unique_hives:
        hive_mask = (test_df["hive_id"] == hive_id).values
        n_samples = int(hive_mask.sum())
        y_true = test_df.loc[hive_mask, "frames_count"].values

        row = {
            "hive_id": hive_id,
            "samples": n_samples,
            "mean_true_frames": round(float(y_true.mean()), 1),
            "min_true_frames": round(float(y_true.min()), 1),
            "max_true_frames": round(float(y_true.max()), 1),
        }

        for exp_key in reg_results.keys():
            preds = reg_results[exp_key]["test_predictions"][model_key][hive_mask]
            mae = float(mean_absolute_error(y_true, preds))
            row[f"{exp_key}_mae"] = round(mae, 2)

        hive_records.append(row)

    return pd.DataFrame(hive_records)


# =========================================================================
# MAIN
# =========================================================================

def main():
    print("=" * 80)
    print("MSPB2 COMPREHENSIVE ABLATION BENCHMARK EXPERIMENTS")
    print("Tasks: (1) Colony Strength Classification & (2) Frame Count Regression")
    print("=" * 80)

    train_df, val_df, test_df, all_features = load_datasets()
    feature_groups = get_feature_groups(all_features)

    print("\nIdentified Feature Ablation Groups:")
    for g_name, g_feats in feature_groups.items():
        print(f"  - {g_name:<30}: {len(g_feats)} features -> {g_feats[:3]}...")

    # 1. Run Classification Ablation
    print("\n" + "#" * 80)
    print("PART 1: BINARY COLONY STRENGTH CLASSIFICATION ABLATION")
    print("#" * 80)
    clf_results = {}
    for g_name, g_feats in feature_groups.items():
        res = run_classification_experiment(g_name, g_feats, train_df, val_df, test_df)
        clf_results[g_name] = res

    # 2. Run Regression Ablation
    print("\n" + "#" * 80)
    print("PART 2: EXACT COLONY SIZE & FRAME COUNT REGRESSION ABLATION")
    print("#" * 80)
    reg_results = {}
    for g_name, g_feats in feature_groups.items():
        res = run_regression_experiment(g_name, g_feats, train_df, val_df, test_df)
        reg_results[g_name] = res

    # 3. Print Summary Tables
    print("\n" + "=" * 96)
    print("ABLATION SUMMARY 1: BINARY COLONY STRENGTH CLASSIFICATION (14 HELD-OUT TEST HIVES)")
    print("=" * 96)
    c_header = f"{'Feature Group':<32} | {'Feats':<5} | {'Model':<16} | {'Macro F1':<8} | {'Weak Rec':<8} | {'ROC-AUC':<8} | {'Accuracy':<8}"
    print(c_header)
    print("-" * len(c_header))
    for name in feature_groups.keys():
        rf_m = clf_results[name]["rf_test_metrics"]
        fc = clf_results[name]["feature_count"]
        print(
            f"{name:<32} | {fc:<5} | {'Random Forest':<16} | {rf_m['macro_f1']:8.3f} | {rf_m['weak_recall']*100:6.1f}% | {rf_m['roc_auc']:8.3f} | {rf_m['accuracy']*100:5.1f}%"
        )

    print("\n" + "=" * 96)
    print("ABLATION SUMMARY 2: CONTINUOUS FRAME COUNT REGRESSION (14 HELD-OUT TEST HIVES)")
    print("=" * 96)
    r_header = f"{'Feature Group':<32} | {'Feats':<5} | {'Model':<24} | {'MAE (frames)':<12} | {'RMSE':<8} | {'R²':<8} | {'Pearson r':<10} | {'±2 frames':<10}"
    print(r_header)
    print("-" * len(r_header))
    for name in feature_groups.keys():
        rf_m = reg_results[name]["rf_test_metrics"]
        fc = reg_results[name]["feature_count"]
        print(
            f"{name:<32} | {fc:<5} | {'Random Forest Regressor':<24} | {rf_m['mae']:<12.3f} | {rf_m['rmse']:<8.3f} | {rf_m['r2_score']:<8.3f} | {rf_m['pearson_r']:<10.3f} | {rf_m['within_2_frames_pct']:5.1f}%"
        )

    # 4. Print Per-Hive Breakdown Tables
    print("\n" + "=" * 96)
    print(f"PER-HIVE TEST BREAKDOWN: CLASSIFICATION ACCURACY ({test_df['hive_id'].nunique()} HELD-OUT HIVES)")
    print("=" * 96)
    clf_hive_df = compute_per_hive_classification(test_df, clf_results, model_key="Random Forest")
    exp_names = list(feature_groups.keys())
    col_headers = [f"{name[:14]:<14}" for name in exp_names]
    hive_header = f"{'Hive ID':<10} | {'Total':<5} | {'Strong':<6} | {'Weak':<5} | " + " | ".join(col_headers)
    print(hive_header)
    print("-" * len(hive_header))
    for _, r in clf_hive_df.iterrows():
        acc_strs = [f"{r.get(f'{name}_acc', 0.0)*100:13.1f}%" for name in exp_names]
        print(f"{r['hive_id']:<10} | {r['samples']:<5} | {r['strong_samples']:<6} | {r['weak_samples']:<5} | " + " | ".join(acc_strs))
    print("-" * len(hive_header))
    avg_strs = [f"{clf_hive_df[f'{name}_acc'].mean()*100:13.1f}%" for name in exp_names]
    print(f"{'AVERAGE':<10} | {len(test_df):<5} | {(test_df['label']==1).sum():<6} | {(test_df['label']==0).sum():<5} | " + " | ".join(avg_strs))

    print("\n" + "=" * 96)
    print(f"PER-HIVE TEST BREAKDOWN: REGRESSION MAE (FRAMES) ({test_df['hive_id'].nunique()} HELD-OUT HIVES)")
    print("=" * 96)
    reg_hive_df = compute_per_hive_regression(test_df, reg_results, model_key="Random Forest Regressor")
    col_headers_reg = [f"{name[:14]:<14}" for name in exp_names]
    hive_header_reg = f"{'Hive ID':<10} | {'Total':<5} | {'True Mean':<10} | {'Range':<10} | " + " | ".join(col_headers_reg)
    print(hive_header_reg)
    print("-" * len(hive_header_reg))
    for _, r in reg_hive_df.iterrows():
        mae_strs = [f"{r.get(f'{name}_mae', 0.0):13.2f}f" for name in exp_names]
        range_str = f"{r['min_true_frames']}-{r['max_true_frames']}"
        print(f"{r['hive_id']:<10} | {r['samples']:<5} | {r['mean_true_frames']:<10.1f} | {range_str:<10} | " + " | ".join(mae_strs))
    print("-" * len(hive_header_reg))
    avg_mae_strs = [f"{reg_hive_df[f'{name}_mae'].mean():13.2f}f" for name in exp_names]
    print(f"{'AVERAGE':<10} | {len(test_df):<5} | {test_df['frames_count'].mean():<10.1f} | {'1.0-30.0':<10} | " + " | ".join(avg_mae_strs))

    # 5. Export JSON
    export_payload = {
        "classification": {
            "experiments": {},
            "per_hive_breakdown": clf_hive_df.to_dict(orient="records"),
            "hive_level_average_accuracy": {
                name: float(clf_hive_df[f"{name}_acc"].mean() * 100) for name in exp_names
            },
        },
        "regression": {
            "experiments": {},
            "per_hive_breakdown": reg_hive_df.to_dict(orient="records"),
            "hive_level_average_mae": {
                name: float(reg_hive_df[f"{name}_mae"].mean()) for name in exp_names
            },
        },
    }

    for exp_name, data in clf_results.items():
        export_payload["classification"]["experiments"][exp_name] = {
            "feature_count": data["feature_count"],
            "feature_names": data["feature_names"],
            "champion_model": data["champion_model"],
            "rf_test_metrics": data["rf_test_metrics"],
            "champion_test_metrics": data["champion_test_metrics"],
            "all_models_test_metrics": data["test_metrics_by_model"],
        }

    for exp_name, data in reg_results.items():
        export_payload["regression"]["experiments"][exp_name] = {
            "feature_count": data["feature_count"],
            "feature_names": data["feature_names"],
            "champion_model": data["champion_model"],
            "rf_test_metrics": data["rf_test_metrics"],
            "champion_test_metrics": data["champion_test_metrics"],
            "all_models_test_metrics": data["test_metrics_by_model"],
        }

    results_path = ARTIFACTS_DIR / "ablation_benchmark_results.json"
    with open(results_path, "w") as f:
        json.dump(export_payload, f, indent=2)
    print(f"\n[OK] Full dual-task ablation results exported to: {results_path.resolve()}")


if __name__ == "__main__":
    main()
