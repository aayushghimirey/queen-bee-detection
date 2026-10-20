"""
run_ablation_experiments.py
Runs three comparative experiments using identical hive splits:
1. Environmental only: 10 temperature and humidity features.
2. Audio only: 53 frequency-domain features (plus 64 all-audio supplementary benchmark).
3. Combined: all 75 features.

Evaluates Test Macro F1, Weak Recall, ROC-AUC, PR-AUC, and per-hive breakdown
across the 14 unseen held-out test hives.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    HistGradientBoostingClassifier,
    GradientBoostingClassifier,
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

BASE_DIR = Path("/Users/aayushghimire/Datasets/MSPB2")
DATA_DIR = BASE_DIR / "data"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

METADATA_COLS = [
    "sample_id",
    "hive_id",
    "raw_hive_id",
    "season",
    "apiary",
    "inspection_date",
    "window_readings_count",
    "frames_count",
    "label",
]


def load_datasets() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str]]:
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    val_df = pd.read_csv(DATA_DIR / "val.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")

    all_features = [c for c in train_df.columns if c not in METADATA_COLS]

    print(f"Loaded datasets with identical splits:")
    print(f"  Train: {len(train_df)} samples, {train_df['hive_id'].nunique()} hives")
    print(f"  Val:   {len(val_df)} samples, {val_df['hive_id'].nunique()} hives")
    print(f"  Test:  {len(test_df)} samples, {test_df['hive_id'].nunique()} hives")
    print(f"  Available feature count: {len(all_features)}")
    return train_df, val_df, test_df, all_features


def get_feature_groups(all_features: List[str]) -> Dict[str, List[str]]:
    groups: Dict[str, List[str]] = {}

    # 1. Environmental: Temperature and Humidity features
    temp_features = [
        "temperature_mean",
        "temperature_std",
        "temperature_range",
        "temp_cv",
        "brood_temp_adherence",
        "temp_diurnal_drop",
        "temp_linear_trend",
    ]
    humidity_features = [
        "humidity_mean",
        "humidity_std",
        "humidity_range",
    ]
    env_features = [f for f in (temp_features + humidity_features) if f in all_features]
    available_temp = [f for f in temp_features if f in all_features]
    available_hum = [f for f in humidity_features if f in all_features]

    # Check for audio features
    freq_band_features = [c for c in all_features if c.startswith("hz_")]
    freq_aggregates = [
        "spectral_centroid",
        "piping_to_fanning_ratio",
        "low_band_power_sum",
        "high_band_power_sum",
        "dominant_freq_bin",
    ]
    freq_domain_features = [c for c in (freq_band_features + freq_aggregates) if c in all_features]

    audio_density_features = [
        "audio_density_mean",
        "audio_density_std",
        "audio_density_range",
        "audio_density_ratio_mean",
        "audio_density_ratio_std",
        "audio_density_ratio_range",
        "density_variation_mean",
        "density_variation_std",
        "density_variation_range",
        "audio_diurnal_dynamism",
        "audio_density_iqr",
    ]
    audio_density_features = [c for c in audio_density_features if c in all_features]
    all_audio_features = freq_domain_features + audio_density_features

    has_audio = len(freq_domain_features) > 0

    if has_audio:
        # Full multi-modal ablation
        groups["Environmental Only"] = env_features
        groups["Audio Only (Freq-Domain)"] = freq_domain_features
        if len(all_audio_features) > len(freq_domain_features):
            groups["Audio Only (All Audio)"] = all_audio_features
        groups["Combined (All Features)"] = list(all_features)
    else:
        # Environmental microclimate component ablation
        if available_temp:
            groups["Temperature Only (7)"] = available_temp
        if available_hum:
            groups["Humidity Only (3)"] = available_hum
        groups["Environmental Microclimate (10)"] = env_features

    return groups


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, Any]:
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


def create_model_suite() -> Dict[str, Pipeline]:
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


def run_experiment(
    exp_name: str,
    feature_cols: List[str],
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> Dict[str, Any]:
    print("\n" + "=" * 80)
    print(f"RUNNING: {exp_name.upper()} ({len(feature_cols)} FEATURES)")
    print("=" * 80)

    X_train = train_df[feature_cols].values
    y_train = train_df["label"].values

    X_val = val_df[feature_cols].values
    y_val = val_df["label"].values

    X_test = test_df[feature_cols].values
    y_test = test_df["label"].values

    model_suite = create_model_suite()
    val_metrics = {}
    fitted_pipes = {}

    for m_name, pipe in model_suite.items():
        pipe.fit(X_train, y_train)
        fitted_pipes[m_name] = pipe

        y_val_pred = pipe.predict(X_val)
        y_val_prob = pipe.predict_proba(X_val)[:, 1] if hasattr(pipe, "predict_proba") else y_val_pred
        val_metrics[m_name] = evaluate_predictions(y_val, y_val_pred, y_val_prob)

    # Pick champion on validation Macro F1
    champion_name = max(val_metrics.keys(), key=lambda k: val_metrics[k]["macro_f1"])
    champion_pipe = fitted_pipes[champion_name]

    # Evaluate on held-out test
    test_metrics_by_model = {}
    test_predictions_by_model = {}

    for m_name, pipe in fitted_pipes.items():
        y_test_pred = pipe.predict(X_test)
        y_test_prob = pipe.predict_proba(X_test)[:, 1] if hasattr(pipe, "predict_proba") else y_test_pred
        test_metrics_by_model[m_name] = evaluate_predictions(y_test, y_test_pred, y_test_prob)
        test_predictions_by_model[m_name] = {
            "pred": y_test_pred,
            "prob": y_test_prob,
        }

    # Focus on Random Forest (consistent across all) and Champion
    rf_test = test_metrics_by_model["Random Forest"]
    champ_test = test_metrics_by_model[champion_name]

    print(f"Validation Champion: {champion_name} (Val Macro F1: {val_metrics[champion_name]['macro_f1']:.3f})")
    print(f"Held-out Test Performance ({champion_name}):")
    print(f"  Test Accuracy:    {champ_test['accuracy']*100:5.1f}%")
    print(f"  Test Macro F1:    {champ_test['macro_f1']:7.3f}")
    print(f"  Test Weak Recall: {champ_test['weak_recall']*100:5.1f}% ({champ_test['confusion_matrix'][0][0]}/{champ_test['confusion_matrix'][0][0]+champ_test['confusion_matrix'][0][1]})")
    print(f"  Test Weak F1:     {champ_test['weak_f1']:7.3f}")
    print(f"  Test ROC-AUC:     {champ_test['roc_auc']:7.3f}")
    print(f"  Test PR-AUC:      {champ_test['pr_auc']:7.3f}")

    print(f"Held-out Test Performance (Random Forest standard comparator):")
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


def compute_per_hive_metrics(
    test_df: pd.DataFrame,
    exp_results: Dict[str, Dict[str, Any]],
    model_key: str = "Random Forest",
) -> pd.DataFrame:
    """
    Computes per-hive accuracy, sample count, label composition, and predictions across experiments.
    """
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

        for exp_key in exp_results.keys():
            preds = exp_results[exp_key]["test_predictions"][model_key]["pred"][hive_mask]
            probs = exp_results[exp_key]["test_predictions"][model_key]["prob"][hive_mask]
            acc = float(accuracy_score(y_true, preds))
            mean_p = float(probs.mean())
            correct_cnt = int((y_true == preds).sum())

            row[f"{exp_key}_acc"] = acc
            row[f"{exp_key}_correct"] = f"{correct_cnt}/{n_samples}"
            row[f"{exp_key}_mean_prob_strong"] = mean_p

        hive_records.append(row)

    return pd.DataFrame(hive_records)


def main():
    print("=" * 80)
    print("ABLATION STUDY: SENSOR COMPONENT & ARCHITECTURE BENCHMARK")
    print("Zero-Leakage Hive Partitioning (MSPB2 Benchmark)")
    print("=" * 80)

    train_df, val_df, test_df, all_features = load_datasets()
    feature_groups = get_feature_groups(all_features)

    experiments = [(name, fcols) for name, fcols in feature_groups.items()]

    exp_results = {}
    for name, fcols in experiments:
        exp_results[name] = run_experiment(name, fcols, train_df, val_df, test_df)

    # ---------------------------------------------------------
    # COMPARATIVE SUMMARY TABLE
    # ---------------------------------------------------------
    print("\n" + "=" * 92)
    print(f"OVERALL HELD-OUT TEST PERFORMANCE COMPARISON ({test_df['hive_id'].nunique()} UNSEEN HIVES, {len(test_df)} SAMPLES)")
    print("=" * 92)
    header = f"{'Experiment':<32} | {'Feats':<5} | {'Model':<16} | {'Macro F1':<8} | {'Weak Rec':<8} | {'ROC-AUC':<8} | {'PR-AUC':<8} | {'Acc':<6}"
    print(header)
    print("-" * len(header))

    # Show Random Forest (Standard fixed architecture)
    print("--- Standard Architecture: Random Forest (Comparable across all sets) ---")
    for name, _ in experiments:
        rf_m = exp_results[name]["rf_test_metrics"]
        fc = exp_results[name]["feature_count"]
        print(
            f"{name:<32} | {fc:<5} | {'Random Forest':<16} | {rf_m['macro_f1']:8.3f} | {rf_m['weak_recall']*100:6.1f}% | {rf_m['roc_auc']:8.3f} | {rf_m['pr_auc']:8.3f} | {rf_m['accuracy']*100:5.1f}%"
        )

    print("\n--- Validation Selected Champion Models ---")
    for name, _ in experiments:
        champ_name = exp_results[name]["champion_model"]
        champ_m = exp_results[name]["champion_test_metrics"]
        fc = exp_results[name]["feature_count"]
        print(
            f"{name:<32} | {fc:<5} | {champ_name:<16} | {champ_m['macro_f1']:8.3f} | {champ_m['weak_recall']*100:6.1f}% | {champ_m['roc_auc']:8.3f} | {champ_m['pr_auc']:8.3f} | {champ_m['accuracy']*100:5.1f}%"
        )

    # ---------------------------------------------------------
    # PER-HIVE BREAKDOWN TABLE
    # ---------------------------------------------------------
    print("\n" + "=" * 96)
    print(f"PER-HIVE TEST SET BREAKDOWN (RANDOM FOREST ON {test_df['hive_id'].nunique()} HELD-OUT HIVES)")
    print("=" * 96)
    hive_df = compute_per_hive_metrics(test_df, exp_results, model_key="Random Forest")

    exp_names = list(exp_results.keys())
    col_headers = [f"{name[:14]:<14}" for name in exp_names]
    hive_header = f"{'Hive ID':<10} | {'Total':<5} | {'Strong':<6} | {'Weak':<5} | " + " | ".join(col_headers)
    print(hive_header)
    print("-" * len(hive_header))
    for _, r in hive_df.iterrows():
        acc_strs = [f"{r.get(f'{name}_acc', 0.0)*100:13.1f}%" for name in exp_names]
        print(
            f"{r['hive_id']:<10} | {r['samples']:<5} | {r['strong_samples']:<6} | {r['weak_samples']:<5} | "
            + " | ".join(acc_strs)
        )

    # Summary of Hive-Level Accuracy
    print("-" * len(hive_header))
    avg_strs = [f"{hive_df[f'{name}_acc'].mean()*100:13.1f}%" for name in exp_names]
    total_test_samples = len(test_df)
    total_strong = int((test_df['label'] == 1).sum())
    total_weak = int((test_df['label'] == 0).sum())
    print(
        f"{'AVERAGE':<10} | {total_test_samples:<5} | {total_strong:<6} | {total_weak:<5} | "
        + " | ".join(avg_strs)
    )

    # ---------------------------------------------------------
    # DETAILED PER-HIVE ANALYSIS & CASE STUDIES
    # ---------------------------------------------------------
    print("\n" + "=" * 80)
    print("HIVE-BY-HIVE COMPARATIVE ANALYSIS")
    print("=" * 80)
    baseline_name = exp_names[0]
    for _, r in hive_df.iterrows():
        hid = r["hive_id"]
        base_acc = r[f"{baseline_name}_acc"] * 100
        line_parts = [f"Hive {hid:<10}: {baseline_name}={base_acc:5.1f}%"]
        for other_name in exp_names[1:]:
            other_acc = r[f"{other_name}_acc"] * 100
            diff = other_acc - base_acc
            diff_str = f"(+{diff:.1f}%)" if diff > 0 else (f"({diff:.1f}%)" if diff < 0 else "(=)")
            line_parts.append(f"{other_name}={other_acc:5.1f}% {diff_str}")
        print(" | ".join(line_parts))

    # ---------------------------------------------------------
    # EXPORT RESULTS TO JSON
    # ---------------------------------------------------------
    export_payload = {
        "experiments": {},
        "per_hive_breakdown": hive_df.to_dict(orient="records"),
        "hive_level_average_accuracy": {
            name: float(hive_df[f"{name}_acc"].mean() * 100) for name in exp_names
        },
    }

    for exp_name, data in exp_results.items():
        # Strip numpy arrays for json serialization
        export_payload["experiments"][exp_name] = {
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
    print(f"\n[OK] Full ablation results exported to: {results_path.resolve()}")


if __name__ == "__main__":
    main()
