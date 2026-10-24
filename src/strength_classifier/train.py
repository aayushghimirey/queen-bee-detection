
"""
src/strength_classifier/train.py

Train and benchmark binary honey bee colony strength classifiers.

Labels:
    0 = Weak
    1 = Strong

Decision rule:
    Predict Weak when P(Weak) >= weak_threshold.

Metrics:
    Weak Recall, Weak Precision and Weak F1 treat Weak (class 0)
    as the positive class.
    False Alarm Rate = Strong samples incorrectly predicted Weak
                       / total actual Strong samples.

Artifacts:
    artifacts/colony_strength_model.joblib
    artifacts/feature_names.json
    artifacts/evaluation_metrics.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np

# Make the project root importable when this file is executed directly.
BASE_DIR = Path(__file__).resolve().parent.parent.parent

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.shared.config import settings
from src.shared.dataset import load_datasets
from src.strength_classifier.model import (
    ColonyStrengthClassifier,
    build_classifier_pipeline,
    HAVE_XGB,
)

ARTIFACTS_DIR = settings.MSPB2_ARTIFACTS_DIR
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

WEAK = 0
STRONG = 1
DEFAULT_WEAK_THRESHOLD = 0.425


def get_class_probability(
    pipeline: Any,
    X: np.ndarray,
    class_label: int,
) -> Optional[np.ndarray]:
    """
    Return probabilities for a particular class using the estimator's
    actual class ordering instead of assuming a fixed column position.
    """
    if not hasattr(pipeline, "predict_proba"):
        return None

    probabilities = np.asarray(pipeline.predict_proba(X))
    classes = list(pipeline.classes_)

    if class_label not in classes:
        raise ValueError(
            f"Class {class_label} not found in estimator classes {classes}"
        )

    class_index = classes.index(class_label)
    return probabilities[:, class_index]


def evaluate_model(
    pipeline: Any,
    X: np.ndarray,
    y: np.ndarray,
    split_name: str = "Validation",
    weak_threshold: float = DEFAULT_WEAK_THRESHOLD,
) -> Dict[str, Any]:
    """
    Evaluate a classifier using Weak=0 and Strong=1.

    Confusion matrix format from sklearn:
        [[actual Weak predicted Weak, actual Weak predicted Strong],
         [actual Strong predicted Weak, actual Strong predicted Strong]]

    For this project:
        TP_weak = actual Weak predicted Weak
        FN_weak = actual Weak predicted Strong
        FP_weak = actual Strong predicted Weak (false alarm)
        TN_weak = actual Strong predicted Strong
    """
    if not 0.0 <= weak_threshold <= 1.0:
        raise ValueError("weak_threshold must be between 0.0 and 1.0")

    y = np.asarray(y).astype(int)

    if y.size == 0:
        raise ValueError(f"{split_name} dataset is empty")

    if not np.isin(y, [WEAK, STRONG]).all():
        raise ValueError("Labels must be 0 (Weak) or 1 (Strong)")

    prob_weak = get_class_probability(pipeline, X, WEAK)
    prob_strong = get_class_probability(pipeline, X, STRONG)

    if prob_weak is not None:
        # The decision threshold applies to P(Weak), not P(Strong).
        y_pred = np.where(prob_weak >= weak_threshold, WEAK, STRONG)
    else:
        # A model without probability support uses its own predictions.
        y_pred = np.asarray(pipeline.predict(X)).astype(int)

    accuracy = float(accuracy_score(y, y_pred))
    macro_f1 = float(
        f1_score(y, y_pred, average="macro", zero_division=0)
    )
    weighted_f1 = float(
        f1_score(y, y_pred, average="weighted", zero_division=0)
    )

    weak_precision = float(
        precision_score(y, y_pred, pos_label=WEAK, zero_division=0)
    )
    weak_recall = float(
        recall_score(y, y_pred, pos_label=WEAK, zero_division=0)
    )
    weak_f1 = float(
        f1_score(y, y_pred, pos_label=WEAK, zero_division=0)
    )

    strong_precision = float(
        precision_score(y, y_pred, pos_label=STRONG, zero_division=0)
    )
    strong_recall = float(
        recall_score(y, y_pred, pos_label=STRONG, zero_division=0)
    )
    strong_f1 = float(
        f1_score(y, y_pred, pos_label=STRONG, zero_division=0)
    )

    # Fixed class order: rows are actual labels; columns are predicted labels.
    cm = confusion_matrix(
        y, y_pred, labels=[WEAK, STRONG]
    ).astype(int)

    # Project-specific confusion matrix interpretation:
    tp_weak = int(cm[0, 0])  # Weak correctly predicted Weak
    fn_weak = int(cm[0, 1])  # Weak incorrectly predicted Strong
    fp_weak = int(cm[1, 0])  # Strong incorrectly predicted Weak
    tn_weak = int(cm[1, 1])  # Strong correctly predicted Strong

    total_weak = int((y == WEAK).sum())
    total_strong = int((y == STRONG).sum())

    false_alarm_rate = (
        float(fp_weak / total_strong)
        if total_strong > 0
        else 0.0
    )

    # ROC-AUC uses Strong=1 as the positive class.
    # Weak PR-AUC explicitly uses Weak as the positive class.
    # Strong PR-AUC is also provided, with its meaning made explicit.
    roc_auc = None
    weak_pr_auc = None
    strong_pr_auc = None

    if prob_strong is not None and len(np.unique(y)) == 2:
        roc_auc = float(roc_auc_score(y, prob_strong))
        strong_pr_auc = float(
            average_precision_score(y, prob_strong)
        )

    if prob_weak is not None and total_weak > 0:
        y_weak_binary = (y == WEAK).astype(int)
        weak_pr_auc = float(
            average_precision_score(y_weak_binary, prob_weak)
        )

    return {
        "split": split_name,
        "weak_threshold": float(weak_threshold),
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "roc_auc": roc_auc,
        "weak_pr_auc": weak_pr_auc,
        "strong_pr_auc": strong_pr_auc,
        "false_alarm_rate": false_alarm_rate,
        "false_alarm_count": fp_weak,
        "total_weak": total_weak,
        "total_strong": total_strong,
        "weak_detected": tp_weak,
        "weak_missed": fn_weak,
        "strong_correct": tn_weak,
        "weak_class_metrics": {
            "precision": weak_precision,
            "recall": weak_recall,
            "f1_score": weak_f1,
        },
        "strong_class_metrics": {
            "precision": strong_precision,
            "recall": strong_recall,
            "f1_score": strong_f1,
        },
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_definition": {
            "rows": ["actual_weak_0", "actual_strong_1"],
            "columns": ["predicted_weak_0", "predicted_strong_1"],
            "tp_weak": tp_weak,
            "fn_weak": fn_weak,
            "fp_weak": fp_weak,
            "tn_weak": tn_weak,
        },
    }


def select_champion_model(
    val_results: Dict[str, Any],
    priority_metric: str = "weak_recall",
    explicit_choice: str = "auto",
) -> str:
    """
    Select a champion using validation results only.

    Tie-breakers:
        weak_recall: Weak F1, Macro F1, ROC-AUC
        macro_f1:    Weak Recall, Weak F1, ROC-AUC
        accuracy:    Macro F1, Weak Recall, ROC-AUC
        roc_auc:     Weak Recall, Macro F1, Accuracy
    """
    alias_map = {
        "rf": "Random Forest",
        "random_forest": "Random Forest",
        "logistic": "Baseline (Logistic Regression)",
        "lr": "Baseline (Logistic Regression)",
        "hgb": "HistGradientBoosting",
        "hist_gbdt": "HistGradientBoosting",
        "gbdt": "Gradient Boosting (GBDT)",
        "xgb": "XGBoost",
        "xgboost": "XGBoost",
    }

    if explicit_choice != "auto":
        resolved = alias_map.get(
            explicit_choice.lower(), explicit_choice
        )

        if resolved in val_results:
            return resolved

        raise ValueError(
            f"Champion '{explicit_choice}' not found. "
            f"Available candidates: {list(val_results.keys())}"
        )

    def metric_value(value: Optional[float]) -> float:
        return float(value) if value is not None else -1.0

    def score_key(item):
        _, metrics = item

        weak_recall = metrics["weak_class_metrics"]["recall"]
        weak_f1 = metrics["weak_class_metrics"]["f1_score"]
        macro_f1 = metrics["macro_f1"]
        accuracy = metrics["accuracy"]
        roc_auc = metric_value(metrics["roc_auc"])

        if priority_metric == "weak_recall":
            return weak_recall, weak_f1, macro_f1, roc_auc

        if priority_metric == "macro_f1":
            return macro_f1, weak_recall, weak_f1, roc_auc

        if priority_metric == "accuracy":
            return accuracy, macro_f1, weak_recall, roc_auc

        if priority_metric == "roc_auc":
            return roc_auc, weak_recall, macro_f1, accuracy

        raise ValueError(f"Unsupported priority metric: {priority_metric}")

    return max(val_results.items(), key=score_key)[0]


def print_confusion_matrix(
    metrics: Dict[str, Any],
    title: str,
) -> None:
    """Print the matrix with clear class meanings and correct terminology."""
    cm = metrics["confusion_matrix"]

    tp_weak = cm[0][0]
    fn_weak = cm[0][1]
    fp_weak = cm[1][0]
    tn_weak = cm[1][1]

    print(f"\n{title}")
    print("Rows = actual class; columns = predicted class")
    print("                    Predicted Weak   Predicted Strong")
    print(f"Actual Weak (0)          {tp_weak:4d}               {fn_weak:4d}")
    print(f"Actual Strong (1)        {fp_weak:4d}               {tn_weak:4d}")

    print("\nWeak-detection interpretation:")
    print(f"  True positives (weak detected): {tp_weak}")
    print(f"  False negatives (weak missed):  {fn_weak}")
    print(f"  False positives (false alarms): {fp_weak}")
    print(f"  True negatives (strong correct): {tn_weak}")


def print_benchmark_table(
    val_results: Dict[str, Any],
    weak_threshold: float,
) -> None:
    print("\n" + "=" * 100)
    print(
        "VALIDATION BENCHMARK COMPARISON "
        f"(P(Weak) >= {weak_threshold:.2f})"
    )
    print("=" * 100)

    header = (
        f"{'Model':<30} | {'Acc':>6} | {'Macro F1':>8} | "
        f"{'Weak Rec':>8} | {'Weak F1':>8} | "
        f"{'False Alarms':>15} | {'ROC-AUC':>7}"
    )
    print(header)
    print("-" * len(header))

    for name, metrics in val_results.items():
        weak_recall = metrics["weak_class_metrics"]["recall"]
        weak_f1 = metrics["weak_class_metrics"]["f1_score"]
        total_strong = metrics["total_strong"]
        false_alarms = metrics["false_alarm_count"]

        false_alarm_text = (
            f"{false_alarms}/{total_strong} "
            f"({metrics['false_alarm_rate'] * 100:.1f}%)"
        )

        roc_auc = metrics["roc_auc"]
        roc_text = f"{roc_auc:.3f}" if roc_auc is not None else "N/A"

        print(
            f"{name:<30} | "
            f"{metrics['accuracy'] * 100:5.1f}% | "
            f"{metrics['macro_f1']:8.3f} | "
            f"{weak_recall * 100:7.1f}% | "
            f"{weak_f1:8.3f} | "
            f"{false_alarm_text:>15} | "
            f"{roc_text:>7}"
        )


def print_evaluation_summary(
    metrics: Dict[str, Any],
    prefix: str,
) -> None:
    """Print sample counts dynamically, without assuming dataset size."""
    weak = metrics["weak_class_metrics"]
    strong = metrics["strong_class_metrics"]

    print(f"{prefix} Accuracy:           {metrics['accuracy'] * 100:.1f}%")
    print(f"{prefix} Macro F1:            {metrics['macro_f1']:.3f}")

    if metrics["roc_auc"] is not None:
        print(f"{prefix} ROC-AUC (Strong):    {metrics['roc_auc']:.3f}")

    if metrics["weak_pr_auc"] is not None:
        print(f"{prefix} PR-AUC (Weak):       {metrics['weak_pr_auc']:.3f}")

    if metrics["strong_pr_auc"] is not None:
        print(f"{prefix} PR-AUC (Strong):     {metrics['strong_pr_auc']:.3f}")

    print(
        f"{prefix} Weak Recall:         {weak['recall'] * 100:.1f}% "
        f"({metrics['weak_detected']}/{metrics['total_weak']} detected)"
    )
    print(f"{prefix} Weak Precision:      {weak['precision'] * 100:.1f}%")
    print(f"{prefix} Weak F1:             {weak['f1_score']:.3f}")
    print(
        f"{prefix} False Alarm Rate:    "
        f"{metrics['false_alarm_rate'] * 100:.1f}% "
        f"({metrics['false_alarm_count']}/{metrics['total_strong']} strong samples)"
    )
    print(
        f"{prefix} Strong Recall:       {strong['recall'] * 100:.1f}%"
    )
    print(f"{prefix} Strong F1:           {strong['f1_score']:.3f}")


def get_feature_importances(
    champion_pipe: Any,
    feature_cols: list,
) -> list:
    """Extract model-native feature importance or normalized absolute coefficients."""
    model_obj = (
        champion_pipe.named_steps["model"]
        if hasattr(champion_pipe, "named_steps")
        and "model" in champion_pipe.named_steps
        else champion_pipe
    )

    results = []

    if hasattr(model_obj, "feature_importances_"):
        importances = np.asarray(model_obj.feature_importances_, dtype=float)
        method = "feature_importance"

    elif hasattr(model_obj, "coef_"):
        coefficients = np.asarray(model_obj.coef_, dtype=float)

        if coefficients.ndim == 2:
            coefficients = coefficients[0]

        importances = np.abs(coefficients)
        total = float(importances.sum())

        if total > 0:
            importances = importances / total

        method = "normalized_absolute_coefficient"

    else:
        return results

    if len(importances) != len(feature_cols):
        print(
            "Warning: importance count does not match feature count; "
            "skipping feature importance report."
        )
        return results

    for index in np.argsort(importances)[::-1]:
        results.append({
            "rank": len(results) + 1,
            "feature": str(feature_cols[index]),
            "importance": float(importances[index]),
            "method": method,
        })

    return results


def main(
    priority_metric: str = "weak_recall",
    champion_choice: str = "auto",
    weak_threshold: float = DEFAULT_WEAK_THRESHOLD,
) -> None:
    if not 0.0 <= weak_threshold <= 1.0:
        raise ValueError("weak_threshold must be between 0.0 and 1.0")

    print("=" * 80)
    print("TRAINING & BENCHMARKING COLONY STRENGTH CLASSIFIERS")
    print(
        f"Selection Policy: Priority Metric='{priority_metric}', "
        f"Explicit Choice='{champion_choice}', "
        f"Weak Threshold={weak_threshold:.2f}"
    )
    print("=" * 80)

    train_df, val_df, test_df, feature_cols = load_datasets()

    X_train = train_df[feature_cols].to_numpy()
    y_train = train_df["label"].to_numpy(dtype=int)

    X_val = val_df[feature_cols].to_numpy()
    y_val = val_df["label"].to_numpy(dtype=int)

    X_test = test_df[feature_cols].to_numpy()
    y_test = test_df["label"].to_numpy(dtype=int)

    candidates = {
        "Baseline (Logistic Regression)": build_classifier_pipeline("logistic"),
        "Random Forest": build_classifier_pipeline("rf"),
        "HistGradientBoosting": build_classifier_pipeline("hist_gbdt"),
        "Gradient Boosting (GBDT)": build_classifier_pipeline("gbdt"),
    }

    if HAVE_XGB:
        weak_count = int((y_train == WEAK).sum())
        strong_count = int((y_train == STRONG).sum())

        scale_pos = (
            weak_count / strong_count if strong_count > 0 else 1.0
        )
        candidates["XGBoost"] = build_classifier_pipeline(
            "xgboost",
            scale_pos_weight=scale_pos,
        )

    val_results: Dict[str, Any] = {}

    print("\n--- Training Candidate Classifiers ---")

    for name, pipeline in candidates.items():
        print(f"Training {name}...")
        pipeline.fit(X_train, y_train)

        val_results[name] = evaluate_model(
            pipeline,
            X_val,
            y_val,
            split_name="Validation",
            weak_threshold=weak_threshold,
        )

    print_benchmark_table(val_results, weak_threshold)

    # Select using validation results only.
    champion_name = select_champion_model(
        val_results,
        priority_metric=priority_metric,
        explicit_choice=champion_choice,
    )
    champion_pipe = candidates[champion_name]

    print("\n" + "=" * 80)
    print(f"SELECTED CHAMPION: {champion_name}")
    print(f"Selection priority: {priority_metric}")
    print("=" * 80)

    print_confusion_matrix(
        val_results[champion_name],
        f"Validation Confusion Matrix ({champion_name})",
    )

    # Evaluate the selected champion on the test set once.
    # Do not use these results to choose another model or threshold.
    print("\n" + "=" * 80)
    print(
        f"HELD-OUT TEST EVALUATION ({champion_name}, "
        f"threshold={weak_threshold:.2f})"
    )
    print("=" * 80)

    test_metrics = evaluate_model(
        champion_pipe,
        X_test,
        y_test,
        split_name="Test",
        weak_threshold=weak_threshold,
    )

    print_evaluation_summary(test_metrics, "Test")
    print_confusion_matrix(
        test_metrics,
        "Test Confusion Matrix",
    )

    print("\n" + "=" * 80)
    print(f"FEATURE IMPORTANCE ({champion_name})")
    print("=" * 80)

    top_features = get_feature_importances(
        champion_pipe,
        feature_cols,
    )

    for item in top_features:
        print(
            f"{item['rank']:3d}. "
            f"{item['feature']:<30} : "
            f"{item['importance'] * 100:6.2f}%"
        )

    # Save the wrapper so the threshold travels with the estimator.
    champion_deployed = ColonyStrengthClassifier(
        base_estimator=champion_pipe,
        weak_threshold=weak_threshold,
    )

    # Preserve the estimator's actual class ordering.
    if hasattr(champion_pipe, "classes_"):
        champion_deployed.classes_ = np.asarray(champion_pipe.classes_)
    else:
        champion_deployed.classes_ = np.array([WEAK, STRONG])

    if top_features:
        model_obj = (
            champion_pipe.named_steps["model"]
            if hasattr(champion_pipe, "named_steps")
            and "model" in champion_pipe.named_steps
            else champion_pipe
        )

        if hasattr(model_obj, "feature_importances_"):
            champion_deployed.feature_importances_ = (
                model_obj.feature_importances_
            )
        elif hasattr(model_obj, "coef_"):
            champion_deployed.coef_ = model_obj.coef_

    model_out_path = ARTIFACTS_DIR / "colony_strength_model.joblib"
    joblib.dump(champion_deployed, model_out_path)

    features_json_path = ARTIFACTS_DIR / "feature_names.json"
    with features_json_path.open("w", encoding="utf-8") as file:
        json.dump(list(feature_cols), file, indent=2)

    report = {
        "champion_model": champion_name,
        "selection_priority_metric": priority_metric,
        "weak_threshold": weak_threshold,
        "label_mapping": {
            "0": "Weak",
            "1": "Strong",
        },
        "feature_count": len(feature_cols),
        "feature_names": list(feature_cols),
        "validation_benchmarks": val_results,
        "test_evaluation": test_metrics,
        "top_features": top_features,
        "evaluation_notes": [
            "Model selection used validation metrics only.",
            "Test metrics are reported for the selected champion.",
            "Weak PR-AUC treats Weak (class 0) as the positive class.",
            "False Alarm Rate is the proportion of actual Strong samples predicted Weak.",
            "Confusion matrix rows are actual classes [Weak, Strong].",
            "Confusion matrix columns are predicted classes [Weak, Strong].",
        ],
    }

    report_json_path = ARTIFACTS_DIR / "evaluation_metrics.json"
    with report_json_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, allow_nan=False)

    print("\n" + "=" * 80)
    print("ARTIFACTS SAVED")
    print("=" * 80)
    print(f"Model:          {model_out_path.resolve()}")
    print(f"Feature schema: {features_json_path.resolve()}")
    print(f"Evaluation:     {report_json_path.resolve()}")
    print(f"Saved threshold: {weak_threshold:.2f}")
    print("\nTraining and evaluation completed.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train and benchmark the MSPB2 colony strength classifier"
    )

    parser.add_argument(
        "--priority-metric",
        choices=["weak_recall", "macro_f1", "accuracy", "roc_auc"],
        default="weak_recall",
        help="Validation metric used to select the champion.",
    )
    parser.add_argument(
        "--champion",
        choices=["auto", "rf", "logistic", "hist_gbdt", "gbdt", "xgboost"],
        default="auto",
        help="Optional explicit champion override.",
    )
    parser.add_argument(
        "--weak-threshold",
        type=float,
        default=DEFAULT_WEAK_THRESHOLD,
        help=f"Predict Weak when P(Weak) is at least this value (default: {DEFAULT_WEAK_THRESHOLD}).",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    main(
        priority_metric=args.priority_metric,
        champion_choice=args.champion,
        weak_threshold=args.weak_threshold,
    )
