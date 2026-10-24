"""Select a balanced Weak decision threshold using validation data only.

The fixed candidate is Logistic Regression with ``class_weight={0: 2.0,
1: 1.0}``. Labels are fixed by the MSPB2 data schema: 0=Weak, 1=Strong.
This experiment neither reads test labels nor creates/replaces model artifacts.
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.shared.config import settings
from src.shared.feature_extraction import ENVIRONMENTAL_FEATURES

WEAK, STRONG = 0, 1
CANDIDATE_NAME = "LR (weak_w=2.0)"
MAX_FALSE_ALARM_RATE = 0.20
THRESHOLDS = tuple(round(0.10 + 0.025 * index, 3) for index in range(21))
RESULT_PATH = settings.MSPB2_ARTIFACTS_DIR / "balanced_threshold_results.json"


def class_probability(estimator: Any, X: np.ndarray, class_label: int) -> np.ndarray:
    """Return a class probability using the estimator's actual ``classes_`` order."""
    probabilities = np.asarray(estimator.predict_proba(X), dtype=float)
    classes = np.asarray(estimator.classes_)
    matches = np.flatnonzero(classes == class_label)
    if len(matches) != 1:
        raise ValueError(f"Expected class {class_label} exactly once; got classes={classes.tolist()}")
    return probabilities[:, int(matches[0])]


def validate_train_validation(train_df: pd.DataFrame, val_df: pd.DataFrame, feature_cols: List[str]) -> None:
    """Ensure the existing hive-separated train/validation split and 40-feature schema."""
    if len(feature_cols) != 40:
        raise AssertionError(f"Expected 40 features, found {len(feature_cols)}")
    for name, frame in (("train", train_df), ("validation", val_df)):
        missing = {"hive_id", "label", *feature_cols} - set(frame.columns)
        if missing:
            raise ValueError(f"{name} split is missing columns: {sorted(missing)}")
        labels = set(frame["label"].astype(int).unique())
        if labels != {WEAK, STRONG}:
            raise ValueError(f"{name} split must contain labels {{0, 1}}; got {sorted(labels)}")
    overlap = set(train_df["hive_id"].astype(str)) & set(val_df["hive_id"].astype(str))
    if overlap:
        raise AssertionError(f"Hive leakage between train and validation: {sorted(overlap)[:5]}")


def load_train_validation() -> tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """Load only the existing train and validation files, retaining the 40-feature schema."""
    data_dir = settings.MSPB2_BASE_DIR / "data"
    train_df = pd.read_csv(data_dir / "train.csv")
    val_df = pd.read_csv(data_dir / "val.csv")
    feature_cols = [column for column in ENVIRONMENTAL_FEATURES if column in train_df.columns]
    return train_df, val_df, feature_cols


def make_candidate() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            C=0.1, class_weight={WEAK: 2.0, STRONG: 1.0}, max_iter=1000, random_state=42
        )),
    ])


def evaluate_threshold(y_true: np.ndarray, probability_weak: np.ndarray, threshold: float) -> Dict[str, Any]:
    """Evaluate predictions; confusion matrix rows/columns are Weak then Strong."""
    y_pred = np.where(np.asarray(probability_weak) >= threshold, WEAK, STRONG)
    cm = confusion_matrix(y_true, y_pred, labels=[WEAK, STRONG]).astype(int)
    weak_detected, weak_missed = (int(value) for value in cm[0])
    false_alarms, strong_correct = (int(value) for value in cm[1])
    total_strong = int((y_true == STRONG).sum())
    return {
        "weak_threshold": float(threshold),
        "weak_recall": float(recall_score(y_true, y_pred, pos_label=WEAK, zero_division=0)),
        "weak_detected": weak_detected,
        "weak_missed": weak_missed,
        "weak_precision": float(precision_score(y_true, y_pred, pos_label=WEAK, zero_division=0)),
        "weak_f1": float(f1_score(y_true, y_pred, pos_label=WEAK, zero_division=0)),
        "false_alarm_count": false_alarms,
        "false_alarm_rate": float(false_alarms / total_strong),
        "strong_recall": float(recall_score(y_true, y_pred, pos_label=STRONG, zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_definition": {
            "rows": ["actual_weak", "actual_strong"],
            "columns": ["predicted_weak", "predicted_strong"],
        },
        "total_strong": total_strong,
        "strong_correct": strong_correct,
    }


def select_threshold(results: List[Dict[str, Any]]) -> Dict[str, Any] | None:
    """Maximize Weak F1 under the false-alarm cap; break ties by macro F1 then recall."""
    eligible = [row for row in results if row["false_alarm_rate"] <= MAX_FALSE_ALARM_RATE]
    if not eligible:
        return None
    return max(eligible, key=lambda row: (row["weak_f1"], row["macro_f1"], row["weak_recall"]))


def print_row(label: str, row: Dict[str, Any]) -> None:
    cm = row["confusion_matrix"]
    print(
        f"{label} (P(Weak) >= {row['weak_threshold']:.3f}): "
        f"weak recall={row['weak_recall']:.3f} ({row['weak_detected']} detected, {row['weak_missed']} missed), "
        f"precision={row['weak_precision']:.3f}, weak F1={row['weak_f1']:.3f}; "
        f"false alarms={row['false_alarm_count']}/{row['total_strong']} ({row['false_alarm_rate']:.1%}), "
        f"strong recall={row['strong_recall']:.3f}, macro F1={row['macro_f1']:.3f}, accuracy={row['accuracy']:.3f}"
    )
    print(f"  Confusion matrix [[actual Weak: predicted Weak={cm[0][0]}, predicted Strong={cm[0][1]}], "
          f"[actual Strong: predicted Weak={cm[1][0]}, predicted Strong={cm[1][1]}]]")


def main() -> int:
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    train_df, val_df, feature_cols = load_train_validation()
    validate_train_validation(train_df, val_df, feature_cols)

    model = make_candidate()
    X_train = train_df[feature_cols].to_numpy()
    y_train = train_df["label"].to_numpy(dtype=int)
    X_val = val_df[feature_cols].to_numpy()
    y_val = val_df["label"].to_numpy(dtype=int)
    model.fit(X_train, y_train)
    probability_weak = class_probability(model, X_val, WEAK)
    results = [evaluate_threshold(y_val, probability_weak, threshold) for threshold in THRESHOLDS]
    selected = select_threshold(results)

    by_threshold = {row["weak_threshold"]: row for row in results}
    payload: Dict[str, Any] = {
        "candidate": CANDIDATE_NAME,
        "data_policy": {
            "fit_split": "train_only",
            "threshold_evaluation_split": "validation_only",
            "test_data_used": False,
            "feature_count": len(feature_cols),
            "feature_schema": list(feature_cols),
            "hive_separation_verified": True,
        },
        "selection_policy": {
            "maximize": "validation_weak_f1",
            "maximum_false_alarm_rate": MAX_FALSE_ALARM_RATE,
            "tie_breakers": ["macro_f1", "weak_recall"],
            "invalid_when_no_candidate_meets_false_alarm_limit": True,
        },
        "validation_threshold_results": results,
    }
    if selected is None:
        payload["selection"] = {
            "status": "no_valid_threshold",
            "reason": f"No threshold had validation false_alarm_rate <= {MAX_FALSE_ALARM_RATE:.0%}.",
        }
        print(f"No valid threshold exists: none met the {MAX_FALSE_ALARM_RATE:.0%} validation false-alarm limit.")
    else:
        payload["selection"] = {"status": "selected", "validation_metrics": selected}
        print_row("Best valid threshold", selected)

    print_row("Existing threshold", by_threshold[0.25])
    print_row("Default threshold", by_threshold[0.50])
    if selected is not None:
        print(
            "Trade-off: the selected threshold maximizes validation Weak F1 among thresholds at or below "
            f"the {MAX_FALSE_ALARM_RATE:.0%} false-alarm cap. Relative to 0.25 and 0.50, use the printed "
            "Weak precision/recall and false-alarm rates to see the observed balance."
        )
    RESULT_PATH.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Saved full validation threshold results: {RESULT_PATH}")
    return 0 if selected is not None else 2


if __name__ == "__main__":
    raise SystemExit(main())
