"""Compare 6h, 12h, and 24h environmental feature windows for classification.

All variants use the existing sample IDs and hive-separated train/validation/test
splits. The LR (weak_w=2.0) classifier and its validation-selected 0.425 Weak
threshold are fixed before this window comparison. Window selection uses
validation metrics only; the test split is evaluated once for the selected
window and is never used to choose it.
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

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

from scripts.data_prep import (
    _find_dataset_path,
    match_windows_and_extract_features,
    parse_d3_inspections,
    parse_d5_d6_inspections,
)
from src.shared.config import settings
from src.shared.feature_extraction import ENVIRONMENTAL_FEATURES

WEAK, STRONG = 0, 1
WEAK_THRESHOLD = 0.425
WINDOW_HOURS = (6, 12, 24)
RESULT_PATH = settings.MSPB2_ARTIFACTS_DIR / "window_duration_experiment_results.json"


def probability_for_class(model: Pipeline, X: np.ndarray, class_label: int) -> np.ndarray:
    """Look up the requested probability column through the fitted classes_."""
    matches = np.flatnonzero(np.asarray(model.classes_) == class_label)
    if len(matches) != 1:
        raise ValueError(f"Class {class_label} is not uniquely present in {model.classes_.tolist()}")
    return np.asarray(model.predict_proba(X), dtype=float)[:, int(matches[0])]


def make_model() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            C=0.1, class_weight={WEAK: 2.0, STRONG: 1.0}, max_iter=1000, random_state=42
        )),
    ])


def evaluate(y_true: np.ndarray, p_weak: np.ndarray) -> Dict[str, Any]:
    y_pred = np.where(p_weak >= WEAK_THRESHOLD, WEAK, STRONG)
    cm = confusion_matrix(y_true, y_pred, labels=[WEAK, STRONG]).astype(int)
    strong_count = int((y_true == STRONG).sum())
    return {
        "weak_threshold": WEAK_THRESHOLD,
        "weak_recall": float(recall_score(y_true, y_pred, pos_label=WEAK, zero_division=0)),
        "weak_precision": float(precision_score(y_true, y_pred, pos_label=WEAK, zero_division=0)),
        "weak_f1": float(f1_score(y_true, y_pred, pos_label=WEAK, zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "strong_recall": float(recall_score(y_true, y_pred, pos_label=STRONG, zero_division=0)),
        "false_alarm_count": int(cm[1, 0]),
        "false_alarm_rate": float(cm[1, 0] / strong_count),
        "confusion_matrix": cm.tolist(),
    }


def load_sensor_sources() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    d3_inspections, _ = parse_d3_inspections()
    d3_path = _find_dataset_path("D3_D4/D3_D4/D3/D3_sensor_data.csv")
    d3_sensor = pd.read_csv(d3_path, usecols=["tag_number", "published_at", "temperature", "humidity"], low_memory=False)
    d3_sensor["published_at"] = pd.to_datetime(d3_sensor["published_at"], utc=True)

    d5_inspections, _ = parse_d5_d6_inspections()
    d5_path = _find_dataset_path("D5_D6/D5_D6/D5_D6_sensor_data.pkl")
    d5_sensor = pd.read_pickle(d5_path)
    if "published_at" not in d5_sensor.columns:
        d5_sensor = d5_sensor.reset_index()
    d5_sensor = d5_sensor[[column for column in ["tag_number", "published_at", "temperature", "humidity"] if column in d5_sensor.columns]]
    d5_sensor["published_at"] = pd.to_datetime(d5_sensor["published_at"], utc=True)
    return d3_sensor, d3_inspections, d5_sensor, d5_inspections


def recreate_window_dataset(
    hours: int,
    sources: Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> pd.DataFrame:
    """Recompute the unchanged 40-feature schema using only each preceding window."""
    d3_sensor, d3_inspections, d5_sensor, d5_inspections = sources
    # Keep the established sample IDs across all three horizons. A short window
    # can contain fewer than the 24h pipeline's 30 readings, but its available
    # temperature/humidity statistics are still a valid early-prediction input.
    min_readings = 1
    d3 = match_windows_and_extract_features(d3_sensor, d3_inspections, min_readings, hours)
    d5 = match_windows_and_extract_features(d5_sensor, d5_inspections, min_readings, hours)
    return pd.concat([d3, d5], ignore_index=True)


def split_like_existing(dataset: pd.DataFrame, eligible_sample_ids: set[str]) -> Dict[str, pd.DataFrame]:
    """Attach regenerated rows to the unchanged hive-separated split assignments."""
    data_dir = settings.MSPB2_BASE_DIR / "data"
    splits = {name: pd.read_csv(data_dir / f"{name}.csv") for name in ("train", "val", "test")}
    generated = dataset.set_index("sample_id")
    if not generated.index.is_unique:
        raise ValueError("Regenerated data contains duplicate sample IDs")
    output: Dict[str, pd.DataFrame] = {}
    for name, existing in splits.items():
        existing = existing[existing["sample_id"].astype(str).isin(eligible_sample_ids)].reset_index(drop=True)
        sample_ids = existing["sample_id"].astype(str).tolist()
        missing = sorted(set(sample_ids) - set(generated.index.astype(str)))
        if missing:
            raise ValueError(f"Regenerated data misses {len(missing)} {name} sample IDs; e.g. {missing[:3]}")
        frame = generated.loc[sample_ids].reset_index()
        if not frame["label"].astype(int).equals(existing["label"].astype(int).reset_index(drop=True)):
            raise AssertionError(f"Labels changed while reconstructing the {name} split")
        if set(frame["label"].astype(int)) != {WEAK, STRONG}:
            raise ValueError(f"The common cohort's {name} split lacks a class")
        output[name] = frame
    if set(output["train"].hive_id) & set(output["val"].hive_id):
        raise AssertionError("Hive leakage between train and validation")
    if set(output["train"].hive_id) & set(output["test"].hive_id):
        raise AssertionError("Hive leakage between train and test")
    if set(output["val"].hive_id) & set(output["test"].hive_id):
        raise AssertionError("Hive leakage between validation and test")
    return output


def run_window(hours: int, splits: Dict[str, pd.DataFrame]) -> Tuple[Dict[str, Any], Pipeline]:
    train, val = splits["train"], splits["val"]
    model = make_model()
    model.fit(train[ENVIRONMENTAL_FEATURES].to_numpy(), train["label"].to_numpy(dtype=int))
    val_p_weak = probability_for_class(model, val[ENVIRONMENTAL_FEATURES].to_numpy(), WEAK)
    return {
        "window_hours": hours,
        "feature_count": len(ENVIRONMENTAL_FEATURES),
        "feature_schema": list(ENVIRONMENTAL_FEATURES),
        "sample_counts": {name: len(frame) for name, frame in splits.items()},
        "validation": evaluate(val["label"].to_numpy(dtype=int), val_p_weak),
    }, model


def main() -> int:
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    sources = load_sensor_sources()
    validation_results: List[Dict[str, Any]] = []
    regenerated = {hours: recreate_window_dataset(hours, sources) for hours in WINDOW_HOURS}
    existing_ids = set()
    for name in ("train", "val", "test"):
        existing_ids.update(pd.read_csv(settings.MSPB2_BASE_DIR / "data" / f"{name}.csv")["sample_id"].astype(str))
    eligible_ids = existing_ids.intersection(*(set(frame["sample_id"].astype(str)) for frame in regenerated.values()))
    dropped_ids = sorted(existing_ids - eligible_ids)
    if dropped_ids:
        print(f"Common cohort excludes {len(dropped_ids)} existing sample(s) with no readings in at least one window: {dropped_ids}")

    for hours in WINDOW_HOURS:
        print(f"\nRunning Experiment {'ABC'[WINDOW_HOURS.index(hours)]}: {hours}-hour window")
        splits = split_like_existing(regenerated[hours], eligible_ids)
        result, _model = run_window(hours, splits)
        validation_results.append(result)
        metric = result["validation"]
        print(f"  validation weak F1={metric['weak_f1']:.3f}, weak recall={metric['weak_recall']:.3f}, "
              f"false alarms={metric['false_alarm_count']} ({metric['false_alarm_rate']:.1%}), macro F1={metric['macro_f1']:.3f}")
    selected = max(validation_results, key=lambda row: (
        row["validation"]["weak_f1"], row["validation"]["macro_f1"], row["validation"]["weak_recall"]
    ))
    # Refit the selected window once from its train split and evaluate test only after selection.
    selected_splits = split_like_existing(regenerated[selected["window_hours"]], eligible_ids)
    selected_model = make_model().fit(
        selected_splits["train"][ENVIRONMENTAL_FEATURES].to_numpy(), selected_splits["train"]["label"].to_numpy(dtype=int)
    )
    test = selected_splits["test"]
    selected["one_time_test_metrics"] = evaluate(
        test["label"].to_numpy(dtype=int), probability_for_class(selected_model, test[ENVIRONMENTAL_FEATURES].to_numpy(), WEAK)
    )
    payload = {
        "experiments": validation_results,
        "selection_policy": {
            "selected_on": "validation_only",
            "maximize": "weak_f1",
            "tie_breakers": ["macro_f1", "weak_recall"],
            "classifier": "LR (weak_w=2.0)",
            "weak_threshold": WEAK_THRESHOLD,
            "test_evaluated_only_after_selection": True,
        },
        "selected_window_hours": selected["window_hours"],
        "common_cohort": {
            "existing_sample_count": len(existing_ids),
            "common_sample_count": len(eligible_ids),
            "excluded_sample_ids_without_readings_in_all_windows": dropped_ids,
            "hive_split_assignments_preserved": True,
        },
    }
    RESULT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSelected on validation: {selected['window_hours']}h window")
    print(f"Saved results: {RESULT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
