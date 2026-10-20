import json
from datetime import datetime, timedelta, timezone
from threading import Lock
from typing import Dict, Any, Optional

import joblib
import numpy as np
import pandas as pd
from sqlalchemy.orm import Session
from sqlalchemy import select

from src.config import settings
from src.models import Hive, HiveTelemetry, HivePrediction
from src.feature_extraction import compute_24h_environmental_features, ENVIRONMENTAL_FEATURES

# ============================================================
# CONCURRENCY LOCK PER HIVE
# ============================================================
hive_locks: Dict[str, Lock] = {}
hive_locks_guard = Lock()


def get_hive_lock(hive_id: str) -> Lock:
    with hive_locks_guard:
        if hive_id not in hive_locks:
            hive_locks[hive_id] = Lock()
        return hive_locks[hive_id]


# ============================================================
# MODEL & FEATURE SCHEMA LOADING
# ============================================================
MODEL_PATH = settings.MSPB2_ARTIFACTS_DIR / "colony_strength_model.joblib"
FEATURE_NAMES_PATH = settings.MSPB2_ARTIFACTS_DIR / "feature_names.json"

if not MODEL_PATH.exists():
    raise RuntimeError(f"Trained model not found at {MODEL_PATH}")

if not FEATURE_NAMES_PATH.exists():
    raise RuntimeError(f"Feature schema not found at {FEATURE_NAMES_PATH}")

model = joblib.load(MODEL_PATH)
FEATURE_NAMES = json.loads(FEATURE_NAMES_PATH.read_text())

if len(FEATURE_NAMES) != 10:
    raise RuntimeError(f"Expected 10 environmental feature names; got {len(FEATURE_NAMES)}")


# ============================================================
# PREDICTION GENERATION SERVICE
# ============================================================

def get_latest_prediction(db: Session, hive_id: str) -> Optional[HivePrediction]:
    return (
        db.query(HivePrediction)
        .filter(HivePrediction.hive_id == hive_id)
        .order_by(HivePrediction.predicted_at.desc(), HivePrediction.id.desc())
        .first()
    )


def generate_hive_prediction(db: Session, hive_id: str) -> Dict[str, Any]:
    """
    On-demand prediction logic:
    1. Checks if prediction is due or new telemetry is available.
    2. Builds 24-hour window from PostgreSQL.
    3. Computes 75 engineered physical features.
    4. Executes calibrated Random Forest model.
    5. Persists prediction history.
    """
    with get_hive_lock(hive_id):
        previous = get_latest_prediction(db, hive_id)

        # Get the latest telemetry reading timestamp
        latest_reading = (
            db.query(HiveTelemetry)
            .filter(HiveTelemetry.hive_id == hive_id)
            .order_by(HiveTelemetry.published_at.desc())
            .first()
        )

        if not latest_reading:
            if previous:
                return _format_prediction_result(previous, "previous")
            return {
                "hive_id": hive_id,
                "prediction_status": "pending",
                "message": "Waiting for the first telemetry reading.",
            }

        now = datetime.now(timezone.utc)
        latest_dt = latest_reading.published_at
        if latest_dt.tzinfo is None:
            latest_dt = latest_dt.replace(tzinfo=timezone.utc)

        if previous:
            prev_pred_at = previous.predicted_at
            if prev_pred_at.tzinfo is None:
                prev_pred_at = prev_pred_at.replace(tzinfo=timezone.utc)

            prev_read_at = previous.last_reading_at
            if prev_read_at.tzinfo is None:
                prev_read_at = prev_read_at.replace(tzinfo=timezone.utc)

            prediction_interval = timedelta(hours=settings.PREDICTION_INTERVAL_HOURS)
            prediction_due = (now - prev_pred_at) >= prediction_interval
            new_data_available = latest_dt > prev_read_at
        else:
            prediction_due = True
            new_data_available = True

        if not (prediction_due and new_data_available):
            return _format_prediction_result(previous, "previous")

        # Telemetry window: [latest_dt - 24 hours, latest_dt]
        window_start = latest_dt - timedelta(hours=settings.WINDOW_HOURS)

        # Query readings in the window
        readings = (
            db.query(HiveTelemetry)
            .filter(
                HiveTelemetry.hive_id == hive_id,
                HiveTelemetry.published_at > window_start,
                HiveTelemetry.published_at <= latest_dt,
            )
            .order_by(HiveTelemetry.published_at.asc())
            .all()
        )

        if not readings:
            if previous:
                return {
                    **_format_prediction_result(previous, "previous"),
                    "refresh_message": "No telemetry readings available in the 24-hour window.",
                }
            return {
                "hive_id": hive_id,
                "prediction_status": "pending",
                "message": "No telemetry readings available in the 24-hour window.",
            }

        # Validate minimum readings and temporal span
        earliest_dt = readings[0].published_at
        if earliest_dt.tzinfo is None:
            earliest_dt = earliest_dt.replace(tzinfo=timezone.utc)

        coverage_hours = (latest_dt - earliest_dt).total_seconds() / 3600.0

        if len(readings) < settings.MIN_READINGS or coverage_hours < settings.MIN_COVERAGE_HOURS:
            err_msg = (
                f"Insufficient 24h data: have {len(readings)} readings spanning {coverage_hours:.1f}h. "
                f"Required: >= {settings.MIN_READINGS} readings spanning >= {settings.MIN_COVERAGE_HOURS}h."
            )
            if previous:
                return {
                    **_format_prediction_result(previous, "previous"),
                    "refresh_message": err_msg,
                }
            return {
                "hive_id": hive_id,
                "prediction_status": "pending",
                "message": err_msg,
            }

        # Build DataFrame for environmental feature extraction
        rows = []
        for r in readings:
            rows.append({
                "published_at": r.published_at,
                "temperature": r.temperature,
                "humidity": r.humidity,
            })

        df = pd.DataFrame(rows)
        df["published_at"] = pd.to_datetime(df["published_at"], utc=True)

        try:
            feats = compute_24h_environmental_features(df, time_col="published_at")
        except Exception as exc:
            if previous:
                return {
                    **_format_prediction_result(previous, "previous"),
                    "refresh_message": f"Feature computation error: {exc}",
                }
            return {
                "hive_id": hive_id,
                "prediction_status": "pending",
                "message": f"Feature computation error: {exc}",
            }

        # Check for missing model features
        missing = [f for f in FEATURE_NAMES if f not in feats]
        if missing:
            raise ValueError(f"Extracted features missing expected model inputs: {missing[:5]}")

        # Assemble feature vector in exact training order
        feature_vector = np.array([float(feats[name]) for name in FEATURE_NAMES], dtype=np.float64)
        X = pd.DataFrame([feature_vector], columns=FEATURE_NAMES)

        # Inference
        label = int(model.predict(X)[0])
        probabilities = model.predict_proba(X)[0]
        class_probs = {int(c): float(p) for c, p in zip(model.classes_, probabilities)}

        p_weak = class_probs.get(0, 0.0)
        p_strong = class_probs.get(1, 0.0)

        # Alert level definition
        if p_weak >= 0.70:
            alert = "RED_ALERT"
        elif p_weak >= 0.40:
            alert = "YELLOW"
        else:
            alert = "GREEN"

        # Persist new prediction record
        prediction_record = HivePrediction(
            hive_id=hive_id,
            prediction="Strong" if label == 1 else "Weak",
            class_label=label,
            weak_probability=round(p_weak, 4),
            strong_probability=round(p_strong, 4),
            alert_level=alert,
            window_start=window_start,
            window_end=latest_dt,
            last_reading_at=latest_dt,
            features_json=feats,
            predicted_at=now,
        )
        db.add(prediction_record)
        db.commit()
        db.refresh(prediction_record)

        return _format_prediction_result(prediction_record, "updated")


def _format_prediction_result(record: HivePrediction, status: str) -> Dict[str, Any]:
    return {
        "hive_id": record.hive_id,
        "prediction": record.prediction,
        "class_label": record.class_label,
        "weak_probability": record.weak_probability,
        "strong_probability": record.strong_probability,
        "alert_level": record.alert_level,
        "window_start": record.window_start,
        "window_end": record.window_end,
        "predicted_at": record.predicted_at,
        "prediction_status": status,
    }
