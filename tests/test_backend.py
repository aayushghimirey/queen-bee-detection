import json
from datetime import datetime, timedelta, timezone
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from src.models import HiveTelemetry, HiveAudioRecord, HiveAudioFeature, HivePrediction, HiveDeviceAssignment
from src.audio_processing import DB_BAND_COLUMNS


def test_health_check(client: TestClient):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert data["database_connected"] is True
    assert data["model_loaded"] is True
    assert data["model_feature_count"] == 10


def test_auth_registration_and_login(client: TestClient):
    # Register new user
    reg_payload = {
        "email": "newuser@example.com",
        "password": "SecurePassword123!",
        "full_name": "Bob Builder",
    }
    reg_res = client.post("/api/v1/auth/register", json=reg_payload)
    assert reg_res.status_code == 201
    user_data = reg_res.json()
    assert user_data["email"] == "newuser@example.com"
    assert user_data["full_name"] == "Bob Builder"

    # Duplicate registration returns 409
    dup_res = client.post("/api/v1/auth/register", json=reg_payload)
    assert dup_res.status_code == 409

    # Login JSON
    login_res = client.post(
        "/api/v1/auth/login",
        json={"email": "newuser@example.com", "password": "SecurePassword123!"},
    )
    assert login_res.status_code == 200
    token = login_res.json()["access_token"]
    assert len(token) > 20

    # Login with wrong password returns 401
    bad_login = client.post(
        "/api/v1/auth/login",
        json={"email": "newuser@example.com", "password": "WrongPassword"},
    )
    assert bad_login.status_code == 401

    # Get /auth/me
    me_res = client.get(
        "/api/v1/auth/me",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert me_res.status_code == 200
    assert me_res.json()["email"] == "newuser@example.com"


def test_hive_crud(client: TestClient, auth_headers: dict):
    # Create hive
    payload = {
        "id": "HIVE_BETA",
        "name": "Beta Field Hive",
        "apiary_name": "South Orchard",
        "latitude": 45.1234,
        "longitude": -75.9876,
        "description": "Standard Langstroth 10-frame",
        "metadata_json": {"treatment": "oxalic_acid"},
    }
    create_res = client.post("/api/v1/hives", json=payload, headers=auth_headers)
    assert create_res.status_code == 201
    assert create_res.json()["id"] == "HIVE_BETA"

    # Duplicate hive ID returns 409
    dup_res = client.post("/api/v1/hives", json=payload, headers=auth_headers)
    assert dup_res.status_code == 409

    # List hives
    list_res = client.get("/api/v1/hives", headers=auth_headers)
    assert list_res.status_code == 200
    hives = list_res.json()
    assert any(h["id"] == "HIVE_BETA" for h in hives)

    # Get single hive detail
    detail_res = client.get("/api/v1/hives/HIVE_BETA", headers=auth_headers)
    assert detail_res.status_code == 200
    assert detail_res.json()["name"] == "Beta Field Hive"


def test_iot_device_registration_and_assignment(
    client: TestClient, auth_headers: dict, test_hive: dict
):
    # Register device
    dev_payload = {
        "id": "ESP32_BETA_NODE",
        "device_name": "Orchard Node 2",
        "hardware_version": "ESP32-S3",
        "firmware_version": "v2.0.1",
    }
    dev_res = client.post("/api/v1/iot-devices", json=dev_payload, headers=auth_headers)
    assert dev_res.status_code == 201
    dev_data = dev_res.json()
    assert dev_data["id"] == "ESP32_BETA_NODE"
    assert "api_key" in dev_data
    assert dev_data["api_key"].startswith("esp_")

    # Duplicate device ID returns 409
    dup_res = client.post("/api/v1/iot-devices", json=dev_payload, headers=auth_headers)
    assert dup_res.status_code == 409

    # List devices
    list_res = client.get("/api/v1/iot-devices", headers=auth_headers)
    assert list_res.status_code == 200
    assert any(d["id"] == "ESP32_BETA_NODE" for d in list_res.json())

    # Assign device to hive
    assign_res = client.post(
        f"/api/v1/hives/{test_hive['id']}/assignments",
        json={"device_id": "ESP32_BETA_NODE"},
        headers=auth_headers,
    )
    assert assign_res.status_code == 200
    assert assign_res.json()["hive_id"] == test_hive["id"]
    assert assign_res.json()["device_id"] == "ESP32_BETA_NODE"
    assert assign_res.json()["is_active"] is True

    # Check hive now shows active_device_id
    hive_res = client.get(f"/api/v1/hives/{test_hive['id']}", headers=auth_headers)
    assert hive_res.json()["active_device_id"] == "ESP32_BETA_NODE"

    # Unassign device
    unassign_res = client.delete(
        f"/api/v1/hives/{test_hive['id']}/assignments/ESP32_BETA_NODE",
        headers=auth_headers,
    )
    assert unassign_res.status_code == 204

    # Check hive active_device_id is now None
    hive_res_after = client.get(f"/api/v1/hives/{test_hive['id']}", headers=auth_headers)
    assert hive_res_after.json()["active_device_id"] is None


def test_telemetry_access_control(
    client: TestClient, auth_headers: dict, test_hive: dict, test_device: dict, create_synthetic_wav
):
    reading_json = {
        "published_at": datetime.now(timezone.utc).isoformat(),
        "temperature": 34.5,
        "humidity": 55.0,
    }

    # Case 1: Missing credentials -> 401
    res1 = client.post(
        f"/api/v1/hives/{test_hive['id']}/telemetry",
        json=reading_json,
    )
    assert res1.status_code == 401

    # Case 2: Invalid credentials -> 401
    res2 = client.post(
        f"/api/v1/hives/{test_hive['id']}/telemetry",
        headers={"X-Device-Id": test_device["id"], "X-Device-Key": "invalid_secret"},
        json=reading_json,
    )
    assert res2.status_code == 401

    # Case 3: Valid credentials, but device is NOT assigned to this hive -> 403
    res3 = client.post(
        f"/api/v1/hives/{test_hive['id']}/telemetry",
        headers={"X-Device-Id": test_device["id"], "X-Device-Key": test_device["api_key"]},
        json=reading_json,
    )
    assert res3.status_code == 403
    assert "not actively assigned" in res3.json()["detail"]


def test_audio_upload_and_feature_extraction(
    client: TestClient, auth_headers: dict, test_hive: dict, test_device: dict, create_synthetic_wav, db_session: Session
):
    """Test dedicated acoustic endpoint ingesting WAV burst and extracting spectral features."""
    # First, assign device to hive
    client.post(
        f"/api/v1/hives/{test_hive['id']}/assignments",
        json={"device_id": test_device["id"]},
        headers=auth_headers,
    )

    wav_buf = create_synthetic_wav(frequency_hz=213.62)

    # Post acoustic burst to dedicated /audio endpoint
    res = client.post(
        f"/api/v1/hives/{test_hive['id']}/audio",
        headers={"X-Device-Id": test_device["id"], "X-Device-Key": test_device["api_key"]},
        files={"audio": ("reading_burst.wav", wav_buf.getvalue(), "audio/wav")},
    )
    assert res.status_code == 201
    data = res.json()
    assert data["status"] == "received"
    assert data["features_extracted"] is True
    assert data["filename"] is not None
    assert data["sample_rate"] == 15625
    audio_id = data["audio_id"]

    # Verify database entry in hive_audio_records
    db_audio = db_session.query(HiveAudioRecord).filter(HiveAudioRecord.id == audio_id).first()
    assert db_audio is not None
    assert db_audio.hive_id == test_hive["id"]
    assert db_audio.device_id == test_device["id"]
    assert db_audio.sample_rate == 15625
    assert len(DB_BAND_COLUMNS) == 16
    for col in DB_BAND_COLUMNS:
        assert getattr(db_audio, col) is not None


def test_audio_with_precomputed_frequency_bands_in_dto(
    client: TestClient, auth_headers: dict, test_hive: dict, test_device: dict, db_session: Session
):
    """Test dedicated acoustic endpoint receiving precomputed FFT JSON."""
    # Assign device
    client.post(
        f"/api/v1/hives/{test_hive['id']}/assignments",
        json={"device_id": test_device["id"]},
        headers=auth_headers,
    )

    published_dt = datetime.now(timezone.utc) + timedelta(minutes=5)
    # Send pre-computed 16 frequency bands directly in JSON using the original dotted names
    audio_json = {
        "recorded_at": published_dt.isoformat(),
        "audio_density": 16.2,
        "audio_density_ratio": 0.85,
        "density_variation": 4.1,
        "hz_122.0703125": -15.2,
        "hz_152.587890625": -11.0,
        "hz_183.10546875": -7.5,
        "hz_213.623046875": 14.8,
        "hz_244.140625": 19.2,
        "hz_274.658203125": 12.1,
        "hz_305.17578125": 5.4,
        "hz_335.693359375": 1.2,
        "hz_366.2109375": -2.3,
        "hz_396.728515625": -5.1,
        "hz_427.24609375": -6.2,
        "hz_457.763671875": -8.0,
        "hz_488.28125": -10.4,
        "hz_518.798828125": -12.1,
        "hz_549.31640625": -14.5,
        "hz_579.833984375": -16.8,
    }

    # Post acoustic features as JSON
    res = client.post(
        f"/api/v1/hives/{test_hive['id']}/audio",
        headers={
            "X-Device-Id": test_device["id"],
            "X-Device-Key": test_device["api_key"],
            "Content-Type": "application/json",
        },
        json=audio_json,
    )
    assert res.status_code == 201
    data = res.json()
    assert data["status"] == "received"
    assert data["features_extracted"] is True
    audio_id = data["audio_id"]

    # Verify extracted frequency bands in hive_audio_records match the values
    db_audio = db_session.query(HiveAudioRecord).filter(HiveAudioRecord.id == audio_id).first()
    assert db_audio is not None
    assert db_audio.hz_122_07 == -15.2
    assert db_audio.hz_213_62 == 14.8
    assert db_audio.hz_244_14 == 19.2
    assert db_audio.hz_579_83 == -16.8


def test_telemetry_direct_json_body(
    client: TestClient, auth_headers: dict, test_hive: dict, test_device: dict, db_session: Session
):
    """Test pure environmental telemetry ingestion via JSON."""
    # Assign device
    client.post(
        f"/api/v1/hives/{test_hive['id']}/assignments",
        json={"device_id": test_device["id"]},
        headers=auth_headers,
    )

    published_dt = datetime.now(timezone.utc) + timedelta(minutes=10)
    direct_json_payload = {
        "published_at": published_dt.isoformat(),
        "temperature": 35.2,
        "humidity": 51.8,
    }

    # Post telemetry directly as Content-Type: application/json
    res = client.post(
        f"/api/v1/hives/{test_hive['id']}/telemetry",
        headers={
            "X-Device-Id": test_device["id"],
            "X-Device-Key": test_device["api_key"],
            "Content-Type": "application/json",
        },
        json=direct_json_payload,
    )
    assert res.status_code == 201
    data = res.json()
    assert data["status"] == "received"

    reading_id = data["reading_id"]
    db_telem = db_session.query(HiveTelemetry).filter(HiveTelemetry.id == reading_id).first()
    assert db_telem is not None
    assert db_telem.temperature == 35.2
    assert db_telem.humidity == 51.8


def test_prediction_pending_then_generated(
    client: TestClient, auth_headers: dict, test_hive: dict, test_device: dict, db_session: Session
):
    hive_id = test_hive["id"]

    # Assign device
    client.post(
        f"/api/v1/hives/{hive_id}/assignments",
        json={"device_id": test_device["id"]},
        headers=auth_headers,
    )

    # Check prediction on hive with 0 readings -> status: pending
    pred_res1 = client.get(f"/api/v1/hives/{hive_id}/prediction", headers=auth_headers)
    assert pred_res1.status_code == 200
    assert pred_res1.json()["prediction_status"] == "pending"

    # Seed 96 readings spanning 24 hours (one reading every 15 minutes)
    base_time = datetime.now(timezone.utc) - timedelta(hours=24)
    for i in range(96):
        timestamp = base_time + timedelta(minutes=15 * i)
        t = HiveTelemetry(
            hive_id=hive_id,
            device_id=test_device["id"],
            published_at=timestamp,
            temperature=34.6 + (0.3 if i % 2 == 0 else -0.2),
            humidity=55.0,
        )
        db_session.add(t)
    db_session.commit()

    # Now request prediction
    pred_res2 = client.get(f"/api/v1/hives/{hive_id}/prediction", headers=auth_headers)
    assert pred_res2.status_code == 200
    pred_data = pred_res2.json()

    assert pred_data["prediction_status"] == "updated"
    assert pred_data["prediction"] in ["Strong", "Weak"]
    assert pred_data["class_label"] in [0, 1]
    assert 0.0 <= pred_data["weak_probability"] <= 1.0
    assert 0.0 <= pred_data["strong_probability"] <= 1.0
    assert pred_data["alert_level"] in ["GREEN", "YELLOW", "RED_ALERT"]

    # Calling again immediately without new telemetry -> prediction_status: "previous"
    pred_res3 = client.get(f"/api/v1/hives/{hive_id}/prediction", headers=auth_headers)
    assert pred_res3.status_code == 200
    assert pred_res3.json()["prediction_status"] == "previous"


def test_history_pagination(
    client: TestClient, auth_headers: dict, test_hive: dict, db_session: Session
):
    hive_id = test_hive["id"]

    # Seed 25 telemetry records
    base_time = datetime.now(timezone.utc) - timedelta(days=2)
    for i in range(25):
        t = HiveTelemetry(
            hive_id=hive_id,
            published_at=base_time + timedelta(hours=i),
            temperature=34.0 + (i * 0.1),
            humidity=60.0,
        )
        db_session.add(t)
    db_session.commit()

    # Query page 1 (size 10)
    page1 = client.get(
        f"/api/v1/hives/{hive_id}/history?page=1&page_size=10",
        headers=auth_headers,
    )
    assert page1.status_code == 200
    d1 = page1.json()
    assert d1["total_items"] == 25
    assert d1["total_pages"] == 3
    assert len(d1["items"]) == 10

    # Query page 3 (size 10 -> remaining 5 items)
    page3 = client.get(
        f"/api/v1/hives/{hive_id}/history?page=3&page_size=10",
        headers=auth_headers,
    )
    assert page3.status_code == 200
    d3 = page3.json()
    assert len(d3["items"]) == 5


def test_sensor_data_payload_schema_environmental_only():
    """Verify SensorDataPayload schema contains only environmental fields."""
    from src.schemas import SensorDataPayload

    field_names = set(SensorDataPayload.model_fields.keys())
    assert field_names == {"published_at", "temperature", "humidity"}

    # Assert no audio-derived fields exist on the payload schema
    assert "audio_density" not in field_names
    assert "audio_density_ratio" not in field_names
    assert "density_variation" not in field_names
    assert not any(f.startswith("hz_") for f in field_names)


def test_telemetry_pure_environmental_payload_ingestion(
    client: TestClient, auth_headers: dict, test_hive: dict, test_device: dict, db_session: Session
):
    """Verify telemetry ingestion with environmental-only SensorDataPayload."""
    # Assign device
    client.post(
        f"/api/v1/hives/{test_hive['id']}/assignments",
        json={"device_id": test_device["id"]},
        headers=auth_headers,
    )

    published_dt = datetime.now(timezone.utc) + timedelta(minutes=20)
    env_payload = {
        "published_at": published_dt.isoformat(),
        "temperature": 34.7,
        "humidity": 58.2,
    }

    res = client.post(
        f"/api/v1/hives/{test_hive['id']}/telemetry",
        headers={
            "X-Device-Id": test_device["id"],
            "X-Device-Key": test_device["api_key"],
            "Content-Type": "application/json",
        },
        json=env_payload,
    )
    assert res.status_code == 201
    data = res.json()
    assert data["status"] == "received"

    reading_id = data["reading_id"]
    db_telem = db_session.query(HiveTelemetry).filter(HiveTelemetry.id == reading_id).first()
    assert db_telem is not None
    assert db_telem.temperature == 34.7
    assert db_telem.humidity == 58.2


def test_all_input_features_saved_to_history(
    client: TestClient, auth_headers: dict, test_hive: dict, test_device: dict
):
    """Verify all input features sent in telemetry POST request are saved and returned in history."""
    hive_id = test_hive["id"]

    # Assign device
    client.post(
        f"/api/v1/hives/{hive_id}/assignments",
        json={"device_id": test_device["id"]},
        headers=auth_headers,
    )

    published_dt = datetime.now(timezone.utc) + timedelta(minutes=25)
    full_input_payload = {
        "published_at": published_dt.isoformat(),
        "temperature": 35.4,
        "humidity": 53.1,
        "battery_voltage": 3.95,
        "weight_kg": 42.5,
        "solar_mv": 4800,
        "ambient_temp": 24.3,
    }

    post_res = client.post(
        f"/api/v1/hives/{hive_id}/telemetry",
        headers={
            "X-Device-Id": test_device["id"],
            "X-Device-Key": test_device["api_key"],
            "Content-Type": "application/json",
        },
        json=full_input_payload,
    )
    assert post_res.status_code == 201

    # Fetch history
    hist_res = client.get(
        f"/api/v1/hives/{hive_id}/history?page=1&page_size=10",
        headers=auth_headers,
    )
    assert hist_res.status_code == 200
    hist_data = hist_res.json()
    assert hist_data["total_items"] >= 1

    latest_item = hist_data["items"][0]
    assert latest_item["temperature"] == 35.4
    assert latest_item["humidity"] == 53.1
    # Verify extra input features are preserved in extra_json
    assert latest_item["extra_json"] is not None
    assert latest_item["extra_json"]["battery_voltage"] == 3.95
    assert latest_item["extra_json"]["weight_kg"] == 42.5
    assert latest_item["extra_json"]["solar_mv"] == 4800
    assert latest_item["extra_json"]["ambient_temp"] == 24.3


def test_audio_history_retrieval(
    client: TestClient, auth_headers: dict, test_hive: dict, test_device: dict, db_session: Session
):
    """Verify paginated retrieval of acoustic records via GET /audio."""
    hive_id = test_hive["id"]

    # Assign device
    client.post(
        f"/api/v1/hives/{hive_id}/assignments",
        json={"device_id": test_device["id"]},
        headers=auth_headers,
    )

    base_time = datetime.now(timezone.utc) - timedelta(hours=5)
    for i in range(5):
        rec = HiveAudioRecord(
            hive_id=hive_id,
            device_id=test_device["id"],
            recorded_at=base_time + timedelta(hours=i),
            sample_rate=15625,
            duration_seconds=3.0,
            audio_density=10.0 + i,
            audio_density_ratio=0.8,
            density_variation=3.0,
            **{col: -10.0 + i for col in DB_BAND_COLUMNS},
        )
        db_session.add(rec)
    db_session.commit()

    # Query audio history
    audio_res = client.get(
        f"/api/v1/hives/{hive_id}/audio?page=1&page_size=3",
        headers=auth_headers,
    )
    assert audio_res.status_code == 200
    audio_data = audio_res.json()
    assert audio_data["total_items"] == 5
    assert audio_data["total_pages"] == 2
    assert len(audio_data["items"]) == 3
    assert "frequency_bands" in audio_data["items"][0]
    assert audio_data["items"][0]["sample_rate"] == 15625


def test_telemetry_and_audio_complete_independence(
    client: TestClient, auth_headers: dict, test_hive: dict, test_device: dict, db_session: Session
):
    """Verify that climate telemetry and audio records can exist and function completely independently."""
    hive_id = test_hive["id"]

    # Assign device
    client.post(
        f"/api/v1/hives/{hive_id}/assignments",
        json={"device_id": test_device["id"]},
        headers=auth_headers,
    )

    # 1. Post only telemetry
    telem_res = client.post(
        f"/api/v1/hives/{hive_id}/telemetry",
        headers={
            "X-Device-Id": test_device["id"],
            "X-Device-Key": test_device["api_key"],
            "Content-Type": "application/json",
        },
        json={
            "published_at": datetime.now(timezone.utc).isoformat(),
            "temperature": 34.9,
            "humidity": 56.1,
        },
    )
    assert telem_res.status_code == 201

    # Verify no audio records were created by telemetry
    audio_count = db_session.query(HiveAudioRecord).filter(HiveAudioRecord.hive_id == hive_id).count()
    assert audio_count == 0

    # 2. Post only audio
    audio_res = client.post(
        f"/api/v1/hives/{hive_id}/audio",
        headers={
            "X-Device-Id": test_device["id"],
            "X-Device-Key": test_device["api_key"],
            "Content-Type": "application/json",
        },
        json={
            "recorded_at": (datetime.now(timezone.utc) + timedelta(minutes=1)).isoformat(),
            "audio_density": 14.5,
            **{col: 0.0 for col in DB_BAND_COLUMNS},
        },
    )
    assert audio_res.status_code == 201

    # Verify audio record exists
    audio_count_after = db_session.query(HiveAudioRecord).filter(HiveAudioRecord.hive_id == hive_id).count()
    assert audio_count_after == 1

    # Verify telemetry count did not increase
    telem_count = db_session.query(HiveTelemetry).filter(HiveTelemetry.hive_id == hive_id).count()
    assert telem_count == 1


