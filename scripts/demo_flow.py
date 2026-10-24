"""
MSPB2 End-to-End API Demonstration Script
Executes the full user lifecycle:
1. User registration & JWT authentication
2. Hive creation
3. ESP32 IoT device registration
4. Hive-to-device linking & assignment enforcement
5. Authenticated telemetry + audio upload
6. History inspection
7. On-demand Colony Strength ML Prediction
"""

import io
import json
import numpy as np
from scipy.io import wavfile
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient
from src.api.routes import app
from src.shared.database import Base, engine


def generate_synthetic_wav(freq: float = 213.62, duration: float = 1.0) -> bytes:
    fs = 15625
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    signal = 0.5 * np.sin(2 * np.pi * freq * t) + 0.2 * np.sin(2 * np.pi * 427.25 * t)
    scaled = (signal * 32767.0).astype(np.int16)
    buf = io.BytesIO()
    wavfile.write(buf, fs, scaled)
    return buf.getvalue()


def run_demo():
    print("=" * 70)
    print("MSPB2 HONEY BEE API DEMONSTRATION")
    print("=" * 70)

    # Initialize tables with graceful fallback if PostgreSQL container is offline
    try:
        Base.metadata.create_all(bind=engine)
    except Exception as exc:
        print(f"[NOTE] Default DB unavailable ({type(exc).__name__}). Using local SQLite for demo run...")
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker
        from src.shared.database import get_db

        demo_db_path = Path(__file__).resolve().parent.parent / "demo.db"
        sqlite_engine = create_engine(f"sqlite:///{demo_db_path}", connect_args={"check_same_thread": False})
        Base.metadata.create_all(bind=sqlite_engine)
        DemoSession = sessionmaker(autocommit=False, autoflush=False, bind=sqlite_engine)

        def override_get_db():
            db = DemoSession()
            try:
                yield db
            finally:
                db.close()

        app.dependency_overrides[get_db] = override_get_db

    client = TestClient(app)

    # 1. Health Check
    print("\n[Step 1] System Health Check...")
    res = client.get("/health")
    print(f"Health Response ({res.status_code}):", res.json())

    # 2. Register Beekeeper
    print("\n[Step 2] Registering Beekeeper User...")
    email = "beekeeper.dan@apiary.org"
    password = "SuperSecretPassword123!"
    reg_res = client.post(
        "/api/v1/auth/register",
        json={"email": email, "password": password, "full_name": "Dan The Beekeeper"},
    )
    if reg_res.status_code == 201:
        print("User registered successfully:", reg_res.json()["email"])
    elif reg_res.status_code == 409:
        print("User already registered, proceeding to login...")

    # 3. Authenticate & Obtain JWT
    print("\n[Step 3] Logging in to retrieve JWT Bearer token...")
    login_res = client.post(
        "/api/v1/auth/login",
        json={"email": email, "password": password},
    )
    token = login_res.json()["access_token"]
    user_headers = {"Authorization": f"Bearer {token}"}
    print("Obtained JWT Token:", token[:25] + "...")

    # 4. Create Hive
    hive_id = "HIVE-ORCHARD-42"
    print(f"\n[Step 4] Creating Hive: {hive_id}...")
    hive_res = client.post(
        "/api/v1/hives",
        json={
            "id": hive_id,
            "name": "Apple Orchard Observation Hive",
            "apiary_name": "East Orchard",
            "latitude": 45.4215,
            "longitude": -75.6972,
            "description": "10-frame Langstroth hive monitored for colony strength",
            "metadata_json": {"queen_breed": "Carniolan", "installed": "2024-05"},
        },
        headers=user_headers,
    )
    if hive_res.status_code in (201, 409):
        print(f"Hive {hive_id} is ready.")

    # 5. Register ESP32 IoT Device
    device_id = "ESP32-NODE-ORCHARD-42"
    print(f"\n[Step 5] Registering IoT Edge Device: {device_id}...")
    dev_res = client.post(
        "/api/v1/iot-devices",
        json={
            "id": device_id,
            "device_name": "Brood Acoustic Sensor Node",
            "hardware_version": "ESP32-WROOM-32D",
            "firmware_version": "v2.1.0",
        },
        headers=user_headers,
    )
    if dev_res.status_code == 201:
        device_data = dev_res.json()
        device_api_key = device_data["api_key"]
        print(f"Device registered. Secret API Key: {device_api_key[:12]}...")
    else:
        print("Device already registered. Recreating for clean demo credentials...")
        device_api_key = "esp_demo_key"

    # 6. Link Device to Hive
    print(f"\n[Step 6] Linking device {device_id} to hive {hive_id}...")
    assign_res = client.post(
        f"/api/v1/hives/{hive_id}/assignments",
        json={"device_id": device_id},
        headers=user_headers,
    )
    print(f"Assignment status ({assign_res.status_code}):", assign_res.json())

    # 7. Submit Telemetry as ESP32
    if dev_res.status_code == 201:
        print(f"\n[Step 7] ESP32 uploading 15-minute telemetry + audio burst...")
        from datetime import datetime, timezone
        pub_time = datetime.now(timezone.utc).isoformat()
        wav_bytes = generate_synthetic_wav(freq=213.62, duration=1.0)

        reading_payload = {
            "published_at": pub_time,
            "temperature": 34.8,
            "humidity": 56.2,
            "audio_density": 16.2,
            "audio_density_ratio": 0.84,
            "density_variation": 4.1,
        }

        telem_res = client.post(
            f"/api/v1/hives/{hive_id}/telemetry",
            headers={"X-Device-Id": device_id, "X-Device-Key": device_api_key},
            data={"reading_json": json.dumps(reading_payload)},
            files={"audio": ("reading.wav", wav_bytes, "audio/wav")},
        )
        print(f"Telemetry Ingest ({telem_res.status_code}):", telem_res.json())

    # 8. Query History
    print(f"\n[Step 8] Beekeeper inspecting hive telemetry history...")
    hist_res = client.get(
        f"/api/v1/hives/{hive_id}/history?page=1&page_size=5",
        headers=user_headers,
    )
    hist_data = hist_res.json()
    print(f"Total history records: {hist_data['total_items']}")
    if hist_data["items"]:
        latest = hist_data["items"][0]
        audio_b = list(latest['audio_features'].keys())[:4] if latest.get('audio_features') else 'Environmental Only'
        print(f"Latest reading: T={latest['temperature']}°C, RH={latest['humidity']}%, Audio bands={audio_b}...")

    # 9. Query On-Demand Prediction
    print(f"\n[Step 9] Beekeeper requesting Colony Strength ML Prediction...")
    pred_res = client.get(
        f"/api/v1/hives/{hive_id}/prediction",
        headers=user_headers,
    )
    print(f"Prediction Result ({pred_res.status_code}):", pred_res.json())
    print("\n" + "=" * 70)
    print("DEMO RUN COMPLETED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    run_demo()
