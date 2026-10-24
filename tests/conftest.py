import io
from pathlib import Path
from typing import Generator
import numpy as np
from scipy.io import wavfile
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool

from src.shared.database import Base, get_db
from src.api.routes import app
from src.shared.config import settings
from src.shared.security import hash_password, hash_api_key
from src.shared.models import User, Hive, IoTDevice, HiveDeviceAssignment

# Use an in-memory SQLite database with StaticPool for thread-safe test isolation
SQLALCHEMY_DATABASE_URL = "sqlite:///:memory:"

test_engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)


@pytest.fixture(scope="session", autouse=True)
def setup_test_directories(tmp_path_factory):
    """Set temporary directory for audio storage during tests."""
    temp_audio_dir = tmp_path_factory.mktemp("test_audio")
    settings.MSPB2_AUDIO_DIR = temp_audio_dir
    yield


@pytest.fixture(autouse=True)
def setup_database():
    """Create fresh tables for every test."""
    Base.metadata.create_all(bind=test_engine)
    yield
    Base.metadata.drop_all(bind=test_engine)


@pytest.fixture
def db_session() -> Generator[Session, None, None]:
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


@pytest.fixture
def client(db_session: Session) -> Generator[TestClient, None, None]:
    def override_get_db():
        try:
            yield db_session
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


@pytest.fixture
def test_user(db_session: Session) -> User:
    user = User(
        id="user-test-12345",
        email="beekeeper@example.com",
        hashed_password=hash_password("ValidPassword123!"),
        full_name="Beekeeper Alice",
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user


@pytest.fixture
def auth_headers(client: TestClient, test_user: User) -> dict:
    response = client.post(
        "/api/v1/auth/login",
        json={"email": test_user.email, "password": "ValidPassword123!"},
    )
    assert response.status_code == 200
    token = response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def test_hive(client: TestClient, auth_headers: dict) -> dict:
    payload = {
        "id": "HIVE_ALPHA",
        "name": "Alpha Observation Hive",
        "apiary_name": "North Meadow",
        "latitude": 45.4215,
        "longitude": -75.6972,
        "description": "Production test colony",
        "metadata_json": {"brood_frames_baseline": 18},
    }
    response = client.post("/api/v1/hives", json=payload, headers=auth_headers)
    assert response.status_code == 201
    return response.json()


@pytest.fixture
def test_device(client: TestClient, auth_headers: dict) -> dict:
    payload = {
        "id": "ESP32_NODE_01",
        "device_name": "Brood Chamber Node 1",
        "hardware_version": "ESP32-WROOM-32D",
        "firmware_version": "v1.2.0",
    }
    response = client.post("/api/v1/iot-devices", json=payload, headers=auth_headers)
    assert response.status_code == 201
    return response.json()


@pytest.fixture
def create_synthetic_wav():
    """Factory fixture to produce a 1-second synthetic WAV file buffer at 15,625 Hz."""
    def _generator(frequency_hz: float = 244.14, duration_sec: float = 1.0) -> io.BytesIO:
        fs = 15625
        t = np.linspace(0, duration_sec, int(fs * duration_sec), endpoint=False)
        # Generate composite wave: fundamental + harmonic
        signal = (
            0.6 * np.sin(2 * np.pi * frequency_hz * t)
            + 0.3 * np.sin(2 * np.pi * (frequency_hz * 2) * t)
        )
        scaled = (signal * 32767.0).astype(np.int16)
        buf = io.BytesIO()
        wavfile.write(buf, fs, scaled)
        buf.seek(0)
        return buf
    return _generator
