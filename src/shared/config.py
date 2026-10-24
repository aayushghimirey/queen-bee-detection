import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


ROOT_DIR = Path(__file__).resolve().parent.parent.parent
ENV_FILE = ROOT_DIR / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(str(ENV_FILE), ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Database
    DATABASE_URL: str = "postgresql+psycopg://mspb2_user:mspb2_password@localhost:5432/mspb2_db"

    # Security
    JWT_SECRET: str = "super-secret-key-change-this-in-production-min-32-chars"
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 1440  # 24 hours

    # Paths & Storage
    MSPB2_BASE_DIR: Path = ROOT_DIR
    MSPB2_AUDIO_DIR: Path = ROOT_DIR / "audio"
    MSPB2_ARTIFACTS_DIR: Path = ROOT_DIR / "artifacts"

    # API Metadata
    API_TITLE: str = "MSPB2 Beehive Monitoring & Colony Strength API"
    API_VERSION: str = "1.0.0"
    DEBUG: bool = False

    # Pipeline Thresholds
    MAX_AUDIO_BYTES: int = 20 * 1024 * 1024  # 20 MB
    PREDICTION_INTERVAL_HOURS: int = 1
    WINDOW_HOURS: int = 24
    MIN_READINGS: int = 60
    MIN_COVERAGE_HOURS: float = 23.5


settings = Settings()

LOCAL_ROOT = ROOT_DIR

if not (settings.MSPB2_BASE_DIR / "data").exists():
    settings.MSPB2_BASE_DIR = LOCAL_ROOT

try:
    settings.MSPB2_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
except (PermissionError, OSError):
    settings.MSPB2_AUDIO_DIR = LOCAL_ROOT / "audio"
    settings.MSPB2_AUDIO_DIR.mkdir(parents=True, exist_ok=True)

try:
    if not settings.MSPB2_ARTIFACTS_DIR.exists():
        settings.MSPB2_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
except (PermissionError, OSError):
    settings.MSPB2_ARTIFACTS_DIR = LOCAL_ROOT / "artifacts"
    settings.MSPB2_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

if not (settings.MSPB2_ARTIFACTS_DIR / "colony_strength_model.joblib").exists():
    settings.MSPB2_ARTIFACTS_DIR = LOCAL_ROOT / "artifacts"
