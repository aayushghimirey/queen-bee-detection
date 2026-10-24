"""
src.shared
Common foundational infrastructure, ORM models, Pydantic schemas, and data utilities.
"""

from src.shared.config import settings
from src.shared.database import Base, engine, SessionLocal, get_db
from src.shared.models import (
    User,
    Hive,
    IoTDevice,
    HiveDeviceAssignment,
    HiveTelemetry,
    HiveAudioRecord,
    HiveAudioFeature,
    HivePrediction,
)
from src.shared.schemas import (
    UserRegister,
    UserLogin,
    Token,
    UserResponse,
    HiveCreate,
    HiveResponse,
    IoTDeviceCreate,
    IoTDeviceResponse,
    DeviceAssignmentCreate,
    DeviceAssignmentResponse,
    SensorDataPayload,
    TelemetryResponse,
    AudioDataPayload,
    AudioUploadResponse,
    AudioRecordResponse,
    AudioHistoryResponse,
    PredictionResponse,
    TelemetryItemResponse,
    HistoryResponse,
)
from src.shared.security import (
    hash_password,
    verify_password,
    create_access_token,
    decode_access_token,
    get_current_user,
    generate_device_api_key,
    hash_api_key,
    verify_api_key,
    get_authenticated_device,
)
from src.shared.feature_extraction import (
    compute_24h_environmental_features,
    compute_24h_features,
    ENVIRONMENTAL_FEATURES,
)
from src.shared.audio_processing import DB_BAND_COLUMNS
from src.shared.dataset import load_datasets, METADATA_COLS
