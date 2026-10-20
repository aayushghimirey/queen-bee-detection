from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, ConfigDict, Field


# ============================================================
# USER & AUTH SCHEMAS
# ============================================================

class UserRegister(BaseModel):
    email: str = Field(..., max_length=255)
    password: str = Field(..., min_length=6, max_length=128)
    full_name: Optional[str] = Field(None, max_length=255)


class UserLogin(BaseModel):
    email: str
    password: str


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    email: str
    full_name: Optional[str] = None
    is_active: bool
    created_at: datetime


# ============================================================
# HIVE SCHEMAS
# ============================================================

class HiveCreate(BaseModel):
    id: str = Field(..., pattern=r"^[A-Za-z0-9_-]{1,80}$", description="Unique Hive identifier")
    name: str = Field(..., max_length=100)
    apiary_name: Optional[str] = Field(None, max_length=100)
    latitude: Optional[float] = Field(None, ge=-90.0, le=90.0)
    longitude: Optional[float] = Field(None, ge=-180.0, le=180.0)
    description: Optional[str] = None
    metadata_json: Optional[Dict[str, Any]] = Field(default_factory=dict)


class HiveResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    user_id: str
    name: str
    apiary_name: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    description: Optional[str] = None
    metadata_json: Dict[str, Any] = Field(default_factory=dict)
    active_device_id: Optional[str] = None
    created_at: datetime
    updated_at: datetime


# ============================================================
# IOT DEVICE SCHEMAS
# ============================================================

class IoTDeviceCreate(BaseModel):
    id: str = Field(..., pattern=r"^[A-Za-z0-9_-]{1,80}$", description="Unique Device identifier, e.g. serial or MAC")
    device_name: str = Field(..., max_length=100)
    hardware_version: str = Field("ESP32-WROOM-32D", max_length=50)
    firmware_version: Optional[str] = Field(None, max_length=50)


class IoTDeviceCreatedResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    device_name: str
    hardware_version: str
    firmware_version: Optional[str] = None
    api_key: str = Field(..., description="Secret device API key; store securely, not shown again.")
    is_active: bool
    created_at: datetime


class IoTDeviceResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    device_name: str
    hardware_version: str
    firmware_version: Optional[str] = None
    is_active: bool
    last_seen_at: Optional[datetime] = None
    assigned_hive_id: Optional[str] = None
    created_at: datetime


# ============================================================
# ASSIGNMENT SCHEMAS
# ============================================================

class DeviceAssignmentCreate(BaseModel):
    device_id: str


class DeviceAssignmentResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    hive_id: str
    device_id: str
    is_active: bool
    assigned_at: datetime
    unassigned_at: Optional[datetime] = None


# ============================================================
# TELEMETRY SCHEMAS
# ============================================================

class SensorDataPayload(BaseModel):
    """
    Telemetry payload for the active colony-strength model.
    Contains microclimate environmental features (temperature and humidity).
    Audio-derived fields are removed from the active model payload to reduce
    edge bandwidth, power, and unnecessary feature overhead.
    """
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    published_at: datetime
    temperature: float = Field(..., ge=-40.0, le=85.0, description="Internal hive temperature in °C")
    humidity: float = Field(..., ge=0.0, le=100.0, description="Internal relative humidity in %")


class TelemetryResponse(BaseModel):
    status: str = "received"
    hive_id: str
    device_id: Optional[str] = None
    reading_id: int
    published_at: datetime


# ============================================================
# ACOUSTIC / AUDIO SCHEMAS
# ============================================================

class AudioDataPayload(BaseModel):
    """
    Optional direct acoustic payload if an edge node performs on-device FFT extraction.
    """
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    recorded_at: Optional[datetime] = None
    sample_rate: int = 15625
    duration_seconds: Optional[float] = None
    audio_density: float = 0.0
    audio_density_ratio: float = 0.0
    density_variation: float = 0.0


class AudioUploadResponse(BaseModel):
    status: str = "received"
    audio_id: int
    hive_id: str
    device_id: Optional[str] = None
    recorded_at: datetime
    filename: Optional[str] = None
    file_size_bytes: Optional[int] = None
    duration_seconds: Optional[float] = None
    sample_rate: int = 15625
    features_extracted: bool = True
    audio_density: Optional[float] = None
    audio_density_ratio: Optional[float] = None
    density_variation: Optional[float] = None
    frequency_bands: Optional[Dict[str, float]] = None


class AudioRecordResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    hive_id: str
    device_id: Optional[str] = None
    recorded_at: datetime
    created_at: datetime
    audio_path: Optional[str] = None
    file_size_bytes: Optional[int] = None
    sample_rate: int
    duration_seconds: Optional[float] = None
    audio_density: float
    audio_density_ratio: float
    density_variation: float
    frequency_bands: Dict[str, float]
    notes_json: Optional[Dict[str, Any]] = None


class AudioHistoryResponse(BaseModel):
    hive_id: str
    page: int
    page_size: int
    total_items: int
    total_pages: int
    items: List[AudioRecordResponse]


# ============================================================
# PREDICTION SCHEMAS
# ============================================================

class PredictionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    hive_id: str
    prediction: Optional[str] = None
    class_label: Optional[int] = None
    weak_probability: Optional[float] = None
    strong_probability: Optional[float] = None
    alert_level: Optional[str] = None
    window_start: Optional[datetime] = None
    window_end: Optional[datetime] = None
    predicted_at: Optional[datetime] = None
    prediction_status: str  # "updated" | "previous" | "pending"
    message: Optional[str] = None
    refresh_message: Optional[str] = None


# ============================================================
# HISTORY SCHEMAS
# ============================================================

class TelemetryItemResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    hive_id: str
    device_id: Optional[str] = None
    published_at: datetime
    received_at: datetime
    temperature: float
    humidity: float
    audio_density: Optional[float] = None
    audio_density_ratio: Optional[float] = None
    density_variation: Optional[float] = None
    audio_features: Optional[Dict[str, float]] = None
    extra_json: Optional[Dict[str, Any]] = None


class HistoryResponse(BaseModel):
    hive_id: str
    page: int
    page_size: int
    total_items: int
    total_pages: int
    items: List[TelemetryItemResponse]
