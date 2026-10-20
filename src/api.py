import json
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional
from uuid import uuid4

from fastapi import (
    APIRouter,
    Depends,
    FastAPI,
    File,
    Form,
    Header,
    HTTPException,
    Query,
    Request,
    UploadFile,
    status,
)
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy import desc, func, select
from sqlalchemy.orm import Session

from src.audio_processing import (
    DB_BAND_COLUMNS,
    FREQ_BAND_MAPPING,
    extract_audio_features,
    validate_wav_header,
)
from src.config import settings
from src.database import get_db
from src.models import (
    Hive,
    HiveAudioRecord,
    HiveAudioFeature,
    HiveDeviceAssignment,
    HivePrediction,
    HiveTelemetry,
    IoTDevice,
    User,
)
from src.prediction_service import (
    FEATURE_NAMES,
    generate_hive_prediction,
    get_latest_prediction,
    model,
)
from src.schemas import (
    AudioDataPayload,
    AudioHistoryResponse,
    AudioRecordResponse,
    AudioUploadResponse,
    DeviceAssignmentCreate,
    DeviceAssignmentResponse,
    HistoryResponse,
    HiveCreate,
    HiveResponse,
    IoTDeviceCreate,
    IoTDeviceCreatedResponse,
    IoTDeviceResponse,
    PredictionResponse,
    SensorDataPayload,
    TelemetryItemResponse,
    TelemetryResponse,
    Token,
    UserLogin,
    UserRegister,
    UserResponse,
)
from src.security import (
    create_access_token,
    generate_device_api_key,
    get_authenticated_device,
    get_current_user,
    hash_api_key,
    hash_password,
    verify_password,
)

# Initialize FastAPI application
app = FastAPI(
    title=settings.API_TITLE,
    version=settings.API_VERSION,
    description="MSPB2 Precision Beekeeping IoT Telemetry Ingestion & Colony Strength Classification API",
)

api_router = APIRouter(prefix="/api/v1")


# ============================================================
# 1. USER AUTHENTICATION & MANAGEMENT
# ============================================================


@api_router.post(
    "/auth/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED
)
def register_user(payload: UserRegister, db: Session = Depends(get_db)):
    """Register a new beekeeper account."""
    existing = (
        db.query(User).filter(User.email == payload.email.lower().strip()).first()
    )
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="A user with this email address already exists.",
        )

    user = User(
        email=payload.email.lower().strip(),
        hashed_password=hash_password(payload.password),
        full_name=payload.full_name,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


@api_router.post("/auth/login", response_model=Token)
def login_json(payload: UserLogin, db: Session = Depends(get_db)):
    """Authenticate with JSON payload and receive JWT Bearer token."""
    user = db.query(User).filter(User.email == payload.email.lower().strip()).first()
    if not user or not verify_password(payload.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user account",
        )

    access_token = create_access_token(data={"sub": str(user.id), "email": user.email})
    return {"access_token": access_token, "token_type": "bearer"}


@api_router.post("/auth/token", response_model=Token)
def login_oauth2(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db),
):
    """OAuth2 compatible token login for Swagger UI and API clients."""
    user = (
        db.query(User).filter(User.email == form_data.username.lower().strip()).first()
    )
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token = create_access_token(data={"sub": str(user.id), "email": user.email})
    return {"access_token": access_token, "token_type": "bearer"}


@api_router.get("/auth/me", response_model=UserResponse)
def get_current_user_profile(current_user: User = Depends(get_current_user)):
    """Retrieve the authenticated user's profile."""
    return current_user


# ============================================================
# 2. HIVE MANAGEMENT
# ============================================================


def _populate_hive_response(hive: Hive, db: Session) -> HiveResponse:
    # Find active assigned device
    assignment = (
        db.query(HiveDeviceAssignment)
        .filter(
            HiveDeviceAssignment.hive_id == hive.id,
            HiveDeviceAssignment.is_active == True,
        )
        .first()
    )
    return HiveResponse(
        id=hive.id,
        user_id=hive.user_id,
        name=hive.name,
        apiary_name=hive.apiary_name,
        latitude=hive.latitude,
        longitude=hive.longitude,
        description=hive.description,
        metadata_json=hive.metadata_json or {},
        active_device_id=assignment.device_id if assignment else None,
        created_at=hive.created_at,
        updated_at=hive.updated_at,
    )


@api_router.post(
    "/hives", response_model=HiveResponse, status_code=status.HTTP_201_CREATED
)
def create_hive(
    payload: HiveCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Create a new user-owned beehive."""
    existing = db.query(Hive).filter(Hive.id == payload.id).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Hive with ID '{payload.id}' already exists.",
        )

    hive = Hive(
        id=payload.id,
        user_id=current_user.id,
        name=payload.name,
        apiary_name=payload.apiary_name,
        latitude=payload.latitude,
        longitude=payload.longitude,
        description=payload.description,
        metadata_json=payload.metadata_json or {},
    )
    db.add(hive)
    db.commit()
    db.refresh(hive)
    return _populate_hive_response(hive, db)


@api_router.get("/hives", response_model=List[HiveResponse])
def list_hives(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """List all beehives belonging to the current user."""
    hives = (
        db.query(Hive)
        .filter(Hive.user_id == current_user.id)
        .order_by(Hive.created_at.desc())
        .all()
    )
    return [_populate_hive_response(h, db) for h in hives]


@api_router.get("/hives/{hive_id}", response_model=HiveResponse)
def get_hive_detail(
    hive_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Retrieve details of a specific hive."""
    hive = (
        db.query(Hive)
        .filter(Hive.id == hive_id, Hive.user_id == current_user.id)
        .first()
    )
    if not hive:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Hive not found"
        )
    return _populate_hive_response(hive, db)


# ============================================================
# 3. IOT DEVICE REGISTRATION & MANAGEMENT
# ============================================================


def _populate_device_response(device: IoTDevice, db: Session) -> IoTDeviceResponse:
    assignment = (
        db.query(HiveDeviceAssignment)
        .filter(
            HiveDeviceAssignment.device_id == device.id,
            HiveDeviceAssignment.is_active == True,
        )
        .first()
    )
    return IoTDeviceResponse(
        id=device.id,
        device_name=device.device_name,
        hardware_version=device.hardware_version,
        firmware_version=device.firmware_version,
        is_active=device.is_active,
        last_seen_at=device.last_seen_at,
        assigned_hive_id=assignment.hive_id if assignment else None,
        created_at=device.created_at,
    )


@api_router.post(
    "/iot-devices",
    response_model=IoTDeviceCreatedResponse,
    status_code=status.HTTP_201_CREATED,
)
def register_iot_device(
    payload: IoTDeviceCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Register a new ESP32 edge device.
    Generates a secure API key returned ONCE in the response.
    """
    existing = db.query(IoTDevice).filter(IoTDevice.id == payload.id).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"IoT device with ID '{payload.id}' is already registered.",
        )

    raw_api_key = generate_device_api_key()
    hashed_key = hash_api_key(raw_api_key)

    device = IoTDevice(
        id=payload.id,
        user_id=current_user.id,
        device_name=payload.device_name,
        hardware_version=payload.hardware_version,
        firmware_version=payload.firmware_version,
        api_key_hash=hashed_key,
    )
    db.add(device)
    db.commit()
    db.refresh(device)

    return IoTDeviceCreatedResponse(
        id=device.id,
        device_name=device.device_name,
        hardware_version=device.hardware_version,
        firmware_version=device.firmware_version,
        api_key=raw_api_key,
        is_active=device.is_active,
        created_at=device.created_at,
    )


@api_router.get("/iot-devices", response_model=List[IoTDeviceResponse])
def list_iot_devices(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """List all IoT devices registered by the current user."""
    devices = (
        db.query(IoTDevice)
        .filter(IoTDevice.user_id == current_user.id)
        .order_by(IoTDevice.created_at.desc())
        .all()
    )
    return [_populate_device_response(d, db) for d in devices]


# ============================================================
# 4. HIVE-DEVICE ASSIGNMENT (LINKING)
# ============================================================


@api_router.post(
    "/hives/{hive_id}/assignments", response_model=DeviceAssignmentResponse
)
def assign_device_to_hive(
    hive_id: str,
    payload: DeviceAssignmentCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Link an IoT device to a specific hive.
    Enforces that the user owns both the hive and the device.
    Deactivates any previous assignments for this hive or device.
    """
    hive = (
        db.query(Hive)
        .filter(Hive.id == hive_id, Hive.user_id == current_user.id)
        .first()
    )
    if not hive:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Hive not found"
        )

    device = (
        db.query(IoTDevice)
        .filter(IoTDevice.id == payload.device_id, IoTDevice.user_id == current_user.id)
        .first()
    )
    if not device:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="IoT device not found"
        )

    now = datetime.now(timezone.utc)

    # Deactivate existing assignments for this hive
    db.query(HiveDeviceAssignment).filter(
        HiveDeviceAssignment.hive_id == hive_id,
        HiveDeviceAssignment.is_active == True,
    ).update({"is_active": False, "unassigned_at": now})

    # Deactivate existing assignments for this device
    db.query(HiveDeviceAssignment).filter(
        HiveDeviceAssignment.device_id == payload.device_id,
        HiveDeviceAssignment.is_active == True,
    ).update({"is_active": False, "unassigned_at": now})

    # Create new active assignment
    assignment = HiveDeviceAssignment(
        hive_id=hive_id,
        device_id=payload.device_id,
        is_active=True,
        assigned_at=now,
    )
    db.add(assignment)
    db.commit()
    db.refresh(assignment)

    return assignment


@api_router.delete(
    "/hives/{hive_id}/assignments/{device_id}", status_code=status.HTTP_204_NO_CONTENT
)
def unassign_device_from_hive(
    hive_id: str,
    device_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Unassign an IoT device from a hive."""
    hive = (
        db.query(Hive)
        .filter(Hive.id == hive_id, Hive.user_id == current_user.id)
        .first()
    )
    if not hive:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Hive not found"
        )

    assignment = (
        db.query(HiveDeviceAssignment)
        .filter(
            HiveDeviceAssignment.hive_id == hive_id,
            HiveDeviceAssignment.device_id == device_id,
            HiveDeviceAssignment.is_active == True,
        )
        .first()
    )
    if not assignment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Active assignment not found"
        )

    assignment.is_active = False
    assignment.unassigned_at = datetime.now(timezone.utc)
    db.commit()


# ============================================================
# 5. TELEMETRY INGESTION (DEVICE-AUTHENTICATED)
# ============================================================


@api_router.post("/hives/{hive_id}/telemetry", response_model=TelemetryResponse, status_code=status.HTTP_201_CREATED)
async def submit_telemetry(
    hive_id: str,
    request: Request,
    authenticated_device: IoTDevice = Depends(get_authenticated_device),
    db: Session = Depends(get_db),
):
    """
    Ingest environmental sensor telemetry (microclimate: temperature, humidity, and custom sensors).
    Dedicated lightweight endpoint for climate sensor nodes.
    """
    # Verify active assignment
    assignment = (
        db.query(HiveDeviceAssignment)
        .filter(
            HiveDeviceAssignment.hive_id == hive_id,
            HiveDeviceAssignment.device_id == authenticated_device.id,
            HiveDeviceAssignment.is_active == True,
        )
        .first()
    )
    if not assignment:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Device '{authenticated_device.id}' is not actively assigned to hive '{hive_id}'.",
        )

    content_type = request.headers.get("content-type", "").lower()
    data = {}

    if "application/json" in content_type:
        try:
            data = await request.json()
            payload = SensorDataPayload.model_validate(data)
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Invalid JSON payload: {exc}",
            )
    elif "multipart/form-data" in content_type or "application/x-www-form-urlencoded" in content_type:
        form = await request.form()
        reading_json = form.get("reading_json")
        if not reading_json:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Missing reading_json in form data",
            )
        try:
            data = json.loads(reading_json)
            payload = SensorDataPayload.model_validate(data)
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Invalid reading_json in form data: {exc}",
            )
    else:
        try:
            data = await request.json()
            payload = SensorDataPayload.model_validate(data)
        except Exception:
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail=f"Unsupported Content-Type: {content_type}. Use application/json.",
            )

    # Normalize published_at to UTC
    published_dt = payload.published_at
    if published_dt.tzinfo is None:
        published_dt = published_dt.replace(tzinfo=timezone.utc)
    else:
        published_dt = published_dt.astimezone(timezone.utc)

    # Idempotency check
    existing = (
        db.query(HiveTelemetry)
        .filter(HiveTelemetry.hive_id == hive_id, HiveTelemetry.published_at == published_dt)
        .first()
    )
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"A telemetry reading for hive '{hive_id}' at timestamp '{published_dt.isoformat()}' already exists.",
        )

    # Collect extra payload properties (e.g., battery_voltage, weight_kg, ambient_temp)
    standard_keys = {
        "published_at",
        "temperature",
        "humidity",
    }
    extra_json = {k: v for k, v in data.items() if k not in standard_keys}

    # Persist pure environmental telemetry record
    telemetry = HiveTelemetry(
        hive_id=hive_id,
        device_id=authenticated_device.id,
        published_at=published_dt,
        temperature=payload.temperature,
        humidity=payload.humidity,
        extra_json=extra_json,
    )
    db.add(telemetry)
    db.commit()
    db.refresh(telemetry)

    return TelemetryResponse(
        status="received",
        hive_id=hive_id,
        device_id=authenticated_device.id,
        reading_id=telemetry.id,
        published_at=published_dt,
    )


# ============================================================
# 5B. ACOUSTIC / AUDIO RECORDING INGESTION & QUERY
# ============================================================


@api_router.post("/hives/{hive_id}/audio", response_model=AudioUploadResponse, status_code=status.HTTP_201_CREATED)
async def submit_audio(
    hive_id: str,
    request: Request,
    authenticated_device: IoTDevice = Depends(get_authenticated_device),
    db: Session = Depends(get_db),
):
    """
    Ingest acoustic audio recordings or precomputed FFT features from an acoustic logger node.
    Accepts:
    1. Multipart form data with an 'audio' WAV file burst and optional 'recorded_at'.
    2. Direct JSON with precomputed 16 frequency bands and acoustic density metrics.
    """
    # Verify active assignment
    assignment = (
        db.query(HiveDeviceAssignment)
        .filter(
            HiveDeviceAssignment.hive_id == hive_id,
            HiveDeviceAssignment.device_id == authenticated_device.id,
            HiveDeviceAssignment.is_active == True,
        )
        .first()
    )
    if not assignment:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Device '{authenticated_device.id}' is not actively assigned to hive '{hive_id}'.",
        )

    content_type = request.headers.get("content-type", "").lower()
    audio = None
    data = {}
    recorded_dt = datetime.now(timezone.utc)
    audio_path = None
    file_size_bytes = 0
    saved_filename = None

    if "multipart/form-data" in content_type:
        form = await request.form()
        audio = form.get("audio")
        if not audio or not hasattr(audio, "filename") or not audio.filename:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Missing 'audio' file in multipart form data",
            )
        rec_time_str = form.get("recorded_at")
        if rec_time_str:
            try:
                recorded_dt = datetime.fromisoformat(str(rec_time_str))
            except Exception:
                pass

        if recorded_dt.tzinfo is None:
            recorded_dt = recorded_dt.replace(tzinfo=timezone.utc)
        else:
            recorded_dt = recorded_dt.astimezone(timezone.utc)

        # Save audio file to storage directory
        hive_audio_folder = settings.MSPB2_AUDIO_DIR / hive_id
        hive_audio_folder.mkdir(parents=True, exist_ok=True)
        saved_filename = f"{recorded_dt.strftime('%Y%m%dT%H%M%S')}_{uuid4().hex[:8]}.wav"
        saved_path = hive_audio_folder / saved_filename

        try:
            with saved_path.open("wb") as out:
                while True:
                    chunk = await audio.read(1024 * 1024)
                    if not chunk:
                        break
                    file_size_bytes += len(chunk)
                    if file_size_bytes > settings.MAX_AUDIO_BYTES:
                        raise HTTPException(
                            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                            detail=f"Audio file exceeds {settings.MAX_AUDIO_BYTES // (1024*1024)} MB limit",
                        )
                    out.write(chunk)
        except Exception:
            saved_path.unlink(missing_ok=True)
            raise
        finally:
            await audio.close()

        # Validate RIFF/WAVE header
        try:
            validate_wav_header(saved_path)
        except HTTPException:
            saved_path.unlink(missing_ok=True)
            raise

        # Extract 16 frequency bands and acoustic properties from WAV
        try:
            audio_feats = extract_audio_features(saved_path)
        except HTTPException:
            saved_path.unlink(missing_ok=True)
            raise

        audio_path = str(saved_path)

    elif "application/json" in content_type:
        try:
            data = await request.json()
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Invalid JSON payload: {exc}",
            )
        if "recorded_at" in data and data["recorded_at"]:
            try:
                recorded_dt = datetime.fromisoformat(str(data["recorded_at"]))
            except Exception:
                pass
        if recorded_dt.tzinfo is None:
            recorded_dt = recorded_dt.replace(tzinfo=timezone.utc)
        else:
            recorded_dt = recorded_dt.astimezone(timezone.utc)

        audio_feats = {
            "sample_rate": int(data.get("sample_rate", 15625)),
            "duration_seconds": float(data.get("duration_seconds", 3.0)) if data.get("duration_seconds") is not None else None,
            "audio_density": float(data.get("audio_density", 0.0)),
            "audio_density_ratio": float(data.get("audio_density_ratio", 0.0)),
            "density_variation": float(data.get("density_variation", 0.0)),
        }
        for db_col, feat_col in FREQ_BAND_MAPPING.items():
            val = data.get(db_col)
            if val is None:
                val = data.get(feat_col)
            audio_feats[db_col] = float(val) if val is not None else 0.0
    else:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported Content-Type: {content_type}. Use multipart/form-data or application/json.",
        )

    # Persist audio record
    record = HiveAudioRecord(
        hive_id=hive_id,
        device_id=authenticated_device.id,
        recorded_at=recorded_dt,
        audio_path=audio_path,
        file_size_bytes=file_size_bytes if file_size_bytes > 0 else None,
        sample_rate=audio_feats["sample_rate"],
        duration_seconds=audio_feats.get("duration_seconds"),
        audio_density=audio_feats.get("audio_density", 0.0),
        audio_density_ratio=audio_feats.get("audio_density_ratio", 0.0),
        density_variation=audio_feats.get("density_variation", 0.0),
        **{col: audio_feats.get(col, 0.0) for col in DB_BAND_COLUMNS},
    )
    db.add(record)
    db.commit()
    db.refresh(record)

    freq_dict = {col: getattr(record, col) for col in DB_BAND_COLUMNS}

    return AudioUploadResponse(
        status="received",
        audio_id=record.id,
        hive_id=hive_id,
        device_id=authenticated_device.id,
        recorded_at=recorded_dt,
        filename=saved_filename,
        file_size_bytes=file_size_bytes if file_size_bytes > 0 else None,
        duration_seconds=record.duration_seconds,
        sample_rate=record.sample_rate,
        features_extracted=True,
        audio_density=record.audio_density,
        audio_density_ratio=record.audio_density_ratio,
        density_variation=record.density_variation,
        frequency_bands=freq_dict,
    )


@api_router.get("/hives/{hive_id}/audio", response_model=AudioHistoryResponse)
def get_audio_history(
    hive_id: str,
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(50, ge=1, le=500, description="Items per page"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Paginated retrieval of acoustic records and 16-band spectral features for a hive.
    """
    hive = (
        db.query(Hive)
        .filter(Hive.id == hive_id, Hive.user_id == current_user.id)
        .first()
    )
    if not hive:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Hive not found"
        )

    offset = (page - 1) * page_size
    total = (
        db.query(func.count(HiveAudioRecord.id))
        .filter(HiveAudioRecord.hive_id == hive_id)
        .scalar()
        or 0
    )

    records = (
        db.query(HiveAudioRecord)
        .filter(HiveAudioRecord.hive_id == hive_id)
        .order_by(HiveAudioRecord.recorded_at.desc(), HiveAudioRecord.id.desc())
        .offset(offset)
        .limit(page_size)
        .all()
    )

    items = []
    for r in records:
        freq_dict = {col: getattr(r, col) for col in DB_BAND_COLUMNS}
        items.append(
            AudioRecordResponse(
                id=r.id,
                hive_id=r.hive_id,
                device_id=r.device_id,
                recorded_at=r.recorded_at,
                created_at=r.created_at,
                audio_path=r.audio_path,
                file_size_bytes=r.file_size_bytes,
                sample_rate=r.sample_rate,
                duration_seconds=r.duration_seconds,
                audio_density=r.audio_density,
                audio_density_ratio=r.audio_density_ratio,
                density_variation=r.density_variation,
                frequency_bands=freq_dict,
                notes_json=r.notes_json,
            )
        )

    total_pages = (total + page_size - 1) // page_size if total > 0 else 0

    return AudioHistoryResponse(
        hive_id=hive_id,
        page=page,
        page_size=page_size,
        total_items=total,
        total_pages=total_pages,
        items=items,
    )


# ============================================================
# 6. COLONY STRENGTH PREDICTION (ON-DEMAND)
# ============================================================


@api_router.get("/hives/{hive_id}/prediction", response_model=PredictionResponse)
def get_or_generate_prediction(
    hive_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    On-demand colony strength prediction for a beehive.
    - Protected by user authentication.
    - Evaluates 24-hour sensor window across 75 features.
    - Evaluates model only when new telemetry is available and prediction interval is due.
    - Returns Strong (1) vs Weak (0) prediction with probability and alert tier.
    """
    hive = (
        db.query(Hive)
        .filter(Hive.id == hive_id, Hive.user_id == current_user.id)
        .first()
    )
    if not hive:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Hive not found"
        )

    result = generate_hive_prediction(db, hive_id)
    return result


# ============================================================
# 7. PAGINATED TELEMETRY HISTORY
# ============================================================


@api_router.get("/hives/{hive_id}/history", response_model=HistoryResponse)
def get_hive_history(
    hive_id: str,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Retrieve paginated historical telemetry and acoustic features for a hive."""
    hive = (
        db.query(Hive)
        .filter(Hive.id == hive_id, Hive.user_id == current_user.id)
        .first()
    )
    if not hive:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Hive not found"
        )

    offset = (page - 1) * page_size
    total = (
        db.query(func.count(HiveTelemetry.id))
        .filter(HiveTelemetry.hive_id == hive_id)
        .scalar()
        or 0
    )

    readings = (
        db.query(HiveTelemetry)
        .filter(HiveTelemetry.hive_id == hive_id)
        .order_by(HiveTelemetry.published_at.desc(), HiveTelemetry.id.desc())
        .offset(offset)
        .limit(page_size)
        .all()
    )

    items = []
    for r in readings:
        items.append(
            TelemetryItemResponse(
                id=r.id,
                hive_id=r.hive_id,
                device_id=r.device_id,
                published_at=r.published_at,
                received_at=r.received_at,
                temperature=r.temperature,
                humidity=r.humidity,
                audio_density=None,
                audio_density_ratio=None,
                density_variation=None,
                audio_features=None,
                extra_json=r.extra_json,
            )
        )

    total_pages = (total + page_size - 1) // page_size if total > 0 else 0

    return HistoryResponse(
        hive_id=hive_id,
        page=page,
        page_size=page_size,
        total_items=total,
        total_pages=total_pages,
        items=items,
    )


# ============================================================
# SYSTEM HEALTH CHECK
# ============================================================


@app.get("/health", tags=["System"])
def health_check(db: Session = Depends(get_db)):
    """System liveness and dependency status check."""
    db_ok = True
    try:
        db.execute(select(1)).scalar()
    except Exception:
        db_ok = False

    return {
        "status": "online" if db_ok else "degraded",
        "database_connected": db_ok,
        "model_loaded": model is not None,
        "model_feature_count": len(FEATURE_NAMES),
        "artifacts_dir": str(settings.MSPB2_ARTIFACTS_DIR),
        "audio_dir": str(settings.MSPB2_AUDIO_DIR),
    }


# Include versioned API router
app.include_router(api_router)
