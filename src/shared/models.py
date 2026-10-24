import uuid
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any

from sqlalchemy import (
    String,
    Boolean,
    Float,
    Integer,
    BigInteger,
    Text,
    DateTime,
    ForeignKey,
    UniqueConstraint,
    Index,
    JSON,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.shared.database import Base


def get_utc_now():
    return datetime.now(timezone.utc)


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    email: Mapped[str] = mapped_column(
        String(255), unique=True, nullable=False, index=True
    )
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    full_name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=get_utc_now, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=get_utc_now,
        onupdate=get_utc_now,
        nullable=False,
    )

    # Relationships
    hives: Mapped[List["Hive"]] = relationship(
        "Hive", back_populates="owner", cascade="all, delete-orphan"
    )
    devices: Mapped[List["IoTDevice"]] = relationship(
        "IoTDevice", back_populates="owner", cascade="all, delete-orphan"
    )


class Hive(Base):
    __tablename__ = "hives"

    id: Mapped[str] = mapped_column(String(80), primary_key=True)
    user_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    apiary_name: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    latitude: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    longitude: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    metadata_json: Mapped[Dict[str, Any]] = mapped_column(
        JSON, default=dict, nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=get_utc_now, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=get_utc_now,
        onupdate=get_utc_now,
        nullable=False,
    )

    # Relationships
    owner: Mapped["User"] = relationship("User", back_populates="hives")
    assignments: Mapped[List["HiveDeviceAssignment"]] = relationship(
        "HiveDeviceAssignment", back_populates="hive", cascade="all, delete-orphan"
    )
    telemetry: Mapped[List["HiveTelemetry"]] = relationship(
        "HiveTelemetry", back_populates="hive", cascade="all, delete-orphan"
    )
    predictions: Mapped[List["HivePrediction"]] = relationship(
        "HivePrediction", back_populates="hive", cascade="all, delete-orphan"
    )
    audio_records: Mapped[List["HiveAudioRecord"]] = relationship(
        "HiveAudioRecord", back_populates="hive", cascade="all, delete-orphan"
    )


class IoTDevice(Base):
    __tablename__ = "iot_devices"

    id: Mapped[str] = mapped_column(String(80), primary_key=True)
    user_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    device_name: Mapped[str] = mapped_column(String(100), nullable=False)
    hardware_version: Mapped[str] = mapped_column(
        String(50), default="ESP32-WROOM-32D", nullable=False
    )
    firmware_version: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    api_key_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    last_seen_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=get_utc_now, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=get_utc_now,
        onupdate=get_utc_now,
        nullable=False,
    )

    # Relationships
    owner: Mapped["User"] = relationship("User", back_populates="devices")
    assignments: Mapped[List["HiveDeviceAssignment"]] = relationship(
        "HiveDeviceAssignment", back_populates="device", cascade="all, delete-orphan"
    )
    telemetry: Mapped[List["HiveTelemetry"]] = relationship(
        "HiveTelemetry", back_populates="device"
    )
    audio_records: Mapped[List["HiveAudioRecord"]] = relationship(
        "HiveAudioRecord", back_populates="device"
    )


class HiveDeviceAssignment(Base):
    __tablename__ = "hive_device_assignments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    hive_id: Mapped[str] = mapped_column(
        String(80), ForeignKey("hives.id", ondelete="CASCADE"), nullable=False, index=True
    )
    device_id: Mapped[str] = mapped_column(
        String(80), ForeignKey("iot_devices.id", ondelete="CASCADE"), nullable=False, index=True
    )
    is_active: Mapped[bool] = mapped_column(
        Boolean, default=True, nullable=False, index=True
    )
    assigned_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=get_utc_now, nullable=False
    )
    unassigned_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # Relationships
    hive: Mapped["Hive"] = relationship("Hive", back_populates="assignments")
    device: Mapped["IoTDevice"] = relationship("IoTDevice", back_populates="assignments")


class HiveTelemetry(Base):
    __tablename__ = "hive_telemetry"

    id: Mapped[int] = mapped_column(
        BigInteger().with_variant(Integer, "sqlite"), primary_key=True, autoincrement=True
    )
    hive_id: Mapped[str] = mapped_column(
        String(80), ForeignKey("hives.id", ondelete="CASCADE"), nullable=False, index=True
    )
    device_id: Mapped[Optional[str]] = mapped_column(
        String(80), ForeignKey("iot_devices.id", ondelete="SET NULL"), nullable=True, index=True
    )
    published_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )
    received_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=get_utc_now, nullable=False
    )
    temperature: Mapped[float] = mapped_column(Float, nullable=False)
    humidity: Mapped[float] = mapped_column(Float, nullable=False)
    extra_json: Mapped[Dict[str, Any]] = mapped_column(
        JSON, default=dict, nullable=False
    )

    __table_args__ = (
        UniqueConstraint("hive_id", "published_at", name="uq_hive_published_at"),
        Index("idx_telemetry_hive_published", "hive_id", "published_at"),
    )

    # Relationships
    hive: Mapped["Hive"] = relationship("Hive", back_populates="telemetry")
    device: Mapped[Optional["IoTDevice"]] = relationship("IoTDevice", back_populates="telemetry")


class HiveAudioRecord(Base):
    __tablename__ = "hive_audio_records"

    id: Mapped[int] = mapped_column(
        BigInteger().with_variant(Integer, "sqlite"), primary_key=True, autoincrement=True
    )
    hive_id: Mapped[str] = mapped_column(
        String(80), ForeignKey("hives.id", ondelete="CASCADE"), nullable=False, index=True
    )
    device_id: Mapped[Optional[str]] = mapped_column(
        String(80), ForeignKey("iot_devices.id", ondelete="SET NULL"), nullable=True, index=True
    )
    recorded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=get_utc_now, nullable=False
    )
    audio_path: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    file_size_bytes: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    sample_rate: Mapped[int] = mapped_column(Integer, default=15625, nullable=False)
    duration_seconds: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Acoustic energy and density indicators
    audio_density: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    audio_density_ratio: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    density_variation: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)

    # 16 Frequency Bands matching MSPB2 FFT spacing (122.07 Hz to 579.83 Hz)
    hz_122_07: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    hz_152_59: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    hz_183_11: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    hz_213_62: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    hz_244_14: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    hz_274_66: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    hz_305_18: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    hz_335_69: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    hz_366_21: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    hz_396_73: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    hz_427_25: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    hz_457_76: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    hz_488_28: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    hz_518_80: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    hz_549_32: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    hz_579_83: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)

    notes_json: Mapped[Dict[str, Any]] = mapped_column(
        JSON, default=dict, nullable=False
    )

    __table_args__ = (
        Index("idx_audio_records_hive_recorded", "hive_id", "recorded_at"),
    )

    # Relationships
    hive: Mapped["Hive"] = relationship("Hive", back_populates="audio_records")
    device: Mapped[Optional["IoTDevice"]] = relationship("IoTDevice", back_populates="audio_records")


# Backward compatibility alias
HiveAudioFeature = HiveAudioRecord


class HivePrediction(Base):
    __tablename__ = "hive_predictions"

    id: Mapped[int] = mapped_column(
        BigInteger().with_variant(Integer, "sqlite"), primary_key=True, autoincrement=True
    )
    hive_id: Mapped[str] = mapped_column(
        String(80), ForeignKey("hives.id", ondelete="CASCADE"), nullable=False, index=True
    )
    prediction: Mapped[str] = mapped_column(String(20), nullable=False)  # "Strong" | "Weak"
    class_label: Mapped[int] = mapped_column(Integer, nullable=False)    # 1 | 0
    weak_probability: Mapped[float] = mapped_column(Float, nullable=False)
    strong_probability: Mapped[float] = mapped_column(Float, nullable=False)
    alert_level: Mapped[str] = mapped_column(String(20), nullable=False) # "GREEN" | "YELLOW" | "RED_ALERT"
    predicted_frame_count: Mapped[Optional[float]] = mapped_column(Float, nullable=True) # Continuous frame count prediction
    window_start: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    window_end: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    last_reading_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    features_json: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    predicted_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=get_utc_now, nullable=False, index=True
    )

    __table_args__ = (
        Index("idx_predictions_hive_predicted", "hive_id", "predicted_at"),
    )

    # Relationships
    hive: Mapped["Hive"] = relationship("Hive", back_populates="predictions")
