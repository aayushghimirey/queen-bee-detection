"""
MSPB2 Honey Bee Monitoring and Colony Strength System
"""

import sys

# Ensure ClippedRegressor is resolvable from src.train_frame_regressor or src.frame_regressor.model
from src.frame_regressor.model import ClippedRegressor
from src.shared.config import settings
from src.shared.database import Base, engine, get_db
from src.shared.models import (
    User,
    Hive,
    IoTDevice,
    HiveDeviceAssignment,
    HiveTelemetry,
    HiveAudioRecord,
    HivePrediction,
)
