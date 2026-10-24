"""
src.frame_regressor
Exact Honey Bee Colony Size & Frame Count Regressor (Continuous Strength).
"""

from src.frame_regressor.model import ClippedRegressor
from src.frame_regressor.train import (
    main as train_frame_regressor,
    evaluate_regressor,
)
