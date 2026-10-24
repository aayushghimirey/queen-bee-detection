from pathlib import Path
from typing import List, Tuple
import pandas as pd
from src.shared.config import settings
from src.shared.feature_extraction import ENVIRONMENTAL_FEATURES

# Metadata columns to exclude from training features
METADATA_COLS: List[str] = [
    "sample_id",
    "hive_id",
    "raw_hive_id",
    "season",
    "apiary",
    "inspection_date",
    "window_readings_count",
    "frames_count",
    "label",
]


def load_datasets(data_dir: Path = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Loads train, validation, and test datasets with identical hive-grouped splits.
    Returns (train_df, val_df, test_df, feature_cols).
    """
    if data_dir is None:
        data_dir = settings.MSPB2_BASE_DIR / "data"

    train_df = pd.read_csv(data_dir / "train.csv")
    val_df = pd.read_csv(data_dir / "val.csv")
    test_df = pd.read_csv(data_dir / "test.csv")

    feature_cols = [c for c in ENVIRONMENTAL_FEATURES if c in train_df.columns]
    assert len(feature_cols) == 40, f"Expected 40 environmental features, found {len(feature_cols)}"

    return train_df, val_df, test_df, feature_cols
