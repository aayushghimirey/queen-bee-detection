"""
data_prep.py
Builds the unified labeled dataset for Honey Bee Colony Strength classification.
Pairs ground-truth inspection records with preceding 24-hour sensor windows,
computes 10 environmental microclimate features, and outputs data/labeled_dataset.csv.
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd

# Add src to python path
SRC_DIR = Path(__file__).resolve().parent
sys.path.append(str(SRC_DIR))
from feature_extraction import (
    compute_24h_environmental_features,
    ENVIRONMENTAL_FEATURES,
)

BASE_DIR = Path("/Users/aayushghimire/Datasets/MSPB2")
DATASET_DIR = BASE_DIR / "dataset"
DATA_OUT_DIR = BASE_DIR / "data"
DATA_OUT_DIR.mkdir(parents=True, exist_ok=True)


def _find_dataset_path(rel_path: str) -> Path:
    p = DATASET_DIR / rel_path
    if p.exists():
        return p
    return BASE_DIR / rel_path


def parse_d3_inspections() -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Parses D3 phenotypic inspections for occupied bee frames.
    Threshold: Strong = 1 if frames >= 16.0, Weak = 0 otherwise.
    """
    pheno_path = _find_dataset_path("D3_D4/D3_D4/D3/D3_phenotypic_data.xlsx")
    df_pheno = pd.read_excel(pheno_path)
    print(f"[D3] Loaded phenotypic data: {len(df_pheno)} hives")

    # Map Hive -> Brood Chamber tag -> Sensor Tag
    # Rule: sensor_tag = brood_tag + 200000
    hive_to_sensor: Dict[Any, int] = {}
    for _, row in df_pheno.iterrows():
        hive_id = row['Hive']
        brood_tag = row['Brood chamber']
        if pd.notna(brood_tag):
            try:
                hive_to_sensor[hive_id] = int(brood_tag) + 200000
            except ValueError:
                pass

    # Inspection columns and corresponding cutoff timestamps
    # Beekeepers inspect during morning hours (~10-12 UTC).
    # Cutoff strictly sets the end of the 24-hour evaluation window.
    inspections_cfg = [
        ("June 16th bees frames", "2021-06-16 11:00:00+00:00"),
        ("June 28th bees frames", "2021-06-28 11:00:00+00:00"),
        ("June 29th bees frames", "2021-06-29 11:00:00+00:00"),
        ("Bees frames July 12th or 13th", "2021-07-12 11:00:00+00:00"),
        ("Bees frames July 26th", "2021-07-26 11:00:00+00:00"),
        ("Bees frames August 9th", "2021-08-09 11:00:00+00:00"),
        ("Bees frames August 23rd", "2021-08-23 11:00:00+00:00"),
        ("Bees frames September 14th", "2021-09-14 11:00:00+00:00"),
    ]

    records = []
    for col_name, cutoff_time in inspections_cfg:
        cutoff_dt = pd.to_datetime(cutoff_time)
        for _, row in df_pheno.iterrows():
            hive_id = row['Hive']
            if hive_id not in hive_to_sensor:
                continue

            frame_val = row.get(col_name)
            if pd.isna(frame_val):
                continue

            try:
                frames_count = float(frame_val)
            except (ValueError, TypeError):
                continue

            # Ground-truth binarization
            label = 1 if frames_count >= 16.0 else 0
            apiary = str(row.get('Apiary', 'Unknown'))

            records.append({
                'season': 'D3_2021',
                'hive_id': f"D3_{hive_id}",
                'raw_hive_id': hive_id,
                'sensor_tag': hive_to_sensor[hive_id],
                'apiary': apiary,
                'inspection_col': col_name,
                'cutoff_time': cutoff_dt,
                'frames_count': frames_count,
                'label': label
            })

    df_inspections = pd.DataFrame(records)
    print(f"[D3] Extracted {len(df_inspections)} candidate inspection events across {df_inspections['hive_id'].nunique()} hives.")
    return df_inspections, hive_to_sensor


def parse_d5_d6_inspections() -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Parses D5_D6 phenotypic performance data for occupied bee frames.
    Sums brood chamber + super 1 + super 2 bee frames.
    Threshold: Strong = 1 if frames >= 16.0, Weak = 0 otherwise.
    """
    pheno_path = _find_dataset_path("D5_D6/D5_D6/D5_D6_phenotypic_data.xlsx")
    df_perf = pd.read_excel(pheno_path, sheet_name='Performance data')
    print(f"[D5_D6] Loaded phenotypic data: {len(df_perf)-1} hives")

    # Map Hive -> Brood Chamber tag (direct match)
    hive_to_sensor: Dict[Any, int] = {}
    for idx in range(1, len(df_perf)):
        hive_id = df_perf.iloc[idx, 0]  # # Hive
        brood_val = df_perf.iloc[idx, 4] # Unnamed: 4
        if pd.notna(brood_val):
            try:
                hive_to_sensor[hive_id] = int(brood_val)
            except (ValueError, TypeError):
                pass

    # Dates and column indices in Performance data
    d5_inspections = [
        ("2022-06-01 11:00:00+00:00", 55, 57, 58, "June 1st 2022"),
        ("2022-06-20 11:00:00+00:00", 60, 62, 63, "June 20 2022"),
        ("2022-07-06 11:00:00+00:00", 65, 67, 68, "July 6 2022"),
        ("2022-07-18 11:00:00+00:00", 70, 72, 73, "July 18 2022"),
        ("2022-08-15 11:00:00+00:00", 75, 77, 78, "August 15 2022"),
        ("2023-06-07 11:00:00+00:00", 80, 82, 83, "June 7 2023"),
    ]

    records = []
    for cutoff_str, b_idx, s1_idx, s2_idx, desc in d5_inspections:
        cutoff_dt = pd.to_datetime(cutoff_str)
        for idx in range(1, len(df_perf)):
            hive_id = df_perf.iloc[idx, 0]
            if hive_id not in hive_to_sensor:
                continue

            b_val = df_perf.iloc[idx, b_idx]
            if pd.isna(b_val):
                continue

            try:
                b_frames = float(b_val)
            except (ValueError, TypeError):
                continue

            s1_val = df_perf.iloc[idx, s1_idx]
            s2_val = df_perf.iloc[idx, s2_idx]
            s1_frames = float(s1_val) if pd.notna(s1_val) and str(s1_val).strip() != '' else 0.0
            s2_frames = float(s2_val) if pd.notna(s2_val) and str(s2_val).strip() != '' else 0.0

            total_frames = b_frames + s1_frames + s2_frames
            label = 1 if total_frames >= 16.0 else 0
            apiary = str(df_perf.iloc[idx, 1])

            records.append({
                'season': 'D5_D6_2022_2023',
                'hive_id': f"D5D6_{hive_id}",
                'raw_hive_id': hive_id,
                'sensor_tag': hive_to_sensor[hive_id],
                'apiary': apiary,
                'inspection_col': desc,
                'cutoff_time': cutoff_dt,
                'frames_count': total_frames,
                'label': label
            })

    df_inspections = pd.DataFrame(records)
    print(f"[D5_D6] Extracted {len(df_inspections)} candidate inspection events across {df_inspections['hive_id'].nunique()} hives.")
    return df_inspections, hive_to_sensor


def match_windows_and_extract_features(
    df_sensor: pd.DataFrame,
    df_inspections: pd.DataFrame,
    min_readings: int = 30
) -> pd.DataFrame:
    """
    For each inspection event, extracts sensor records in [cutoff - 24h, cutoff],
    computes 10 environmental features, and pairs with ground-truth label.
    """
    print(f"Aligning {len(df_inspections)} inspections against {len(df_sensor):,} sensor readings...")
    
    # Ensure datetime with UTC timezone
    if not pd.api.types.is_datetime64_any_dtype(df_sensor['published_at']):
        df_sensor['published_at'] = pd.to_datetime(df_sensor['published_at'], utc=True)
    elif df_sensor['published_at'].dt.tz is None:
        df_sensor['published_at'] = df_sensor['published_at'].dt.tz_localize('UTC')

    sensor_by_tag = {tag: grp for tag, grp in df_sensor.groupby('tag_number')}

    labeled_rows: List[Dict[str, Any]] = []
    skipped_coverage = 0

    for idx, insp in df_inspections.iterrows():
        tag = insp['sensor_tag']
        cutoff = insp['cutoff_time']
        start_time = cutoff - pd.Timedelta(hours=24)

        tag_df = sensor_by_tag.get(tag)
        if tag_df is None or len(tag_df) == 0:
            skipped_coverage += 1
            continue

        # Strictly preceding 24 hours: [cutoff - 24h, cutoff]
        mask = (tag_df['published_at'] >= start_time) & (tag_df['published_at'] <= cutoff)
        window_df = tag_df.loc[mask]

        if len(window_df) < min_readings:
            skipped_coverage += 1
            continue

        # Compute the 10 environmental features
        try:
            feats = compute_24h_environmental_features(window_df, time_col='published_at')
        except Exception as e:
            print(f"Warning: feature extraction failed for hive {insp['hive_id']} at {cutoff}: {e}")
            continue

        sample_row: Dict[str, Any] = {
            'sample_id': f"{insp['hive_id']}_{insp['cutoff_time'].strftime('%Y%m%d')}",
            'hive_id': insp['hive_id'],
            'raw_hive_id': str(insp['raw_hive_id']),
            'season': insp['season'],
            'apiary': insp['apiary'],
            'inspection_date': insp['cutoff_time'].strftime('%Y-%m-%d'),
            'window_readings_count': len(window_df),
            'frames_count': insp['frames_count'],
            'label': insp['label']  # 1 = Strong, 0 = Weak
        }
        sample_row.update(feats)
        labeled_rows.append(sample_row)

    df_result = pd.DataFrame(labeled_rows)
    print(f"Extraction complete: Created {len(df_result)} valid 24h labeled samples (skipped {skipped_coverage} with <{min_readings} readings).")
    return df_result


def main():
    print("=" * 80)
    print("STEP 1: BUILDING UNIFIED DATASET WITH 10 24-HOUR ENVIRONMENTAL FEATURES")
    print("=" * 80)

    # 1. Parse D3 Inspections
    d3_inspections, _ = parse_d3_inspections()

    # 2. Load D3 Sensor CSV (strictly environmental microclimate columns)
    d3_sensor_path = _find_dataset_path("D3_D4/D3_D4/D3/D3_sensor_data.csv")
    print(f"\nLoading D3 sensor data from {d3_sensor_path.name}...")
    use_cols = ['tag_number', 'published_at', 'temperature', 'humidity']
    d3_sensor = pd.read_csv(d3_sensor_path, usecols=use_cols, low_memory=False)
    d3_sensor['published_at'] = pd.to_datetime(d3_sensor['published_at'], utc=True)
    print(f"Loaded D3 sensor records: {len(d3_sensor):,}")

    # 3. Match D3
    d3_samples = match_windows_and_extract_features(d3_sensor, d3_inspections, min_readings=30)
    del d3_sensor  # Free memory

    # 4. Parse D5_D6 Inspections
    d5_inspections, _ = parse_d5_d6_inspections()

    # 5. Load D5_D6 Sensor PKL
    d5_sensor_path = _find_dataset_path("D5_D6/D5_D6/D5_D6_sensor_data.pkl")
    print(f"\nLoading D5_D6 sensor data from {d5_sensor_path.name}...")
    try:
        d5_sensor = pd.read_pickle(d5_sensor_path)
        # Check if published_at is in index
        if 'published_at' not in d5_sensor.columns:
            d5_sensor = d5_sensor.reset_index()
        d5_sensor['published_at'] = pd.to_datetime(d5_sensor['published_at'], utc=True)
        # Filter strictly down to microclimate columns
        keep_cols = [c for c in ['tag_number', 'published_at', 'temperature', 'humidity'] if c in d5_sensor.columns]
        d5_sensor = d5_sensor[keep_cols]
        print(f"Loaded D5_D6 sensor records: {len(d5_sensor):,}")

        # Match D5_D6
        d5_samples = match_windows_and_extract_features(d5_sensor, d5_inspections, min_readings=30)
        del d5_sensor  # Free memory
        
        # Combine D3 and D5_D6
        df_combined = pd.concat([d3_samples, d5_samples], ignore_index=True)
    except Exception as e:
        print(f"Note: Could not process D5_D6 sensor pickle ({e}). Proceeding with D3 dataset.")
        df_combined = d3_samples

    print("\n" + "=" * 80)
    print("UNIFIED DATASET SUMMARY")
    print("=" * 80)
    print(f"Total labeled samples: {len(df_combined)}")
    print(f"Unique physical hives: {df_combined['hive_id'].nunique()}")
    print("Class balance (Strong=1 vs Weak=0):")
    counts = df_combined['label'].value_counts()
    props = df_combined['label'].value_counts(normalize=True)
    for lbl in [1, 0]:
        c = counts.get(lbl, 0)
        p = props.get(lbl, 0.0)
        name = "Strong (>= 16 frames)" if lbl == 1 else "Weak (< 16 frames)"
        print(f"  Class {lbl} ({name}): {c} samples ({p*100:.1f}%)")

    # Verify 10 environmental features are present
    missing_feats = [f for f in ENVIRONMENTAL_FEATURES if f not in df_combined.columns]
    print(f"\nFeature verification: 10 environmental features expected, {len(ENVIRONMENTAL_FEATURES) - len(missing_feats)}/10 present.")
    if missing_feats:
        print(f"Warning: Missing features: {missing_feats}")

    # Save to CSV
    out_csv = DATA_OUT_DIR / "labeled_dataset.csv"
    df_combined.to_csv(out_csv, index=False)
    print(f"\nSaved labeled dataset to: {out_csv.resolve()}")


if __name__ == "__main__":
    main()
