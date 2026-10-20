"""
split_data.py
Splits the 75-feature labeled dataset strictly by unique physical hive ID into
Training, Validation, and Test sets.
Enforces and programmatically asserts ZERO hive overlap across splits.
Outputs data/train.csv, data/val.csv, data/test.csv, and data/splits.json.
"""

import json
from pathlib import Path
from typing import Dict, List, Set, Any
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

BASE_DIR = Path("/Users/aayushghimire/Datasets/MSPB2")
DATA_DIR = BASE_DIR / "data"


def split_dataset_by_hive(
    dataset_path: Path,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42
):
    print("=" * 80)
    print("STEP 2: LEAKAGE-FREE DATASET PARTITIONING BY UNIQUE HIVE ID")
    print("=" * 80)

    df = pd.read_csv(dataset_path)
    print(f"Loaded dataset: {len(df)} samples across {df['hive_id'].nunique()} unique hives.")

    # 1. Summarize hives to enable stratified hive assignment
    hive_summary = []
    for hive_id, grp in df.groupby('hive_id'):
        total_samples = len(grp)
        strong_samples = (grp['label'] == 1).sum()
        weak_samples = (grp['label'] == 0).sum()
        majority_label = 1 if strong_samples >= weak_samples else 0
        hive_summary.append({
            'hive_id': hive_id,
            'total_samples': total_samples,
            'strong_samples': strong_samples,
            'weak_samples': weak_samples,
            'majority_label': majority_label,
            'season': grp['season'].iloc[0]
        })

    df_hives = pd.DataFrame(hive_summary)
    all_unique_hives: List[str] = sorted(df_hives['hive_id'].tolist())
    print(f"Total physical hives to partition: {len(all_unique_hives)}")

    # 2. Stratified split of hives into Train+Val vs Test
    # Test gets exactly test_size (~15% of hives)
    train_val_hives_df, test_hives_df = train_test_split(
        df_hives,
        test_size=test_size,
        random_state=random_state,
        stratify=df_hives['majority_label']
    )

    # 3. Stratified split of Train+Val into Train vs Val
    # Val gets val_size / (1 - test_size)
    relative_val_size = val_size / (1.0 - test_size)
    train_hives_df, val_hives_df = train_test_split(
        train_val_hives_df,
        test_size=relative_val_size,
        random_state=random_state,
        stratify=train_val_hives_df['majority_label']
    )

    train_hives: Set[str] = set(train_hives_df['hive_id'])
    val_hives: Set[str] = set(val_hives_df['hive_id'])
    test_hives: Set[str] = set(test_hives_df['hive_id'])

    # 4. CRITICAL LEAKAGE ASSERTIONS: Programmatic verification of ZERO hive overlap
    train_val_overlap = train_hives & val_hives
    train_test_overlap = train_hives & test_hives
    val_test_overlap = val_hives & test_hives

    assert len(train_val_overlap) == 0, f"DATA LEAKAGE DETECTED: Overlap between Train and Val: {train_val_overlap}"
    assert len(train_test_overlap) == 0, f"DATA LEAKAGE DETECTED: Overlap between Train and Test: {train_test_overlap}"
    assert len(val_test_overlap) == 0, f"DATA LEAKAGE DETECTED: Overlap between Val and Test: {val_test_overlap}"
    assert (len(train_hives) + len(val_hives) + len(test_hives)) == len(all_unique_hives), "Mismatch in total hive count!"

    print("\n" + "=" * 80)
    print("PROGRAMMATIC LEAKAGE VERIFICATION: PASSED (ZERO OVERLAP)")
    print("=" * 80)
    print(f"Train vs Val Overlap: {len(train_val_overlap)} hives")
    print(f"Train vs Test Overlap: {len(train_test_overlap)} hives")
    print(f"Val vs Test Overlap: {len(val_test_overlap)} hives")

    # 5. Partition the sample DataFrames
    df_train = df[df['hive_id'].isin(train_hives)].copy().reset_index(drop=True)
    df_val = df[df['hive_id'].isin(val_hives)].copy().reset_index(drop=True)
    df_test = df[df['hive_id'].isin(test_hives)].copy().reset_index(drop=True)

    print("\n--- Split Sizes & Proportions ---")
    print(f"Training Set:   {len(df_train)} samples across {len(train_hives)} hives ({len(train_hives)/len(all_unique_hives)*100:.1f}% of hives)")
    print(f"Validation Set: {len(df_val)} samples across {len(val_hives)} hives ({len(val_hives)/len(all_unique_hives)*100:.1f}% of hives)")
    print(f"Test Set:       {len(df_test)} samples across {len(test_hives)} hives ({len(test_hives)/len(all_unique_hives)*100:.1f}% of hives)")

    print("\n--- Class Balance Across Splits ---")
    for name, split_df in [("Train", df_train), ("Val", df_val), ("Test", df_test)]:
        cnts = split_df['label'].value_counts()
        props = split_df['label'].value_counts(normalize=True)
        s_cnt = cnts.get(1, 0)
        w_cnt = cnts.get(0, 0)
        s_pct = props.get(1, 0.0) * 100
        w_pct = props.get(0, 0.0) * 100
        print(f"  {name:5s}: Total={len(split_df):3d} | Strong (1) = {s_cnt:3d} ({s_pct:5.1f}%) | Weak (0) = {w_cnt:3d} ({w_pct:5.1f}%)")

    # 6. Save split files
    train_path = DATA_DIR / "train.csv"
    val_path = DATA_DIR / "val.csv"
    test_path = DATA_DIR / "test.csv"
    splits_json_path = DATA_DIR / "splits.json"

    df_train.to_csv(train_path, index=False)
    df_val.to_csv(val_path, index=False)
    df_test.to_csv(test_path, index=False)

    split_manifest = {
        'total_samples': len(df),
        'total_unique_hives': len(all_unique_hives),
        'hive_overlap_train_val': 0,
        'hive_overlap_train_test': 0,
        'hive_overlap_val_test': 0,
        'splits': {
            'train': {
                'samples_count': len(df_train),
                'hives_count': len(train_hives),
                'hives': sorted(list(train_hives)),
                'class_1_strong': int((df_train['label'] == 1).sum()),
                'class_0_weak': int((df_train['label'] == 0).sum())
            },
            'validation': {
                'samples_count': len(df_val),
                'hives_count': len(val_hives),
                'hives': sorted(list(val_hives)),
                'class_1_strong': int((df_val['label'] == 1).sum()),
                'class_0_weak': int((df_val['label'] == 0).sum())
            },
            'test': {
                'samples_count': len(df_test),
                'hives_count': len(test_hives),
                'hives': sorted(list(test_hives)),
                'class_1_strong': int((df_test['label'] == 1).sum()),
                'class_0_weak': int((df_test['label'] == 0).sum())
            }
        }
    }

    with open(splits_json_path, 'w') as f:
        json.dump(split_manifest, f, indent=2)

    print(f"\nSaved Train CSV to: {train_path.resolve()}")
    print(f"Saved Val CSV to:   {val_path.resolve()}")
    print(f"Saved Test CSV to:  {test_path.resolve()}")
    print(f"Saved Split JSON to: {splits_json_path.resolve()}")

    print("\n--- Complete List of Hive IDs per Split ---")
    print(f"\n[TRAIN HIVES ({len(train_hives)})]:")
    print(", ".join(sorted(list(train_hives))))

    print(f"\n[VALIDATION HIVES ({len(val_hives)})]:")
    print(", ".join(sorted(list(val_hives))))

    print(f"\n[TEST HIVES ({len(test_hives)})]:")
    print(", ".join(sorted(list(test_hives))))


if __name__ == "__main__":
    labeled_data_file = DATA_DIR / "labeled_dataset.csv"
    split_dataset_by_hive(labeled_data_file)
