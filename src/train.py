"""
train.py
Unified Machine Learning training entrypoint for the MSPB2 Beehive Colony Strength system:
1. Binary Colony Strength Classifier (Strong vs. Weak): src.strength_classifier.train
2. Exact Colony Size & Frame Count Regressor (Continuous Strength): src.frame_regressor.train

Usage:
  python src/train.py --mode both          # Trains both models (default)
  python src/train.py --mode classification # Trains binary Strong/Weak classifier
  python src/train.py --mode regression     # Trains continuous Frame Count regressor
"""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.strength_classifier.train import (
    DEFAULT_WEAK_THRESHOLD,
    main as train_strength,
)
from src.frame_regressor.train import main as train_regressor


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train MSPB2 Beehive ML models (Colony Strength Classifier & Frame Count Regressor)"
    )
    parser.add_argument(
        "--mode",
        choices=["both", "classification", "regression"],
        default="both",
        help="Model training target: 'both' (default), 'classification' (Strong/Weak), or 'regression' (Frame Count)",
    )
    parser.add_argument(
        "--priority-metric",
        choices=["weak_recall", "macro_f1", "accuracy", "roc_auc"],
        default="weak_recall",
        help="Priority metric for classifier deployment selection (default: weak_recall)",
    )
    parser.add_argument(
        "--weak-threshold",
        type=float,
        default=DEFAULT_WEAK_THRESHOLD,
        help=(
            "Probability threshold on weak class for classifying a colony as Weak "
            f"(default: {DEFAULT_WEAK_THRESHOLD})"
        ),
    )
    parser.add_argument(
        "--classifier",
        choices=["auto", "rf", "logistic", "hist_gbdt", "gbdt", "xgboost"],
        default="auto",
        help="Explicit classifier champion selection override (default: auto)",
    )
    parser.add_argument(
        "--regressor-metric",
        choices=["mae", "rmse", "r2", "pearson_r", "tolerance"],
        default="mae",
        help="Priority metric for regressor deployment selection (default: mae)",
    )
    parser.add_argument(
        "--regressor",
        choices=["auto", "rf", "ridge", "gbdt", "hgb", "xgboost"],
        default="auto",
        help="Explicit regressor champion selection override (default: auto)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.mode in ["classification", "both"]:
        print("\n" + "=" * 80)
        print("STAGE 1: BINARY COLONY STRENGTH CLASSIFIER")
        print("=" * 80)
        train_strength(
            priority_metric=args.priority_metric,
            champion_choice=args.classifier,
            weak_threshold=args.weak_threshold,
        )

    if args.mode in ["regression", "both"]:
        print("\n" + "=" * 80)
        print("STAGE 2: EXACT COLONY SIZE & FRAME COUNT REGRESSOR")
        print("=" * 80)
        train_regressor(
            priority_metric=args.regressor_metric, champion_choice=args.regressor
        )

    print("\n" + "=" * 80)
    print("ALL REQUESTED MODEL TRAINING STAGES COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    main()
