"""Drift Check Script (CSL236 Bonus Stage - up to 5 marks).

Compares mean and standard deviation of logged prediction inputs against baseline training data.
Run with: python drift_check.py
"""

import json
from pathlib import Path
import pandas as pd
import numpy as np

from schemas import NUMERICAL_FEATURES

BASE_DIR = Path(__file__).resolve().parent
TRAIN_DATA_PATH = BASE_DIR / "data" / "student_dropout.csv"
LOG_FILE = BASE_DIR / "logs" / "prediction_inputs.log"


def load_training_data():
    if not TRAIN_DATA_PATH.exists():
        raise FileNotFoundError(f"Training data not found at {TRAIN_DATA_PATH}")
    df = pd.read_csv(TRAIN_DATA_PATH)
    return df


def load_logged_inputs():
    if not LOG_FILE.exists():
        raise FileNotFoundError(f"Log file not found at {LOG_FILE}. Make some predictions first!")

    records = []
    with open(LOG_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entry = json.loads(line)
                records.append(entry.get("input_features", {}))

    return pd.DataFrame(records)


def run_drift_check(drift_threshold_std: float = 1.5):
    print("=" * 75)
    print("CSL236 DATA DRIFT MONITORING REPORT")
    print("=" * 75)

    df_train = load_training_data()
    df_live = load_logged_inputs()

    print(f"Baseline Training Records: {len(df_train)}")
    print(f"Logged Live Predictions:   {len(df_live)}")
    print("-" * 75)
    print(f"{'Feature':<40} {'Train Mean (Std)':<20} {'Live Mean (Std)':<20} {'Status'}")
    print("-" * 75)

    drift_detected = False

    for feature in NUMERICAL_FEATURES:
        if feature in df_train.columns and feature in df_live.columns:
            train_col = df_train[feature].dropna()
            live_col = df_live[feature].dropna()

            if len(live_col) == 0:
                continue

            train_mean = train_col.mean()
            train_std = train_col.std()
            live_mean = live_col.mean()
            live_std = live_col.std() if len(live_col) > 1 else 0.0

            # Z-score of mean shift
            std_ref = train_std if train_std > 1e-6 else 1.0
            shift = abs(live_mean - train_mean) / std_ref

            if shift > drift_threshold_std:
                status = "⚠️ DRIFT DETECTED"
                drift_detected = True
            else:
                status = "✅ Normal"

            train_str = f"{train_mean:.2f} ({train_std:.2f})"
            live_str = f"{live_mean:.2f} ({live_std:.2f})"
            print(f"{feature:<40} {train_str:<20} {live_str:<20} {status}")

    print("=" * 75)
    if drift_detected:
        print("ALERT: One or more features exhibit distribution drift. Consider retraining.")
    else:
        print("SUMMARY: No significant data drift detected. The model is serving within expected bounds.")
    print("=" * 75)


if __name__ == "__main__":
    run_drift_check()
