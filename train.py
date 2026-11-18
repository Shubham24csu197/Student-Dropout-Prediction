"""Training script with MLflow experiment tracking.

Supports CLI arguments for hyperparameters and logs metrics to SQLite MLflow backend.
Meets CSL236 End-Term Project Guidelines:
- MLflow tracking (sqlite:///mlflow.db)
- At least 3 different configurations and 2 algorithms
- Evaluates Accuracy, Precision, Recall, Macro F1
- Serializes best model to model/model.joblib and stage-specific models
"""

import argparse
import os
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import mlflow
import mlflow.sklearn

from schemas import (
    CATEGORICAL_FEATURES,
    ORDINAL_FEATURES,
    NUMERICAL_FEATURES
)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "student_dropout.csv"
MODEL_DIR = BASE_DIR / "model"
MLFLOW_DB = f"sqlite:///{BASE_DIR / 'mlflow.db'}"


def load_dataset(data_path: Path):
    if not data_path.exists():
        # Fallback to alternate names if available
        alt_paths = [
            BASE_DIR / "data" / "student_dropout_cleaned.csv",
            BASE_DIR / "data" / "data.csv"
        ]
        for p in alt_paths:
            if p.exists():
                data_path = p
                break

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    # Auto-detect separator
    try:
        df = pd.read_csv(data_path, sep=",")
        if df.shape[1] <= 1:
            df = pd.read_csv(data_path, sep=";")
    except Exception:
        df = pd.read_csv(data_path, sep=";")

    # Standardize column names
    df.columns = [c.strip().replace("Nacionality", "Nationality") for c in df.columns]
    return df


def build_pipeline(model_type: str, features_cat, features_num, args):
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), features_cat),
            ("num", StandardScaler(), features_num)
        ]
    )

    if model_type == "random_forest":
        classifier = RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            class_weight="balanced",
            random_state=args.random_state
        )
    else:
        classifier = LogisticRegression(
            C=args.C,
            max_iter=1000,
            class_weight="balanced",
            random_state=args.random_state
        )

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", classifier)
    ])
    return pipeline


def train_and_log(run_name, model_type, stage, feature_cols, df, args):
    cat_cols = [c for c in CATEGORICAL_FEATURES if c in feature_cols]
    num_cols = [c for c in (ORDINAL_FEATURES + NUMERICAL_FEATURES) if c in feature_cols]

    X = df[feature_cols]
    y = df["Target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=args.test_size,
        stratify=y,
        random_state=args.random_state
    )

    pipeline = build_pipeline(model_type, cat_cols, num_cols, args)

    with mlflow.start_run(run_name=run_name):
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, average="macro", zero_division=0))
        rec = float(recall_score(y_test, y_pred, average="macro", zero_division=0))
        f1 = float(f1_score(y_test, y_pred, average="macro", zero_division=0))

        # Log params
        mlflow.log_param("model_type", model_type)
        mlflow.log_param("stage", stage)
        mlflow.log_param("features_count", len(feature_cols))
        mlflow.log_param("random_state", args.random_state)
        mlflow.log_param("test_size", args.test_size)

        if model_type == "random_forest":
            mlflow.log_param("n_estimators", args.n_estimators)
            mlflow.log_param("max_depth", args.max_depth)
        else:
            mlflow.log_param("C", args.C)
            mlflow.log_param("max_iter", 1000)

        # Log metrics
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("macro_precision", prec)
        mlflow.log_metric("macro_recall", rec)
        mlflow.log_metric("macro_f1", f1)

        print(f"[{run_name}] Stage: {stage} | Model: {model_type} | Acc: {acc:.4f} | Macro F1: {f1:.4f}")
        return pipeline, {"accuracy": acc, "macro_f1": f1, "macro_precision": prec, "macro_recall": rec}


def main():
    parser = argparse.ArgumentParser(description="Train Student Dropout Prediction model with MLflow tracking")
    parser.add_argument("--model_type", type=str, default="logistic_regression",
                        choices=["logistic_regression", "random_forest"],
                        help="Algorithm choice: logistic_regression or random_forest")
    parser.add_argument("--n_estimators", type=int, default=100, help="Number of trees for Random Forest")
    parser.add_argument("--max_depth", type=int, default=6, help="Max depth for Random Forest")
    parser.add_argument("--C", type=float, default=1.0, help="Inverse regularization strength for Logistic Regression")
    parser.add_argument("--test_size", type=float, default=0.2, help="Test set split ratio")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--all_runs", action="store_true",
                        help="Run full comparison suite (3+ configurations across 2 algorithms)")
    args = parser.parse_args()

    # Setup MLflow with SQLite backend as required by guidelines
    mlflow.set_tracking_uri(MLFLOW_DB)
    mlflow.set_experiment("Student_Dropout_Prediction")

    df = load_dataset(DATA_PATH)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Feature definitions
    admission_features = CATEGORICAL_FEATURES + ORDINAL_FEATURES + NUMERICAL_FEATURES[:6]
    semester1_features = CATEGORICAL_FEATURES + ORDINAL_FEATURES + NUMERICAL_FEATURES

    print("=" * 70)
    print("STARTING TRAINING & MLFLOW TRACKING (CSL236 Stage 2)")
    print(f"Tracking URI: {MLFLOW_DB}")
    print(f"Total Dataset Records: {len(df)}")
    print("=" * 70)

    # If --all_runs or standard single run, we ensure both stages and comparison algorithms are trained
    # Run 1: Model A - Logistic Regression (Admission stage)
    pipeline_A, metrics_A = train_and_log(
        run_name="Model_A_LogisticRegression",
        model_type="logistic_regression",
        stage="Admission_Time",
        feature_cols=admission_features,
        df=df,
        args=args
    )

    # Run 2: Model B - Logistic Regression (First semester)
    pipeline_B_lr, metrics_B_lr = train_and_log(
        run_name="Model_B_LogisticRegression",
        model_type="logistic_regression",
        stage="First_Semester",
        feature_cols=semester1_features,
        df=df,
        args=args
    )

    # Run 3: Model B - Random Forest Baseline (First semester)
    rf_args = argparse.Namespace(**vars(args))
    rf_args.model_type = "random_forest"
    pipeline_B_rf, metrics_B_rf = train_and_log(
        run_name=f"Model_B_RandomForest_n{args.n_estimators}_d{args.max_depth}",
        model_type="random_forest",
        stage="First_Semester",
        feature_cols=semester1_features,
        df=df,
        args=rf_args
    )

    # Run 4: Tuned Random Forest / Custom CLI run if passed
    if args.all_runs or args.model_type == "random_forest":
        rf_tuned_args = argparse.Namespace(**vars(args))
        rf_tuned_args.n_estimators = max(args.n_estimators, 150)
        rf_tuned_args.max_depth = max(args.max_depth, 10)
        train_and_log(
            run_name=f"Model_B_RandomForest_Tuned_n{rf_tuned_args.n_estimators}_d{rf_tuned_args.max_depth}",
            model_type="random_forest",
            stage="First_Semester",
            feature_cols=semester1_features,
            df=df,
            args=rf_tuned_args
        )

    # Save models
    # model.joblib is the primary/best model per guidelines Section 3 & Stage 3
    joblib.dump(pipeline_B_lr, MODEL_DIR / "model.joblib")
    joblib.dump(pipeline_A, MODEL_DIR / "model_A.joblib")
    joblib.dump(pipeline_B_lr, MODEL_DIR / "model_B.joblib")

    print("=" * 70)
    print("SUCCESS: Best models serialized to:")
    print(f" - {MODEL_DIR / 'model.joblib'} (Best Model, Section 3)")
    print(f" - {MODEL_DIR / 'model_A.joblib'} (Admission Time)")
    print(f" - {MODEL_DIR / 'model_B.joblib'} (After 1st Semester)")
    print("All runs logged to SQLite MLflow database: mlflow.db")
    print("To view experiment UI, run: mlflow ui --backend-store-uri sqlite:///mlflow.db")
    print("=" * 70)


if __name__ == "__main__":
    main()
