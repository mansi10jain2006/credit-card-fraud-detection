
"""
Model training runner with optional MLflow experiment tracking.

Trains the project's four evaluation models:
  1. Logistic Regression
  2. Decision Tree
  3. Random Forest
  4. K-Nearest Neighbors (KNN)

If an MLflow server is active at http://127.0.0.1:5000, experiment runs are logged.
Otherwise, training runs locally and saves all models to the models/ directory.
"""

import time
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

try:
    import mlflow
    import mlflow.sklearn

    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT_NAME = "credit_card_fraud_detection"
TARGET_COLUMN = "Class"
RANDOM_STATE = 42
TEST_SIZE = 0.2


def load_and_split():
    """Load genuine dataset and produce a stratified 80/20 train/test split."""
    raw_path = Path("data") / "creditcard.csv"
    cleaned_path = Path("data") / "cleaned.csv"
    features_path = Path("data") / "features.csv"

    if features_path.exists():
        df = pd.read_csv(features_path)
    elif cleaned_path.exists():
        from src.features.engineering import create_features

        df_clean = pd.read_csv(cleaned_path)
        df = create_features(df_clean)
        df.to_csv(features_path, index=False)
    elif raw_path.exists():
        from src.data.cleaner import clean_data
        from src.features.engineering import create_features

        df_raw = pd.read_csv(raw_path)
        df_clean, _ = clean_data(df_raw)
        df_clean.to_csv(cleaned_path, index=False)
        df = create_features(df_clean)
        df.to_csv(features_path, index=False)
    else:
        raise FileNotFoundError(
            "Genuine dataset not found!\n"
            "Please download 'creditcard.csv' from Kaggle:\n"
            "  https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud\n"
            "and place it at:\n"
            "  data/creditcard.csv\n"
            "Then run this script."
        )

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )


def compute_metrics(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    prefix: str,
) -> dict[str, Any]:
    """Compute standard classification metrics."""
    y_pred = model.predict(X)

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X)[:, 1]
    else:
        y_proba = y_pred.astype(float)

    cm = confusion_matrix(y, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    return {
        f"{prefix}_accuracy": float(accuracy_score(y, y_pred)),
        f"{prefix}_precision": float(
            precision_score(y, y_pred, zero_division=0)
        ),
        f"{prefix}_recall": float(recall_score(y, y_pred, zero_division=0)),
        f"{prefix}_f1": float(f1_score(y, y_pred, zero_division=0)),
        f"{prefix}_roc_auc": float(roc_auc_score(y, y_proba)),
        f"{prefix}_pr_auc": float(average_precision_score(y, y_proba)),
        f"{prefix}_tp": int(tp),
        f"{prefix}_fp": int(fp),
        f"{prefix}_fn": int(fn),
        f"{prefix}_tn": int(tn),
    }


def main():
    print("=" * 60)
    print("RUNNING MODEL TRAINING PIPELINE")
    print("=" * 60)

    try:
        X_train, X_test, y_train, y_test = load_and_split()
    except FileNotFoundError as error:
        print(f"\n[Notice] {error}")
        return

    models = {
        "Logistic Regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=1000,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "Decision Tree": DecisionTreeClassifier(
            class_weight="balanced",
            max_depth=8,
            min_samples_leaf=5,
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            class_weight="balanced",
            max_depth=12,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "KNN": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    KNeighborsClassifier(
                        n_neighbors=5,
                        weights="distance",
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }

    use_mlflow = False

    if MLFLOW_AVAILABLE:
        try:
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            mlflow.set_experiment(EXPERIMENT_NAME)
            use_mlflow = True
            print(f"Connected to MLflow at {MLFLOW_TRACKING_URI}")
        except (OSError, ConnectionError, TimeoutError, mlflow.exceptions.MlflowException):
            print("MLflow server not active; proceeding with local training.")

    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)

    best_pr_auc = -1.0
    best_model = None
    best_name = ""

    for name, model in models.items():
        print(f"\nTraining {name}...")
        start_time = time.time()
        model.fit(X_train, y_train)
        elapsed = time.time() - start_time

        test_metrics = compute_metrics(model, X_test, y_test, prefix="test")
        pr_auc = test_metrics["test_pr_auc"]

        print(
            f"  Completed in {elapsed:.2f}s | "
            f"Test PR-AUC: {pr_auc:.4f} | "
            f"Recall: {test_metrics['test_recall']:.4f}"
        )

        filename = name.lower().replace(" ", "_") + ".pkl"
        joblib.dump(model, models_dir / filename)

        if pr_auc > best_pr_auc:
            best_pr_auc = pr_auc
            best_model = model
            best_name = name

        if use_mlflow:
            try:
                with mlflow.start_run(run_name=name):
                    mlflow.log_metric("train_time", elapsed)

                    for key, value in test_metrics.items():
                        mlflow.log_metric(key, value)

                    mlflow.sklearn.log_model(model, artifact_path="model")
            except (
                OSError,
                ConnectionError,
                TimeoutError,
                mlflow.exceptions.MlflowException,
            ) as error:
                print(f"  Note: MLflow logging skipped ({error})")

    if best_model is not None:
        print(f"\nBest model: {best_name} (PR-AUC: {best_pr_auc:.4f})")


if __name__ == "__main__":
    main()
