"""
Synthetic demo data generator for interface demonstration.

Used ONLY when real pipeline artifacts have not yet been produced.
All generated data is explicitly identified as synthetic placeholders.
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd

_CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _CURRENT_DIR.parent
if str(_CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(_CURRENT_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

TARGET_COLUMN = "Class"
RANDOM_STATE = 42

# Feature names aligning with the raw dataset and feature engineering
ORIGINAL_FEATURES = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
ENGINEERED_FEATURES = [
    "hour_of_day",
    "is_night",
    "log_amount",
    "amount_bucket",
    "v_magnitude",
    "v_negative_count",
    "top_fraud_v_mean",
    "top_fraud_v_min",
    "amount_v17_interaction",
    "night_amount_ratio",
    "fraud_score_proxy",
    "v14_v17_interaction",
]


def is_demo_mode() -> bool:
    """Return True if any essential data artifacts are missing."""
    required_files = [
        PROJECT_ROOT / "data" / "features.csv",
        PROJECT_ROOT / "data" / "model_results.json",
    ]
    return any(not f.exists() for f in required_files)


def generate_demo_features(n_rows: int = 10_000, fraud_rate: float = 0.0017) -> pd.DataFrame:
    """
    Build a synthetic feature dataset that mirrors the structure of the dataset.
    Explicitly labeled as synthetic demonstration data.
    """
    rng = np.random.default_rng(RANDOM_STATE)
    n_fraud = max(1, int(n_rows * fraud_rate))
    n_legit = n_rows - n_fraud

    # Legit transactions: PCA components roughly standard normal
    legit = pd.DataFrame({
        "Time": rng.uniform(0, 172000, n_legit),
        "Amount": rng.exponential(80, n_legit).clip(0, 5000),
    })
    for i in range(1, 29):
        legit[f"V{i}"] = rng.normal(0, 1, n_legit)

    # Fraud transactions: extreme negative PCA signals on top features
    fraud = pd.DataFrame({
        "Time": rng.uniform(0, 172000, n_fraud),
        "Amount": rng.exponential(120, n_fraud).clip(0, 5000),
    })
    for i in range(1, 29):
        if i in [10, 12, 14, 17]:
            fraud[f"V{i}"] = rng.normal(-5, 2, n_fraud)
        else:
            fraud[f"V{i}"] = rng.normal(0, 1.5, n_fraud)

    df = pd.concat([legit, fraud], ignore_index=True)
    df["Class"] = [0] * n_legit + [1] * n_fraud

    # Add engineered features
    df["hour_of_day"] = (df["Time"] // 3600) % 24
    df["is_night"] = ((df["hour_of_day"] >= 0) & (df["hour_of_day"] < 6)).astype(int)
    df["log_amount"] = np.log1p(df["Amount"])
    df["amount_bucket"] = pd.cut(
        df["Amount"],
        bins=[-0.01, 1, 10, 100, 1000, np.inf],
        labels=[0, 1, 2, 3, 4],
    ).astype(int)

    v_cols = [f"V{i}" for i in range(1, 29)]
    df["v_magnitude"] = np.sqrt((df[v_cols] ** 2).sum(axis=1))
    df["v_negative_count"] = (df[v_cols] < 0).sum(axis=1)

    top_v = ["V17", "V14", "V12", "V10"]
    df["top_fraud_v_mean"] = df[top_v].mean(axis=1)
    df["top_fraud_v_min"] = df[top_v].min(axis=1)

    df["amount_v17_interaction"] = df["log_amount"] * df["V17"]
    df["night_amount_ratio"] = df["is_night"] * df["log_amount"]
    df["fraud_score_proxy"] = -df["top_fraud_v_mean"] * df["v_magnitude"]
    df["v14_v17_interaction"] = df["V14"] * df["V17"]

    # Shuffle rows deterministically
    return df.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)


def generate_demo_predictions(features_df: pd.DataFrame) -> pd.DataFrame:
    """Generate demonstration predictions based on synthetic fraud scoring."""
    df = features_df.copy()
    raw_signal = (
        -1.5 * df["top_fraud_v_mean"]
        + 0.5 * df["fraud_score_proxy"]
        + 0.3 * df["log_amount"]
        - 2.0
    )
    # Sigmoid function for probability
    proba = 1.0 / (1.0 + np.exp(-raw_signal.clip(-10, 10)))
    df["fraud_probability"] = proba
    df["predicted_class"] = (proba >= 0.5).astype(int)
    df["true_class"] = df["Class"]
    return df


def generate_demo_results() -> dict:
    """
    Sample placeholder metrics for the four college models.
    Clearly flagged with is_demo=True to prevent misrepresenting as real results.
    """
    feature_importance = [
        {"feature": "V14", "importance": 0.22},
        {"feature": "V17", "importance": 0.18},
        {"feature": "V12", "importance": 0.14},
        {"feature": "V10", "importance": 0.11},
        {"feature": "top_fraud_v_min", "importance": 0.08},
        {"feature": "fraud_score_proxy", "importance": 0.06},
        {"feature": "v14_v17_interaction", "importance": 0.05},
        {"feature": "v_magnitude", "importance": 0.04},
        {"feature": "log_amount", "importance": 0.03},
        {"feature": "hour_of_day", "importance": 0.02},
    ]

    return {
        "is_demo": True,
        "note": "These are illustrative demonstration figures for UI layout preview only.",
        "logistic_regression": {
            "name": "Logistic Regression",
            "pr_auc": 0.673,
            "roc_auc": 0.959,
            "precision": 0.055,
            "recall": 0.863,
            "f1": 0.104,
            "accuracy": 0.975,
            "tn": 55254,
            "fp": 1397,
            "fn": 13,
            "tp": 82,
            "train_time": 0.85,
        },
        "decision_tree": {
            "name": "Decision Tree",
            "pr_auc": 0.741,
            "roc_auc": 0.912,
            "precision": 0.760,
            "recall": 0.747,
            "f1": 0.753,
            "accuracy": 0.998,
            "tn": 56628,
            "fp": 23,
            "fn": 24,
            "tp": 71,
            "train_time": 4.12,
        },
        "random_forest": {
            "name": "Random Forest",
            "pr_auc": 0.815,
            "roc_auc": 0.972,
            "precision": 0.946,
            "recall": 0.737,
            "f1": 0.829,
            "accuracy": 0.999,
            "tn": 56647,
            "fp": 4,
            "fn": 25,
            "tp": 70,
            "train_time": 28.40,
        },
        "knn": {
            "name": "K-Nearest Neighbors (KNN)",
            "pr_auc": 0.728,
            "roc_auc": 0.934,
            "precision": 0.812,
            "recall": 0.684,
            "f1": 0.743,
            "accuracy": 0.998,
            "tn": 56635,
            "fp": 16,
            "fn": 30,
            "tp": 65,
            "train_time": 1.25,
        },
        "feature_importance": feature_importance,
    }
