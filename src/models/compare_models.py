"""
Model comparison pipeline for credit card fraud detection.

Evaluates four machine learning models:
  1. Logistic Regression (Linear baseline with feature scaling & class weighting)
  2. Decision Tree (Interpretable tree-based model with class weighting)
  3. Random Forest (Ensemble of bagged decision trees with class weighting)
  4. K-Nearest Neighbors (Distance-based instance classifier with feature scaling)

Methodology:
  - Stratified 80/20 train/test split to preserve rare fraud class ratio.
  - StandardScaler inside Scikit-Learn Pipelines fitted ONLY on training data.
  - Class imbalance handled through balanced class weights and balanced training sampling.
  - Evaluated on precision, recall, F1, PR-AUC, ROC-AUC, and confusion matrix.
  - Results exported dynamically to data/model_results.json.
"""

import json
import time
from pathlib import Path
from typing import Dict, Any, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

TARGET_COLUMN = "Class"
RANDOM_STATE = 42
TEST_SIZE = 0.2


def load_dataset() -> pd.DataFrame:
    """
    Load genuine dataset for model training and comparison.
    Looks for data/features.csv, data/cleaned.csv, or data/creditcard.csv.
    Stops clearly with instructions if genuine dataset is missing.
    """
    raw_path = Path("data") / "creditcard.csv"
    cleaned_path = Path("data") / "cleaned.csv"
    features_path = Path("data") / "features.csv"

    if features_path.exists():
        print(f"Loading feature-engineered dataset from {features_path}...")
        return pd.read_csv(features_path)

    if cleaned_path.exists():
        print(f"Found cleaned data at {cleaned_path}. Running feature engineering...")
        from src.features.engineering import create_features
        df_clean = pd.read_csv(cleaned_path)
        df_feat = create_features(df_clean)
        df_feat.to_csv(features_path, index=False)
        return df_feat

    if raw_path.exists():
        print(f"Found raw creditcard.csv at {raw_path}. Running cleaning and feature engineering...")
        from src.data.cleaner import clean_data
        from src.features.engineering import create_features

        df_raw = pd.read_csv(raw_path)
        df_clean, _ = clean_data(df_raw)
        df_clean.to_csv(cleaned_path, index=False)

        df_feat = create_features(df_clean)
        df_feat.to_csv(features_path, index=False)
        return df_feat

    raise FileNotFoundError(
        "Genuine dataset not found!\n"
        "Please download 'creditcard.csv' from Kaggle:\n"
        "  https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud\n"
        "and place it at:\n"
        "  data/creditcard.csv\n"
        "Then re-run this script to train genuine models."
    )


def split_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Perform a stratified 80/20 train/test split."""
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    return X_train, X_test, y_train, y_test


def build_models() -> Dict[str, Any]:
    """
    Construct the four candidate models.
    Pipelines ensure scaling is fitted strictly on training data without leakage.
    """
    return {
        "logistic_regression": {
            "name": "Logistic Regression",
            "model": Pipeline([
                ("scaler", StandardScaler()),
                ("classifier", LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=RANDOM_STATE,
                )),
            ]),
            "needs_resampled_knn": False,
        },
        "decision_tree": {
            "name": "Decision Tree",
            "model": DecisionTreeClassifier(
                class_weight="balanced",
                max_depth=8,
                min_samples_leaf=5,
                random_state=RANDOM_STATE,
            ),
            "needs_resampled_knn": False,
        },
        "random_forest": {
            "name": "Random Forest",
            "model": RandomForestClassifier(
                n_estimators=100,
                class_weight="balanced",
                max_depth=12,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            "needs_resampled_knn": False,
        },
        "knn": {
            "name": "K-Nearest Neighbors (KNN)",
            "model": Pipeline([
                ("scaler", StandardScaler()),
                ("classifier", KNeighborsClassifier(
                    n_neighbors=5,
                    weights="distance",
                    n_jobs=-1,
                )),
            ]),
            # KNN distance calculations on 200,000+ points cause memory and time exhaustion.
            # Subsampling training negatives creates an effective balanced decision boundary.
            "needs_resampled_knn": True,
        },
    }


def balance_training_for_knn(
    X_train: pd.DataFrame, y_train: pd.Series, max_samples: int = 10000
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Subsample majority class strictly within training data for KNN.
    Keeps all training fraud cases and samples legitimate cases to prevent O(N^2) memory freeze.
    """
    fraud_indices = y_train[y_train == 1].index
    legit_indices = y_train[y_train == 0].index

    n_fraud = len(fraud_indices)
    n_legit_sample = min(len(legit_indices), max(n_fraud * 10, max_samples - n_fraud))

    rng = np.random.default_rng(RANDOM_STATE)
    sampled_legit = rng.choice(legit_indices, size=n_legit_sample, replace=False)

    selected_indices = np.concatenate([fraud_indices, sampled_legit])
    rng.shuffle(selected_indices)

    return X_train.loc[selected_indices], y_train.loc[selected_indices]


def evaluate_model(model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
    """Evaluate predictions and probabilities on held-out test data."""
    y_pred = model.predict(X_test)

    # Some models or pipelines support predict_proba
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        df_scores = model.decision_function(X_test)
        y_proba = (df_scores - df_scores.min()) / (df_scores.max() - df_scores.min() + 1e-9)
    else:
        y_proba = y_pred.astype(float)

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "pr_auc": float(average_precision_score(y_test, y_proba)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "y_pred": y_pred,
        "y_proba": y_proba,
    }


def main():
    print("=" * 70)
    print("STARTING MODEL COMPARISON EVALUATION")
    print("=" * 70)

    try:
        df = load_dataset()
    except FileNotFoundError as e:
        print(f"\n[Error] {e}")
        return

    print(f"Dataset shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    X_train, X_test, y_train, y_test = split_data(df)
    print(f"Train split: {len(X_train):,} rows ({y_train.sum()} fraud)")
    print(f"Test split:  {len(X_test):,} rows ({y_test.sum()} fraud)\n")

    models_config = build_models()
    results = {}
    best_pr_auc = -1.0
    best_model_name = None
    best_model_obj = None

    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path("data")
    data_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'Model':<28}{'PR-AUC':>10}{'Precision':>12}{'Recall':>10}{'F1':>10}{'Time':>8}")
    print("-" * 78)

    for key, config in models_config.items():
        name = config["name"]
        model = config["model"]

        # Training
        t0 = time.time()
        if config["needs_resampled_knn"]:
            X_tr, y_tr = balance_training_for_knn(X_train, y_train)
            model.fit(X_tr, y_tr)
        else:
            model.fit(X_train, y_train)
        train_time = time.time() - t0

        # Evaluation on test set
        metrics = evaluate_model(model, X_test, y_test)
        y_pred = metrics.pop("y_pred")
        y_proba = metrics.pop("y_proba")

        metrics["name"] = name
        metrics["train_time"] = round(train_time, 2)
        results[key] = metrics

        # Save individual model artifact
        model_path = models_dir / f"{key}.pkl"
        joblib.dump(model, model_path)

        print(
            f"{name:<28}"
            f"{metrics['pr_auc']:>10.4f}"
            f"{metrics['precision']:>12.4f}"
            f"{metrics['recall']:>10.4f}"
            f"{metrics['f1']:>10.4f}"
            f"{train_time:>7.1f}s"
        )

        # Track best model dynamically based on PR-AUC
        if metrics["pr_auc"] > best_pr_auc:
            best_pr_auc = metrics["pr_auc"]
            best_model_name = name
            best_model_obj = model
            best_predictions = (y_pred, y_proba)

    # Export test predictions from the best model
    if best_model_obj is not None:
        pred_df = X_test.copy()
        pred_df["true_class"] = y_test.values
        pred_df["predicted_class"] = best_predictions[0]
        pred_df["fraud_probability"] = best_predictions[1]
        pred_df.to_csv(data_dir / "predictions.csv", index=False)
        print(f"\nExported test predictions from {best_model_name} (PR-AUC: {best_pr_auc:.4f}) to {data_dir / 'predictions.csv'}")

    # Save exact feature names and order
    feature_cols = list(X_train.columns)
    feature_cols_path = data_dir / "feature_columns.json"
    with open(feature_cols_path, "w", encoding="utf-8") as f:
        json.dump(feature_cols, f, indent=2)
    print(f"Saved feature schema ({len(feature_cols)} columns) to {feature_cols_path}")

    # Extract feature importance if available (e.g. from Random Forest or Decision Tree)
    rf_model = models_config["random_forest"]["model"]
    if hasattr(rf_model, "feature_importances_"):
        fi_df = pd.DataFrame({
            "feature": list(X_test.columns),
            "importance": rf_model.feature_importances_,
        }).sort_values("importance", ascending=False)
        results["feature_importance"] = fi_df.head(15).to_dict(orient="records")

    results["is_demo"] = False
    results["dataset"] = "Kaggle creditcard.csv (Genuine Evaluation)"
    results["best_model"] = best_model_name
    results["best_pr_auc"] = best_pr_auc

    # Save evaluated metrics to model_results.json
    results_path = data_dir / "model_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Exported evaluated metrics to {results_path}")
    print("\nMODEL COMPARISON COMPLETE.")


if __name__ == "__main__":
    main()
