
"""
Export model artifacts for application display:
  1. data/predictions.csv    — Test-set predictions and probabilities
  2. data/model_results.json — Comparative metrics across the 4 college models

Run from the project root with: python src/models/export_for_app.py
"""

import json
from pathlib import Path

import joblib
import pandas as pd
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

TARGET_COLUMN = "Class"
RANDOM_STATE = 42
TEST_SIZE = 0.2


def main():
    features_path = Path("data") / "features.csv"

    if not features_path.exists():
        print(
            f"[Notice] Features file not found at {features_path}. "
            "Run the pipeline first."
        )
        return

    # Load data and reproduce the stratified test split.
    df = pd.read_csv(features_path)
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    models_dir = Path("models")
    data_dir = Path("data")
    data_dir.mkdir(parents=True, exist_ok=True)

    # 1. Save predictions from the production model if available.
    prod_path = models_dir / "production_model.pkl"

    if prod_path.exists():
        production_model = joblib.load(prod_path)
        y_pred = production_model.predict(X_test)

        if hasattr(production_model, "predict_proba"):
            y_proba = production_model.predict_proba(X_test)[:, 1]
        else:
            y_proba = y_pred.astype(float)

        predictions = X_test.copy()
        predictions["true_class"] = y_test.to_numpy()
        predictions["predicted_class"] = y_pred
        predictions["fraud_probability"] = y_proba

        predictions.to_csv(data_dir / "predictions.csv", index=False)
        print(f"Saved data/predictions.csv ({len(predictions):,} rows)")
    else:
        print(
            "[Notice] Production model not found. "
            "Skipping data/predictions.csv export."
        )

    # 2. Recompute metrics for the four comparison models.
    model_files = {
        "logistic_regression": (
            "Logistic Regression",
            models_dir / "logistic_regression.pkl",
        ),
        "decision_tree": (
            "Decision Tree",
            models_dir / "decision_tree.pkl",
        ),
        "random_forest": (
            "Random Forest",
            models_dir / "random_forest.pkl",
        ),
        "knn": (
            "K-Nearest Neighbors (KNN)",
            models_dir / "knn.pkl",
        ),
    }

    results = {}

    for key, (name, path) in model_files.items():
        if not path.exists():
            print(f"[Notice] Model not found: {path}. Skipping.")
            continue

        model = joblib.load(path)
        y_pred_m = model.predict(X_test)

        if hasattr(model, "predict_proba"):
            y_proba_m = model.predict_proba(X_test)[:, 1]
        else:
            y_proba_m = y_pred_m.astype(float)

        cm = confusion_matrix(
            y_test,
            y_pred_m,
            labels=[0, 1],
        ).ravel()

        results[key] = {
            "name": name,
            "pr_auc": float(average_precision_score(y_test, y_proba_m)),
            "roc_auc": float(roc_auc_score(y_test, y_proba_m)),
            "precision": float(
                precision_score(y_test, y_pred_m, zero_division=0)
            ),
            "recall": float(
                recall_score(y_test, y_pred_m, zero_division=0)
            ),
            "f1": float(f1_score(y_test, y_pred_m, zero_division=0)),
            "accuracy": float(accuracy_score(y_test, y_pred_m)),
            "tn": int(cm[0]),
            "fp": int(cm[1]),
            "fn": int(cm[2]),
            "tp": int(cm[3]),
        }

    # Extract feature importance from the random forest, if available.
    rf_path = models_dir / "random_forest.pkl"

    if rf_path.exists():
        rf = joblib.load(rf_path)

        if hasattr(rf, "feature_importances_"):
            fi_df = pd.DataFrame(
                {
                    "feature": list(X_test.columns),
                    "importance": rf.feature_importances_,
                }
            ).sort_values("importance", ascending=False)

            results["feature_importance"] = (
                fi_df.head(15).to_dict(orient="records")
            )

    # Save the comparative metrics.
    results_path = data_dir / "model_results.json"

    with results_path.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    print(f"Saved {results_path}")


if __name__ == "__main__":
    main()
