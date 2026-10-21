"""
Hyperparameter tuning for Random Forest on credit card fraud detection.

Uses Optuna with TPE (Tree-structured Parzen Estimator) for Bayesian search.
Optimizes for PR-AUC via 5-fold stratified cross-validation on the training set.

Run from the project root with: python src/models/tuning.py
"""

import json
import time
from pathlib import Path

import joblib
import optuna
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

TARGET_COLUMN = "Class"
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
N_TRIALS = 20

# Suppress verbose Optuna logging output
optuna.logging.set_verbosity(optuna.logging.WARNING)


def load_and_split(path: Path = Path("data") / "features.csv"):
    """Load features and produce a stratified 80/20 train/test split."""
    if not path.exists():
        raise FileNotFoundError(
            f"Features file not found at {path}. Run data cleaning and feature engineering first."
        )
    df = pd.read_csv(path)
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    return train_test_split(
        X, y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )


def objective(trial, X_train, y_train):
    """Optuna objective function: proposes hyperparameters and evaluates via CV."""
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 150, step=25),
        "max_depth": trial.suggest_int("max_depth", 6, 16),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
        "class_weight": "balanced",
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
    }

    model = RandomForestClassifier(**params)
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="average_precision", n_jobs=-1)
    return scores.mean()


def train_final_model(best_params, X_train, y_train):
    """Retrain Random Forest with the best hyperparameters on the full training set."""
    model = RandomForestClassifier(
        **best_params,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def evaluate_on_test(model, X_test, y_test):
    """Evaluate final model on held-out test data."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    return {
        "pr_auc": float(average_precision_score(y_test, y_proba)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


def main():
    print("=" * 60)
    print("HYPERPARAMETER TUNING (Random Forest via Optuna)")
    print("=" * 60)

    try:
        X_train, X_test, y_train, y_test = load_and_split()
    except FileNotFoundError as e:
        print(f"\n[Notice] {e}")
        return

    print(f"Train: {X_train.shape[0]:,} rows ({y_train.sum()} fraud)")
    print(f"Test:  {X_test.shape[0]:,} rows ({y_test.sum()} fraud)\n")

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    )

    print(f"Running {N_TRIALS} Optuna trials with {CV_FOLDS}-fold stratified CV...")
    t0 = time.time()
    study.optimize(lambda trial: objective(trial, X_train, y_train), n_trials=N_TRIALS)
    elapsed = time.time() - t0

    print(f"\nTuning completed in {elapsed:.1f}s")
    print(f"Best CV PR-AUC: {study.best_value:.4f}")
    print("Best hyperparameters:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    # Retrain and evaluate
    model = train_final_model(study.best_params, X_train, y_train)
    metrics = evaluate_on_test(model, X_test, y_test)
    print(f"\nTest PR-AUC:   {metrics['pr_auc']:.4f}")
    print(f"Test Precision: {metrics['precision']:.4f}")
    print(f"Test Recall:    {metrics['recall']:.4f}")

    # Save params and tuned model
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    with open(models_dir / "best_params.json", "w") as f:
        json.dump(study.best_params, f, indent=2)
    joblib.dump(model, models_dir / "random_forest.pkl")
    print("\nSaved models/best_params.json and updated models/random_forest.pkl")


if __name__ == "__main__":
    main()
