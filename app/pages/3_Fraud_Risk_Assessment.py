"""
Page 3: Fraud Risk Assessment.

Main demonstration tool allowing professors and students to:
  1. Select a sample transaction or upload a transaction CSV.
  2. Select one of the four trained comparison models.
  3. Click 'Predict' to assess the risk.
  4. View classification result (Potential Fraud vs Likely Legitimate).
  5. View probability output if supported by the model.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_CURRENT_DIR = Path(__file__).resolve().parent
_APP_DIR = _CURRENT_DIR.parent
_PROJECT_ROOT = _APP_DIR.parent
for p in [str(_APP_DIR), str(_PROJECT_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import joblib
import numpy as np
import pandas as pd
import streamlit as st

try:
    from utils import (
        COLORS,
        PROJECT_ROOT,
        get_available_models,
        page_header,
        render_footer,
    )
except ImportError:
    from app.utils import (
        COLORS,
        PROJECT_ROOT,
        get_available_models,
        page_header,
        render_footer,
    )

try:
    from src.features.engineering import create_features
except ImportError:
    create_features = None

try:
    st.set_page_config(
        page_title="Fraud Risk Assessment - Credit Card Fraud Detection",
        layout="wide",
    )
except Exception:
    pass

page_header(
    "Fraud Risk Assessment",
    "Interactive transaction risk scoring using trained machine learning models",
)

# 1. Model Selection
st.markdown("### 1. Select Model")
available_models = get_available_models()

if not available_models:
    st.error(
        "No Trained Models Found\n\n"
        "Model files (logistic_regression.pkl, decision_tree.pkl, random_forest.pkl, knn.pkl) "
        "were not found in models/. Please train the models first using:\n\n"
        "```powershell\n"
        "python src/models/compare_models.py\n"
        "```"
    )
    render_footer()
    st.stop()

selected_model_name = st.selectbox(
    "Choose evaluation algorithm:",
    list(available_models.keys()),
    index=0,
)
model_path = available_models[selected_model_name]

try:
    loaded_model = joblib.load(model_path)
except Exception as e:
    st.error(f"Error loading model from {model_path.name}: {e}")
    render_footer()
    st.stop()

st.caption(f"Loaded {selected_model_name} successfully.")
st.markdown("---")

# 2. Transaction Input Selection
st.markdown("### 2. Select or Upload Transaction")

input_tab1, input_tab2 = st.tabs(["Select Sample Transaction", "Upload Custom CSV"])

selected_df: Optional[pd.DataFrame] = None
is_sample_demo = False

with input_tab1:
    sample_path = PROJECT_ROOT / "data" / "sample_transactions.csv"
    if sample_path.exists():
        sample_df = pd.read_csv(sample_path)
        labels = sample_df["Transaction_Label"].tolist() if "Transaction_Label" in sample_df.columns else [
            f"Transaction #{i+1}" for i in range(len(sample_df))
        ]
        chosen_label = st.selectbox("Choose a sample transaction:", labels, index=0)
        idx = labels.index(chosen_label)
        selected_df = sample_df.iloc[[idx]].copy()
        is_sample_demo = True

        st.caption("Selected Transaction Data (Amount, Time, and key PCA signals):")
        cols_to_show = ["Time", "Amount"] + [f"V{i}" for i in [14, 17, 12, 10] if f"V{i}" in selected_df.columns]
        st.dataframe(selected_df[cols_to_show], use_container_width=True, hide_index=True)
    else:
        st.info("Sample transactions file (data/sample_transactions.csv) not found. Please upload a CSV.")

with input_tab2:
    uploaded_file = st.file_uploader(
        "Upload CSV transaction file:",
        type=["csv"],
        help="CSV must contain Amount, Time, and PCA components V1 to V28.",
    )
    if uploaded_file is not None:
        try:
            uploaded_df = pd.read_csv(uploaded_file)
            required_cols = ["Time", "Amount"] + [f"V{i}" for i in range(1, 29)]
            missing = [c for c in required_cols if c not in uploaded_df.columns]

            if missing:
                st.error(f"Validation Error: Uploaded file is missing {len(missing)} required column(s): {', '.join(missing[:5])}...")
            elif uploaded_df[required_cols].isnull().any().any():
                st.error("Validation Error: Uploaded transaction data contains null or empty values.")
            else:
                st.success(f"Uploaded file valid: {len(uploaded_df)} transaction(s) loaded.")
                st.dataframe(uploaded_df.head(5), use_container_width=True)
                selected_df = uploaded_df.copy()
                is_sample_demo = False
        except Exception as e:
            st.error(f"Error reading uploaded CSV: {e}")

st.markdown("---")

# 3. Feature Preparation Function (Error A Fixed)
def prepare_features(df: pd.DataFrame, model: Any) -> pd.DataFrame:
    """Preprocess transaction columns to match model expectations."""
    clean = df.copy()
    drop_cols = [c for c in ["Class", "Transaction_Label", "predicted_class", "fraud_probability"] if c in clean.columns]
    features_only = clean.drop(columns=drop_cols)

    # Apply feature engineering if raw V columns and Time/Amount exist
    if create_features is not None:
        raw_v = [f"V{i}" for i in range(1, 29)]
        if all(c in features_only.columns for c in ["Time", "Amount"] + raw_v):
            try:
                features_only = create_features(features_only)
            except Exception:
                pass

    # Align columns with model's expected feature names
    expected = None
    if hasattr(model, "feature_names_in_"):
        expected = list(model.feature_names_in_)
    elif hasattr(model, "named_steps"):
        for step in model.named_steps.values():
            if hasattr(step, "feature_names_in_"):
                expected = list(step.feature_names_in_)
                break

    if expected:
        for c in expected:
            if c not in features_only.columns:
                features_only[c] = 0.0
        features_only = features_only[expected]

    return features_only

# 4. Prediction Execution
st.markdown("### 3. Run Prediction")

if selected_df is None or selected_df.empty:
    st.info("Please select or upload a transaction above to enable prediction.")
    render_footer()
    st.stop()

predict_clicked = st.button("Predict", type="primary", use_container_width=True)

if predict_clicked:
    try:
        X_ready = prepare_features(selected_df, loaded_model)
        predictions = loaded_model.predict(X_ready)

        # Check genuine support for predict_proba
        has_proba = hasattr(loaded_model, "predict_proba")
        probabilities = loaded_model.predict_proba(X_ready)[:, 1] if has_proba else None

        if len(selected_df) == 1:
            pred = int(predictions[0])
            col_res, col_prob = st.columns([2, 1])

            with col_res:
                if pred == 1:
                    st.error("### Result: Potential Fraud")
                    st.write("The model identified anomaly patterns consistent with fraudulent behavior.")
                else:
                    st.success("### Result: Likely Legitimate")
                    st.write("The transaction attributes align with normal cardholder purchase profiles.")

                if is_sample_demo:
                    st.caption("Evaluation run on a sample demonstration transaction.")
                else:
                    st.caption("Evaluation run on custom uploaded transaction data.")

            with col_prob:
                if probabilities is not None:
                    prob_val = float(probabilities[0])
                    st.metric(
                        label="Fraud Probability",
                        value=f"{prob_val * 100:.1f}%",
                    )
                else:
                    st.info("Probability output not supported by this classifier.")

        else:
            # Batch CSV results
            batch_df = selected_df.copy()
            batch_df["Prediction"] = ["Potential Fraud" if p == 1 else "Likely Legitimate" for p in predictions]
            if probabilities is not None:
                batch_df["Fraud Probability"] = [f"{p*100:.1f}%" for p in probabilities]
            st.dataframe(batch_df, use_container_width=True)
            n_flagged = sum(predictions)
            st.info(f"Summary: Flagged {n_flagged} of {len(predictions)} transactions as potential fraud.")

    except Exception as e:
        st.error(f"Prediction failed: {e}")

render_footer()
