
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
from typing import Any

_CURRENT_DIR = Path(__file__).resolve().parent
_APP_DIR = _CURRENT_DIR.parent
_PROJECT_ROOT = _APP_DIR.parent

for path in [str(_APP_DIR), str(_PROJECT_ROOT)]:
    if path not in sys.path:
        sys.path.insert(0, path)

import joblib
import pandas as pd
import streamlit as st

try:
    from utils import (
        PROJECT_ROOT,
        get_available_models,
        page_header,
        render_footer,
    )
except ImportError:
    from app.utils import (
        PROJECT_ROOT,
        get_available_models,
        page_header,
        render_footer,
    )

try:
    from src.features.engineering import create_features
except ImportError:
    create_features = None


st.set_page_config(
    page_title="Fraud Risk Assessment - Credit Card Fraud Detection",
    layout="wide",
)

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
        "Model files (logistic_regression.pkl, decision_tree.pkl, "
        "random_forest.pkl, knn.pkl) were not found in models/. "
        "Please train the models first using:\n\n"
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
except (OSError, ValueError, EOFError, ImportError, AttributeError) as exc:
    st.error(f"Error loading model from {model_path.name}: {exc}")
    render_footer()
    st.stop()

st.caption(f"Loaded {selected_model_name} successfully.")
st.markdown("---")

# 2. Transaction Input Selection
st.markdown("### 2. Select or Upload Transaction")

input_tab1, input_tab2 = st.tabs(
    ["Select Sample Transaction", "Upload Custom CSV"]
)

selected_df: pd.DataFrame | None = None
is_sample_demo = False

with input_tab1:
    sample_path = PROJECT_ROOT / "data" / "sample_transactions.csv"

    if sample_path.exists():
        sample_df = pd.read_csv(sample_path)

        if "Transaction_Label" in sample_df.columns:
            labels = sample_df["Transaction_Label"].tolist()
        else:
            labels = [
                f"Transaction #{index + 1}"
                for index in range(len(sample_df))
            ]

        if labels:
            chosen_label = st.selectbox(
                "Choose a sample transaction:",
                labels,
                index=0,
            )
            idx = labels.index(chosen_label)
            selected_df = sample_df.iloc[[idx]].copy()
            is_sample_demo = True

            st.caption(
                "Selected Transaction Data (Amount, Time, and key PCA signals):"
            )
            cols_to_show = [
                "Time",
                "Amount",
                *[
                    f"V{index}"
                    for index in [14, 17, 12, 10]
                    if f"V{index}" in selected_df.columns
                ],
            ]
            st.dataframe(
                selected_df[cols_to_show],
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("The sample transactions file contains no transactions.")
    else:
        st.info(
            "Sample transactions file (data/sample_transactions.csv) not found. "
            "Please upload a CSV."
        )

with input_tab2:
    uploaded_file = st.file_uploader(
        "Upload CSV transaction file:",
        type=["csv"],
        help="CSV must contain Amount, Time, and PCA components V1 to V28.",
    )

    if uploaded_file is not None:
        try:
            uploaded_df = pd.read_csv(uploaded_file)
            required_cols = [
                "Time",
                "Amount",
                *[f"V{index}" for index in range(1, 29)],
            ]
            missing = [
                column
                for column in required_cols
                if column not in uploaded_df.columns
            ]

            if missing:
                st.error(
                    "Validation Error: Uploaded file is missing "
                    f"{len(missing)} required column(s): "
                    f"{', '.join(missing[:5])}..."
                )
            elif uploaded_df[required_cols].isnull().any().any():
                st.error(
                    "Validation Error: Uploaded transaction data contains "
                    "null or empty values."
                )
            elif uploaded_df.empty:
                st.error("Validation Error: The uploaded CSV contains no transactions.")
            else:
                st.success(
                    f"Uploaded file valid: {len(uploaded_df)} "
                    "transaction(s) loaded."
                )
                st.dataframe(
                    uploaded_df.head(5),
                    use_container_width=True,
                )
                selected_df = uploaded_df.copy()
                is_sample_demo = False

        except (pd.errors.ParserError, UnicodeDecodeError, ValueError, OSError) as exc:
            st.error(f"Error reading uploaded CSV: {exc}")

st.markdown("---")


# 3. Feature Preparation Function
def prepare_features(df: pd.DataFrame, model: Any) -> pd.DataFrame:
    """Preprocess transaction columns to match model expectations."""
    clean = df.copy()
    drop_cols = [
        column
        for column in [
            "Class",
            "Transaction_Label",
            "predicted_class",
            "fraud_probability",
        ]
        if column in clean.columns
    ]
    features_only = clean.drop(columns=drop_cols)

    # Apply feature engineering when raw transaction columns are available.
    if create_features is not None:
        raw_v = [f"V{index}" for index in range(1, 29)]
        required_raw = ["Time", "Amount", *raw_v]

        if all(column in features_only.columns for column in required_raw):
            try:
                features_only = create_features(features_only)
            except (ValueError, TypeError, KeyError) as exc:
                raise ValueError(
                    f"Feature engineering failed: {exc}"
                ) from exc

    # Align columns with the model's expected feature names.
    expected = None

    if hasattr(model, "feature_names_in_"):
        expected = list(model.feature_names_in_)
    elif hasattr(model, "named_steps"):
        for step in model.named_steps.values():
            if hasattr(step, "feature_names_in_"):
                expected = list(step.feature_names_in_)
                break

    if expected:
        for column in expected:
            if column not in features_only.columns:
                features_only[column] = 0.0
        features_only = features_only[expected]

    return features_only


# 4. Prediction Execution
st.markdown("### 3. Run Prediction")

if selected_df is None or selected_df.empty:
    st.info("Please select or upload a transaction above to enable prediction.")
    render_footer()
    st.stop()

predict_clicked = st.button(
    "Predict",
    type="primary",
    use_container_width=True,
)

if predict_clicked:
    try:
        X_ready = prepare_features(selected_df, loaded_model)
        predictions = loaded_model.predict(X_ready)

        has_proba = hasattr(loaded_model, "predict_proba")
        probabilities = (
            loaded_model.predict_proba(X_ready)[:, 1]
            if has_proba
            else None
        )

        if len(selected_df) == 1:
            pred = int(predictions[0])
            col_res, col_prob = st.columns([2, 1])

            with col_res:
                if pred == 1:
                    st.error("### Result: Potential Fraud")
                    st.write(
                        "The model identified anomaly patterns consistent "
                        "with fraudulent behavior."
                    )
                else:
                    st.success("### Result: Likely Legitimate")
                    st.write(
                        "The transaction attributes align with normal "
                        "cardholder purchase profiles."
                    )

                if is_sample_demo:
                    st.caption(
                        "Evaluation run on a sample demonstration transaction."
                    )
                else:
                    st.caption(
                        "Evaluation run on custom uploaded transaction data."
                    )

            with col_prob:
                if probabilities is not None:
                    prob_val = float(probabilities[0])
                    st.metric(
                        label="Fraud Probability",
                        value=f"{prob_val * 100:.1f}%",
                    )
                else:
                    st.info(
                        "Probability output not supported by this classifier."
                    )

        else:
            batch_df = selected_df.copy()
            batch_df["Prediction"] = [
                "Potential Fraud" if prediction == 1 else "Likely Legitimate"
                for prediction in predictions
            ]

            if probabilities is not None:
                batch_df["Fraud Probability"] = [
                    f"{probability * 100:.1f}%"
                    for probability in probabilities
                ]

            st.dataframe(batch_df, use_container_width=True)
            n_flagged = int(sum(predictions))
            st.info(
                f"Summary: Flagged {n_flagged} of {len(predictions)} "
                "transactions as potential fraud."
            )

    except (ValueError, TypeError, KeyError, AttributeError, OSError) as exc:
        st.error(f"Prediction failed: {exc}")

render_footer()
