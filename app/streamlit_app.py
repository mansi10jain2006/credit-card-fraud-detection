
"""
Credit Card Fraud Detection - Machine Learning Mini-Project.

Main application entry point. Provides four simple views:
  1. Overview: Problem, dataset status, models, and one-line workflow.
  2. Explore Dataset: Summary metrics, class distribution chart, and sample preview.
  3. Model Comparison: Side-by-side evaluation of LR, Decision Tree, Random Forest, and KNN.
  4. Fraud Risk Assessment: Live interactive transaction prediction tool.
"""

import sys
from pathlib import Path

# Add project root and app directory to path
APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent

if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import runpy

import streamlit as st

try:
    from utils import (
        RAW_DATA_PATH,
        is_real_dataset_available,
        page_header,
        render_footer,
    )
except ImportError:
    from app.utils import (
        RAW_DATA_PATH,
        is_real_dataset_available,
        page_header,
        render_footer,
    )


# ---------- Page Configuration ----------
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------- Simple Sidebar Navigation ----------
with st.sidebar:
    st.markdown("### Fraud Detection")
    st.caption("College ML Mini-Project")
    st.markdown("---")

    selected_page = st.radio(
        "Navigation",
        [
            "Overview",
            "Explore Dataset",
            "Model Comparison",
            "Fraud Risk Assessment",
        ],
        index=0,
    )

    st.markdown("---")

    # Clean, simple dataset indicator
    if is_real_dataset_available():
        st.success("Status: Real Dataset Available")
    else:
        st.info("Status: Demonstration Mode (Synthetic Data)")


# ==============================================================================
# VIEW 1: OVERVIEW
# ==============================================================================
def render_overview():
    page_header(
        "Credit Card Fraud Detection",
        "Machine Learning Mini-Project",
    )

    # 1. Short problem explanation
    st.markdown("### Problem Statement")
    st.write(
        "Credit card fraud causes billions of dollars in losses annually. "
        "Detecting fraudulent transactions is difficult because fraud is "
        "extremely rare, accounting for approximately 0.17% of all transactions. "
        "Because legitimate transactions represent 99.83% of the volume, a naive "
        "model that predicts 'legitimate' for every transaction achieves 99.83% "
        "accuracy while detecting zero fraud. Therefore, models must be evaluated "
        "using metrics like Precision-Recall AUC (PR-AUC), Recall, and Precision "
        "rather than standard accuracy."
    )

    st.markdown("---")

    # 2. Dataset Status
    st.markdown("### Dataset Status")

    if RAW_DATA_PATH.exists():
        st.success(
            "Real Dataset Available: creditcard.csv is present in the data/ directory."
        )
    else:
        st.warning(
            "Real Dataset Missing: creditcard.csv was not found in data/. "
            "The application is running in demonstration mode using synthetic "
            "sample data. To evaluate on real data, place creditcard.csv into "
            "the data/ directory."
        )

    st.markdown("---")

    # 3. Models Compared
    st.markdown("### Comparison Models")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("**1. Logistic Regression**")
        st.caption(
            "Linear probabilistic classifier with feature scaling and balanced class weights."
        )

    with col2:
        st.markdown("**2. Decision Tree**")
        st.caption(
            "Interpretable tree-based model with depth control to prevent overfitting."
        )

    with col3:
        st.markdown("**3. Random Forest**")
        st.caption(
            "Ensemble of bagged decision trees for robust classification."
        )

    with col4:
        st.markdown("**4. K-Nearest Neighbors (KNN)**")
        st.caption(
            "Distance-based classifier with scaled features and balanced sampling."
        )

    st.markdown("---")

    # 4. Simple One-Line Workflow
    st.markdown("### Project Workflow")
    st.info(
        "Dataset -> Preprocessing & Scaling -> Model Training -> Evaluation -> Prediction"
    )

    render_footer()


# ==============================================================================
# ROUTER
# ==============================================================================
if selected_page == "Overview":
    render_overview()

elif selected_page == "Explore Dataset":
    p1_path = APP_DIR / "pages" / "1_Explore_Dataset.py"

    if p1_path.exists():
        runpy.run_path(str(p1_path), run_name="__main__")
    else:
        st.error(f"Page file not found: {p1_path}")

elif selected_page == "Model Comparison":
    p2_path = APP_DIR / "pages" / "2_Model_Comparison.py"

    if p2_path.exists():
        runpy.run_path(str(p2_path), run_name="__main__")
    else:
        st.error(f"Page file not found: {p2_path}")

elif selected_page == "Fraud Risk Assessment":
    p3_path = APP_DIR / "pages" / "3_Fraud_Risk_Assessment.py"

    if p3_path.exists():
        runpy.run_path(str(p3_path), run_name="__main__")
    else:
        st.error(f"Page file not found: {p3_path}")
