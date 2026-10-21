"""
Explore Dataset view: Simple dataset inspection for college presentation.

Displays summary metrics, class distribution chart, and dataset preview.
"""

import sys
from pathlib import Path

_CURRENT_DIR = Path(__file__).resolve().parent
_APP_DIR = _CURRENT_DIR.parent
_PROJECT_ROOT = _APP_DIR.parent
for p in [str(_APP_DIR), str(_PROJECT_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import plotly.graph_objects as go
import streamlit as st

try:
    from utils import COLORS, load_features, page_header, render_footer
except ImportError:
    from app.utils import COLORS, load_features, page_header, render_footer

try:
    st.set_page_config(
        page_title="Explore Dataset - Credit Card Fraud Detection",
        layout="wide",
    )
except Exception:
    pass

page_header(
    "Explore Dataset",
    "Summary of transaction counts, class distribution, and sample data",
)

# Fix Error B: Load features consistently returning (df, is_synthetic)
df, is_synthetic = load_features()

# Label synthetic data clearly
if is_synthetic:
    st.info(
        "Notice: The real Kaggle dataset was not found on disk. "
        "Displaying synthetic demonstration data for interface preview."
    )
else:
    st.success("Displaying real processed transactions.")

# 1. Summary Metrics
total_tx = len(df)
if "Class" in df.columns:
    n_fraud = int(df["Class"].sum())
    n_legit = total_tx - n_fraud
    fraud_pct = (n_fraud / total_tx) * 100
else:
    n_fraud = 0
    n_legit = total_tx
    fraud_pct = 0.0

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Transactions", f"{total_tx:,}")
with col2:
    st.metric("Legitimate Transactions", f"{n_legit:,}")
with col3:
    st.metric("Fraudulent Transactions", f"{n_fraud:,}")
with col4:
    st.metric("Fraud Rate", f"{fraud_pct:.3f}%")

st.markdown("---")

# 2. Class Distribution Chart
st.markdown("### Class Distribution")
col_chart, col_exp = st.columns([3, 2])

with col_chart:
    fig = go.Figure(data=[
        go.Bar(
            x=["Legitimate (Class 0)", "Fraudulent (Class 1)"],
            y=[n_legit, n_fraud],
            marker_color=[COLORS["legit"], COLORS["fraud"]],
            text=[f"{n_legit:,}", f"{n_fraud:,}"],
            textposition="auto",
        )
    ])
    fig.update_layout(
        title="Class Counts",
        yaxis_title="Count",
        height=350,
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)

with col_exp:
    st.markdown("#### Understanding Class Imbalance")
    st.write(
        "Fraud occurs in only a tiny fraction of transactions (~0.17%). "
        "Because 99.83% of transactions are legitimate, accuracy is not a reliable evaluation metric: "
        "a dummy classifier predicting 'legitimate' 100% of the time gets 99.83% accuracy but fails completely. "
        "To properly measure detection efficacy, models must be evaluated on Recall (percentage of fraud caught) "
        "and Precision (avoiding false alarms)."
    )

st.markdown("---")

# 3. Small Dataset Preview
st.markdown("### Dataset Preview")
st.caption("First 5 rows of the dataset (displaying Time, Amount, selected PCA components, and Class):")
preview_cols = [c for c in ["Time", "Amount", "V1", "V2", "V3", "V14", "V17", "Class"] if c in df.columns]
if not preview_cols:
    preview_cols = df.columns[:8].tolist()
st.dataframe(df[preview_cols].head(5), use_container_width=True)

render_footer()
