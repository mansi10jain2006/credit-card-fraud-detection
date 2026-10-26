
"""
Model Comparison view: Simple, honest evaluation of the four comparison algorithms.

Displays Logistic Regression, Decision Tree, Random Forest, and KNN metrics
without hardcoding winners, fabricating metrics, or showing mysterious aliases.
"""

import sys
from pathlib import Path

_CURRENT_DIR = Path(__file__).resolve().parent
_APP_DIR = _CURRENT_DIR.parent
_PROJECT_ROOT = _APP_DIR.parent

for path in [str(_APP_DIR), str(_PROJECT_ROOT)]:
    if path not in sys.path:
        sys.path.insert(0, path)

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

try:
    from utils import load_model_results, page_header, render_footer
except ImportError:
    from app.utils import load_model_results, page_header, render_footer


st.set_page_config(
    page_title="Model Comparison - Credit Card Fraud Detection",
    layout="wide",
)

page_header(
    "Model Comparison",
    "Comparative evaluation of Logistic Regression, Decision Tree, Random Forest, and KNN",
)

# Load real evaluation results from disk.
results = load_model_results()

# The four required college mini-project models.
TARGET_MODELS = [
    ("logistic_regression", "Logistic Regression"),
    ("decision_tree", "Decision Tree"),
    ("random_forest", "Random Forest"),
    ("knn", "K-Nearest Neighbors (KNN)"),
]

if results is None:
    st.warning(
        "Evaluation Results Not Yet Generated on Disk\n\n"
        "The comparison models have not yet been evaluated on the real dataset. "
        "To generate verified results:\n\n"
        "1. Ensure creditcard.csv is in the data/ directory.\n"
        "2. Run the comparison script in your terminal:\n\n"
        "```powershell\n"
        "python src/models/compare_models.py\n"
        "```\n\n"
        "This trains the four models, evaluates them on the held-out test split, "
        "and exports metrics to data/model_results.json."
    )
    render_footer()
    st.stop()

# Build the comparison table from the available model results.
rows = []

for key, display_name in TARGET_MODELS:
    if key in results:
        metrics = results[key]
        rows.append(
            {
                "Model": display_name,
                "PR-AUC": float(metrics.get("pr_auc", 0.0)),
                "ROC-AUC": float(metrics.get("roc_auc", 0.0)),
                "Precision": float(metrics.get("precision", 0.0)),
                "Recall": float(metrics.get("recall", 0.0)),
                "F1-Score": float(metrics.get("f1", 0.0)),
            }
        )

if not rows:
    st.info(
        "Evaluation file exists, but results for the four required models "
        "were not found."
    )
    render_footer()
    st.stop()

df_comp = pd.DataFrame(rows)

st.markdown("### Performance Metrics Table")

st.dataframe(
    df_comp.style.format(
        {
            "PR-AUC": "{:.4f}",
            "ROC-AUC": "{:.4f}",
            "Precision": "{:.4f}",
            "Recall": "{:.4f}",
            "F1-Score": "{:.4f}",
        }
    ),
    use_container_width=True,
    hide_index=True,
)

st.markdown("---")

# Display a comparison chart using the actual evaluation metrics.
st.markdown("### Model Comparison Chart")

fig = go.Figure()

fig.add_trace(
    go.Bar(
        name="PR-AUC",
        x=df_comp["Model"],
        y=df_comp["PR-AUC"],
        marker_color="#1f77b4",
    )
)

fig.add_trace(
    go.Bar(
        name="F1-Score",
        x=df_comp["Model"],
        y=df_comp["F1-Score"],
        marker_color="#2ca02c",
    )
)

fig.add_trace(
    go.Bar(
        name="Recall",
        x=df_comp["Model"],
        y=df_comp["Recall"],
        marker_color="#ff7f0e",
    )
)

fig.update_layout(
    title="Key Imbalance Metrics Across Classifiers",
    barmode="group",
    yaxis_title="Score",
    yaxis={"range": [0, 1.05]},
    height=380,
    legend={
        "orientation": "h",
        "yanchor": "bottom",
        "y": 1.02,
        "xanchor": "right",
        "x": 1,
    },
)

st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# Summarize the results dynamically rather than hardcoding a winner.
st.markdown("### Evaluation Summary")

if not df_comp.empty:
    best_pr = df_comp.loc[df_comp["PR-AUC"].idxmax()]

    st.info(
        f"Based on the empirical evaluation on the held-out test data, "
        f"**{best_pr['Model']}** achieved the highest PR-AUC score of "
        f"**{best_pr['PR-AUC']:.4f}** with an F1-score of "
        f"**{best_pr['F1-Score']:.4f}**. In imbalanced fraud detection, "
        f"PR-AUC and F1-score are useful evaluation criteria because they "
        f"reflect the model's ability to identify fraud while controlling "
        f"false alarms."
    )

render_footer()
