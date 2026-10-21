# Credit Card Fraud Detection
## Machine Learning Mini-Project

An end-to-end machine learning system designed to identify fraudulent credit card transactions in a heavily imbalanced dataset (~0.17% fraud rate). Built as a complete academic pipeline featuring automated data validation, exploratory data analysis, domain and statistical feature engineering, model comparison across four foundational algorithms, and an interactive Streamlit web dashboard.

---

## 1. Project Overview

- **The Problem:** Credit card fraud accounts for billions of dollars in annual losses worldwide. Detecting fraud is inherently difficult because genuine transactions outnumber fraudulent events by approximately 600 to 1 (~0.17% fraud rate). Under such severe class imbalance, naive models that always predict "legitimate" achieve 99.83% accuracy while failing to detect any fraud.
- **The Operational Challenge:** False positives incur real customer friction (legitimate card declines), while false negatives allow fraudulent monetary loss. As a result, model selection cannot rely on accuracy alone; it must be driven by **Precision-Recall AUC (PR-AUC)**, **Recall**, and **Precision**.
- **The Dataset:** European cardholder transaction dataset comprising 284,807 transactions over two days. Features `V1` through `V28` are anonymized principal components derived via PCA to safeguard cardholder confidentiality. Only `Time` (seconds elapsed) and `Amount` ($) are in their original units. Target: `Class = 1` (Fraud), `Class = 0` (Legitimate).

---

## 2. Machine Learning Models Compared

This project compares four classic, interpretable machine learning algorithms taught in academic curricula:

| Model | Type | Imbalance Handling | Preprocessing Pipeline |
| :--- | :--- | :--- | :--- |
| **Logistic Regression** | Linear Probabilistic | `class_weight="balanced"` | `StandardScaler` inside Pipeline (Leakage-Free) |
| **Decision Tree** | Non-Linear Tree Split | `class_weight="balanced"` | Max depth & min samples leaf regularization |
| **Random Forest** | Bagged Ensemble | `class_weight="balanced"` | Multi-tree bootstrap aggregation (100 estimators) |
| **K-Nearest Neighbors (KNN)** | Instance Distance-Based | Distance weights + balanced train subsampling | `StandardScaler` inside Pipeline (Leakage-Free) |

> **Anti-Leakage Design:** For distance-sensitive models (Logistic Regression and KNN), feature scaling is encapsulated inside Scikit-Learn `Pipeline` objects. The scalers are fitted strictly on training data (`X_train`) and applied without refitting to test partitions.

---

## 3. System Architecture

```
┌─────────────────────────────────┐
│     Raw Data: creditcard.csv    │  284,807 transactions, 31 columns
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│       Quality Gate Validation   │  5 Automated Checks (Schema, Nulls,
└────────────────┬────────────────┘   Ranges, Row Count, Target)
                 │
                 ▼
┌─────────────────────────────────┐
│      Cleaner & Deduplication    │  Removes 1,081 exact duplicates
└────────────────┬────────────────┘  Enforces numeric data types
                 │
                 ▼
┌─────────────────────────────────┐
│       Feature Engineering       │  13 engineered domain, statistical,
└────────────────┬────────────────┘   and interaction attributes
                 │
                 ▼
┌─────────────────────────────────┐
│     Stratified Split (80/20)    │  Preserves 0.17% fraud ratio
└────────────────┬────────────────┘   across train and test splits
                 │
                 ▼
┌─────────────────────────────────┐
│     Model Training & Tuning     │  Logistic Regression, Decision Tree,
└────────────────┬────────────────┘   Random Forest, and KNN
                 │
                 ▼
┌─────────────────────────────────┐
│   Evaluation & Export Artifacts │  PR-AUC, ROC-AUC, F1, Precision,
└────────────────┬────────────────┘   Recall, and Confusion Matrices
                 │
                 ▼
┌─────────────────────────────────┐
│   Streamlit Web Dashboard       │  Multi-page dashboard with EDA,
│   (app/streamlit_app.py)        │  Model Comparison, & Live Checker
└─────────────────────────────────┘
```

---

## 4. Project Structure

```
end-to-end-credit-card-fraud-detection/
├── app/
│   ├── pages/
│   │   ├── 1_Explore_Dataset.py         # Summary metrics & class distribution
│   │   ├── 2_Model_Comparison.py        # Side-by-side evaluation table & charts
│   │   └── 3_Fraud_Risk_Assessment.py   # Live prediction & CSV batch upload tool
│   ├── demo_data.py                     # Demonstration data generators
│   ├── streamlit_app.py                 # Main dashboard entrypoint (Overview)
│   └── utils.py                         # Data loaders, caching & helpers
├── data/                             # Dataset directory
│   ├── sample_transactions.csv       # Pre-configured sample transactions
│   ├── creditcard.csv                # (User provided) Raw Kaggle dataset
│   ├── cleaned.csv                   # Cleaned data
│   ├── features.csv                  # Engineered features
│   └── model_results.json            # Dynamic evaluation results
├── models/                           # Serialized model artifacts (.pkl)
├── notebooks/
│   └── eda.ipynb                     # Jupyter notebook for exploratory analysis
├── src/
│   ├── data/
│   │   ├── loader.py                 # Dataset inspection script
│   │   ├── quality.py                # 5-gate automated validation
│   │   └── cleaner.py                # Deduplication and type casting
│   ├── features/
│   │   ├── engineering.py            # Feature transformation logic
│   │   └── run_features.py           # Feature pipeline execution script
│   └── models/
│       ├── baseline.py               # Logistic Regression baseline
│       ├── compare_models.py         # 4-model comparative training & evaluation
│       └── run_training.py           # Experiment runner & model exporter
├── tests/
│   ├── conftest.py                   # Pytest fixtures
│   ├── test_data_quality.py          # Validation gate test cases
│   ├── test_features.py              # Feature transformation unit tests
│   └── test_model.py                 # Model prediction & probability tests
├── requirements.txt                  # Python dependencies
└── setup.py                          # Local package installation configuration
```

---

## 5. Setup & Installation (Windows PowerShell)

### Step 1: Create and Activate Virtual Environment
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```
*(If PowerShell displays a script execution policy restriction, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` and then run the activate command again).*

### Step 2: Install Dependencies
```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

---

## 6. How to Run

### Option A: Launch the Streamlit Dashboard Directly
You can run the interactive web application immediately:
```powershell
streamlit run app/streamlit_app.py
```
*Open your browser at `http://localhost:8501`. If the raw dataset is not yet present, the application runs in a clearly marked **Interface Demonstration Mode** using synthetic sample structures.*

---

### Option B: Execute the Full Machine Learning Pipeline

1. **Obtain the Dataset:**
   Download `creditcard.csv` from Kaggle's [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) dataset and place it in the `data/` directory:
   ```
   data/creditcard.csv
   ```

2. **Run Pipeline Steps Sequentially:**
   ```powershell
   # Step 1: Verify data quality against 5 validation checks
   python src/data/quality.py

   # Step 2: Clean data and remove duplicate rows (produces data/cleaned.csv)
   python src/data/cleaner.py

   # Step 3: Engineer domain and statistical features (produces data/features.csv)
   python src/features/run_features.py

   # Step 4: Train and evaluate Logistic Regression, Decision Tree, Random Forest, & KNN
   python src/models/compare_models.py

   # Step 5: Run automated test suite
   pytest tests/ -v
   ```

---

## 7. Fraud Risk Assessment

The **Fraud Risk Assessment** page provides:
- **Sample Transaction Selector:** Test predefined transactions (known legitimate and fraudulent cases) without manually typing 30 numerical values.
- **CSV Batch Upload:** Upload a CSV file containing transactions for automatic column validation, null checking, and batch prediction.
- **Risk Assessment Output:** Displays clear classification (*Likely Legitimate* vs. *Potential Fraud*), estimated fraud probability percentage, and feature inspection.
- **Academic Disclaimer:** Reminds users that ML predictions are statistical anomaly indicators and require analyst verification in production environments.

---

## 8. Evaluation Metrics Rationale

- **PR-AUC (Precision-Recall Area Under Curve):** The headline metric. Prioritizes performance on the minority fraud class without being artificially inflated by the massive volume of true negatives.
- **Recall:** Proportion of actual fraud caught. High recall minimizes monetary loss from undetected fraud.
- **Precision:** Proportion of flagged transactions that were genuine fraud. High precision avoids unnecessary account freezes and legitimate customer declines.
- **F1-Score:** Harmonic mean balancing precision and recall.
- **Confusion Matrix:** Provides explicit counts of True Negatives, False Positives (false alarms), False Negatives (missed fraud), and True Positives.
