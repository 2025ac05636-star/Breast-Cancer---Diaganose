"""
Streamlit App — ML Assignment 2
Breast Cancer Classification Dashboard

Features:
  a. CSV upload (test data)
  b. Model selection dropdown
  c. Evaluation metrics display
  d. Confusion matrix + classification report
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)

st.set_page_config(page_title="Breast Cancer Classifier Comparison", layout="wide")

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "kNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Random Forest (Ensemble)": "random_forest_ensemble.pkl",
    "SVM": "svm.pkl",
}


@st.cache_resource
def load_model(name):
    path = os.path.join(MODEL_DIR, MODEL_FILES[name])
    with open(path, "rb") as f:
        return pickle.load(f)


@st.cache_resource
def load_scaler():
    with open(os.path.join(MODEL_DIR, "scaler.pkl"), "rb") as f:
        return pickle.load(f)


st.title("🔬 Breast Cancer Classification — Model Comparison App")
st.markdown(
    "This app demonstrates **6 classification models** trained on the "
    "**Breast Cancer Wisconsin (Diagnostic)** dataset (569 rows, 30 features). "
    "Upload the provided `test_data.csv`, pick a model, and view its performance."
)

# -------------------------------------------------------------------------
# a. Dataset upload
# -------------------------------------------------------------------------
st.header("1. Upload Test Data (CSV)")
uploaded_file = st.file_uploader(
    "Upload test_data.csv (must contain the 30 feature columns + a 'target' column)",
    type=["csv"],
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success(f"Loaded {df.shape[0]} rows and {df.shape[1]} columns.")
    st.dataframe(df.head())

    if "target" not in df.columns:
        st.error("The uploaded CSV must contain a 'target' column with true labels.")
    else:
        X = df.drop(columns=["target"])
        y_true = df["target"]

        # -----------------------------------------------------------------
        # b. Model selection dropdown
        # -----------------------------------------------------------------
        st.header("2. Select a Model")
        model_choice = st.selectbox("Choose a classification model:", list(MODEL_FILES.keys()))

        model = load_model(model_choice)
        scaler = load_scaler()

        X_scaled = scaler.transform(X)
        y_pred = model.predict(X_scaled)
        y_proba = model.predict_proba(X_scaled)[:, 1]

        # -----------------------------------------------------------------
        # c. Evaluation metrics display
        # -----------------------------------------------------------------
        st.header("3. Evaluation Metrics")
        acc = accuracy_score(y_true, y_pred)
        auc = roc_auc_score(y_true, y_proba)
        prec = precision_score(y_true, y_pred)
        rec = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        mcc = matthews_corrcoef(y_true, y_pred)

        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Accuracy", f"{acc:.4f}")
        c2.metric("AUC", f"{auc:.4f}")
        c3.metric("Precision", f"{prec:.4f}")
        c4.metric("Recall", f"{rec:.4f}")
        c5.metric("F1 Score", f"{f1:.4f}")
        c6.metric("MCC", f"{mcc:.4f}")

        # -----------------------------------------------------------------
        # d. Confusion matrix + classification report
        # -----------------------------------------------------------------
        st.header("4. Confusion Matrix & Classification Report")
        col1, col2 = st.columns(2)

        with col1:
            cm = confusion_matrix(y_true, y_pred)
            fig, ax = plt.subplots(figsize=(4, 3.5))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                        xticklabels=["Malignant", "Benign"],
                        yticklabels=["Malignant", "Benign"])
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            ax.set_title(f"Confusion Matrix — {model_choice}")
            st.pyplot(fig)

        with col2:
            report = classification_report(y_true, y_pred, target_names=["Malignant", "Benign"], output_dict=True)
            report_df = pd.DataFrame(report).transpose().round(3)
            st.dataframe(report_df)

        # -----------------------------------------------------------------
        # Bonus: compare ALL models on the uploaded data
        # -----------------------------------------------------------------
        st.header("5. Compare All Models on This Data")
        if st.checkbox("Show comparison across all 6 models"):
            rows = []
            for name in MODEL_FILES:
                m = load_model(name)
                yp = m.predict(X_scaled)
                ypr = m.predict_proba(X_scaled)[:, 1]
                rows.append({
                    "Model": name,
                    "Accuracy": round(accuracy_score(y_true, yp), 4),
                    "AUC": round(roc_auc_score(y_true, ypr), 4),
                    "Precision": round(precision_score(y_true, yp), 4),
                    "Recall": round(recall_score(y_true, yp), 4),
                    "F1": round(f1_score(y_true, yp), 4),
                    "MCC": round(matthews_corrcoef(y_true, yp), 4),
                })
            st.dataframe(pd.DataFrame(rows).set_index("Model"))
else:
    st.info("👆 Upload the `test_data.csv` file (found in the project root of this repo) to get started.")
