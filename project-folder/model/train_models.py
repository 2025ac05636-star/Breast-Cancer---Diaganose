"""
train_models.py
----------------
Trains 6 classification models on the Breast Cancer Wisconsin (Diagnostic) dataset,
evaluates them with Accuracy, AUC, Precision, Recall, F1 and MCC, and saves:
  - each trained model as a .pkl file (in model/)
  - the fitted StandardScaler (model/scaler.pkl)
  - a held-out test set as test_data.csv (project root) — used by the Streamlit app
  - a metrics comparison table as model/metrics_comparison.csv

Dataset: Breast Cancer Wisconsin (Diagnostic)
Source : scikit-learn built-in (originally UCI ML Repository)
Shape  : 569 instances, 30 numeric features, binary target (malignant / benign)
This satisfies the assignment's minimum requirement of >=500 instances and >=12 features.
"""

import pandas as pd
import numpy as np
import pickle
import os

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef
)

RANDOM_STATE = 42
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(MODEL_DIR)

# ---------------------------------------------------------------------------
# 1. Load dataset
# ---------------------------------------------------------------------------
data = load_breast_cancer(as_frame=True)
X = data.data
y = data.target  # 0 = malignant, 1 = benign
feature_names = list(X.columns)

print(f"Dataset shape: {X.shape}, classes: {np.unique(y)}")

# ---------------------------------------------------------------------------
# 2. Train / Test split (stratified)
# ---------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

# ---------------------------------------------------------------------------
# 3. Scale features (fit on train only)
# ---------------------------------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

with open(os.path.join(MODEL_DIR, "scaler.pkl"), "wb") as f:
    pickle.dump(scaler, f)

# ---------------------------------------------------------------------------
# 4. Save test data (features + true label) as CSV for the Streamlit app
#    (this is the "test data" mentioned in the assignment instructions)
# ---------------------------------------------------------------------------
test_df = X_test.copy()
test_df["target"] = y_test.values
test_df.to_csv(os.path.join(ROOT_DIR, "test_data.csv"), index=False)
print(f"Saved test_data.csv with shape {test_df.shape}")

# ---------------------------------------------------------------------------
# 5. Define models
#    NOTE: The assignment text lists 5 models but also says "all the 6 ML
#    models" — to be safe we implement 6, adding SVM as the extra model.
# ---------------------------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=5000, random_state=RANDOM_STATE),
    "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
    "kNN": KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes": GaussianNB(),
    "Random Forest (Ensemble)": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
    "SVM": SVC(probability=True, random_state=RANDOM_STATE),
}

results = []

for name, model in models.items():
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    metrics = {
        "ML Model Name": name,
        "Accuracy": round(accuracy_score(y_test, y_pred), 4),
        "AUC": round(roc_auc_score(y_test, y_proba), 4),
        "Precision": round(precision_score(y_test, y_pred), 4),
        "Recall": round(recall_score(y_test, y_pred), 4),
        "F1": round(f1_score(y_test, y_pred), 4),
        "MCC": round(matthews_corrcoef(y_test, y_pred), 4),
    }
    results.append(metrics)
    print(metrics)

    # Save model
    fname = name.lower().replace(" ", "_").replace("(", "").replace(")", "") + ".pkl"
    with open(os.path.join(MODEL_DIR, fname), "wb") as f:
        pickle.dump(model, f)

# ---------------------------------------------------------------------------
# 6. Save comparison table
# ---------------------------------------------------------------------------
results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(MODEL_DIR, "metrics_comparison.csv"), index=False)
print("\nFinal comparison table:")
print(results_df.to_string(index=False))
