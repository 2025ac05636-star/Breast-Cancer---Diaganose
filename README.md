# ML Assignment 2 — Breast Cancer Classification Dashboard

## a. Problem Statement

Breast cancer diagnosis is a critical binary classification task in healthcare —
correctly distinguishing **malignant** from **benign** tumors directly affects
patient treatment decisions. This project implements and compares six machine
learning classification models to predict tumor malignancy from digitized
features of a fine needle aspirate (FNA) of a breast mass, and deploys an
interactive Streamlit dashboard so predictions and performance metrics can be
explored live.

## b. Dataset Description

- **Name:** Breast Cancer Wisconsin (Diagnostic) Data Set
- **Source:** UCI Machine Learning Repository (bundled via `sklearn.datasets.load_breast_cancer`, originally from UCI)
- **Instances:** 569 (≥ 500 required ✅)
- **Features:** 30 numeric features (≥ 12 required ✅) — computed from digitized images of FNA of breast masses, describing characteristics of the cell nuclei present (e.g. `mean radius`, `mean texture`, `mean perimeter`, `mean area`, `mean smoothness`, `worst radius`, `worst concavity`, etc.)
- **Target:** Binary — `0 = malignant`, `1 = benign`
- **Class balance:** 212 malignant / 357 benign
- **Train/Test split:** 80% / 20%, stratified, `random_state = 42`

## c. GitHub Repository Link

https://github.com/2025ac05636-star/Breast-Cancer---Diaganose
## d. Models Used

> Note: the assignment brief lists 5 models but also states "all the 6 ML
> models" — to stay safe, a 6th model (**SVM**) has been added in addition to
> the 5 explicitly named models.

### Comparison Table

| ML Model Name             | Accuracy | AUC    | Precision | Recall | F1     | MCC    |
|----------------------------|----------|--------|-----------|--------|--------|--------|
| Logistic Regression        | 0.9825   | 0.9954 | 0.9861    | 0.9861 | 0.9861 | 0.9623 |
| Decision Tree               | 0.9123   | 0.9157 | 0.9559    | 0.9028 | 0.9286 | 0.8174 |
| kNN                         | 0.9561   | 0.9788 | 0.9589    | 0.9722 | 0.9655 | 0.9054 |
| Naive Bayes                 | 0.9298   | 0.9868 | 0.9444    | 0.9444 | 0.9444 | 0.8492 |
| Random Forest (Ensemble)    | 0.9561   | 0.9932 | 0.9589    | 0.9722 | 0.9655 | 0.9054 |
| SVM                         | 0.9825   | 0.9950 | 0.9861    | 0.9861 | 0.9861 | 0.9623 |

### Observations

| ML Model Name             | Observation about model performance |
|----------------------------|--------------------------------------|
| Logistic Regression        | Best overall performer — the classes are almost linearly separable in this feature space (after standardization), and the model achieves 98.25% accuracy with the highest AUC/MCC alongside SVM. Fast to train and highly interpretable. |
| Decision Tree               | Weakest model of the six (91.23% accuracy, MCC 0.8174). A single tree overfits to the training split and is sensitive to small variations in feature thresholds, producing more false negatives (lower recall) than the other models. |
| kNN                          | Solid performance (95.61% accuracy) once features are scaled, since kNN is distance-based and sensitive to feature magnitude. Performs almost identically to Random Forest here. |
| Naive Bayes                  | Reasonable accuracy (92.98%) despite the Gaussian independence assumption being violated (many of the 30 features are correlated, e.g. radius/perimeter/area). Still achieves a strong AUC (0.9868), meaning it ranks the classes well even where its default 0.5 threshold isn't perfectly calibrated. |
| Random Forest (Ensemble)     | Strong and stable (95.61% accuracy, AUC 0.9932) — averaging many trees fixes most of the single Decision Tree's overfitting, and it is the most robust ensemble choice without needing feature scaling. |
| SVM                           | Ties with Logistic Regression for best accuracy/F1/MCC. The RBF kernel finds a clean separating boundary in the standardized feature space, benefiting from the same near-linear separability as Logistic Regression. |
| **Overall Winner for this dataset** | **Logistic Regression** (tied with SVM) — highest Accuracy, F1, and MCC, with near-perfect AUC (0.9954). Given its simplicity, speed, and interpretability compared to SVM, **Logistic Regression is the recommended model** for this dataset. |

## Project Structure

```
project-folder/
│-- app.py                     # Streamlit app
│-- requirements.txt
│-- README.md
│-- test_data.csv              # held-out test set (features + target)
│-- model/
│   │-- train_models.py        # training + evaluation script
│   │-- logistic_regression.pkl
│   │-- decision_tree.pkl
│   │-- knn.pkl
│   │-- naive_bayes.pkl
│   │-- random_forest_ensemble.pkl
│   │-- svm.pkl
│   │-- scaler.pkl
│   │-- metrics_comparison.csv
```

## How to Run Locally

```bash
pip install -r requirements.txt
python model/train_models.py   # regenerates models + test_data.csv (optional, already included)
streamlit run app.py
```
## Live App

**Live App:**
https://breast-cancer---diaganose-nggnmuytczaupk3pjqxs4b.streamlit.app/

## App Features

1. **CSV Upload** — upload `test_data.csv` to run inference.
2. **Model Selection Dropdown** — choose from all 6 trained models.
3. **Evaluation Metrics Display** — Accuracy, AUC, Precision, Recall, F1, MCC shown live.
4. **Confusion Matrix & Classification Report** — visual + tabular breakdown of predictions.
5. **Bonus:** side-by-side comparison of all 6 models on the uploaded data.
