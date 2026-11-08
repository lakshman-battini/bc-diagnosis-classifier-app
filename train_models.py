"""Train 5 classifiers on one dataset, save models + test data + metrics.

Run:  python train_models.py
Outputs:
    model/*.joblib        trained pipelines (scaler + classifier)
    test_data.csv         held-out test set (features + target) for the Streamlit app
    full_dataset.csv      full dataset, for reference
    model/metrics.csv     evaluation metrics for all models
"""
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

SEED = 42
TARGET = "target"
MODEL_DIR = "model"

MODELS = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=SEED),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=SEED),
    "KNN": KNeighborsClassifier(n_neighbors=7),
    "Naive Bayes": GaussianNB(),
    "Support Vector Machine": SVC(kernel="rbf", probability=True, random_state=SEED),
}


def compute_metrics(y_true, y_pred, y_prob):
    """Works for binary and multi-class targets."""
    n_classes = len(np.unique(y_true))
    avg = "binary" if n_classes == 2 else "macro"
    if n_classes == 2:
        auc = roc_auc_score(y_true, y_prob[:, 1])
    else:
        auc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": auc,
        "Precision": precision_score(y_true, y_pred, average=avg, zero_division=0),
        "Recall": recall_score(y_true, y_pred, average=avg, zero_division=0),
        "F1": f1_score(y_true, y_pred, average=avg, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    data = load_breast_cancer(as_frame=True)
    df = data.frame.copy()  # 30 features + 'target'
    df.to_csv("full_dataset.csv", index=False)
    print(f"Dataset: {df.shape[0]} rows, {df.shape[1] - 1} features")

    X, y = df.drop(columns=[TARGET]), df[TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED)

    # Test data saved with the target column, as the app requires
    pd.concat([X_te, y_te], axis=1).to_csv("test_data.csv", index=False)

    rows = []
    for name, clf in MODELS.items():
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
        pipe.fit(X_tr, y_tr)
        fname = name.lower().replace(" ", "_") + ".joblib"
        joblib.dump(pipe, os.path.join(MODEL_DIR, fname))
        m = compute_metrics(y_te, pipe.predict(X_te), pipe.predict_proba(X_te))
        rows.append({"ML Model Name": name, **m})
        print(name, {k: round(v, 4) for k, v in m.items()})

    res = pd.DataFrame(rows).round(4)
    res.to_csv(os.path.join(MODEL_DIR, "metrics.csv"), index=False)
    print("\n", res.to_string(index=False))


if __name__ == "__main__":
    main()
