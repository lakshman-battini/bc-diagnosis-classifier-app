import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, f1_score,
                             matthews_corrcoef, precision_score,
                             recall_score, roc_auc_score)

TARGET = "target"
MODEL_DIR = "model"
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}

st.set_page_config(page_title="Breast Cancer Classifier Lab", page_icon="🩺", layout="wide")


@st.cache_resource
def load_model(name):
    return joblib.load(os.path.join(MODEL_DIR, MODEL_FILES[name]))


def evaluate(model, X, y):
    pred = model.predict(X)
    prob = model.predict_proba(X)
    n_classes = len(np.unique(y))
    avg = "binary" if n_classes == 2 else "macro"
    if n_classes == 2:
        auc = roc_auc_score(y, prob[:, 1])
    else:
        auc = roc_auc_score(y, prob, multi_class="ovr", average="macro")
    metrics = {
        "Accuracy": accuracy_score(y, pred),
        "AUC": auc,
        "Precision": precision_score(y, pred, average=avg, zero_division=0),
        "Recall": recall_score(y, pred, average=avg, zero_division=0),
        "F1": f1_score(y, pred, average=avg, zero_division=0),
        "MCC": matthews_corrcoef(y, pred),
    }
    return pred, metrics


st.title("🩺 Breast Cancer Diagnosis: Classifier Comparison")
st.caption("ML Assignment 2 · M.Tech (AIML) · Models are pre-trained; "
           "the uploaded CSV is used only for evaluation (no retraining).")

# ---- Sidebar: data upload + model selection ----
with st.sidebar:
    st.header("1. Test data")
    uploaded = st.file_uploader("Upload test CSV (features + 'target')", type=["csv"])
    st.header("2. Model")
    model_name = st.selectbox("Choose a model", list(MODEL_FILES.keys()))
    compare_all = st.checkbox("Also compare all 5 models", value=True)
    if os.path.exists("test_data.csv"):
        with open("test_data.csv", "rb") as f:
            st.download_button("Download sample test_data.csv", f, "test_data.csv", "text/csv")

if uploaded is None:
    st.info("Upload a test CSV in the sidebar to begin (the sample test_data.csv "
            "from the repo works). It must contain all 30 feature columns and a "
            f"`{TARGET}` column (0 = malignant, 1 = benign).")
    st.stop()

df = pd.read_csv(uploaded)
if TARGET not in df.columns:
    st.error(f"Column `{TARGET}` not found in the uploaded file.")
    st.stop()

model = load_model(model_name)
expected = list(model.feature_names_in_)
missing = [c for c in expected if c not in df.columns]
if missing:
    st.error(f"Missing feature columns: {missing}")
    st.stop()

X, y = df[expected], df[TARGET]
st.subheader("Data preview")
st.write(f"{df.shape[0]} rows × {df.shape[1]} columns")
st.dataframe(df.head())

pred, metrics = evaluate(model, X, y)

st.subheader(f"Evaluation metrics: {model_name}")
cols = st.columns(6)
for col, (k, v) in zip(cols, metrics.items()):
    col.metric(k, f"{v:.4f}")

left, right = st.columns(2)
with left:
    st.markdown("**Confusion matrix**")
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y, pred, display_labels=["Malignant", "Benign"], cmap="Blues", ax=ax)
    ax.grid(False)
    st.pyplot(fig)
with right:
    st.markdown("**Classification report**")
    rep = classification_report(y, pred, target_names=["Malignant", "Benign"],
                                output_dict=True, zero_division=0)
    st.dataframe(pd.DataFrame(rep).T.round(3))

if compare_all:
    st.subheader("All models on the uploaded test data")
    rows = []
    for name in MODEL_FILES:
        _, m = evaluate(load_model(name), X, y)
        rows.append({"ML Model Name": name, **m})
    comp = pd.DataFrame(rows).set_index("ML Model Name")
    st.dataframe(comp.style.format("{:.4f}").highlight_max(axis=0, color="#c6efce"))
    st.bar_chart(comp[["Accuracy", "F1", "MCC"]])
