from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    matthews_corrcoef,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

st.set_page_config(page_title="Classification Model Evaluator", layout="wide")
st.title("Classification Model Evaluator")
st.write("Upload a labeled test CSV to evaluate a trained classification model.")

bundle_path = Path(__file__).with_name("model_bundle.joblib")

if not bundle_path.exists():
    st.error("model_bundle.joblib was not found. Put it in the same folder as app.py.")
    st.stop()

bundle = joblib.load(bundle_path)
preprocessor = bundle["preprocessor"]
models = bundle["models"]
feature_columns = bundle["feature_columns"]

selected_name = st.selectbox("Select a model", list(models.keys()))
uploaded_file = st.file_uploader("Upload test data (CSV)", type=["csv"])

if uploaded_file is not None:
    try:
        test_df = pd.read_csv(uploaded_file, sep=None, engine="python")
    except (UnicodeDecodeError, pd.errors.ParserError, ValueError) as exc:
        st.error(f"Could not read the CSV: {exc}")
        st.stop()

    if "y" not in test_df.columns:
        st.error("The CSV must include the target column 'y' so the app can calculate evaluation metrics.")
        st.stop()

    missing_columns = [col for col in feature_columns if col not in test_df.columns]
    if missing_columns:
        st.error(f"Test data is missing required feature columns: {missing_columns}")
        st.stop()

    raw_labels = test_df["y"].astype("string").str.strip().str.lower()
    label_map = {
        "yes": 1, "no": 0,
        "1": 1, "0": 0,
        "1.0": 1, "0.0": 0,
    }
    y_true = raw_labels.map(label_map)

    if y_true.isna().any():
        invalid_labels = raw_labels[y_true.isna()].dropna().unique().tolist()
        st.error(f"Missing or unrecognized values in 'y': {invalid_labels[:10]}")
        st.stop()

    X_test = test_df.loc[:, feature_columns]

    if X_test.isna().any().any():
        st.error("The test data contains missing feature values. Please clean them before evaluation.")
        st.stop()

    try:
        X_test_prepared = preprocessor.transform(X_test)
        model = models[selected_name]
        y_pred = model.predict(X_test_prepared)
    except (ValueError, KeyError) as exc:
        st.error(f"Could not make predictions: {exc}")
        st.stop()

    y_true = y_true.astype(int)

    st.subheader(f"Evaluation: {selected_name}")
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1 Score": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }

    if y_true.nunique() == 2:
        if hasattr(model, "predict_proba"):
            positive_index = list(model.classes_).index(1)
            y_score = model.predict_proba(X_test_prepared)[:, positive_index]
            metrics["ROC AUC"] = roc_auc_score(y_true, y_score)
        elif hasattr(model, "decision_function"):
            metrics["ROC AUC"] = roc_auc_score(
                y_true, model.decision_function(X_test_prepared)
            )

    st.dataframe(
        pd.DataFrame([metrics]).round(4),
        hide_index=True,
        use_container_width=True,
    )

    st.subheader("Confusion matrix")
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    st.dataframe(
        pd.DataFrame(
            matrix,
            index=["Actual No", "Actual Yes"],
            columns=["Predicted No", "Predicted Yes"],
        ),
        use_container_width=True,
    )

    st.subheader("Classification report")
    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["No", "Yes"],
        output_dict=True,
        zero_division=0,
    )
    st.dataframe(pd.DataFrame(report).transpose().round(4), use_container_width=True)
