# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from io import BytesIO
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"

st.set_page_config(page_title="Heart Disease Predictor", layout="wide")

@st.cache_resource
def load_artifacts():
    # load models and scaler
    artifacts = {}
    try:
        artifacts["lr"] = joblib.load(MODELS_DIR / "logistic_regression.pkl")
        artifacts["dt"] = joblib.load(MODELS_DIR / "decision_tree.pkl")
        artifacts["knn"] = joblib.load(MODELS_DIR / "knn.pkl")
    except Exception as e:
        st.error(f"Error loading models: {e}")
        raise
    try:
        artifacts["scaler"] = joblib.load(MODELS_DIR / "scaler.pkl")
    except Exception as e:
        st.error(f"Error loading scaler: {e}")
        raise

    with open(BASE_DIR / "metadata.json", "r") as f:
        artifacts["metadata"] = json.load(f)

    try:
        artifacts["cleaned"] = pd.read_csv(DATA_DIR / "cleaned_heart_disease_dataset.csv")
    except Exception:
        artifacts["cleaned"] = None

    return artifacts

art = load_artifacts()
FEATURE_ORDER = art["metadata"]["features"]
models = {
    "Logistic Regression": art["lr"],
    "Decision Tree": art["dt"],
    "KNN": art["knn"]
}
scaler = art["scaler"]
cleaned_df = art["cleaned"]

st.title("💓 Heart Disease Prediction")

st.markdown(
    """
    This app loads pre-trained models and a scaler to make predictions.
    - Provide a single patient's data (left), or
    - Upload a CSV with multiple rows (right).
    """
)

# helper preprocessing function
def preprocess_input(df: pd.DataFrame, cleaned_reference: pd.DataFrame = None) -> pd.DataFrame:
    # common rename map (tries to handle alternate column names)
    rename_map = {
        "trtbps": "trestbps",
        "exng": "exang",
        "slp": "slope",
        "caa": "ca",
        "thall": "thal",
        "thalach": "thalachh"
    }
    df = df.rename(columns=rename_map)
    # ensure expected features exist
    for col in FEATURE_ORDER:
        if col not in df.columns:
            df[col] = np.nan
    df = df[FEATURE_ORDER].copy()
    df = df.apply(pd.to_numeric, errors="coerce")

    # fill missing using medians from cleaned dataset when available, else column medians
    if cleaned_reference is not None:
        medians = cleaned_reference[FEATURE_ORDER].median()
    else:
        medians = df.median()
    df = df.fillna(medians)
    return df

def predict_df(X_df: pd.DataFrame) -> pd.DataFrame:
    X_scaled = scaler.transform(X_df)
    out = X_df.copy()
    for name, clf in models.items():
        preds = clf.predict(X_scaled)
        out[f"{name}_pred"] = preds
        st.write(f"- **{name}** → { 'No Disease' if int(result[f'{name}_pred'].iloc[0])==1 else 'Disease' }")


    return out

# Layout: two columns
col1, col2 = st.columns(2)

with col1:
    st.subheader("Single prediction")
    with st.form("single"):
        # create input controls for each feature (arranged columns)
        c1, c2 = st.columns(2)
        with c1:
            age = st.number_input("age", min_value=0, max_value=120, value=45)
            sex = st.selectbox("sex (0=Female,1=Male)", options=[0,1], index=1)
            cp = st.selectbox("cp (chest pain: 0-3)", options=[0,1,2,3], index=1)
            trestbps = st.number_input("trestbps (resting bp)", value=120)
            chol = st.number_input("chol (cholesterol)", value=250)
            fbs = st.selectbox("fbs (>120 mg/dl) (0/1)", options=[0,1], index=0)
            restecg = st.selectbox("restecg (0-2)", options=[0,1,2], index=1)
        with c2:
            thalachh = st.number_input("thalachh (max heart rate)", value=150)
            exang = st.selectbox("exang (exercise-induced angina 0/1)", options=[0,1], index=0)
            oldpeak = st.number_input("oldpeak (ST depression)", value=1.0, format="%.2f")
            slope = st.selectbox("slope (0-2)", options=[0,1,2], index=1)
            ca = st.selectbox("ca (num major vessels 0-3)", options=[0,1,2,3], index=0)
            thal = st.selectbox("thal (0-3)", options=[0,1,2,3], index=2)

        submitted = st.form_submit_button("Run prediction")

    if submitted:
        input_df = pd.DataFrame([{
            "age": age, "sex": sex, "cp": cp, "trestbps": trestbps,
            "chol": chol, "fbs": fbs, "restecg": restecg,
            "thalachh": thalachh, "exang": exang, "oldpeak": oldpeak,
            "slope": slope, "ca": ca, "thal": thal
        }])
        pre = preprocess_input(input_df, cleaned_df)
        result = predict_df(pre)
        st.markdown("**Predictions:**")
        for name in models.keys():
           st.write(f"- **{name}** → { 'No Disease' if int(result[f'{name}_pred'].iloc[0])==1 else 'Disease' }")

        st.write("Full output:")
        st.dataframe(result)

with col2:
    st.subheader("Batch prediction (CSV)")
    st.markdown("Upload a CSV with columns matching the features in `metadata.json`.")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded is not None:
        try:
            df_in = pd.read_csv(uploaded)
            df_pre = preprocess_input(df_in, cleaned_df)
            df_out = predict_df(df_pre)
            st.write("Preview of predictions:")
            st.dataframe(df_out.head(50))
            # download
            csv_bytes = df_out.to_csv(index=False).encode("utf-8")
            st.download_button("Download predictions CSV", data=csv_bytes, file_name="predictions.csv", mime="text/csv")
        except Exception as e:
            st.error(f"Error processing file: {e}")

st.markdown("---")
st.subheader("Model metrics & Dataset")

if cleaned_df is None:
    st.info("No cleaned dataset found in data/cleaned_heart_disease_dataset.csv — put one there to see metrics and EDA.")
else:
    st.write(f"Loaded cleaned dataset with {cleaned_df.shape[0]:,} rows.")
    if "target" in cleaned_df.columns:
        X = preprocess_input(cleaned_df, cleaned_df)
        y = cleaned_df["target"].astype(int)
        X_scaled = scaler.transform(X)
        metrics = []
        for name, clf in models.items():
            preds = clf.predict(X_scaled)
            metrics.append({
                "Model": name,
                "Accuracy": accuracy_score(y, preds),
                "F1": f1_score(y, preds),
                "Precision": precision_score(y, preds),
                "Recall": recall_score(y, preds)
            })
        metrics_df = pd.DataFrame(metrics)
        st.write("Model performance on the cleaned dataset (quick check):")
        st.dataframe(metrics_df.style.format({
            "Accuracy": "{:.3f}",
            "F1": "{:.3f}",
            "Precision": "{:.3f}",
            "Recall": "{:.3f}"
        }))

        # Confusion matrices
        st.write("Confusion matrices:")
        fig, axes = plt.subplots(1, len(models), figsize=(5*len(models), 4))
        if len(models) == 1:
            axes = [axes]
        for ax, (name, clf) in zip(axes, models.items()):
            preds = clf.predict(X_scaled)
            cm = confusion_matrix(y, preds)
            sns.heatmap(cm, annot=True, fmt="d", ax=ax)
            ax.set_title(name)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
        st.pyplot(fig)
    else:
        st.info("No 'target' column in cleaned dataset — add it to compute model metrics.")
