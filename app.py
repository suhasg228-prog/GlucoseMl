import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)

st.set_page_config(page_title="GlucoseML", page_icon="🩺", layout="wide")
st.title("🩺 GlucoseML")
st.caption("Educational diabetes-risk classification demo using the Pima Indians Diabetes dataset.")
st.warning("For learning and demonstration only. This model is not a medical diagnostic tool. Do not use its predictions to make healthcare decisions.")

DATA_URL = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"

@st.cache_data
def load_default_data():
    return pd.read_csv(DATA_URL)

uploaded = st.sidebar.file_uploader("Upload a diabetes CSV (optional)", type=["csv"])
try:
    df = pd.read_csv(uploaded) if uploaded is not None else load_default_data()
except Exception as e:
    st.error(f"Could not load the online dataset: {e}")
    st.info("Check your internet connection or upload a CSV with the required columns.")
    st.stop()

# Normalize common target-column names
target_candidates = ["Outcome", "outcome", "Diabetes", "diabetes", "target", "Target"]
target = next((c for c in target_candidates if c in df.columns), None)
if target is None:
    st.error("Target column not found. Expected one of: Outcome, Diabetes, target.")
    st.stop()

feature_cols = [c for c in df.columns if c != target]
if len(feature_cols) < 2:
    st.error("The dataset must contain at least two feature columns.")
    st.stop()

# Ensure target is binary and clean rows that cannot be used
data = df.copy()
data[target] = pd.to_numeric(data[target], errors="coerce")
data = data.dropna(subset=[target])
if data.empty or data[target].nunique() != 2:
    st.error("The target column must contain exactly two classes (for example, 0 and 1).")
    st.stop()
data[target] = data[target].astype(int)

# Convert feature columns to numeric for this project
for col in feature_cols:
    data[col] = pd.to_numeric(data[col], errors="coerce")
feature_cols = [c for c in feature_cols if data[c].notna().any()]
data = data.dropna(subset=[target])
if len(feature_cols) < 2:
    st.error("At least two numeric feature columns are required.")
    st.stop()

st.sidebar.header("Model settings")
model_name = st.sidebar.selectbox("Choose model", ["Logistic Regression", "Random Forest"])
test_size = st.sidebar.slider("Test data (%)", 10, 40, 20, 5)
random_state = st.sidebar.number_input("Random seed", min_value=0, max_value=9999, value=42)

X = data[feature_cols]
y = data[target]
stratify = y if y.value_counts().min() >= 2 else None
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size / 100, random_state=int(random_state), stratify=stratify
)

if model_name == "Logistic Regression":
    estimator = LogisticRegression(max_iter=2000, class_weight="balanced")
else:
    estimator = RandomForestClassifier(
        n_estimators=250, random_state=int(random_state), class_weight="balanced"
    )

model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("classifier", estimator)
])
model.fit(X_train, y_train)
pred = model.predict(X_test)
proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Overview", "Data Analysis", "Visualizations", "Model Evaluation", "Predict"
])

with tab1:
    st.subheader("Dataset overview")
    c1, c2, c3 = st.columns(3)
    c1.metric("Records", f"{len(data):,}")
    c2.metric("Features", len(feature_cols))
    c3.metric("Positive class", f"{(y == 1).mean() * 100:.1f}%")
    st.dataframe(data.head(10), use_container_width=True)
    st.write("**Columns:**", ", ".join(data.columns.astype(str)))
    st.write("**Missing values by column**")
    st.dataframe(data.isna().sum().rename("Missing values"), use_container_width=True)

with tab2:
    st.subheader("Exploratory Data Analysis")
    st.write("Descriptive statistics")
    st.dataframe(data[feature_cols].describe().T, use_container_width=True)
    st.write("Class distribution")
    counts = y.value_counts().sort_index()
    fig, ax = plt.subplots()
    ax.bar([str(x) for x in counts.index], counts.values)
    ax.set_xlabel("Class (0 = negative, 1 = positive)")
    ax.set_ylabel("Count")
    ax.set_title("Target class distribution")
    st.pyplot(fig)
    plt.close(fig)

with tab3:
    st.subheader("Feature distributions")
    selected_feature = st.selectbox("Select a feature", feature_cols)
    fig, ax = plt.subplots()
    sns.histplot(data=data, x=selected_feature, hue=target, kde=True, element="step", ax=ax)
    ax.set_title(f"Distribution of {selected_feature} by class")
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Correlation heatmap")
    corr_data = data[feature_cols + [target]].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(corr_data, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    ax.set_title("Feature correlation heatmap")
    st.pyplot(fig)
    plt.close(fig)

with tab4:
    st.subheader(f"{model_name} performance")
    accuracy = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred, zero_division=0)
    recall = recall_score(y_test, pred, zero_division=0)
    f1 = f1_score(y_test, pred, zero_division=0)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{accuracy:.3f}")
    c2.metric("Precision", f"{precision:.3f}")
    c3.metric("Recall", f"{recall:.3f}")
    c4.metric("F1-score", f"{f1:.3f}")

    cm = confusion_matrix(y_test, pred, labels=[0, 1])
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Predicted 0", "Predicted 1"],
                yticklabels=["Actual 0", "Actual 1"], ax=ax)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Actual label")
    st.pyplot(fig)
    plt.close(fig)

    st.write("Classification report")
    report = classification_report(y_test, pred, zero_division=0, output_dict=True)
    st.dataframe(pd.DataFrame(report).T, use_container_width=True)

    if proba is not None and len(np.unique(y_test)) == 2:
        fpr, tpr, _ = roc_curve(y_test, proba)
        roc_auc = auc(fpr, tpr)
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
        ax.plot([0, 1], [0, 1], linestyle="--")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve")
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)

with tab5:
    st.subheader("Try a sample prediction")
    st.caption("Enter feature values from your dataset. The output is a model demonstration, not a health assessment.")
    with st.form("prediction_form"):
        values = {}
        cols = st.columns(2)
        for i, feature in enumerate(feature_cols):
            series = data[feature].dropna()
            default = float(series.median()) if not series.empty else 0.0
            min_val = float(series.min()) if not series.empty else 0.0
            max_val = float(series.max()) if not series.empty else 100.0
            with cols[i % 2]:
                values[feature] = st.number_input(
                    feature, min_value=min_val, max_value=max_val,
                    value=default, key=f"input_{feature}"
                )
        submitted = st.form_submit_button("Run model")
    if submitted:
        sample = pd.DataFrame([values], columns=feature_cols)
        prediction = int(model.predict(sample)[0])
        probability = float(model.predict_proba(sample)[0, 1])
        st.metric("Predicted class", str(prediction))
        st.write(f"Model-estimated probability for class 1: **{probability:.1%}**")
        st.info("This is only an educational model output. Consult a qualified healthcare professional for medical questions.")

st.sidebar.markdown("---")
st.sidebar.caption("GlucoseML • Student ML project")
