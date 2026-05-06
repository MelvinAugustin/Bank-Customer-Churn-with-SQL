import pandas as pd
import streamlit as st
import joblib
import hashlib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from preprocess import preprocess

st.set_page_config(page_title="Churn Model Trainer", layout="wide")
st.title("🔄 Churn Model — Auto-Retrains on New CSV")

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def csv_hash(file_bytes: bytes) -> str:
    """Return an MD5 hash of the raw file bytes."""
    return hashlib.md5(file_bytes).hexdigest()




def run_model(name: str, X_tr, y_tr, X_te, y_te, feature_cols):
    """Train, evaluate, save and return metrics dict."""
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_tr, y_tr)

    preds = model.predict(X_te)
    probs = model.predict_proba(X_te)[:, 1]

    metrics = {
        "Accuracy": round(accuracy_score(y_te, preds), 4),
        "ROC AUC":  round(roc_auc_score(y_te, probs), 4),
        "Report":   classification_report(y_te, preds, output_dict=True),
    }

    save_path = f"{name.lower()}_churn_model.joblib"
    joblib.dump({"model": model, "features": feature_cols}, save_path)
    metrics["path"] = save_path
    return metrics


def train_all(df: pd.DataFrame):
    """Full pipeline: preprocess → split → 3 variants → return results dict."""
    X, y = preprocess(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    features = X.columns.tolist()

    results = {}

    # 1. Normal
    results["NORMAL"] = run_model("NORMAL", X_train, y_train, X_test, y_test, features)

    # 2. Random Over-Sampling
    ros = RandomOverSampler(random_state=42)
    Xr, yr = ros.fit_resample(X_train, y_train)
    results["ROS"] = run_model("ROS", Xr, yr, X_test, y_test, features)

    # 3. Random Under-Sampling
    rus = RandomUnderSampler(random_state=42)
    Xu, yu = rus.fit_resample(X_train, y_train)
    results["RUS"] = run_model("RUS", Xu, yu, X_test, y_test, features)

    return results


# ─────────────────────────────────────────────
# SIDEBAR — FILE UPLOAD
# ─────────────────────────────────────────────

with st.sidebar:
    st.header("📂 Dataset")
    uploaded = st.file_uploader("Upload a CSV (must contain 'churn' column)", type="csv")

    use_default = st.checkbox("Use built-in Churn.csv", value=(uploaded is None))

# ─────────────────────────────────────────────
# LOAD DATA & DETECT CHANGES
# ─────────────────────────────────────────────

if uploaded is not None:
    raw_bytes = uploaded.getvalue()
    file_id   = csv_hash(raw_bytes)

    # Only retrain when the file actually changed
    if st.session_state.get("last_file_id") != file_id:
        st.session_state["last_file_id"] = file_id
        st.session_state["df"]           = pd.read_csv(uploaded)
        st.session_state["results"]      = None   # invalidate old results
        st.session_state["source"]       = uploaded.name
        st.info(f"New file detected: **{uploaded.name}** — will retrain.")
    else:
        st.success(f"Same file as before: **{uploaded.name}** — no retraining needed.")

elif use_default:
    try:
        df_default = pd.read_csv("Churn.csv")
        default_id = "builtin_churn_csv"
        if st.session_state.get("last_file_id") != default_id:
            st.session_state["last_file_id"] = default_id
            st.session_state["df"]           = df_default
            st.session_state["results"]      = None
            st.session_state["source"]       = "Churn.csv (default)"
        st.info("Using built-in **Churn.csv**.")
    except FileNotFoundError:
        st.error("Churn.csv not found. Please upload a CSV file.")
        st.stop()
else:
    st.warning("Please upload a CSV or enable the default dataset.")
    st.stop()

df = st.session_state["df"]
st.write(f"**Rows:** {len(df):,}  |  **Columns:** {len(df.columns)}  |  **Source:** {st.session_state.get('source','—')}")

with st.expander("🔍 Preview columns & first rows"):
    st.write("**Detected columns:**", list(df.columns))
    st.dataframe(df.head(5))

# ─────────────────────────────────────────────
# TRAIN BUTTON  (or auto-train on new data)
# ─────────────────────────────────────────────

force_train = st.session_state.get("results") is None   # auto-train when data is fresh
manual_btn  = st.button("🚀 Train / Retrain Models")

if force_train or manual_btn:
    with st.spinner("Training three model variants…"):
        try:
            results = train_all(df)
            st.session_state["results"] = results
            st.success("✅ Training complete!")
        except Exception as e:
            st.error(f"Training failed: {e}")
            st.stop()

# ─────────────────────────────────────────────
# RESULTS DISPLAY
# ─────────────────────────────────────────────

results = st.session_state.get("results")
if results:
    st.subheader("📊 Model Performance")

    cols = st.columns(3)
    for col, (name, m) in zip(cols, results.items()):
        with col:
            st.markdown(f"### {name}")
            st.metric("Accuracy", m["Accuracy"])
            st.metric("ROC AUC",  m["ROC AUC"])

            report_df = (
                pd.DataFrame(m["Report"])
                .transpose()
                .round(3)
                .drop(index=["accuracy", "macro avg", "weighted avg"], errors="ignore")
            )
            st.dataframe(report_df[["precision", "recall", "f1-score", "support"]])
            st.caption(f"Saved → `{m['path']}`")