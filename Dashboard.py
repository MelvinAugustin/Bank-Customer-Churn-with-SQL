import sys
import os
import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import (
    accuracy_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix,
)



# ── Import shared preprocessing from train_model.py ─────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from train_model import preprocess

# ────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Dashboard",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Font only — no color overrides, let Streamlit & Plotly use their defaults
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
</style>
""", unsafe_allow_html=True)

FONT = "IBM Plex Sans, sans-serif"

def base_layout(**kwargs):
    """Shared Plotly layout — font + margins only, no color overrides."""
    return dict(font=dict(family=FONT), margin=dict(l=40, r=20, t=40, b=40), **kwargs)

# ────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("🏦 Churn Dashboard")
    uploaded     = st.file_uploader("Upload CSV", type="csv")
    st.divider()
    model_choice = st.selectbox("Active Model", ["NORMAL", "ROS", "RUS"])
    top_n        = st.slider("Feature importance — top N", 5, 30, 15)
    prob_thresh  = st.slider("Churn probability threshold", 0.1, 0.9, 0.5, 0.05)
    st.divider()
    st.caption("Reads model files saved by train_model.py")

# ────────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_csv(raw: bytes) -> pd.DataFrame:
    import io
    return pd.read_csv(io.BytesIO(raw))

if uploaded:
    raw_df = load_csv(uploaded.getvalue())
    source = uploaded.name
elif os.path.exists("Churn.csv"):
    raw_df = pd.read_csv("Churn.csv")
    source = "Churn.csv (default)"
else:
    st.error("No data found. Upload a CSV or place Churn.csv in the working directory.")
    st.stop()

# ────────────────────────────────────────────────────────────────────────────
# LOAD MODELS
# ────────────────────────────────────────────────────────────────────────────
model_files = {
    "NORMAL": "normal_churn_model.joblib",
    "ROS":    "ros_churn_model.joblib",
    "RUS":    "rus_churn_model.joblib",
}

missing = [k for k, v in model_files.items() if not os.path.exists(v)]
if missing:
    st.error(f"Model file(s) not found: `{'`, `'.join(missing)}`. Run train_model.py first.")
    st.stop()

@st.cache_resource(show_spinner=False)
def load_models():
    return {k: joblib.load(v) for k, v in model_files.items()}

model_data = load_models()
models     = {k: v["model"] for k, v in model_data.items()}
features   = model_data["NORMAL"]["features"]

# ────────────────────────────────────────────────────────────────────────────
# PREPROCESS
# ────────────────────────────────────────────────────────────────────────────
with st.spinner("Preprocessing…"):
    try:
        X_full, y_full = preprocess(raw_df)
        X_full = X_full.reindex(columns=features, fill_value=0)
    except Exception as e:
        st.error(f"Preprocessing failed: {e}")
        st.stop()

# ────────────────────────────────────────────────────────────────────────────
# PREDICTIONS
# ────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def compute_predictions(_models, _X, _y):
    preds, probs, met = {}, {}, {}
    for name, m in _models.items():
        p  = m.predict(_X)
        pr = m.predict_proba(_X)[:, 1]
        preds[name] = p
        probs[name] = pr
        met[name] = {
            "Accuracy": round(accuracy_score(_y, p),  4),
            "Recall":   round(recall_score(_y, p),     4),
            "F1":       round(f1_score(_y, p),         4),
            "ROC AUC":  round(roc_auc_score(_y, pr),  4),
            "Avg Prob": round(pr.mean(),               4),
        }
    return preds, probs, met

preds, probs, metrics = compute_predictions(models, X_full, y_full)

active_model = models[model_choice]
active_probs = probs[model_choice]
active_preds = (active_probs >= prob_thresh).astype(int)

# ────────────────────────────────────────────────────────────────────────────
# HEADER + KPIs
# ────────────────────────────────────────────────────────────────────────────
st.title("🏦 Bank Churn Prediction Dashboard")
st.caption(f"Source: {source}  ·  Active model: {model_choice}  ·  Threshold: {prob_thresh}")

m = metrics[model_choice]
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total Customers",    f"{len(X_full):,}")
c2.metric("Actual Churners",    f"{int(y_full.sum()):,}")
c3.metric("Predicted Churners", f"{int(active_preds.sum()):,}")
c4.metric(f"{model_choice} Accuracy", f"{m['Accuracy']:.1%}")
c5.metric(f"{model_choice} ROC AUC",  f"{m['ROC AUC']:.3f}")

st.divider()

# ────────────────────────────────────────────────────────────────────────────
# TABS
# ────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Model Comparison",
    "🔍 Feature Importance",
    "📈 ROC & Confusion",
    "💡 Customer Explorer",
    "🌡️ Data Overview",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — MODEL COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Model Performance Comparison")

    categories = ["Accuracy", "Recall", "F1", "ROC AUC"]
    fig_radar = go.Figure()
    for name in ["NORMAL", "ROS", "RUS"]:
        vals = [metrics[name][c] for c in categories]
        fig_radar.add_trace(go.Scatterpolar(
            r=vals + [vals[0]],
            theta=categories + [categories[0]],
            fill="toself",
            name=name,
        ))
    fig_radar.update_layout(
        **base_layout(height=420),
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        legend=dict(orientation="h", y=-0.15, x=0.5, xanchor="center"),
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    metric_names = ["Accuracy", "Recall", "F1", "ROC AUC", "Avg Prob"]
    comp_rows = [
        {"Model": name, "Metric": mn, "Value": metrics[name][mn]}
        for name in ["NORMAL", "ROS", "RUS"]
        for mn in metric_names
    ]
    fig_bar = px.bar(
        pd.DataFrame(comp_rows), x="Metric", y="Value",
        color="Model", barmode="group", text_auto=".3f",
    )
    fig_bar.update_layout(**base_layout(height=380),
                           legend=dict(orientation="h", y=-0.2, x=0.5, xanchor="center"))
    fig_bar.update_traces(textposition="outside", cliponaxis=False)
    st.plotly_chart(fig_bar, use_container_width=True)

    st.subheader("Raw Metrics Table")
    table_df = pd.DataFrame(metrics).T.reset_index().rename(columns={"index": "Model"})
    st.dataframe(
        table_df.style.format({c: "{:.4f}" for c in metric_names if c in table_df.columns}),
        use_container_width=True,
    )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — FEATURE IMPORTANCE
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader(f"Feature Importance — {model_choice} (Top {top_n})")

    imp_df = (
        pd.DataFrame({"Feature": features, "Importance": active_model.feature_importances_})
        .sort_values("Importance", ascending=False)
        .head(top_n)
    )
    fig_imp = px.bar(imp_df[::-1], x="Importance", y="Feature", orientation="h")
    fig_imp.update_layout(**base_layout(height=max(350, top_n * 28)))
    st.plotly_chart(fig_imp, use_container_width=True)

    st.subheader("Feature Importance — All Models (Top 15)")
    fig_imp3 = go.Figure()
    for name in ["NORMAL", "ROS", "RUS"]:
        imp = pd.Series(models[name].feature_importances_, index=features).nlargest(15)
        fig_imp3.add_trace(go.Bar(x=imp.index, y=imp.values, name=name, opacity=0.85))
    fig_imp3.update_layout(
        **base_layout(height=400), barmode="group",
        xaxis=dict(tickangle=-35),
        legend=dict(orientation="h", y=-0.25, x=0.5, xanchor="center"),
    )
    st.plotly_chart(fig_imp3, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — ROC & CONFUSION
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    col_roc, col_cm = st.columns(2)

    with col_roc:
        st.subheader("ROC Curves")
        fig_roc = go.Figure()
        fig_roc.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                           line=dict(dash="dot", width=1))
        for name in ["NORMAL", "ROS", "RUS"]:
            fpr, tpr, _ = roc_curve(y_full, probs[name])
            fig_roc.add_trace(go.Scatter(
                x=fpr, y=tpr, mode="lines",
                name=f"{name} (AUC={metrics[name]['ROC AUC']:.3f})",
                line=dict(width=2),
            ))
        fig_roc.update_layout(
            **base_layout(height=420),
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
        )
        st.plotly_chart(fig_roc, use_container_width=True)

    with col_cm:
        st.subheader(f"Confusion Matrix — {model_choice}")
        cm = confusion_matrix(y_full, active_preds)
        labels = ["No Churn", "Churn"]
        fig_cm = px.imshow(
            cm, text_auto=True, x=labels, y=labels,
            color_continuous_scale="Blues",
            labels=dict(x="Predicted", y="Actual"),
        )
        fig_cm.update_layout(**base_layout(height=420), coloraxis_showscale=False)
        fig_cm.update_traces(textfont=dict(size=18))
        st.plotly_chart(fig_cm, use_container_width=True)

    st.subheader(f"Predicted Churn Probability Distribution — {model_choice}")
    prob_df = pd.DataFrame({
        "Probability": active_probs,
        "Actual": y_full.map({0: "No Churn", 1: "Churn"}).values,
    })
    fig_dist = px.histogram(prob_df, x="Probability", color="Actual",
                             nbins=50, barmode="overlay", opacity=0.75)
    fig_dist.add_vline(x=prob_thresh, line_dash="dash",
                        annotation_text=f"Threshold {prob_thresh}")
    fig_dist.update_layout(**base_layout(height=340))
    st.plotly_chart(fig_dist, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — CUSTOMER EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("Customer-Level Prediction Explorer")

    explore_df = X_full.copy()
    explore_df["Actual Churn"]          = y_full.values
    explore_df["Churn Probability"]     = active_probs.round(4)
    explore_df["Predicted (threshold)"] = active_preds

    fc1, fc2 = st.columns(2)
    with fc1:
        show_only = st.selectbox("Filter by prediction",
                                  ["All", "Predicted Churn", "Predicted No Churn",
                                   "Correct", "Incorrect"])
    with fc2:
        prob_min, prob_max = st.slider("Probability range", 0.0, 1.0, (0.0, 1.0), 0.01)

    filtered = explore_df[explore_df["Churn Probability"].between(prob_min, prob_max)]
    if show_only == "Predicted Churn":
        filtered = filtered[filtered["Predicted (threshold)"] == 1]
    elif show_only == "Predicted No Churn":
        filtered = filtered[filtered["Predicted (threshold)"] == 0]
    elif show_only == "Correct":
        filtered = filtered[filtered["Actual Churn"] == filtered["Predicted (threshold)"]]
    elif show_only == "Incorrect":
        filtered = filtered[filtered["Actual Churn"] != filtered["Predicted (threshold)"]]

    st.caption(f"Showing {len(filtered):,} of {len(explore_df):,} customers")

    num_cols = [c for c in X_full.columns if X_full[c].nunique() > 5][:2]
    if len(num_cols) >= 2:
        fig_scatter = px.scatter(
            filtered.reset_index(), x=num_cols[0], y=num_cols[1],
            color="Churn Probability", size="Churn Probability",
            hover_data=["Actual Churn", "Predicted (threshold)", "Churn Probability"],
            color_continuous_scale="RdBu_r", opacity=0.7,
        )
        fig_scatter.update_layout(**base_layout(height=380))
        st.plotly_chart(fig_scatter, use_container_width=True)

    st.dataframe(
        filtered.sort_values("Churn Probability", ascending=False)
                .head(200)
                .style.background_gradient(subset=["Churn Probability"], cmap="Blues"),
        use_container_width=True, height=380,
    )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — DATA OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.subheader("Data Overview")

    raw_display = raw_df.copy()
    raw_display.columns = raw_display.columns.str.strip().str.lower()

    if "churn" in raw_display.columns:
        churn_counts = raw_display["churn"].value_counts().reset_index()
        churn_counts.columns = ["Churn", "Count"]
        churn_counts["Churn"] = churn_counts["Churn"].map({0: "No Churn", 1: "Churn"})

        oc1, oc2 = st.columns(2)
        with oc1:
            fig_pie = px.pie(churn_counts, names="Churn", values="Count", hole=0.55)
            fig_pie.update_layout(**base_layout(height=320),
                                   legend=dict(orientation="h", y=-0.1, x=0.5, xanchor="center"))
            st.plotly_chart(fig_pie, use_container_width=True)

        with oc2:
            if "balance" in raw_display.columns:
                raw_display["Churn Label"] = raw_display["churn"].map({0: "No Churn", 1: "Churn"})
                fig_box = px.box(raw_display, x="Churn Label", y="balance",
                                  color="Churn Label", points="outliers")
                fig_box.update_layout(**base_layout(height=320), showlegend=False)
                st.plotly_chart(fig_box, use_container_width=True)

    st.subheader("Numeric Feature Distributions")
    num_features = raw_display.select_dtypes(include="number").columns.tolist()
    feat_to_plot = st.multiselect("Select features", num_features, default=num_features[:4])

    if feat_to_plot:
        n_cols = min(2, len(feat_to_plot))
        n_rows = -(-len(feat_to_plot) // n_cols)
        fig_hist = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=feat_to_plot)
        for i, feat in enumerate(feat_to_plot):
            r, c = divmod(i, n_cols)
            if "churn" in raw_display.columns:
                for churn_val, label in [(0, "No Churn"), (1, "Churn")]:
                    subset = raw_display[raw_display["churn"] == churn_val][feat].dropna()
                    fig_hist.add_trace(
                        go.Histogram(x=subset, name=label, opacity=0.7, showlegend=(i == 0)),
                        row=r+1, col=c+1,
                    )
            else:
                fig_hist.add_trace(
                    go.Histogram(x=raw_display[feat].dropna(), showlegend=False),
                    row=r+1, col=c+1,
                )
        fig_hist.update_layout(
            **base_layout(height=280 * n_rows), barmode="overlay",
            legend=dict(orientation="h", y=-0.08, x=0.5, xanchor="center"),
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    st.subheader("Correlation Heatmap")
    corr_cols = st.multiselect("Features for correlation", num_features, default=num_features[:10])
    if len(corr_cols) >= 2:
        fig_corr = px.imshow(
            raw_display[corr_cols].corr(), text_auto=".2f",
            color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
        )
        fig_corr.update_layout(**base_layout(height=500))
        st.plotly_chart(fig_corr, use_container_width=True)