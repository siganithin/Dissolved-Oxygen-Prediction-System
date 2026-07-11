import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

sys.path.append(os.path.dirname(__file__))
from predict import predict, load_artifacts

st.set_page_config(
    page_title="Dissolved Oxygen Predictor",
    page_icon="💧",
    layout="wide",
)

# --- Page styling: subtle blue gradient header and professional spacing ---
st.markdown(
    """
    <style>
    .stApp > header {background: linear-gradient(90deg,#0f62fe,#0052d4);}
    .stApp {background: linear-gradient(180deg,#f6fbff,#ffffff);}
    .css-1aumxhk {padding-top: 0.5rem;}
    h1, h2, h3 {color: #07204a}
    .stButton>button {background: linear-gradient(90deg,#1e90ff,#0066ff); color: white}
    </style>
    """,
    unsafe_allow_html=True,
)

import threading

# ── Header ───────────────────────────────────────────────────────────────────
st.title("Dissolved Oxygen Prediction System")
st.markdown("### Aquaculture Water Quality Management with BiSRU + Attention")
st.caption(
    "Upload sensor data, compare predicted and actual DO values, and monitor alerts in one interactive dashboard."
)
st.divider()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Upload & settings")
    uploaded = st.file_uploader("Upload your sensor CSV", type=["csv"])
    st.markdown("---")
    st.markdown("**Required columns:**")
    st.code(
        "Temperature (cel)\npH (ph units)\nBiochemical Oxygen Demand (mg/l)\n"
        "Ammonia (mg/l)\nNitrate (mg/l)\nNitrogen (mg/l)\nDissolved Oxygen (mg/l)",
    )
    threshold = st.slider(
        "Alert threshold (mg/L)", 2.0, 8.0, 5.0, 0.5
    )
    show_raw = st.checkbox("Show uploaded data preview", value=False)
    st.markdown("---")
    st.caption("The model and scaler load once per session to improve dashboard responsiveness.")

# ── Main ──────────────────────────────────────────────────────────────────────
if uploaded is None:
    st.info("Upload a CSV file from the sidebar to get started.")
    st.stop()

@st.cache_resource(show_spinner=False)
def load_artifacts_cached():
    return load_artifacts()

@st.cache_data(show_spinner=False)
def run_cached_prediction(file_bytes):
    tmp_dir = os.path.join("data", "processed")
    tmp_path = os.path.join(tmp_dir, "_upload_tmp.csv")
    os.makedirs(tmp_dir, exist_ok=True)
    with open(tmp_path, "wb") as f:
        f.write(file_bytes)

    model, scaler, features = load_artifacts_cached()
    return predict(tmp_path, model=model, scaler=scaler, features=features)

with st.spinner("Loading model and generating predictions..."):
    try:
        result = run_cached_prediction(uploaded.getvalue()).copy()
        result["alert"] = result["predicted_DO"] < threshold
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

# Kick off model preload in background (only once per session)
if "model_loading_started" not in st.session_state:
    st.session_state["model_loading_started"] = True

    def _preload():
        try:
            load_artifacts_cached()
            st.session_state["model_loaded"] = True
        except Exception as exc:
            st.session_state["model_error"] = str(exc)

    threading.Thread(target=_preload, daemon=True).start()

# Show model load status in sidebar
with st.sidebar:
    if st.session_state.get("model_loaded"):
        st.success("Model loaded and ready")
    elif st.session_state.get("model_error"):
        st.error("Model load error: " + st.session_state.get("model_error"))
    else:
        st.info("Model warming up in background — first prediction may take longer")

alerts = int(result["alert"].sum())
total = int(len(result))
avg_pred = float(result["predicted_DO"].mean())
min_pred = float(result["predicted_DO"].min())
max_pred = float(result["predicted_DO"].max())

# ── Metric cards ──────────────────────────────────────────────────────────────
metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
metric_col1.metric("Rows analyzed", f"{total:,}")
metric_col2.metric("Average DO", f"{avg_pred:.2f} mg/L")
metric_col3.metric("Min predicted DO", f"{min_pred:.2f} mg/L")
metric_col4.metric("Alerts", f"{alerts:,}", delta=f"{alerts/total*100:.1f}%")

st.divider()

summary_left, summary_right = st.columns([3, 1])
with summary_left:
    if alerts > 0:
        st.error(
            f"⚠️ {alerts} readings predicted below the threshold of {threshold} mg/L."
        )
    else:
        st.success("All predicted DO levels are within the safe range.")

    st.write(
        "This dashboard uses the BiSRU + Attention prediction model, with input features selected from water quality sensor data. "
        "The charts below show predicted DO trends, alert distribution, and model insights."
    )

with summary_right:
    st.subheader("Prediction insights")
    st.write("- Model predicts DO using temporal sequences of past data.")
    st.write("- Alerts are triggered when predicted DO falls below the selected threshold.")
    st.write("- Actual DO values allow prediction accuracy comparison.")

st.divider()

# ── Trend charts ─────────────────────────────────────────────────────────────
trend_col1, trend_col2 = st.columns(2)
plot_df = result.reset_index(drop=True)
if len(plot_df) > 500:
    plot_df = plot_df.iloc[:500]

with trend_col1:
    st.subheader("Predicted DO trend")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(plot_df.index, plot_df["predicted_DO"], label="Predicted DO", color="#0066CC", linewidth=1.9)
    if "actual_DO" in plot_df.columns:
        ax.plot(plot_df.index, plot_df["actual_DO"], label="Actual DO", color="#FF7F0E", linewidth=1.5, alpha=0.8)
    ax.axhline(threshold, color="#D62728", linestyle="--", linewidth=1.3)
    ax.fill_between(plot_df.index, 0, threshold, color="#D62728", alpha=0.1)
    ax.set_xlabel("Reading index")
    ax.set_ylabel("DO (mg/L)")
    ax.set_title("Dissolved Oxygen Prediction Trend")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

with trend_col2:
    st.subheader("Prediction distribution")
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.hist(plot_df["predicted_DO"], bins=20, color="#1F77B4", alpha=0.8)
    ax2.axvline(threshold, color="#D62728", linestyle="--", linewidth=1.3)
    ax2.set_xlabel("Predicted DO (mg/L)")
    ax2.set_ylabel("Count")
    ax2.set_title("Predicted DO Histogram")
    ax2.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)

st.divider()

# (Removed duplicate large trend plot; charts above present the trends already.)

# ── Feature importance chart ──────────────────────────────────────────────────
importance_path = "data/processed/feature_importance.png"
if os.path.exists(importance_path):
    st.subheader("Feature importance (LightGBM)")
    st.image(importance_path, width=520)

# ── Raw predictions table ─────────────────────────────────────────────────────
st.subheader("Predictions table")
display_df = result.copy()
display_df["status"] = display_df["alert"].map({True: "⚠️ Alert", False: "✅ Safe"})
display_df = display_df.drop(columns=["alert"])
st.dataframe(display_df, use_container_width=True, height=300)

# Download button
csv_out = result.to_csv(index=False).encode()
st.download_button("Download predictions CSV", csv_out,
                   file_name="do_predictions.csv", mime="text/csv")