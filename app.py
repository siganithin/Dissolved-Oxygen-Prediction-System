"""
Dissolved Oxygen Prediction System — Streamlit Cloud entry point.
Run locally:   streamlit run app.py
Deploy:        push to GitHub, connect on share.streamlit.io
"""
import sys
import os

# Ensure src/ is on the import path (works from any working directory)
ROOT = os.path.dirname(os.path.abspath(__file__))
SRC  = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from predict import predict, load_artifacts

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Dissolved Oxygen Predictor",
    page_icon="💧",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp {background: linear-gradient(180deg,#f0f8ff,#ffffff);}
    h1, h2, h3 {color: #07204a;}
    .stButton>button {
        background: linear-gradient(90deg,#1e90ff,#0066ff);
        color: white;
        border: none;
        border-radius: 6px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("💧 Dissolved Oxygen Prediction System")
st.markdown("### Aquaculture Water Quality Management — BiSRU + Attention Model")
st.caption(
    "Upload sensor CSV data to predict dissolved oxygen levels, "
    "compare with actual readings, and monitor low-DO alerts."
)
st.divider()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Controls")
    uploaded = st.file_uploader("Upload sensor CSV", type=["csv"])
    st.markdown("---")
    st.markdown("**Required columns:**")
    st.code(
        "Temperature (cel)\n"
        "pH (ph units)\n"
        "Biochemical Oxygen Demand (mg/l)\n"
        "Ammonia (mg/l)\n"
        "Nitrate (mg/l)\n"
        "Nitrogen (mg/l)\n"
        "Dissolved Oxygen (mg/l)  ← optional (for comparison)"
    )
    st.markdown("---")
    threshold = st.slider("Alert threshold (mg/L)", 2.0, 8.0, 5.0, 0.5)
    show_raw  = st.checkbox("Show uploaded data preview", value=False)
    st.markdown("---")
    st.caption("Model loads once per session for faster subsequent predictions.")

# ── Model loading (cached) ─────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_model():
    return load_artifacts()


@st.cache_data(show_spinner=False)
def run_prediction(file_bytes: bytes) -> pd.DataFrame:
    model, scaler, features = get_model()
    return predict(file_bytes, model=model, scaler=scaler, features=features)


# ── No file state ─────────────────────────────────────────────────────────────
if uploaded is None:
    st.info("👈 Upload a CSV file from the sidebar to get started.")

    # Show a sample of what the app does
    with st.expander("ℹ️ About this application"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
**Model Architecture**
- Bidirectional Simple RNN (BiSRU) with Attention
- 24-step temporal sequences
- Trained on aquaculture water quality data

**Feature Selection**
- LightGBM selects top predictive features
- From: temperature, pH, BOD, ammonia, nitrate, nitrogen
""")
        with col2:
            st.markdown("""
**How to use**
1. Prepare a CSV with the required columns
2. Upload via the sidebar
3. Adjust the alert threshold
4. Download predictions as CSV

**Alert logic**
- 🔴 Alert: predicted DO < threshold
- ✅ Safe: predicted DO ≥ threshold
""")
    st.stop()

# ── Run prediction ────────────────────────────────────────────────────────────
with st.spinner("Running prediction..."):
    try:
        result = run_prediction(uploaded.getvalue()).copy()
        result["alert"] = result["predicted_DO"] < threshold
    except ValueError as e:
        st.error(f"❌ {e}")
        st.stop()
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
        st.stop()

# Show uploaded data preview
if show_raw:
    with st.expander("📄 Uploaded data preview"):
        raw_preview = pd.read_csv(uploaded)
        st.dataframe(raw_preview.head(50), use_container_width=True)

# ── Summary metrics ───────────────────────────────────────────────────────────
alerts   = int(result["alert"].sum())
total    = len(result)
avg_pred = float(result["predicted_DO"].mean())
min_pred = float(result["predicted_DO"].min())
max_pred = float(result["predicted_DO"].max())

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Rows analyzed",    f"{total:,}")
c2.metric("Avg predicted DO", f"{avg_pred:.2f} mg/L")
c3.metric("Min DO",           f"{min_pred:.2f} mg/L")
c4.metric("Max DO",           f"{max_pred:.2f} mg/L")
c5.metric("🚨 Alerts",        f"{alerts:,}",
          delta=f"{alerts/total*100:.1f}% of readings",
          delta_color="inverse")

st.divider()

# ── Alert banner ──────────────────────────────────────────────────────────────
if alerts > 0:
    st.error(
        f"⚠️ **{alerts} readings** predicted below the safe threshold of **{threshold} mg/L**. "
        "Immediate action may be required."
    )
else:
    st.success(f"✅ All {total:,} predicted DO levels are within the safe range (≥ {threshold} mg/L).")

st.divider()

# ── Charts ────────────────────────────────────────────────────────────────────
plot_df = result.reset_index(drop=True)
if len(plot_df) > 500:
    plot_df = plot_df.iloc[:500]

chart_col1, chart_col2 = st.columns([2, 1])

with chart_col1:
    st.subheader("📈 Dissolved Oxygen Trend")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(plot_df.index, plot_df["predicted_DO"],
            label="Predicted DO", color="#1565C0", linewidth=1.8)
    if "actual_DO" in plot_df.columns:
        ax.plot(plot_df.index, plot_df["actual_DO"],
                label="Actual DO", color="#E65100", linewidth=1.4, alpha=0.8)
    ax.axhline(threshold, color="#C62828", linestyle="--", linewidth=1.2, label=f"Alert threshold ({threshold})")
    ax.fill_between(plot_df.index, 0, threshold, color="#FFCDD2", alpha=0.35)
    ax.set_xlabel("Reading index")
    ax.set_ylabel("DO (mg/L)")
    ax.set_title("Predicted Dissolved Oxygen Over Time")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

with chart_col2:
    st.subheader("📊 Distribution")
    fig2, ax2 = plt.subplots(figsize=(5, 4))
    ax2.hist(plot_df["predicted_DO"], bins=25, color="#1E88E5", alpha=0.85, edgecolor="white")
    ax2.axvline(threshold, color="#C62828", linestyle="--", linewidth=1.2, label="Threshold")
    ax2.set_xlabel("Predicted DO (mg/L)")
    ax2.set_ylabel("Count")
    ax2.set_title("DO Distribution")
    ax2.legend(frameon=False)
    ax2.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)

# Actual vs Predicted scatter (only if actual values present)
if "actual_DO" in result.columns:
    st.subheader("🎯 Actual vs Predicted DO")
    fig3, ax3 = plt.subplots(figsize=(5, 4))
    ax3.scatter(plot_df["actual_DO"], plot_df["predicted_DO"],
                alpha=0.4, s=15, color="#1E88E5")
    mn = min(plot_df["actual_DO"].min(), plot_df["predicted_DO"].min())
    mx = max(plot_df["actual_DO"].max(), plot_df["predicted_DO"].max())
    ax3.plot([mn, mx], [mn, mx], "r--", linewidth=1)
    ax3.set_xlabel("Actual DO (mg/L)")
    ax3.set_ylabel("Predicted DO (mg/L)")
    ax3.set_title("Actual vs Predicted")
    ax3.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    _, scatter_col, _ = st.columns([1, 2, 1])
    with scatter_col:
        st.pyplot(fig3)
    plt.close(fig3)

st.divider()

# ── Feature importance image ───────────────────────────────────────────────────
importance_path = os.path.join(ROOT, "data", "processed", "feature_importance.png")
if os.path.exists(importance_path):
    with st.expander("📌 Feature Importance (LightGBM)", expanded=True):
        _, img_col, _ = st.columns([1, 2, 1])
        with img_col:
            st.image(importance_path, caption="LightGBM feature importance scores")

# ── Predictions table ─────────────────────────────────────────────────────────
st.subheader("📋 Predictions Table")
display_df = result.copy()
display_df["status"] = display_df["alert"].map({True: "⚠️ Alert", False: "✅ Safe"})
display_df = display_df.drop(columns=["alert"])
st.dataframe(display_df, use_container_width=True, height=320)

dl_col1, dl_col2 = st.columns([1, 3])
with dl_col1:
    st.download_button(
        "⬇️ Download predictions CSV",
        result.to_csv(index=False).encode(),
        file_name="do_predictions.csv",
        mime="text/csv",
    )

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "Dissolved Oxygen Prediction System · BiSRU + Attention · "
    "Built for aquaculture water quality management"
)
