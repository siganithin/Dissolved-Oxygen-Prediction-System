"""
Dissolved Oxygen Prediction System
Streamlit Cloud entry point — compatible with any package version.
"""
import sys, os, traceback
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Dissolved Oxygen Predictor",
    page_icon="💧",
    layout="wide",
)
st.markdown("""
<style>
.stApp {background: linear-gradient(180deg,#f0f8ff,#ffffff);}
h1,h2,h3 {color:#07204a;}
</style>""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("💧 Dissolved Oxygen Prediction System")
st.markdown("### Aquaculture Water Quality Management — BiSRU + Attention Model")
st.caption("Upload sensor CSV data to predict dissolved oxygen levels and monitor alerts.")
st.divider()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Controls")
    uploaded  = st.file_uploader("Upload sensor CSV", type=["csv"])
    st.markdown("---")
    st.markdown("**Required columns:**")
    st.code(
        "Temperature (cel)\n"
        "pH (ph units)\n"
        "Biochemical Oxygen Demand (mg/l)\n"
        "Ammonia (mg/l)\n"
        "Nitrate (mg/l)\n"
        "Nitrogen (mg/l)\n"
        "Dissolved Oxygen (mg/l)  ← optional"
    )
    st.markdown("---")
    threshold = st.slider("Alert threshold (mg/L)", 2.0, 8.0, 5.0, 0.5)
    show_raw  = st.checkbox("Show raw data preview", value=False)

# ── Load model once ───────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model...")
def load_model():
    from predict import load_artifacts
    return load_artifacts()

# ── No file uploaded ──────────────────────────────────────────────────────────
if uploaded is None:
    st.info("👈 Upload a CSV file from the sidebar to get started.")
    with st.expander("ℹ️ About this app"):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("""
**Model**
- Bidirectional SRU + Attention (BiSRU)
- 24-step temporal sequences
- Trained on aquaculture sensor data

**Feature selection**
- LightGBM selects top 4 features from: temperature, pH, BOD, ammonia, nitrate, nitrogen
""")
        with c2:
            st.markdown("""
**How to use**
1. Upload a CSV with the required columns
2. Adjust the alert threshold
3. Download predictions as CSV

**Alerts**
- 🔴 Alert: predicted DO < threshold
- ✅ Safe: predicted DO ≥ threshold
""")
    st.stop()

# ── Run prediction ────────────────────────────────────────────────────────────
file_bytes = uploaded.getvalue()

try:
    model, scaler, features = load_model()
except Exception as e:
    st.error(f"❌ Failed to load model: {e}")
    st.code(traceback.format_exc())
    st.stop()

try:
    from predict import predict
    with st.spinner("Running prediction..."):
        result = predict(file_bytes, model=model, scaler=scaler, features=features)
    result = result.copy()
    result["alert"] = result["predicted_DO"] < threshold
except ValueError as e:
    st.error(f"❌ {e}")
    st.stop()
except Exception as e:
    st.error(f"❌ Prediction error: {e}")
    st.code(traceback.format_exc())
    st.stop()

# ── Raw preview ───────────────────────────────────────────────────────────────
if show_raw:
    with st.expander("📄 Uploaded data preview"):
        import io
        st.dataframe(pd.read_csv(io.BytesIO(file_bytes)).head(50), use_container_width=True)

# ── Metrics ───────────────────────────────────────────────────────────────────
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
c5.metric("🚨 Alerts", f"{alerts:,}",
          delta=f"{alerts/total*100:.1f}% of readings",
          delta_color="inverse")

st.divider()

if alerts > 0:
    st.error(f"⚠️ **{alerts} readings** predicted below **{threshold} mg/L**. Immediate action may be required.")
else:
    st.success(f"✅ All {total:,} predictions are within the safe range (≥ {threshold} mg/L).")

st.divider()

# ── Charts ────────────────────────────────────────────────────────────────────
plot_df = result.reset_index(drop=True).iloc[:500]

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📈 Dissolved Oxygen Trend")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(plot_df.index, plot_df["predicted_DO"], label="Predicted DO",
            color="#1565C0", linewidth=1.8)
    if "actual_DO" in plot_df.columns:
        ax.plot(plot_df.index, plot_df["actual_DO"], label="Actual DO",
                color="#E65100", linewidth=1.4, alpha=0.8)
    ax.axhline(threshold, color="#C62828", linestyle="--", linewidth=1.2,
               label=f"Threshold ({threshold} mg/L)")
    ax.fill_between(plot_df.index, 0, threshold, color="#FFCDD2", alpha=0.3)
    ax.set_xlabel("Reading index")
    ax.set_ylabel("DO (mg/L)")
    ax.set_title("Predicted Dissolved Oxygen Over Time")
    ax.legend(frameon=False)
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

with col2:
    st.subheader("📊 Distribution")
    fig2, ax2 = plt.subplots(figsize=(5, 4))
    ax2.hist(plot_df["predicted_DO"], bins=25, color="#1E88E5", alpha=0.85, edgecolor="white")
    ax2.axvline(threshold, color="#C62828", linestyle="--", linewidth=1.2, label="Threshold")
    ax2.set_xlabel("Predicted DO (mg/L)")
    ax2.set_ylabel("Count")
    ax2.set_title("DO Distribution")
    ax2.legend(frameon=False)
    ax2.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)

if "actual_DO" in result.columns:
    st.subheader("🎯 Actual vs Predicted")
    fig3, ax3 = plt.subplots(figsize=(5, 4))
    ax3.scatter(plot_df["actual_DO"], plot_df["predicted_DO"],
                alpha=0.4, s=15, color="#1E88E5")
    mn = min(plot_df["actual_DO"].min(), plot_df["predicted_DO"].min())
    mx = max(plot_df["actual_DO"].max(), plot_df["predicted_DO"].max())
    ax3.plot([mn, mx], [mn, mx], "r--", linewidth=1)
    ax3.set_xlabel("Actual DO (mg/L)")
    ax3.set_ylabel("Predicted DO (mg/L)")
    ax3.set_title("Actual vs Predicted")
    ax3.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    _, sc, _ = st.columns([1, 2, 1])
    with sc:
        st.pyplot(fig3)
    plt.close(fig3)

st.divider()

# ── Feature importance ────────────────────────────────────────────────────────
img_path = os.path.join(ROOT, "data", "processed", "feature_importance.png")
if os.path.exists(img_path):
    with st.expander("📌 Feature Importance (LightGBM)", expanded=True):
        _, ic, _ = st.columns([1, 2, 1])
        with ic:
            st.image(img_path, caption="LightGBM feature importance")

# ── Predictions table ─────────────────────────────────────────────────────────
st.subheader("📋 Predictions Table")
disp = result.copy()
disp["status"] = disp["alert"].map({True: "⚠️ Alert", False: "✅ Safe"})
disp = disp.drop(columns=["alert"])
st.dataframe(disp, use_container_width=True, height=320)

st.download_button(
    "⬇️ Download predictions CSV",
    result.to_csv(index=False).encode(),
    file_name="do_predictions.csv",
    mime="text/csv",
)

st.divider()
st.caption("Dissolved Oxygen Prediction System · BiSRU + Attention · CVR College of Engineering")
