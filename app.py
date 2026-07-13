import sys, os, io, json, warnings, traceback
warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))

import streamlit as st
st.set_page_config(page_title="DO Prediction System", page_icon="💧", layout="wide")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ══════════════════════════════════════════════════════════════
#  CUSTOM CSS — Beautiful dark-blue aquatic theme
# ══════════════════════════════════════════════════════════════
st.markdown("""
<style>
/* Main background */
.stApp { background: linear-gradient(160deg, #0a1628 0%, #0d2137 40%, #0a2e4a 100%); }

/* All text white by default */
html, body, [class*="css"] { color: #e8f4fd; }

/* Title */
h1 { color: #00d4ff !important; font-size: 2.4rem !important;
     text-shadow: 0 0 20px rgba(0,212,255,0.4); }
h2, h3 { color: #7ecfff !important; }

/* Metric cards */
[data-testid="metric-container"] {
    background: linear-gradient(135deg, rgba(0,100,180,0.35), rgba(0,50,120,0.35));
    border: 1px solid rgba(0,212,255,0.25);
    border-radius: 12px; padding: 16px 20px;
    backdrop-filter: blur(6px);
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
}
[data-testid="metric-container"] label { color: #7ecfff !important; font-size:0.8rem; }
[data-testid="metric-container"] [data-testid="metric-value"] {
    color: #00d4ff !important; font-size: 1.6rem !important; font-weight: 700;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #071525, #0d2137) !important;
    border-right: 1px solid rgba(0,212,255,0.15);
}
[data-testid="stSidebar"] label { color: #7ecfff !important; }

/* Buttons */
.stButton>button, .stDownloadButton>button {
    background: linear-gradient(90deg, #0066cc, #00aaff);
    color: white !important; border: none; border-radius: 8px;
    padding: 0.5rem 1.5rem; font-weight: 600;
    box-shadow: 0 4px 15px rgba(0,170,255,0.3);
    transition: all 0.2s;
}
.stButton>button:hover { transform: translateY(-1px); box-shadow: 0 6px 20px rgba(0,170,255,0.5); }

/* File uploader */
[data-testid="stFileUploader"] {
    background: rgba(0,100,180,0.15);
    border: 2px dashed rgba(0,212,255,0.4);
    border-radius: 12px; padding: 8px;
}

/* Success / error boxes */
.stSuccess { background: rgba(0,180,100,0.15) !important; border-color: #00cc77 !important; }
.stError   { background: rgba(220,50,50,0.15)  !important; border-color: #ff4444 !important; }
.stInfo    { background: rgba(0,120,200,0.15)  !important; border-color: #00aaff !important; }

/* Divider */
hr { border-color: rgba(0,212,255,0.15) !important; }

/* Dataframe */
.stDataFrame { border: 1px solid rgba(0,212,255,0.2); border-radius: 10px; }

/* Expander */
.streamlit-expanderHeader { color: #7ecfff !important; }

/* Code block */
.stCode { background: rgba(0,30,60,0.6) !important; }

/* Slider */
.stSlider [data-baseweb="slider"] div[role="slider"] {
    background-color: #00d4ff !important;
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  CONSTANTS
# ══════════════════════════════════════════════════════════════
SEQ   = 24
FEATS = ["temperature","pH","BOD","ammonia","nitrate","nitrogen"]
SCOLS = FEATS + ["dissolved_oxygen"]
CMAP  = {
    "Temperature (cel)":"temperature",
    "pH (ph units)":"pH",
    "Biochemical Oxygen Demand (mg/l)":"BOD",
    "Ammonia (mg/l)":"ammonia",
    "Nitrate (mg/l)":"nitrate",
    "Nitrogen (mg/l)":"nitrogen",
    "Dissolved Oxygen (mg/l)":"dissolved_oxygen",
}

# ══════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 💧 DO Predictor")
    st.markdown("---")
    uploaded = st.file_uploader("📂 Upload Sensor CSV", type=["csv"])
    st.markdown("---")
    threshold = st.slider("🚨 Alert Threshold (mg/L)", 2.0, 8.0, 5.0, 0.5,
                          help="Readings below this value trigger an alert")
    st.markdown("---")
    show_raw = st.checkbox("Show raw data preview")
    st.markdown("---")
    st.markdown("**Required columns:**")
    st.code("Temperature (cel)\npH (ph units)\nBOD (mg/l)\nAmmonia (mg/l)\nNitrate (mg/l)\nNitrogen (mg/l)\nDissolved Oxygen (mg/l)")
    st.markdown("---")
    st.markdown("""
<small style='color:#4a9abb'>
🎓 CVR College of Engineering<br>
B.Tech CSE — April 2026<br>
BiSRU + Attention Model
</small>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  HEADER
# ══════════════════════════════════════════════════════════════
col_title, col_badge = st.columns([4, 1])
with col_title:
    st.title("💧 Dissolved Oxygen Prediction System")
    st.markdown(
        "<p style='color:#7ecfff;font-size:1.1rem;margin-top:-10px'>"
        "Aquaculture Water Quality Management &nbsp;·&nbsp; "
        "<b style='color:#00d4ff'>BiSRU + Attention Neural Network</b></p>",
        unsafe_allow_html=True
    )
with col_badge:
    st.markdown("""
<div style='background:rgba(0,212,255,0.1);border:1px solid rgba(0,212,255,0.3);
border-radius:10px;padding:12px;text-align:center;margin-top:8px'>
<div style='color:#00d4ff;font-size:1.8rem'>🐠</div>
<div style='color:#7ecfff;font-size:0.75rem'>Aquaculture<br>AI System</div>
</div>""", unsafe_allow_html=True)
st.divider()

# ══════════════════════════════════════════════════════════════
#  MODEL LOADER
# ══════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="🔄 Loading BiSRU model...")
def load_model():
    import onnxruntime as rt
    sess  = rt.InferenceSession(os.path.join(ROOT,"models","bisru_model.onnx"))
    dmin  = np.load(os.path.join(ROOT,"models","scaler_min.npy"))
    dscl  = np.load(os.path.join(ROOT,"models","scaler_scale.npy"))
    with open(os.path.join(ROOT,"models","selected_features.json")) as f:
        feats = json.load(f)
    return sess, dmin, dscl, feats

# ══════════════════════════════════════════════════════════════
#  LANDING PAGE
# ══════════════════════════════════════════════════════════════
if uploaded is None:
    # Pipeline overview
    st.markdown("### 🔬 System Pipeline")
    p1,p2,p3,p4,p5 = st.columns(5)
    steps = [
        ("📥","Data Upload","Upload sensor CSV with water quality readings"),
        ("🧹","Preprocess","Clean, fill nulls, MinMax normalize"),
        ("⚡","LightGBM","Select top 4 influential features"),
        ("🧠","BiSRU+Attn","24-step sequence deep learning prediction"),
        ("📊","Dashboard","Visualize trends & trigger DO alerts"),
    ]
    for col, (icon, title, desc) in zip([p1,p2,p3,p4,p5], steps):
        col.markdown(f"""
<div style='background:rgba(0,100,180,0.2);border:1px solid rgba(0,212,255,0.2);
border-radius:10px;padding:14px;text-align:center;height:130px'>
<div style='font-size:1.8rem'>{icon}</div>
<div style='color:#00d4ff;font-weight:600;font-size:0.85rem'>{title}</div>
<div style='color:#7ecfff;font-size:0.72rem;margin-top:4px'>{desc}</div>
</div>""", unsafe_allow_html=True)

    st.divider()

    # Model info cards
    st.markdown("### 🏗️ Model Architecture")
    m1,m2,m3 = st.columns(3)
    cards = [
        ("🔁","BiSRU Layer","Bidirectional Simple RNN captures temporal patterns in both forward & backward directions across 24 time steps"),
        ("🎯","Attention Mechanism","Assigns importance weights to each time step — focuses on the most critical historical readings"),
        ("⚡","LightGBM Selector","Ranks all 6 water parameters and selects top 4 features, reducing noise and improving speed"),
    ]
    for col,(icon,title,desc) in zip([m1,m2,m3],cards):
        col.markdown(f"""
<div style='background:rgba(0,80,160,0.25);border:1px solid rgba(0,212,255,0.2);
border-radius:12px;padding:18px;height:160px'>
<div style='font-size:1.5rem'>{icon}</div>
<div style='color:#00d4ff;font-weight:700;margin:6px 0 8px'>{title}</div>
<div style='color:#aac8e0;font-size:0.82rem;line-height:1.4'>{desc}</div>
</div>""", unsafe_allow_html=True)

    st.divider()
    st.info("👈 **Upload a CSV file from the sidebar to start predicting dissolved oxygen levels.**")
    st.stop()

# ══════════════════════════════════════════════════════════════
#  PREDICTION
# ══════════════════════════════════════════════════════════════
try:
    sess, dmin, dscl, sel_feats = load_model()
except Exception as e:
    st.error(f"Model load failed: {e}"); st.code(traceback.format_exc()); st.stop()

try:
    with st.spinner("🧠 Running BiSRU prediction..."):
        raw   = uploaded.getvalue()
        df    = pd.read_csv(io.BytesIO(raw), low_memory=False)
        df    = df.rename(columns=CMAP)
        has_do = "dissolved_oxygen" in df.columns
        need  = FEATS + (["dissolved_oxygen"] if has_do else [])
        df    = df[[c for c in need if c in df.columns]].copy()

        for c in FEATS:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
                df[c] = df[c].fillna(float(df[c].median()))

        if len(df) < SEQ + 1:
            st.error(f"Need at least {SEQ+1} rows. Got {len(df)}."); st.stop()

        cols = FEATS + (["dissolved_oxygen"] if has_do else [])
        arr  = df[cols].values.astype(np.float64)
        for i, c in enumerate(cols):
            idx = SCOLS.index(c)
            arr[:, i] = (arr[:, i] - dmin[idx]) * dscl[idx]

        fi  = [cols.index(f) for f in sel_feats]
        fa  = arr[:, fi].astype(np.float32)
        X   = np.stack([fa[i:i+SEQ] for i in range(len(fa)-SEQ)]).astype(np.float32)
        pn  = sess.run(None, {sess.get_inputs()[0].name: X})[0].flatten()
        di  = SCOLS.index("dissolved_oxygen")
        pmg = pn / dscl[di] + dmin[di]

        res = pd.DataFrame({
            "index": np.arange(len(pmg)),
            "predicted_DO": np.round(pmg.astype(float), 3),
            "alert": pmg < threshold,
        })
        if has_do:
            am = arr[SEQ:, cols.index("dissolved_oxygen")] / dscl[di] + dmin[di]
            res["actual_DO"] = np.round(am.astype(float), 3)

except Exception as e:
    st.error(f"Prediction error: {e}"); st.code(traceback.format_exc()); st.stop()

# ══════════════════════════════════════════════════════════════
#  METRICS
# ══════════════════════════════════════════════════════════════
alerts = int(res["alert"].sum()); total = len(res)
pct    = alerts / total * 100

st.markdown("### 📊 Prediction Summary")
c1,c2,c3,c4,c5 = st.columns(5)
c1.metric("📋 Rows Analyzed",  f"{total:,}")
c2.metric("📈 Avg Predicted DO", f"{res['predicted_DO'].mean():.2f} mg/L")
c3.metric("📉 Min DO",          f"{res['predicted_DO'].min():.2f} mg/L")
c4.metric("📊 Max DO",          f"{res['predicted_DO'].max():.2f} mg/L")
c5.metric("🚨 Alerts",          f"{alerts:,}", delta=f"{pct:.1f}% critical", delta_color="inverse")

st.divider()

if alerts > 0:
    st.error(f"⚠️ **{alerts} readings ({pct:.1f}%)** are predicted below the safe threshold of **{threshold} mg/L**. Immediate aeration action recommended.")
else:
    st.success(f"✅ All **{total:,}** predictions are within the safe dissolved oxygen range (≥ {threshold} mg/L).")

st.divider()

# Raw preview
if show_raw:
    with st.expander("📄 Raw uploaded data"):
        st.dataframe(pd.read_csv(io.BytesIO(raw)).head(50), use_container_width=True)

# ══════════════════════════════════════════════════════════════
#  CHARTS  — dark theme
# ══════════════════════════════════════════════════════════════
DARK_BG   = "#0d1f33"
GRID_COL  = "#1a3a5c"
TEXT_COL  = "#7ecfff"

def apply_dark(ax):
    ax.set_facecolor(DARK_BG)
    ax.figure.patch.set_facecolor(DARK_BG)
    ax.tick_params(colors=TEXT_COL)
    ax.xaxis.label.set_color(TEXT_COL)
    ax.yaxis.label.set_color(TEXT_COL)
    ax.title.set_color("#00d4ff")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GRID_COL)
    ax.spines["bottom"].set_color(GRID_COL)
    ax.yaxis.set_tick_params(color=GRID_COL)
    ax.xaxis.set_tick_params(color=GRID_COL)
    ax.grid(True, color=GRID_COL, alpha=0.5, linewidth=0.5)

p = res.reset_index(drop=True).iloc[:600]

st.markdown("### 📈 Dissolved Oxygen Trends")
ch1, ch2 = st.columns([3, 1])

with ch1:
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(p.index, p["predicted_DO"], label="Predicted DO",
            color="#00aaff", lw=2, zorder=3)
    if "actual_DO" in p.columns:
        ax.plot(p.index, p["actual_DO"], label="Actual DO",
                color="#ff8c42", lw=1.5, alpha=0.85, zorder=2)
    ax.axhline(threshold, color="#ff4444", ls="--", lw=1.5,
               label=f"Alert threshold ({threshold} mg/L)", zorder=4)
    ax.fill_between(p.index, 0, threshold, color="#ff2222", alpha=0.07)
    ax.fill_between(p.index, threshold, p["predicted_DO"],
                    where=p["predicted_DO"] >= threshold,
                    color="#00aaff", alpha=0.08)
    ax.set_xlabel("Reading Index", fontsize=10)
    ax.set_ylabel("DO (mg/L)", fontsize=10)
    ax.set_title("Dissolved Oxygen Prediction Over Time", fontsize=12, fontweight="bold")
    leg = ax.legend(frameon=True, facecolor="#0a1e30", edgecolor="#1a3a5c",
                    labelcolor=TEXT_COL, fontsize=9)
    apply_dark(ax)
    plt.tight_layout()
    st.pyplot(fig); plt.close(fig)

with ch2:
    fig2, ax2 = plt.subplots(figsize=(4, 4))
    n, bins, patches = ax2.hist(p["predicted_DO"], bins=22,
                                 color="#00aaff", alpha=0.85, edgecolor="#0d1f33")
    # Colour bars below threshold red
    for patch, left in zip(patches, bins[:-1]):
        if left < threshold:
            patch.set_facecolor("#ff4444")
    ax2.axvline(threshold, color="#ff4444", ls="--", lw=1.5)
    ax2.set_xlabel("Predicted DO (mg/L)", fontsize=9)
    ax2.set_ylabel("Count", fontsize=9)
    ax2.set_title("DO Distribution", fontsize=11, fontweight="bold")
    apply_dark(ax2)
    plt.tight_layout(); st.pyplot(fig2); plt.close(fig2)

# Actual vs Predicted scatter
if "actual_DO" in res.columns:
    st.markdown("### 🎯 Actual vs Predicted Analysis")
    sc1, sc2, sc3 = st.columns([1, 2, 1])
    with sc2:
        fig3, ax3 = plt.subplots(figsize=(6, 5))
        sc = ax3.scatter(p["actual_DO"], p["predicted_DO"],
                         c=p["predicted_DO"], cmap="cool",
                         alpha=0.5, s=18, zorder=3)
        mn = min(float(p["actual_DO"].min()), float(p["predicted_DO"].min()))
        mx = max(float(p["actual_DO"].max()), float(p["predicted_DO"].max()))
        ax3.plot([mn, mx], [mn, mx], "r--", lw=1.5, label="Perfect prediction", zorder=4)
        cbar = plt.colorbar(sc, ax=ax3)
        cbar.ax.yaxis.set_tick_params(color=TEXT_COL)
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color=TEXT_COL)
        cbar.set_label("Predicted DO (mg/L)", color=TEXT_COL)
        ax3.set_xlabel("Actual DO (mg/L)"); ax3.set_ylabel("Predicted DO (mg/L)")
        ax3.set_title("Actual vs Predicted DO", fontsize=12, fontweight="bold")
        leg3 = ax3.legend(frameon=True, facecolor="#0a1e30", edgecolor="#1a3a5c",
                          labelcolor=TEXT_COL, fontsize=9)
        apply_dark(ax3)
        plt.tight_layout(); st.pyplot(fig3); plt.close(fig3)

st.divider()

# ══════════════════════════════════════════════════════════════
#  FEATURE IMPORTANCE
# ══════════════════════════════════════════════════════════════
img_path = os.path.join(ROOT,"data","processed","feature_importance.png")
if os.path.exists(img_path):
    with st.expander("⚡ Feature Importance — LightGBM Selection", expanded=True):
        fi1, fi2 = st.columns([1, 2])
        with fi1:
            st.markdown("""
<div style='background:rgba(0,100,180,0.2);border:1px solid rgba(0,212,255,0.2);
border-radius:10px;padding:16px;margin-top:8px'>
<div style='color:#00d4ff;font-weight:700;margin-bottom:8px'>Selected Features</div>
<div style='color:#aac8e0;font-size:0.85rem;line-height:2'>
🔵 pH<br>🔵 Temperature<br>🔵 Ammonia<br>🔵 Nitrate<br>
<span style='color:#4a8a9a'>⬜ BOD (not selected)</span><br>
<span style='color:#4a8a9a'>⬜ Nitrogen (not selected)</span>
</div>
</div>""", unsafe_allow_html=True)
        with fi2:
            st.image(img_path, caption="LightGBM Feature Importance Scores")

# ══════════════════════════════════════════════════════════════
#  PREDICTIONS TABLE
# ══════════════════════════════════════════════════════════════
st.markdown("### 📋 Full Predictions Table")
disp = res.copy()
disp["status"] = disp["alert"].map({True:"⚠️ Alert",False:"✅ Safe"})
disp = disp.drop(columns=["alert"])
st.dataframe(disp, use_container_width=True, height=320)

dl1, dl2, _ = st.columns([1, 1, 3])
with dl1:
    st.download_button("⬇️ Download Predictions CSV",
                       res.to_csv(index=False).encode(),
                       "do_predictions.csv", "text/csv")
with dl2:
    if show_raw:
        st.download_button("⬇️ Download Raw CSV", raw,
                           "raw_data.csv", "text/csv")

st.divider()
st.markdown("""
<div style='text-align:center;color:#4a8aaa;font-size:0.8rem;padding:10px'>
💧 Dissolved Oxygen Prediction System &nbsp;·&nbsp; BiSRU + Attention &nbsp;·&nbsp;
CVR College of Engineering &nbsp;·&nbsp; B.Tech CSE 2026
</div>""", unsafe_allow_html=True)
