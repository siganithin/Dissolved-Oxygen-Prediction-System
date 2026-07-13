import sys, os, traceback, io, json, warnings
warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))

import streamlit as st
st.set_page_config(page_title="DO Predictor", page_icon="💧", layout="wide")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Controls")
    uploaded  = st.file_uploader("Upload sensor CSV", type=["csv"])
    st.markdown("---")
    st.markdown("**Required columns:**")
    st.code("Temperature (cel)\npH (ph units)\nBiochemical Oxygen Demand (mg/l)\nAmmonia (mg/l)\nNitrate (mg/l)\nNitrogen (mg/l)\nDissolved Oxygen (mg/l)  ← optional")
    st.markdown("---")
    threshold = st.slider("Alert threshold (mg/L)", 2.0, 8.0, 5.0, 0.5)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("💧 Dissolved Oxygen Prediction System")
st.markdown("### Aquaculture Water Quality — BiSRU + Attention Model")
st.divider()

# ── Constants ─────────────────────────────────────────────────────────────────
SEQ   = 24
THR   = 5.0
FEATS = ["temperature","pH","BOD","ammonia","nitrate","nitrogen"]
SCOLS = FEATS + ["dissolved_oxygen"]
CMAP  = {
    "Temperature (cel)":"temperature","pH (ph units)":"pH",
    "Biochemical Oxygen Demand (mg/l)":"BOD","Ammonia (mg/l)":"ammonia",
    "Nitrate (mg/l)":"nitrate","Nitrogen (mg/l)":"nitrogen",
    "Dissolved Oxygen (mg/l)":"dissolved_oxygen",
}

# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model...")
def load_model():
    import onnxruntime as rt
    sess  = rt.InferenceSession(os.path.join(ROOT,"models","bisru_model.onnx"))
    dmin  = np.load(os.path.join(ROOT,"models","scaler_min.npy"))
    dscl  = np.load(os.path.join(ROOT,"models","scaler_scale.npy"))
    with open(os.path.join(ROOT,"models","selected_features.json")) as f:
        feats = json.load(f)
    return sess, dmin, dscl, feats

# ── No file ───────────────────────────────────────────────────────────────────
if uploaded is None:
    st.info("👈 Upload a CSV file from the sidebar to begin.")
    st.stop()

# ── Predict ───────────────────────────────────────────────────────────────────
try:
    sess, dmin, dscl, sel_feats = load_model()
except Exception as e:
    st.error(f"Model load failed: {e}")
    st.code(traceback.format_exc()); st.stop()

try:
    with st.spinner("Running prediction..."):
        df = pd.read_csv(io.BytesIO(uploaded.getvalue()), low_memory=False)
        df = df.rename(columns=CMAP)
        has_do = "dissolved_oxygen" in df.columns
        need   = FEATS + (["dissolved_oxygen"] if has_do else [])
        df     = df[[c for c in need if c in df.columns]].copy()
        for c in FEATS:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(df[c].median())
        if len(df) < SEQ+1:
            st.error(f"Need at least {SEQ+1} rows. Got {len(df)}."); st.stop()

        cols = FEATS + (["dissolved_oxygen"] if has_do else [])
        arr  = df[cols].values.astype(np.float64)
        for i,c in enumerate(cols):
            idx = SCOLS.index(c)
            arr[:,i] = (arr[:,i] - dmin[idx]) * dscl[idx]

        fi   = [cols.index(f) for f in sel_feats]
        fa   = arr[:,fi].astype(np.float32)
        X    = np.stack([fa[i:i+SEQ] for i in range(len(fa)-SEQ)]).astype(np.float32)
        pn   = sess.run(None,{sess.get_inputs()[0].name:X})[0].flatten()
        di   = SCOLS.index("dissolved_oxygen")
        pmg  = pn / dscl[di] + dmin[di]

        res  = pd.DataFrame({"index":np.arange(len(pmg)),
                              "predicted_DO":np.round(pmg.astype(float),3),
                              "alert":pmg < threshold})
        if has_do:
            am = arr[SEQ:, cols.index("dissolved_oxygen")] / dscl[di] + dmin[di]
            res["actual_DO"] = np.round(am.astype(float),3)
except Exception as e:
    st.error(f"Prediction error: {e}")
    st.code(traceback.format_exc()); st.stop()

# ── Metrics ───────────────────────────────────────────────────────────────────
alerts = int(res["alert"].sum()); total = len(res)
c1,c2,c3,c4,c5 = st.columns(5)
c1.metric("Rows",f"{total:,}"); c2.metric("Avg DO",f"{res['predicted_DO'].mean():.2f} mg/L")
c3.metric("Min DO",f"{res['predicted_DO'].min():.2f} mg/L"); c4.metric("Max DO",f"{res['predicted_DO'].max():.2f} mg/L")
c5.metric("🚨 Alerts",f"{alerts:,}",delta=f"{alerts/total*100:.1f}%",delta_color="inverse")
st.divider()
if alerts>0: st.error(f"⚠️ {alerts} readings below {threshold} mg/L.")
else: st.success(f"✅ All {total:,} readings within safe range.")
st.divider()

# ── Charts ────────────────────────────────────────────────────────────────────
p = res.reset_index(drop=True).iloc[:500]
c1,c2 = st.columns([2,1])

with c1:
    st.subheader("📈 DO Trend")
    fig,ax = plt.subplots(figsize=(10,4))
    ax.plot(p.index, p["predicted_DO"], label="Predicted", color="#1565C0", lw=1.8)
    if "actual_DO" in p.columns:
        ax.plot(p.index, p["actual_DO"], label="Actual", color="#E65100", lw=1.4, alpha=0.8)
    ax.axhline(threshold, color="#C62828", ls="--", lw=1.2, label=f"Threshold {threshold}")
    ax.fill_between(p.index,0,threshold,color="#FFCDD2",alpha=0.3)
    ax.set_xlabel("Index"); ax.set_ylabel("DO (mg/L)")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout(); st.pyplot(fig); plt.close(fig)

with c2:
    st.subheader("📊 Distribution")
    fig2,ax2 = plt.subplots(figsize=(5,4))
    ax2.hist(p["predicted_DO"],bins=25,color="#1E88E5",alpha=0.85,edgecolor="white")
    ax2.axvline(threshold,color="#C62828",ls="--",lw=1.2)
    ax2.set_xlabel("Predicted DO (mg/L)"); ax2.set_ylabel("Count")
    ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)
    plt.tight_layout(); st.pyplot(fig2); plt.close(fig2)

if "actual_DO" in res.columns:
    st.subheader("🎯 Actual vs Predicted")
    fig3,ax3 = plt.subplots(figsize=(5,4))
    ax3.scatter(p["actual_DO"],p["predicted_DO"],alpha=0.4,s=15,color="#1E88E5")
    mn=min(float(p["actual_DO"].min()),float(p["predicted_DO"].min()))
    mx=max(float(p["actual_DO"].max()),float(p["predicted_DO"].max()))
    ax3.plot([mn,mx],[mn,mx],"r--",lw=1)
    ax3.set_xlabel("Actual DO"); ax3.set_ylabel("Predicted DO")
    ax3.spines["top"].set_visible(False); ax3.spines["right"].set_visible(False)
    plt.tight_layout()
    _,sc,_ = st.columns([1,2,1])
    with sc: st.pyplot(fig3)
    plt.close(fig3)

st.divider()

# ── Feature importance ────────────────────────────────────────────────────────
img = os.path.join(ROOT,"data","processed","feature_importance.png")
if os.path.exists(img):
    with st.expander("📌 Feature Importance",expanded=True):
        _,ic,_ = st.columns([1,2,1])
        with ic: st.image(img)

# ── Table + download ──────────────────────────────────────────────────────────
st.subheader("📋 Predictions Table")
disp = res.copy()
disp["status"] = disp["alert"].map({True:"⚠️ Alert",False:"✅ Safe"})
st.dataframe(disp.drop(columns=["alert"]), use_container_width=True, height=300)
st.download_button("⬇️ Download CSV", res.to_csv(index=False).encode(),
                   "do_predictions.csv","text/csv")
st.divider()
st.caption("Dissolved Oxygen Prediction · BiSRU+Attention · CVR College of Engineering")
