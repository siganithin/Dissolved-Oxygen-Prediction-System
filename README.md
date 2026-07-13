# Dissolved Oxygen Prediction System

**Aquaculture Water Quality Management using BiSRU + Attention**

A deployed machine learning system that predicts dissolved oxygen (DO) levels in aquaculture ponds using a Bidirectional Simple Recurrent Unit (BiSRU) with Attention mechanism. The app is live on Streamlit Cloud and accepts CSV sensor data for real-time DO prediction and alert monitoring.

**Live App:** https://dissolved-oxygen-prediction-system-kz9w590nithin.streamlit.app

---

## Project Structure

```
DissolvedOxygen_Predictor/
├── app.py                          ← Streamlit app (main entry point)
├── requirements.txt                ← Cloud-compatible dependencies
├── .streamlit/
│   └── config.toml                 ← UI theme
├── src/
│   ├── preprocess.py               ← Data cleaning, normalization, splitting
│   ├── feature_selection.py        ← LightGBM feature ranking
│   ← model.py                      ← BiSRU + Attention architecture (Keras)
│   ├── train.py                    ← Model training pipeline
│   └── predict.py                  ← ONNX inference pipeline
├── models/
│   ├── bisru_model.onnx            ← Trained model (ONNX, used for deployment)
│   ├── bisru_model.keras           ← Trained model (Keras, used for retraining)
│   ├── scaler.pkl                  ← Fitted MinMaxScaler
│   ├── selected_features.pkl       ← Top 4 features selected by LightGBM
│   └── lgbm_selector.pkl           ← Trained LightGBM feature selector
├── data/
│   ├── raw/aquaculture_data.csv    ← Original dataset
│   └── processed/
│       └── feature_importance.png  ← LightGBM feature importance chart
└── notebooks/
    ├── eda.ipynb                   ← Exploratory data analysis
    └── evaluation.ipynb            ← Model evaluation
```

---

## How It Works

### Pipeline Overview

```
Raw CSV  →  Preprocess  →  Feature Selection (LightGBM)
         →  24-step sequences  →  BiSRU + Attention  →  DO prediction (mg/L)
         →  Alert if predicted DO < threshold
```

### Step 1 — Preprocessing (`src/preprocess.py`)

- Loads `aquaculture_data.csv` and renames columns to internal names
- Drops rows with missing target (`dissolved_oxygen`)
- Fills missing feature values with column medians
- Removes rows with negative values
- Applies `MinMaxScaler` to all 7 columns (6 features + 1 target), saves scaler to `models/scaler.pkl`
- Splits data chronologically: **70% train / 15% val / 15% test**

Input columns → internal names:

| Raw Column | Internal Name |
|---|---|
| Temperature (cel) | temperature |
| pH (ph units) | pH |
| Biochemical Oxygen Demand (mg/l) | BOD |
| Ammonia (mg/l) | ammonia |
| Nitrate (mg/l) | nitrate |
| Nitrogen (mg/l) | nitrogen |
| Dissolved Oxygen (mg/l) | dissolved_oxygen |

### Step 2 — Feature Selection (`src/feature_selection.py`)

- Trains a `LGBMRegressor` (300 estimators, lr=0.05, max_depth=6) on the training set
- Extracts feature importance scores and ranks all 6 features
- Saves the **top 4** feature names to `models/selected_features.pkl`
- Saves importance bar chart to `data/processed/feature_importance.png`

LightGBM is used purely as a **feature selector** here, not as the final predictor.

### Step 3 — Model Architecture (`src/model.py`)

The BiSRU-Attention model:

```
Input (24 timesteps × n_features)
    ↓
Bidirectional(SimpleRNN(64, return_sequences=True))   ← BiSRU layer
    ↓
Attention Layer:
    Dense(1, tanh) → score per timestep
    Softmax(axis=1) → attention weights
    Multiply(BiSRU output × weights)
    SumOverTime() → fixed-length context vector
    ↓
Dense(32, relu)
    ↓
Dense(1)  ← predicted DO (normalized)
```

- `SEQUENCE_LEN = 24` time steps
- `SumOverTime` is a custom Keras `Layer` subclass (not Lambda) for safe serialization
- Compiled with Adam + MSE loss

### Step 4 — Training (`src/train.py`)

- Builds overlapping sequences of length 24 from train/val CSVs
- Trains with `EarlyStopping(patience=5)` and `ModelCheckpoint`
- Saves best model to `models/bisru_model.keras`
- Maximum 50 epochs, batch size 256

### Step 5 — Inference (`src/predict.py`)

The deployed inference pipeline uses **ONNX Runtime** instead of TensorFlow:

- Loads `models/bisru_model.onnx` via `onnxruntime.InferenceSession`
- Renames columns, fills NaN, applies saved scaler
- Builds sequences of length 24
- Runs ONNX inference: `session.run(None, {'input': X})`
- Inverse-transforms predictions back to mg/L using the scaler
- Returns a DataFrame with `predicted_DO`, `alert`, and optionally `actual_DO`

> The model was converted from `.keras` to `.onnx` using `tf2onnx` to enable deployment on Streamlit Cloud (Python 3.14), where TensorFlow has no compatible wheels.

### Step 6 — Dashboard (`app.py`)

Streamlit app with:

- **Sidebar** — CSV upload, alert threshold slider (2–8 mg/L, default 5.0)
- **Metric cards** — rows analyzed, avg/min/max predicted DO, alert count
- **Trend chart** — predicted vs actual DO over time, threshold line
- **Distribution histogram** — predicted DO spread
- **Scatter plot** — actual vs predicted (when actual DO is present in CSV)
- **Feature importance image** — from LightGBM
- **Predictions table** — full results with safe/alert status
- **Download button** — export predictions as CSV

Model loads once per session via `@st.cache_resource`. Predictions cache per uploaded file via `@st.cache_data`.

---

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Upload `data/raw/aquaculture_data.csv` (or any CSV with the required columns) to test.

---

## Retrain the Model

```bash
# 1. Preprocess raw data
python src/preprocess.py

# 2. Select top features with LightGBM
python src/feature_selection.py

# 3. Train BiSRU-Attention model
python src/train.py

# 4. Convert to ONNX for deployment
python -c "
import tensorflow as tf, tf2onnx, onnx, sys
sys.path.insert(0,'src')
from src.model import SumOverTime
model = tf.keras.models.load_model('models/bisru_model.keras', custom_objects={'SumOverTime': SumOverTime})
spec = [tf.TensorSpec(model.inputs[0].shape, tf.float32, name='input')]
onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13)
onnx.save(onnx_model, 'models/bisru_model.onnx')
print('Done')
"
```

---

## Dependencies

| Package | Purpose |
|---|---|
| `onnxruntime` | Model inference (replaces TensorFlow at runtime) |
| `lightgbm` | Feature selection |
| `scikit-learn` | MinMaxScaler |
| `pandas` / `numpy` | Data processing |
| `matplotlib` / `seaborn` | Charts |
| `streamlit` | Web dashboard |
| `joblib` | Loading `.pkl` artifacts |
| `tensorflow` | Training only (not needed for inference) |

---

## Team

Siga Nithin (23B81A0590) · Nalla Praneeth (23B81A0595) · Musham Santhosh (23B81A05A4)

CVR College of Engineering — B.Tech Computer Science and Engineering — April 2026

Guide: Ms. M Swapna, Assistant Professor
