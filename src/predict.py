"""
Inference pipeline.
Zero sklearn / joblib dependency at runtime — uses numpy arrays + JSON.
Works on any Python version / any package version.
"""
import io
import os
import json
import traceback
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import onnxruntime as rt

PROJECT_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH    = os.path.join(PROJECT_ROOT, "models", "bisru_model.onnx")
FEATURES_PATH = os.path.join(PROJECT_ROOT, "models", "selected_features.json")
SCALER_MIN    = os.path.join(PROJECT_ROOT, "models", "scaler_min.npy")
SCALER_SCALE  = os.path.join(PROJECT_ROOT, "models", "scaler_scale.npy")

TARGET       = "dissolved_oxygen"
DO_THRESHOLD = 5.0
SEQUENCE_LEN = 24
ALL_FEATURES = ["temperature", "pH", "BOD", "ammonia", "nitrate", "nitrogen"]
SCALER_COLS  = ALL_FEATURES + [TARGET]   # order scaler was fit on

COL_MAP = {
    "Temperature (cel)"                : "temperature",
    "pH (ph units)"                    : "pH",
    "Biochemical Oxygen Demand (mg/l)" : "BOD",
    "Ammonia (mg/l)"                   : "ammonia",
    "Nitrate (mg/l)"                   : "nitrate",
    "Nitrogen (mg/l)"                  : "nitrogen",
    "Dissolved Oxygen (mg/l)"          : "dissolved_oxygen",
    "Date"                             : "timestamp",
}


def load_artifacts():
    sess       = rt.InferenceSession(MODEL_PATH)
    data_min   = np.load(SCALER_MIN)
    scale      = np.load(SCALER_SCALE)
    with open(FEATURES_PATH) as f:
        features = json.load(f)
    return sess, (data_min, scale), features


def _read_csv(source) -> pd.DataFrame:
    if isinstance(source, bytes):
        return pd.read_csv(io.BytesIO(source), low_memory=False)
    return pd.read_csv(source, low_memory=False)


def predict(source, model=None, scaler=None, features=None) -> pd.DataFrame:
    if model is None or scaler is None or features is None:
        model, scaler, features = load_artifacts()

    data_min, scale = scaler

    raw_df = _read_csv(source)
    df = raw_df.rename(columns=COL_MAP)

    has_target = TARGET in df.columns
    needed     = ALL_FEATURES + ([TARGET] if has_target else [])
    df         = df[[c for c in needed if c in df.columns]].copy()

    # Fill NaN in features
    for col in ALL_FEATURES:
        if col in df.columns:
            med = df[col].median()
            df[col] = df[col].fillna(med)

    if len(df) < SEQUENCE_LEN + 1:
        raise ValueError(
            f"Need at least {SEQUENCE_LEN + 1} rows after cleaning. "
            f"Got {len(df)}. Please upload a larger CSV file."
        )

    # Manual MinMax scale — no sklearn
    cols_present = ALL_FEATURES + ([TARGET] if has_target else [])
    arr = df[cols_present].values.astype(np.float64)
    for i, col in enumerate(cols_present):
        idx = SCALER_COLS.index(col)
        arr[:, i] = (arr[:, i] - data_min[idx]) * scale[idx]

    # Build sequences
    feat_idx  = [cols_present.index(f) for f in features]
    feat_arr  = arr[:, feat_idx].astype(np.float32)
    X = np.stack([feat_arr[i: i + SEQUENCE_LEN]
                  for i in range(len(feat_arr) - SEQUENCE_LEN)]).astype(np.float32)

    # ONNX inference
    inp_name   = model.get_inputs()[0].name
    preds_norm = model.run(None, {inp_name: X})[0].flatten()

    # Inverse-transform DO
    do_idx   = SCALER_COLS.index(TARGET)
    preds_mgL = preds_norm / scale[do_idx] + data_min[do_idx]

    result = pd.DataFrame({
        "index"        : np.arange(len(preds_mgL)),
        "predicted_DO" : np.round(preds_mgL.astype(float), 3),
        "alert"        : preds_mgL < DO_THRESHOLD,
    })

    if has_target:
        do_col_idx = cols_present.index(TARGET)
        actual_mgL = arr[SEQUENCE_LEN:, do_col_idx] / scale[do_idx] + data_min[do_idx]
        result["actual_DO"] = np.round(actual_mgL.astype(float), 3)

    return result
