"""
Inference pipeline — zero sklearn dependency at runtime.
Scaling is done manually with saved numpy arrays.
"""
import io
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import onnxruntime as rt

PROJECT_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH    = os.path.join(PROJECT_ROOT, "models", "bisru_model.onnx")
FEATURES_PATH = os.path.join(PROJECT_ROOT, "models", "selected_features.pkl")
SCALER_MIN    = os.path.join(PROJECT_ROOT, "models", "scaler_min.npy")
SCALER_SCALE  = os.path.join(PROJECT_ROOT, "models", "scaler_scale.npy")
SCALER_MAX    = os.path.join(PROJECT_ROOT, "models", "scaler_max.npy")

TARGET       = "dissolved_oxygen"
DO_THRESHOLD = 5.0
SEQUENCE_LEN = 24
ALL_FEATURES = ["temperature", "pH", "BOD", "ammonia", "nitrate", "nitrogen"]
# column order the scaler was fit on: 6 features + target
SCALER_COLS  = ALL_FEATURES + [TARGET]

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


class NumpyScaler:
    """MinMaxScaler reimplemented with pure numpy — no sklearn needed."""
    def __init__(self, min_path, scale_path):
        self.data_min_ = np.load(min_path)
        self.scale_    = np.load(scale_path)

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.data_min_) * self.scale_

    def inverse_col(self, values: np.ndarray, col_idx: int) -> np.ndarray:
        """Inverse-transform a single column by index."""
        return values / self.scale_[col_idx] + self.data_min_[col_idx]


def load_artifacts():
    sess     = rt.InferenceSession(MODEL_PATH)
    scaler   = NumpyScaler(SCALER_MIN, SCALER_SCALE)
    features = joblib.load(FEATURES_PATH)
    return sess, scaler, features


def _read_csv(source) -> pd.DataFrame:
    if isinstance(source, (str, os.PathLike)):
        return pd.read_csv(source, low_memory=False)
    if isinstance(source, bytes):
        return pd.read_csv(io.BytesIO(source), low_memory=False)
    return pd.read_csv(source, low_memory=False)


def predict(source, model=None, scaler=None, features=None) -> pd.DataFrame:
    if model is None or scaler is None or features is None:
        model, scaler, features = load_artifacts()

    raw_df = _read_csv(source)
    df = raw_df.rename(columns=COL_MAP)

    has_target = TARGET in df.columns
    needed = ALL_FEATURES + ([TARGET] if has_target else [])
    df = df[[c for c in needed if c in df.columns]].copy()

    # Fill missing feature values
    for col in ALL_FEATURES:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    if len(df) < SEQUENCE_LEN + 1:
        raise ValueError(
            f"Need at least {SEQUENCE_LEN + 1} rows. Got {len(df)}. Upload a larger file."
        )

    # Scale using numpy arrays — col order must match SCALER_COLS
    cols_to_scale = ALL_FEATURES + ([TARGET] if has_target else [])
    scale_indices = [SCALER_COLS.index(c) for c in cols_to_scale]

    arr = df[cols_to_scale].values.astype(np.float64)
    for i, idx in enumerate(scale_indices):
        arr[:, i] = (arr[:, i] - scaler.data_min_[idx]) * scaler.scale_[idx]

    df_scaled = pd.DataFrame(arr, columns=cols_to_scale)

    # Build sequences
    feat_vals = df_scaled[features].values.astype(np.float32)
    X = np.array(
        [feat_vals[i: i + SEQUENCE_LEN] for i in range(len(df_scaled) - SEQUENCE_LEN)],
        dtype=np.float32
    )

    # ONNX inference
    input_name = model.get_inputs()[0].name
    preds_norm = model.run(None, {input_name: X})[0].flatten()

    # Inverse-transform predictions (DO is last column = index 6)
    do_idx      = SCALER_COLS.index(TARGET)
    do_min      = scaler.data_min_[do_idx]
    do_scale    = scaler.scale_[do_idx]
    preds_mgL   = preds_norm / do_scale + do_min

    result = pd.DataFrame({
        "index"        : range(len(preds_mgL)),
        "predicted_DO" : np.round(preds_mgL, 3),
        "alert"        : preds_mgL < DO_THRESHOLD,
    })

    if has_target:
        actual_norm = df_scaled[TARGET].values[SEQUENCE_LEN:]
        actual_mgL  = actual_norm / do_scale + do_min
        result["actual_DO"] = np.round(actual_mgL, 3)

    return result


if __name__ == "__main__":
    sample = os.path.join(PROJECT_ROOT, "data", "raw", "aquaculture_data.csv")
    r = predict(sample)
    print(r.head(10).to_string(index=False))
    print(f"\nAlerts: {r['alert'].sum()} / {len(r)}")
