import io
import os
import numpy as np
import pandas as pd
import joblib
import onnxruntime as rt
import warnings
warnings.filterwarnings("ignore", category=UserWarning)  # suppress sklearn version warnings

# Project root (parent of src/)
PROJECT_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH    = os.path.join(PROJECT_ROOT, "models", "bisru_model.onnx")
SCALER_PATH   = os.path.join(PROJECT_ROOT, "models", "scaler.pkl")
FEATURES_PATH = os.path.join(PROJECT_ROOT, "models", "selected_features.pkl")

TARGET        = "dissolved_oxygen"
DO_THRESHOLD  = 5.0
ALL_FEATURES  = ["temperature", "pH", "BOD", "ammonia", "nitrate", "nitrogen"]
SEQUENCE_LEN  = 24

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
    sess     = rt.InferenceSession(MODEL_PATH)
    scaler   = joblib.load(SCALER_PATH)
    features = joblib.load(FEATURES_PATH)
    return sess, scaler, features


def _load_csv(source) -> pd.DataFrame:
    """Accept a file path (str), bytes, or file-like object."""
    if isinstance(source, (str, os.PathLike)):
        return pd.read_csv(source, low_memory=False)
    if isinstance(source, bytes):
        return pd.read_csv(io.BytesIO(source), low_memory=False)
    return pd.read_csv(source, low_memory=False)


def preprocess(df: pd.DataFrame, scaler, features: list) -> pd.DataFrame:
    df = df.rename(columns=COL_MAP).copy()
    needed = ALL_FEATURES + ([TARGET] if TARGET in df.columns else [])
    df = df[[c for c in needed if c in df.columns]].copy()
    df[ALL_FEATURES] = df[ALL_FEATURES].fillna(df[ALL_FEATURES].median())
    cols = ALL_FEATURES + ([TARGET] if TARGET in df.columns else [])
    scaled = scaler.transform(df[cols])
    df = df.copy()
    df[cols] = scaled
    return df


def make_sequences(df: pd.DataFrame, features: list) -> np.ndarray:
    values = df[features].values
    return np.array(
        [values[i: i + SEQUENCE_LEN] for i in range(len(df) - SEQUENCE_LEN)],
        dtype="float32"
    )


def inverse_do(scaler, values: np.ndarray) -> np.ndarray:
    dummy = np.zeros((len(values), len(ALL_FEATURES) + 1))
    dummy[:, -1] = values.flatten()
    return scaler.inverse_transform(dummy)[:, -1]


def predict(source, model=None, scaler=None, features=None) -> pd.DataFrame:
    """
    source: file path (str), raw bytes, or file-like object
    """
    if model is None or scaler is None or features is None:
        model, scaler, features = load_artifacts()

    raw_df = _load_csv(source)
    df     = preprocess(raw_df, scaler, features)

    if len(df) < SEQUENCE_LEN + 1:
        raise ValueError(
            f"Need at least {SEQUENCE_LEN + 1} rows after cleaning. Got {len(df)}. "
            "Upload a larger file."
        )

    X          = make_sequences(df, features)
    input_name = model.get_inputs()[0].name
    preds_norm = model.run(None, {input_name: X})[0].flatten()
    preds_mgL  = inverse_do(scaler, preds_norm)

    actual_col = df[TARGET].values[SEQUENCE_LEN:] if TARGET in df.columns else None
    actual_mgL = inverse_do(scaler, actual_col) if actual_col is not None else None

    result = pd.DataFrame({
        "index"        : range(len(preds_mgL)),
        "predicted_DO" : np.round(preds_mgL, 3),
        "alert"        : preds_mgL < DO_THRESHOLD,
    })
    if actual_mgL is not None:
        result["actual_DO"] = np.round(actual_mgL, 3)

    return result


if __name__ == "__main__":
    sample = os.path.join(PROJECT_ROOT, "data", "raw", "aquaculture_data.csv")
    result = predict(sample)
    print(result.head(10).to_string(index=False))
    print(f"\nAlerts: {result['alert'].sum()} / {len(result)}")
