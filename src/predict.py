import numpy as np
import pandas as pd
import joblib
import os
from model import SEQUENCE_LEN, SumOverTime

# Get the project root directory (parent of src/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH    = os.path.join(PROJECT_ROOT, "models/bisru_model.keras")
SCALER_PATH   = os.path.join(PROJECT_ROOT, "models/scaler.pkl")
FEATURES_PATH = os.path.join(PROJECT_ROOT, "models/selected_features.pkl")
TARGET        = "dissolved_oxygen"
DO_THRESHOLD  = 5.0

ALL_FEATURES  = ["temperature", "pH", "BOD", "ammonia", "nitrate", "nitrogen"]


def load_artifacts():
    # Defer heavy TensorFlow import until actually loading the model
    from tensorflow.keras.models import load_model

    model = load_model(MODEL_PATH, custom_objects={"SumOverTime": SumOverTime})
    scaler = joblib.load(SCALER_PATH)
    features = joblib.load(FEATURES_PATH)
    return model, scaler, features


def preprocess_upload(df: pd.DataFrame, scaler, features: list) -> pd.DataFrame:
    col_map = {
        "Temperature (cel)"                : "temperature",
        "pH (ph units)"                    : "pH",
        "Biochemical Oxygen Demand (mg/l)" : "BOD",
        "Ammonia (mg/l)"                   : "ammonia",
        "Nitrate (mg/l)"                   : "nitrate",
        "Nitrogen (mg/l)"                  : "nitrogen",
        "Dissolved Oxygen (mg/l)"          : "dissolved_oxygen",
        "Date"                             : "timestamp",
    }
    df = df.rename(columns=col_map)

    needed = ALL_FEATURES + ([TARGET] if TARGET in df.columns else [])
    df = df[[c for c in needed if c in df.columns]]
    df[ALL_FEATURES] = df[ALL_FEATURES].fillna(df[ALL_FEATURES].median())

    cols_to_scale = ALL_FEATURES + ([TARGET] if TARGET in df.columns else [])
    df[cols_to_scale] = scaler.transform(df[cols_to_scale])
    return df


def make_sequences(df: pd.DataFrame, features: list):
    X = []
    values = df[features].values
    for i in range(len(df) - SEQUENCE_LEN):
        X.append(values[i : i + SEQUENCE_LEN])
    return np.array(X, dtype="float32")


def inverse_do(scaler, values: np.ndarray) -> np.ndarray:
    n = len(ALL_FEATURES)
    dummy = np.zeros((len(values), n + 1))
    dummy[:, -1] = values.flatten()
    return scaler.inverse_transform(dummy)[:, -1]


def predict(csv_path: str, model=None, scaler=None, features=None) -> pd.DataFrame:
    if model is None or scaler is None or features is None:
        model, scaler, features = load_artifacts()

    raw_df = pd.read_csv(csv_path, low_memory=False)
    df     = preprocess_upload(raw_df, scaler, features)

    if len(df) < SEQUENCE_LEN + 1:
        raise ValueError(f"Need at least {SEQUENCE_LEN + 1} rows. Got {len(df)}.")

    X          = make_sequences(df, features)
    preds_norm = model.predict(X, verbose=0).flatten()
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
    # Sample 500 rows from raw data for a clean end-to-end test
    raw = pd.read_csv("data/raw/aquaculture_data.csv", low_memory=False).dropna().head(500)
    sample_path = "data/processed/sample_test.csv"
    raw.to_csv(sample_path, index=False)

    result = predict(sample_path)
    alerts = result["alert"].sum()
    print(result.head(10).to_string(index=False))
    print(f"\n[alerts] {alerts} / {len(result)} predictions below {DO_THRESHOLD} mg/L")