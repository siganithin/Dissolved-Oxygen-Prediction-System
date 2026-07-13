# 💧 Dissolved Oxygen Prediction System

Predicts dissolved oxygen (DO) levels in aquaculture ponds using a **Bidirectional Simple RNN (BiSRU) + Attention** model trained on water quality sensor data.

## Features

- **BiSRU + Attention** deep learning model for temporal DO prediction
- **LightGBM** feature selection to identify top predictors
- **Streamlit dashboard** with real-time alerts, trend charts, and CSV export
- Alert system: flags readings below a configurable DO threshold

## Project Structure

```
DissolvedOxygen_Predictor/
├── app.py                  ← Streamlit entry point (run this)
├── requirements.txt
├── .streamlit/config.toml  ← UI theme config
├── src/
│   ├── preprocess.py       ← Data cleaning & normalization
│   ├── feature_selection.py← LightGBM feature ranking
│   ├── model.py            ← BiSRU + Attention architecture
│   ├── train.py            ← Model training
│   ├── predict.py          ← Inference pipeline
│   └── dashboard.py        ← Legacy dashboard (see app.py)
├── models/
│   ├── bisru_model.keras   ← Trained model
│   ├── scaler.pkl          ← MinMaxScaler
│   ├── selected_features.pkl
│   └── lgbm_selector.pkl
└── data/
    ├── raw/aquaculture_data.csv
    └── processed/
```

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy to Streamlit Cloud

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **New app** → select your repo
4. Set **Main file path** to `app.py`
5. Click **Deploy**

> **Note:** The `models/` folder with `.keras` and `.pkl` files must be committed to the repo for cloud deployment to work.

## Input CSV Format

| Column | Unit |
|---|---|
| Temperature (cel) | °C |
| pH (ph units) | pH |
| Biochemical Oxygen Demand (mg/l) | mg/L |
| Ammonia (mg/l) | mg/L |
| Nitrate (mg/l) | mg/L |
| Nitrogen (mg/l) | mg/L |
| Dissolved Oxygen (mg/l) | mg/L (optional — for comparison) |

## Model Pipeline

```
Raw CSV → Preprocess → Feature Selection (LightGBM)
        → 24-step sequences → BiSRU + Attention → DO prediction
```
