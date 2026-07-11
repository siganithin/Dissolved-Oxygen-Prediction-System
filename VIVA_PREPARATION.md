# Viva Preparation: Dissolved Oxygen Prediction System

This document provides a complete project summary from data collection to the final dashboard, along with likely Viva questions and high-quality answers to help you defend your project confidently.

---

## 1. Project Summary Flow

### Step 1: Data Collection & Preprocessing
- **Data Source:** We use an aquaculture dataset containing parameters like Temperature, pH, Biochemical Oxygen Demand (BOD), Ammonia, Nitrate, Nitrogen, and Dissolved Oxygen (DO).
- **Cleaning:** We remove missing target values (rows where DO is null) and handle missing feature values by replacing them with the column median. Negative, erroneous values are eliminated.
- **Normalization:** Because features are measured on different scales (e.g., pH is 1-14, Temperature is in Celsius, Ammonia is in mg/L), we apply **Min-Max Scaling** to normalize all data between `0 and 1`. This allows the deep learning model to converge much faster.
- **Data Split:** The data is chronologically split into **70% Training, 15% Validation, and 15% Test** sets.

### Step 2: Feature Selection (LightGBM)
- **Why LightGBM?** Using all water quality parameters increases computational overhead and may introduce noise. Light Gradient Boosting Machine (LightGBM) is a highly efficient tree-based algorithm used here not for final prediction, but to rank features based on their importance.
- **Output:** It identifies the **Top 4 features** (e.g., pH, BOD, Ammonia, Nitrate) that most strongly impact Dissolved Oxygen levels. This simplifies the deep learning model.

### Step 3: Model Development (BiSRU + Attention)
- **Sequence Creation:** The data is transformed into sequences of length **24 time steps** (sliding window) since water quality depends on past history.
- **BiSRU (Bidirectional Simple Recurrent Unit):** This is the core sequential model. Unlike LSTM or GRU, SRU is highly parallelizable and faster to train while avoiding the vanishing gradient problem. The "Bidirectional" aspect means it looks at patterns from both standard forward and backward contexts (simultaneously scanning time sequences).
- **Attention Mechanism:** An added intelligent layer that assigns different "weights" to different time steps. Not all past 24 hours are equally important; the Attention Mechanism helps the model focus strictly on the time steps that strongly influence the future DO drop.
- **Performance:** This hybrid model achieves **96.28% Accuracy ($R^2$ = 0.93-0.96)** with remarkably low Mean Squared Error (MSE = 0.0008), outperforming traditional LSTM/GRU models.

### Step 4: The Dashboard (Streamlit)
- **Purpose:** A user-friendly, real-time web application for aquaculturists.
- **Features:** 
   - **CSV Upload:** Users can upload new sensor readings.
   - **Threshold Slider:** Sets a dynamic alert threshold (default 5.0 mg/L).
   - **Automated Caching:** Computations are cached after uploading to prevent redundant reloading of the Heavy ML model, making UI interactions instantly responsive.
   - **Visualizations:** A line chart compares Predicted DO levels against the safety alert threshold, alongside key metric cards.

---

## 2. Potential Viva Questions & Answers

### General & Project Concept 
> **Q: What is the main objective of your project?**
> **A:** The primary objective is to proactively predict Dissolved Oxygen (DO) levels in aquaculture environments using a hybrid deep learning model (BiSRU-Attention). This allows farmers to receive early warning alerts before oxygen drops to critical limits, preventing fish mortality rather than reacting to it after it happens.

> **Q: Why did you choose a Hybrid Model instead of a simple ML model (like Random Forest)?**
> **A:** Dissolved Oxygen time-series data is dynamic, non-linear, and complex. High-end ML models like Random Forest can't effectively capture chronological sequential dependencies (relationships across time). Deep learning models like BiSRU track trends effectively over time, while LightGBM filters out noise before giving the data to the deep neural network.

### Preprocessing & Feature Selection
> **Q: Why was normalization necessary in your preprocessing pipeline?**
> **A:** Our water quality parameters have drastically different scales (e.g., temperature in 20s, pH up to 14, and some chemicals in fractions of mg/L). Without normalization (specifically Min-Max scaling), the neural network would unfairly weight variables with larger numerical values. Scaling forces all values between 0 and 1, ensuring fair treatment and helping the neural network’s optimizer converge faster without getting stuck.

> **Q: What does the LightGBM do in your project?**
> **A:** We use LightGBM specifically as a **Feature Selector**. Instead of putting all environmental factors into the deep learning model, LightGBM evaluates their 'importance scores'. We take only the top 4 most influential features (like pH, BOD). This reduces noise, speeds up training, and prevents overfitting.

### Machine Learning Core (Model)
> **Q: What is BiSRU and why is it better than LSTM?**
> **A:** BiSRU stands for Bidirectional Simple Recurrent Unit. While LSTM (Long Short-Term Memory) is great for time series, it relies on complex internal gates that cannot be computed in parallel, making training very slow. SRU simplifies these gates so time steps can be processed heavily in parallel. We made it Bidirectional to analyze sequences from both the past to the future and vice versa, providing a deeper contextual understanding.

> **Q: Can you explain the "Attention Mechanism"?**
> **A:** In a sequence of 24 time steps, not all historical readings deserve the same significance when predicting the next DO value. The Attention Mechanism behaves mathematically like human focus; it calculates correlation weights and strictly forces the BiSRU to focus only on moments that represent drastic environmental changes, ignoring stable/irrelevant periods.

### Application & Deployment Details
> **Q: In your Streamlit dashboard, a threshold slider is provided. How does that work technically?**
> **A:** When the system predicts DO values for the uploaded data, we compare those predictions against the user-defined threshold (default 5.0 mg/L). For every row where the predicted DO is less than the threshold, we append a boolean `True` (alert state). The dashboard then aggregates these to show total critical warnings.

> **Q: We noticed Streamlit has to rerun scripts completely when the threshold slider moves. Did you handle this?**
> **A:** Yes. We utilized Streamlit's `@st.cache_data` decoration. Instead of repeatedly parsing the CSV and using the heavy Keras deep learning model to redo predictions every time the slider is touched, we hash the uploaded file. The prediction is computed once and cached in memory. Subsequent slider movements just manipulate the cached dataframe making it incredibly fast.

### Future Enhancements
> **Q: What are the limitations, and how would you enhance this in the future?**
> **A:** A limitation is that it currently relies on CSV batch uploads rather than live continuous streaming. In the future, we intend to integrate IoT sensor nodes directly into the dashboard via an API, allowing for true real-time cloud-based monitoring. Furthermore, we could build an automated hardware hook that physically turns on water aerators if the prediction model forecasts a critical drop.
