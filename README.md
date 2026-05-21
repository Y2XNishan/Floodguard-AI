# 🌊 Flood Risk Prediction System — Assam, India

> An end-to-end machine learning system for predicting flood risk in Assam's Brahmaputra basin using XGBoost, LSTM with Attention, and SHAP explainability.

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=pytorch&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-ff4b4b?logo=streamlit&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-337ab7?logo=xgboost&logoColor=white)

---

## 📋 Project Overview

This system predicts flood risk for 10 districts in Assam, India using historical weather, hydrological, and environmental data from the Brahmaputra river basin. It combines traditional ML (XGBoost) with deep learning (LSTM + Attention) and provides explainable predictions through SHAP analysis.

### Key Features
- 🔮 **Real-time flood risk prediction** for any Assam district
- 🗺️ **Interactive risk map** with district-level risk visualization
- 🧠 **Explainable AI** — SHAP values show why a prediction was made
- 📊 **Model comparison** — XGBoost vs LSTM performance dashboard
- 📈 **Data exploration** — interactive charts and trend analysis

---

## 📊 Dataset Sources

| Source | Data Type | Temporal Resolution |
|--------|-----------|-------------------|
| IMD (India Meteorological Dept) | Rainfall, Temperature | Daily |
| CWC (Central Water Commission) | River Levels | Daily |
| NASA MODIS | NDVI (Vegetation Index) | 16-day |
| NASA SMAP | Soil Moisture | Daily |
| SRTM | Elevation | Static |
| **Synthetic (included)** | **All features combined** | **Daily** |

> **Note:** The included `data/sample_data.csv` contains realistic synthetic data for demonstration. Replace with real data for production use.

---

## 🏗️ Model Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    INPUT FEATURES                            │
│  rainfall, river_level, temp, humidity, ndvi, soil_moisture  │
│  + engineered: rolling sums, API, interactions, cyclical     │
└─────────────────────┬────────────────────────────────────────┘
                      │
          ┌───────────┴───────────┐
          │                       │
   ┌──────▼──────┐        ┌──────▼──────┐
   │  XGBoost    │        │ LSTM (x2)   │
   │  Classifier │        │ hidden=128  │
   │  (Baseline) │        │ dropout=0.3 │
   └──────┬──────┘        └──────┬──────┘
          │                      │
          │               ┌──────▼──────┐
          │               │  Attention  │
          │               │  Mechanism  │
          │               └──────┬──────┘
          │                      │
          │               ┌──────▼──────┐
          │               │ FC: 128→64  │
          │               │ ReLU+Drop   │
          │               │ 64→1→Sigm   │
          │               └──────┬──────┘
          │                      │
   ┌──────▼──────────────────────▼──────┐
   │        FLOOD RISK SCORE            │
   │     🟢 Safe  🟡 Moderate  🔴 High  │
   └────────────────┬───────────────────┘
                    │
            ┌───────▼───────┐
            │ SHAP Analysis │
            │ (Explainable) │
            └───────────────┘
```

---

## 🔧 Feature Engineering

| Feature | Description | Method |
|---------|-------------|--------|
| `rainfall_7day_cumsum` | 7-day cumulative rainfall | Rolling sum (window=7) |
| `rainfall_30day_cumsum` | 30-day cumulative rainfall | Rolling sum (window=30) |
| `api` | Antecedent Precipitation Index | EWM (span=7) |
| `river_rise_rate` | Daily river level change | `.diff()` |
| `rainfall_river_interaction` | Rainfall × River level | Product of 7-day rain × level |
| `is_monsoon` | June–September flag | Binary (0/1) |
| `month_sin` | Cyclical month (sine) | `sin(2π × month/12)` |
| `month_cos` | Cyclical month (cosine) | `cos(2π × month/12)` |

---

## 🚀 Installation & Setup

### 1. Clone the Repository
```bash
cd flood-risk-prediction
```

### 2. Create Virtual Environment
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Generate Sample Data
```bash
python data/generate_data.py
```

### 5. Run Preprocessing
```bash
python src/preprocessing.py
```

### 6. Train Models
```bash
# XGBoost (with Optuna tuning — ~5 min)
python src/train_baseline.py

# LSTM (may take 10-30 min depending on hardware)
python src/train_lstm.py
```

### 7. Generate Explainability Plots
```bash
python src/explainability.py
```

### 8. Generate Flood Risk Map
```bash
python src/flood_map.py
```

---

## 🖥️ Running the Streamlit App

```bash
streamlit run app/streamlit_app.py
```

The app will open at `http://localhost:8501` with four pages:
1. **🌊 Risk Predictor** — Make predictions with custom inputs
2. **🗺️ Risk Map** — Interactive Assam flood risk map
3. **📊 Model Performance** — Compare XGBoost vs LSTM
4. **📈 Data Explorer** — Explore historical data with Plotly charts

---

## 📊 Model Performance Results

| Metric | XGBoost | LSTM |
|--------|---------|------|
| Accuracy | TBD | TBD |
| F1 Score | TBD | TBD |
| ROC-AUC | TBD | TBD |
| Recall | TBD | TBD |
| Precision | TBD | TBD |

> Results will be populated after training. Run the training scripts to see actual performance.

---

## 📁 Project Structure

```
flood-risk-prediction/
├── data/
│   ├── raw/                    ← raw data downloads
│   ├── processed/              ← preprocessed splits (.npy)
│   ├── generate_data.py        ← synthetic data generator
│   └── sample_data.csv         ← 5-year daily data (10 districts)
├── notebooks/
│   └── EDA.ipynb               ← exploratory data analysis
├── src/
│   ├── data_collection.py      ← data fetching module
│   ├── preprocessing.py        ← cleaning & feature engineering
│   ├── train_baseline.py       ← XGBoost + Optuna training
│   ├── train_lstm.py           ← PyTorch LSTM + Attention
│   ├── explainability.py       ← SHAP analysis & plots
│   └── flood_map.py            ← Folium risk map generator
├── models/
│   ├── xgb_model.pkl           ← saved XGBoost model
│   ├── lstm_model.pt           ← saved LSTM weights
│   ├── scaler.pkl              ← fitted StandardScaler
│   ├── loss_curve.png          ← LSTM training curve
│   ├── shap_summary_plot.png   ← SHAP feature importance
│   ├── shap_force_plot.html    ← SHAP force explanation
│   └── shap_dependence_rainfall.png
├── app/
│   └── streamlit_app.py        ← Streamlit web dashboard
├── flood_risk_map.html         ← interactive Folium map
├── requirements.txt
└── README.md
```

---

## 🔮 Future Improvements

- [ ] **Real-time data integration** — Connect to live IMD/CWC APIs
- [ ] **Satellite imagery** — Use Sentinel-2/MODIS flood extent mapping
- [ ] **Transformer model** — Replace LSTM with temporal transformer
- [ ] **Multi-task learning** — Predict both occurrence and severity
- [ ] **Spatial features** — Add GNN for district-level spatial relationships
- [ ] **Ensemble model** — Combine XGBoost + LSTM predictions
- [ ] **Alert system** — SMS/email notifications for high-risk predictions
- [ ] **Mobile app** — React Native companion app for field workers

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

<p align="center">
  Built with ❤️ for flood-resilient communities in Assam, India
</p>
