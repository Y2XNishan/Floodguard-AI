hi
---
title: FloodGuard AI
emoji: 🌊
colorFrom: blue
colorTo: cyan
sdk: streamlit
sdk_version: 1.32.0
app_file: app/streamlit_app.py
pinned: true
---

# 🌊 FloodGuard AI
### India Flood Risk Prediction & Agricultural Intelligence System

![Python](https://img.shields.io/badge/Python-3.9+-blue?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red?style=for-the-badge&logo=streamlit)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange?style=for-the-badge&logo=pytorch)
![XGBoost](https://img.shields.io/badge/XGBoost-AUC_0.83-green?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-cyan?style=for-the-badge)

> 🇮🇳 AI-powered flood risk assessment and agricultural intelligence platform covering all 736 districts across India

---

## 🚀 Live Demo
🔗 [FloodGuard AI on HuggingFace Spaces](https://huggingface.co/spaces/Y2XNishan/floodguard-ai) *(Coming Soon)*

---

## 📌 Overview

FloodGuard AI is a comprehensive machine learning platform designed to:
- Predict flood risk for any of India's 736 districts in real-time
- Analyze historical flood patterns using NDMA data (2015–2024)
- Detect crop diseases and predict agricultural yield
- Send email alerts before floods hit your district
- Provide AI-powered flood advisory via chatbot

Built for farmers, disaster management officials, and researchers across India.

---

## ✨ Features

| Tab | Feature | Description |
|---|---|---|
| 🌊 | **Risk Predictor** | XGBoost-based flood risk prediction for any district |
| 🗺️ | **Risk Map** | Interactive dark-theme map of 736 districts |
| 📈 | **Flood Trends** | NDMA historical data (2015–2024) with Plotly charts |
| 📅 | **7-Day Forecast** | Weekly flood risk forecast with rainfall charts |
| 🤖 | **FloodGuard AI** | AI chatbot powered by Groq LLaMA 3.3 |
| 🛰️ | **Damage Classifier** | CNN-based flood damage assessment from images |
| 🌿 | **Crop Disease** | EfficientNetB0 crop disease detector (13 classes) |
| 🌾 | **Yield Predictor** | Random Forest + Gradient Boosting yield prediction |
| 💰 | **Crop Loss Estimator** | Estimate financial loss from flood damage |
| 🔔 | **Alert System** | Email alerts via Gmail SMTP |
| ℹ️ | **About** | Project info and team |

---

## 🤖 ML Models

| Model | Architecture | Performance | Task |
|---|---|---|---|
| Flood Risk Predictor | XGBoost | AUC: **0.83** | Binary flood classification |
| Flood Sequence Model | LSTM + Attention | Recall: **0.70** | Temporal flood prediction |
| Damage Classifier | EfficientNetB0 | Accuracy: **89.6%** | Flood damage severity |
| Crop Disease Detector | EfficientNetB0 | Accuracy: **98.4%** | 13-class disease detection |
| Yield Predictor | RF + Gradient Boosting | R² optimized | Crop yield estimation |

---

## 📊 Coverage

| Metric | Value |
|---|---|
| Districts covered | **736** |
| States & UTs | **36** |
| Training records | **4,695** |
| Historical years | **2015–2024** |
| Crop disease classes | **13** |
| Languages supported | **English + हिंदी** |

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| **Frontend** | Streamlit |
| **ML/DL** | PyTorch, XGBoost, Scikit-learn |
| **Computer Vision** | EfficientNetB0, torchvision |
| **Visualization** | Plotly, Folium |
| **AI Chatbot** | Groq API (LLaMA 3.3 70B) |
| **Weather Data** | OpenWeatherMap API |
| **Alert System** | Gmail SMTP |
| **Maps** | Folium + CartoDB Dark tiles |
| **Data** | NDMA, IMD, CWC |
| **Language** | Python 3.9+ |

---

## 📦 Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/Y2XNishan/Floodguard-AI.git
cd Floodguard-AI
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables
Create a `.env` file in the root directory:
```env
GMAIL_USER=your@gmail.com
GMAIL_APP_PASSWORD=your_gmail_app_password
GROQ_API_KEY=your_groq_api_key
OPENWEATHERMAP_API_KEY=your_openweathermap_key
GEMINI_API_KEY=your_gemini_key
```

### 4. Run the App
```bash
streamlit run app/streamlit_app.py
```

---

## 📁 Project Structure
flood-risk-prediction/
├── app/
│   └── streamlit_app.py      ← Main Streamlit application
├── src/
│   ├── alert_system.py       ← Email alert system
│   ├── chatbot.py            ← Groq AI chatbot
│   ├── forecast.py           ← 7-day forecast logic
│   └── ...
├── data/
│   ├── ndma_flood_history.csv ← Historical flood data
│   ├── india_districts.csv   ← 736 districts data
│   └── processed/            ← Preprocessed datasets
├── models/
│   ├── xgb_real_model.pkl    ← XGBoost model
│   ├── lstm_real_model.pt    ← LSTM model
│   ├── flood_classifier.pth  ← Damage classifier
│   ├── crop_disease_classifier.pth ← Crop disease model
│   └── crop_yield_model.pkl  ← Yield predictor
├── config.py                 ← App configuration
├── requirements.txt          ← Dependencies
└── README.md
flood-risk-prediction/
├── app/
│   └── streamlit_app.py      ← Main Streamlit application
├── src/
│   ├── alert_system.py       ← Email alert system
│   ├── chatbot.py            ← Groq AI chatbot
│   ├── forecast.py           ← 7-day forecast logic
│   └── ...
├── data/
│   ├── ndma_flood_history.csv ← Historical flood data
│   ├── india_districts.csv   ← 736 districts data
│   └── processed/            ← Preprocessed datasets
├── models/
│   ├── xgb_real_model.pkl    ← XGBoost model
│   ├── lstm_real_model.pt    ← LSTM model
│   ├── flood_classifier.pth  ← Damage classifier
│   ├── crop_disease_classifier.pth ← Crop disease model
│   └── crop_yield_model.pkl  ← Yield predictor
├── config.py                 ← App configuration
├── requirements.txt          ← Dependencies
└── README.md
---

## 🔑 API Keys Required

| Service | Purpose | Get Key |
|---|---|---|
| Groq | AI Chatbot | [console.groq.com](https://console.groq.com) |
| OpenWeatherMap | Live weather | [openweathermap.org](https://openweathermap.org) |
| Gmail | Email alerts | Google Account → App Passwords |
| Gemini (optional) | Backup chatbot | [aistudio.google.com](https://aistudio.google.com) |

---

## 🚨 Emergency Contacts

| Service | Number |
|---|---|
| National Disaster Helpline | **1070** |
| Emergency Services | **112** |
| NDMA | **1078** |

---

## 🗺️ Districts Coverage

FloodGuard AI covers all **736 districts** across **36 states and UTs** of India including high-risk zones:

- 🔴 **High Risk:** Assam, Bihar, Odisha, West Bengal, Kerala, Uttar Pradesh
- 🟡 **Moderate Risk:** Maharashtra, Andhra Pradesh, Tamil Nadu, Gujarat
- 🟢 **Low Risk:** Rajasthan, Ladakh, Himachal Pradesh

---

## 📈 Flood Trends Analysis

Historical flood data from **NDMA (2015–2024)** for all districts:
- Year-over-year flood event frequency
- Area affected (hectares)
- People affected
- Economic damage (₹ Crore)
- Trend indicators (Increasing/Decreasing/Stable)

---

## 🌿 Crop Disease Detection

Detects **13 crop diseases** including:
- Apple Scab, Apple Black Rot
- Corn Gray Leaf Spot, Corn Common Rust
- Potato Early Blight, Potato Late Blight
- Tomato diseases (7 classes)
- And more...

---

## ⚠️ Important Notes

- **Windows users:** `num_workers=0` is set for DataLoader compatibility
- **Models:** Pre-trained models not included in repo due to size — use Git LFS or download separately
- **API Keys:** Never commit `.env` file to GitHub

---

## 🔮 Future Improvements

- [ ] Real-time IMD/CWC API integration
- [ ] Satellite imagery flood extent mapping
- [ ] WhatsApp alert integration
- [ ] Mobile app (React Native)
- [ ] Transformer model replacing LSTM
- [ ] GNN for spatial district relationships

---

## 👨‍💻 Developer

**Nishan Kashyap**
- 🎓 BTech Computer Science — KIIT University
- 📍 Jorhat, Assam, India
- 🔗 [GitHub](https://github.com/Y2XNishan)

---

## 📄 License

MIT License — feel free to use, modify and distribute.

---

<div align="center">
  <b>🌊 FloodGuard AI — Protecting Lives & Livelihoods across India</b><br>
  <i>India Flood Risk Prediction & Agricultural Intelligence System</i>
</div>
