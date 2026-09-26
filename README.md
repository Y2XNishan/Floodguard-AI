# FloodGuard AI

FloodGuard AI is a machine learning platform designed for district-level flood risk assessment and agricultural impact forecasting across India. It aggregates historical disaster records, real-time meteorological observations, and deep learning vision models into a unified Streamlit application.

The project addresses two closely linked challenges during monsoon seasons: estimating localized flood probabilities before inundation occurs, and evaluating agricultural consequences (crop disease vulnerability, yield reduction, and financial losses) in affected areas.

## Features

- District Risk Predictor: Computes real-time flood probability for any of India's 736 districts based on rainfall, river levels, terrain elevation, and historical vulnerability. Includes an interactive gauge, model confidence indicators, and actionable preparedness checklists.
- Pan-India Risk Map: Full-screen geospatial map built with Folium, categorizing all 736 districts into low, moderate, and high risk zones with instant search and state-level filtering.
- 7-Day Risk & Weather Forecast: Dynamic daily risk telemetry driven by live weather parameters (temperature, humidity, precipitation) fetched via Open-Meteo.
- Historical Flood Trends: Multi-year disaster analytics covering 2015-2024 from NDMA records, examining flood frequency, affected land area (hectares), human impact, and economic losses.
- Flood Damage Classifier: Computer vision pipeline using transfer learning to classify structural and environmental flood damage severity (mild, moderate, severe) from uploaded ground or aerial photographs.
- Crop Disease Diagnostics: Convolutional neural network recognizing 13 foliar crop pathologies commonly triggered by standing floodwaters and prolonged humidity.
- Crop Yield & Loss Estimator: Quantitative yield prediction and financial loss estimation calibrated against PMFBY (Pradhan Mantri Fasal Bima Yojana) compensation rates.
- Regional Alert System: Automated notification pipeline supporting threshold-configured regional alerts dispatched via email.
- Bilingual Localization: Complete user interface and safety advisories available in English and Hindi.

## Architecture & Tech Stack

- Application & UI: Streamlit, Custom Dark CSS
- Core Machine Learning: PyTorch, XGBoost, Scikit-learn
- Neural Architectures: EfficientNetB0 (Image Classification), LSTM with Temporal Attention (Sequential Modeling)
- Geospatial & Analytics: Folium, Plotly, Pandas, NumPy
- Weather Telemetry: Open-Meteo API
- Notification Dispatch: Python SMTP (Gmail TLS)

## Data Sources & Coverage

- Meteorological Data: India Meteorological Department (IMD) gridded rainfall data and seasonal telemetry.
- Disaster Records: National Disaster Management Authority (NDMA) flood event history from 2015 to 2024.
- Dataset Scale: 4,695 curated district-year records capturing precipitation, inundation history, topography, and reported damage.
- Spatial Resolution: Complete coverage of all 736 districts across 28 states and 8 Union Territories in India.

## Model Performance

| Model | Architecture | Primary Task | Metric |
| :--- | :--- | :--- | :--- |
| Flood Risk Predictor | XGBoost | Binary flood risk classification | AUC: 0.83 |
| Crop Yield Predictor | Gradient Boosting Regressor | District crop yield estimation | R²: 0.937 |
| Sequence Predictor | LSTM + Attention | Multi-day flood persistence | Recall: 0.70 |
| Damage Classifier | EfficientNetB0 | Flood damage severity (3 classes) | Accuracy: 89.6% |
| Crop Disease Detector | EfficientNetB0 | Foliar plant disease (13 classes) | Accuracy: 98.4% |

## Screenshots

| Risk Predictor Dashboard | Pan-India Risk Map |
| :---: | :---: |
| ![Risk Predictor Dashboard](docs/screenshots/risk_predictor.png) | ![Pan-India Risk Map](docs/screenshots/risk_map.png) |

| Crop Disease Detection | Historical Flood Analytics |
| :---: | :---: |
| ![Crop Disease Detection](docs/screenshots/crop_disease.png) | ![Historical Flood Analytics](docs/screenshots/flood_trends.png) |

## Project Structure

```text
flood-risk-prediction/
├── app/
│   ├── streamlit_app.py        # Main dashboard application
│   ├── styles.py               # Dark theme styling and layouts
│   └── ui_overrides.py         # Streamlit overrides and assets
├── src/
│   ├── alert_system.py         # Dispatch daemon and notifications
│   ├── chatbot.py              # Assistant query engine
│   ├── crop_disease_classifier.py # EfficientNet inference for crop diseases
│   ├── crop_loss_estimator.py  # PMFBY loss calculation engine
│   ├── crop_yield_predictor.py # Crop yield regression models
│   ├── flood_classifier.py     # Damage assessment vision pipeline
│   ├── flood_map.py            # Folium map generation
│   ├── forecast.py             # 7-day predictive telemetry
│   ├── pdf_report.py           # Automated district report generator
│   ├── translations.py         # Localization strings (EN / HI)
│   └── weather_api.py          # Open-Meteo live API integration
├── data/
│   ├── india_districts.csv     # 736-district catalog and metadata
│   └── ndma_flood_history.csv  # Historical flood dataset (2015-2024)
├── models/                     # Trained weights, encoders, and scalers
├── requirements.txt            # Dependency manifest
└── README.md
```

## Getting Started

### Prerequisites

- Python 3.10 or 3.11 recommended
- Git

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Y2XNishan/Floodguard-AI.git
   cd Floodguard-AI
   ```

2. Create and activate a virtual environment:
   ```bash
   # Windows (PowerShell)
   python -m venv venv
   .\venv\Scripts\Activate.ps1

   # Linux / macOS
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Configure environment variables (optional):
   Copy `.env.example` to `.env` if you want to configure automated email alerts:
   ```env
   GMAIL_USER=your_email@gmail.com
   GMAIL_APP_PASSWORD=your_16_digit_app_password
   ```

5. Launch the application:
   ```bash
   streamlit run app/streamlit_app.py
   ```

The dashboard will open automatically in your browser at `http://localhost:8501`.

## Limitations & Disclaimer

- Statistical Estimates: All predictions, risk classifications, and agricultural loss assessments produced by FloodGuard AI are statistical estimates generated from historical patterns, satellite-derived weather telemetry, and machine learning models. They are intended for preparedness, decision support, and academic research.
- Not a Replacement for Official Directives: This platform does not issue official disaster alerts. In the event of active floods, severe monsoon depressions, or evacuations, always refer to instructions issued by the National Disaster Management Authority (NDMA), Central Water Commission (CWC), State Disaster Management Authorities (SDMA), and district administrations.
- Official Helplines:
  - NDMA Disaster Helpline: 1078
  - National Disaster Response Force (NDRF): 011-24363260
  - Emergency Services: 112

## Author

Nishan Kashyap
- GitHub: [github.com/Y2XNishan](https://github.com/Y2XNishan)
- LinkedIn: [linkedin.com/in/nishankashyap](https://linkedin.com/in/nishankashyap)

## License

This project is licensed under the MIT License.
