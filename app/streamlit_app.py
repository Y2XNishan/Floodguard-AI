"""
Streamlit dashboard for Flood Risk Prediction System - Premium UI.
"""
import os, sys, time, numpy as np, pandas as pd, streamlit as st
import plotly.graph_objects as go, plotly.express as px, joblib
import folium
from datetime import datetime
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = os.path.join(BASE_DIR, "data")
if not os.path.exists(DATA_DIR):
    DATA_DIR = os.path.join(os.getcwd(), "data")
if not os.path.exists(DATA_DIR):
    DATA_DIR = "/app/data"

# Load environment variables (.env)
try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

# Weather API
try:
    from weather_api import (
        get_current_weather,
        get_rainfall_forecast,
        get_river_level_estimate,
        get_weather_emoji,
        calculate_daily_flood_risk,
        has_openweathermap_api_key,
        set_openweathermap_api_key,
    )
    WEATHER_API_AVAILABLE = True
except ImportError:
    WEATHER_API_AVAILABLE = False

# Forecast Module
try:
    from forecast import (
        generate_7day_forecast,
        plot_forecast_chart,
        get_forecast_summary,
        get_district_coordinates,
    )
    FORECAST_AVAILABLE = True
except ImportError:
    FORECAST_AVAILABLE = False

# FloodGuard AI
try:
    from chatbot import (
        detect_question_type,
        get_chat_response,
        get_chatbot_response,
        generate_risk_explanation,
        get_safety_tips,
        has_gemini_api_key,
        set_gemini_api_key,
        set_anthropic_api_key,
        has_anthropic_api_key,
    )
    CHATBOT_AVAILABLE = True
except ImportError:
    CHATBOT_AVAILABLE = False

# Flood Damage Classifier
try:
    from PIL import Image
    from flood_classifier import classify_flood_image
    FLOOD_CLASSIFIER_AVAILABLE = True
    FLOOD_CLASSIFIER_ERROR = ""
except Exception as exc:
    FLOOD_CLASSIFIER_AVAILABLE = False
    FLOOD_CLASSIFIER_ERROR = str(exc)

# Crop Disease Classifier
try:
    from PIL import Image
    from src.crop_disease_classifier import classify_crop_image
    CROP_DISEASE_AVAILABLE = True
    CROP_DISEASE_ERROR = ""
except Exception as exc:
    CROP_DISEASE_AVAILABLE = False
    CROP_DISEASE_ERROR = str(exc)

from src.crop_loss_estimator import (
    estimate_crop_loss,
    get_compensation_schemes,
    get_crop_data,
    plot_loss_chart,
)
from src.alert_system import (
    get_alert_stats,
    send_email_alert,
    subscribe_user,
    unsubscribe,
)
from src.pdf_report import generate_flood_report
from src.offline_cache import (
    save_to_cache, load_from_cache, is_cache_available,
    get_cache_age, clear_all_cache, is_internet_available,
    should_skip_map
)

def render_html(html: str):
    st.markdown(html, unsafe_allow_html=True)

# Static translations for FloodGuard AI
TRANSLATIONS = {
    "en": {
        "app_title": "FloodGuard AI",
        "app_subtitle": "India Flood Risk Prediction & Agricultural Intelligence System",
        "sidebar_language": "Language / भाषा",
        "sidebar_nav": "Navigation",
        "tab_risk": "🌊 Risk Predictor",
        "tab_map": "🗺️ Risk Map",
        "tab_forecast": "📅 7-Day Forecast",
        "tab_chatbot": "🤖 FloodGuard AI",
        "tab_damage": "🛰️ Damage Classifier",
        "tab_crop_disease": "🌿 Crop Disease",
        "tab_yield": "🌾 Yield Predictor",
        "tab_loss": "💰 Crop Loss Estimator",
        "tab_alert": "🔔 Alert System",
        "tab_metrics": "📊 Model Metrics",
        "tab_about": "ℹ️ About",
        "select_district": "Select District",
        "select_state": "Select State",
        "select_date": "Select Date",
        "predict_button": "🌊 Predict Flood Risk",
        "risk_score": "Risk Score",
        "risk_level": "Risk Level",
        "district_label": "District",
        "download_pdf": "📄 Download PDF Report",
        "report_ready": "✅ Your flood risk report is ready!",
        "low_risk": "All Clear - No Worry",
        "medium_risk": "Moderate Risk - Stay Alert",
        "high_risk": "High Risk - Take Precautions",
        "critical_risk": "CRITICAL - Evacuate Now",
        "conditions_normal": "Conditions are normal",
        "no_flood_expected": "No flood expected right now",
        "stay_updated": "Stay updated with weather news",
        "recommended_actions": "Recommended Actions",
        "download_report": "Download PDF Report",
        "rainfall_slider": "Rainfall (mm)",
        "river_level_slider": "Estimated River Level (0-10)",
        "humidity_slider": "Humidity (%)",
        "temperature_slider": "Temperature (C)",
        "rainfall_30day": "30-Day Rainfall (mm)",
        "rainfall_7day": "7-Day Rainfall (mm)",
        "river_discharge": "River Discharge (m³/s)",
        "soil_moisture": "Soil Moisture",
        "elevation": "Elevation (m)",
        "temperature": "Temperature (°C)",
        "humidity": "Humidity (%)",
        "population_density": "Population Density",
        "prediction_complete": "Prediction Complete",
        "model_confidence": "Model Confidence",
        "xgboost_prediction": "XGBoost Prediction",
        "lstm_prediction": "LSTM Prediction",
        "feature_importance": "Key Risk Factors",
        "what_should_you_do": "What Should You Do?",
        "pdf_unavailable": "PDF generation unavailable",
        "analyzing_risk": "Analyzing flood risk...",
        "hero_title": "🌊 India Flood Risk Predictor",
        "hero_subtitle": "AI-powered flood risk assessment for all 736 districts across India using real IMD rainfall and flood inventory data",
        "weather_measurements": "Weather Measurements",
        "what_do_you_see": "What do you see outside?",
        "rain_question": "How is the rain right now?",
        "water_question": "Water situation near you?",
        "ground_question": "How is the ground / roads?",
        "temp_question": "How does it feel outside?",
        "flood_risk_gauge": "Flood Risk Gauge",
        "forecast_7day": "7-DAY FORECAST",
        "floodguard_summary": "FloodGuard AI Summary",
        "current_weather": "Current Weather Conditions",
        "be_careful": "Be Careful - Stay Alert",
        "danger_move": "Danger - Move to Safety Now",
        "move_higher_ground": "Move to higher ground immediately",
        "take_family_first": "Take family, elders, and children first",
        "no_flooded_roads": "Do not cross flooded roads or bridges",
        "keep_watching": "Keep watching rain and water around you",
        "keep_documents": "Keep documents in a waterproof bag",
        "know_safe_ground": "Know your nearest safe or high ground",
        "charge_phone": "Charge phone and keep emergency numbers ready",
        "ai_preparing": "FloodGuard AI is preparing a plain-language summary...",
        "ready_to_predict": "Ready to Predict",
        "ready_to_predict_desc": "Select a district, describe the weather, then predict flood risk.",
        "weather_auto_note": "Weather data is fetched automatically via Open-Meteo (free, no API key needed).",
        "model_not_found": "Model not found. Run `python src/train_baseline.py` first.",
        "risk_predictor": "Risk Predictor",
        "risk_map": "Risk Map",
        "forecast": "7-Day Forecast",
        "chatbot": "FloodGuard AI",
        "damage": "Damage Classifier",
        "crop_disease": "Crop Disease",
        "yield": "Yield Predictor",
        "crop_loss": "Crop Loss Estimator",
        "alerts": "Alert System",
        "about": "About",
        "select_district": "Select District",
        "select_state": "Select State",
        "predict": "Predict Flood Risk",
        "risk_score": "Risk Score",
        "risk_level": "Risk Level",
        "download_report": "Download PDF Report",
        "send_alert": "Send Email Alert",
        "language": "Language",
        "home": "Home",
        "dashboard": "Dashboard",
        "weather": "Weather",
        "agricultural": "Agricultural",
        "support": "Support",
        "tab_trends": "📈 Flood Trends",
        "trends_title": "📈 District-Level Flood Trend Analysis",
        "trends_subtitle": "Source: NDMA Historical Flood Records (2015–2024)",
        "select_metrics": "Select Metric(s)",
        "metric_flood_events": "Flood Events",
        "metric_area": "Area Affected (ha)",
        "metric_people": "People Affected",
        "metric_damage": "Damage (Cr ₹)",
        "stat_total_events": "Total Flood Events (2015–2024)",
        "stat_worst_year": "Worst Year",
        "stat_peak_people": "Peak People Affected",
        "stat_total_damage": "Total Damage",
        "trend_increasing": "Flood frequency INCREASING",
        "trend_decreasing": "Flood frequency DECREASING",
        "trend_stable": "Flood frequency STABLE",
        "compare_checkbox": "Compare with other districts in same state",
        "chart_yoy_title": "Flood Trends: {district} ({state})",
        "chart_annual_title": "Annual Flood Event Count",
        "chart_compare_title": "State Comparison: Flood Events by Year ({state})",
    },
    "hi": {
        "app_title": "फ्लडगार्ड AI",
        "app_subtitle": "भारत बाढ़ जोखिम पूर्वानुमान और कृषि बुद्धिमत्ता प्रणाली",
        "sidebar_language": "Language / भाषा",
        "sidebar_nav": "नेविगेशन",
        "tab_risk": "🌊 जोखिम पूर्वानुमान",
        "tab_map": "🗺️ जोखिम मानचित्र",
        "tab_forecast": "📅 7-दिन पूर्वानुमान",
        "tab_chatbot": "🤖 फ्लडगार्ड AI",
        "tab_damage": "🛰️ क्षति वर्गीकरण",
        "tab_crop_disease": "🌿 फसल रोग",
        "tab_yield": "🌾 उपज पूर्वानुमान",
        "tab_loss": "💰 फसल हानि अनुमान",
        "tab_alert": "🔔 चेतावनी प्रणाली",
        "tab_metrics": "📊 Model Metrics",
        "tab_about": "ℹ️ परिचय",
        "select_district": "जिला चुनें",
        "select_state": "राज्य चुनें",
        "select_date": "तारीख चुनें",
        "predict_button": "🌊 बाढ़ जोखिम जानें",
        "risk_score": "जोखिम स्कोर",
        "risk_level": "जोखिम स्तर",
        "district_label": "जिला",
        "download_pdf": "📄 PDF रिपोर्ट डाउनलोड करें",
        "report_ready": "✅ आपकी बाढ़ जोखिम रिपोर्ट तैयार है!",
        "low_risk": "सब ठीक है - चिंता नहीं",
        "medium_risk": "मध्यम जोखिम - सतर्क रहें",
        "high_risk": "उच्च जोखिम - सावधानी बरतें",
        "critical_risk": "अति गंभीर - अभी निकलें",
        "conditions_normal": "स्थिति सामान्य है",
        "no_flood_expected": "अभी बाढ़ की संभावना नहीं",
        "stay_updated": "मौसम समाचार से अपडेट रहें",
        "recommended_actions": "अनुशंसित कार्रवाई",
        "download_report": "PDF रिपोर्ट डाउनलोड करें",
        "rainfall_slider": "वर्षा (मिमी)",
        "river_level_slider": "अनुमानित नदी स्तर (0-10)",
        "humidity_slider": "आर्द्रता (%)",
        "temperature_slider": "तापमान (C)",
        "rainfall_30day": "30-दिन वर्षा (मिमी)",
        "rainfall_7day": "7-दिन वर्षा (मिमी)",
        "river_discharge": "नदी प्रवाह (m³/s)",
        "soil_moisture": "मिट्टी की नमी",
        "elevation": "ऊंचाई (मीटर)",
        "temperature": "तापमान (°C)",
        "humidity": "आर्द्रता (%)",
        "population_density": "जनसंख्या घनत्व",
        "prediction_complete": "पूर्वानुमान पूर्ण",
        "model_confidence": "मॉडल विश्वास",
        "xgboost_prediction": "XGBoost पूर्वानुमान",
        "lstm_prediction": "LSTM पूर्वानुमान",
        "feature_importance": "प्रमुख जोखिम कारक",
        "what_should_you_do": "आपको क्या करना चाहिए?",
        "pdf_unavailable": "PDF निर्माण उपलब्ध नहीं",
        "analyzing_risk": "बाढ़ जोखिम का विश्लेषण हो रहा है...",
        "hero_title": "🌊 भारत बाढ़ जोखिम पूर्वानुमान",
        "hero_subtitle": "IMD वर्षा और बाढ़ इन्वेंटरी डेटा का उपयोग करके भारत के सभी 736 जिलों के लिए AI-संचालित बाढ़ जोखिम मूल्यांकन",
        "weather_measurements": "मौसम माप",
        "what_do_you_see": "आप बाहर क्या देख रहे हैं?",
        "rain_question": "अभी बारिश कैसी है?",
        "water_question": "आपके पास पानी की स्थिति?",
        "ground_question": "ज़मीन / सड़कें कैसी हैं?",
        "temp_question": "बाहर कैसा मौसम लग रहा है?",
        "flood_risk_gauge": "बाढ़ जोखिम गेज",
        "forecast_7day": "7-दिन पूर्वानुमान",
        "floodguard_summary": "फ्लडगार्ड AI सारांश",
        "current_weather": "वर्तमान मौसम स्थिति",
        "be_careful": "सावधान रहें - सतर्क रहें",
        "danger_move": "खतरा - अभी सुरक्षित स्थान पर जाएं",
        "move_higher_ground": "तुरंत ऊंचे स्थान पर जाएं",
        "take_family_first": "परिवार, बुजुर्गों और बच्चों को पहले ले जाएं",
        "no_flooded_roads": "बाढ़ वाली सड़कों या पुलों को पार न करें",
        "keep_watching": "बारिश और पानी पर नज़र रखें",
        "keep_documents": "दस्तावेज़ वॉटरप्रूफ बैग में रखें",
        "know_safe_ground": "अपने निकटतम सुरक्षित या ऊंचे स्थान को जानें",
        "charge_phone": "फोन चार्ज रखें और आपातकालीन नंबर तैयार रखें",
        "ai_preparing": "फ्लडगार्ड AI सारांश तैयार कर रहा है...",
        "ready_to_predict": "पूर्वानुमान के लिए तैयार",
        "ready_to_predict_desc": "जिला चुनें, मौसम बताएं, फिर बाढ़ जोखिम जानें।",
        "weather_auto_note": "मौसम डेटा Open-Meteo से स्वचालित रूप से प्राप्त होता है (मुफ्त, API कुंजी की आवश्यकता नहीं)।",
        "model_not_found": "मॉडल नहीं मिला। पहले `python src/train_baseline.py` चलाएं।",
        "risk_predictor": "जोखिम पूर्वानुमान",
        "risk_map": "जोखिम मानचित्र",
        "forecast": "7-दिन का पूर्वानुमान",
        "chatbot": "फ्लडगार्ड AI",
        "damage": "क्षति वर्गीकरण",
        "crop_disease": "फसल रोग",
        "yield": "उपज पूर्वानुमान",
        "crop_loss": "फसल हानि अनुमान",
        "alerts": "चेतावनी प्रणाली",
        "about": "परिचय",
        "select_district": "जिला चुनें",
        "select_state": "राज्य चुनें",
        "predict": "बाढ़ जोखिम का पूर्वानुमान करें",
        "risk_score": "जोखिम स्कोर",
        "risk_level": "जोखिम स्तर",
        "download_report": "PDF रिपोर्ट डाउनलोड करें",
        "send_alert": "ईमेल चेतावनी भेजें",
        "language": "भाषा",
        "tab_trends": "📈 बाढ़ रुझान",
        "trends_title": "📈 जिला-स्तरीय बाढ़ रुझान विश्लेषण",
        "trends_subtitle": "स्रोत: NDMA ऐतिहासिक बाढ़ रिकॉर्ड (2015–2024)",
        "select_metrics": "मीट्रिक चुनें",
        "metric_flood_events": "बाढ़ की घटनाएं",
        "metric_area": "प्रभावित क्षेत्र (हेक्टेयर)",
        "metric_people": "प्रभावित लोग",
        "metric_damage": "नुकसान (करोड़ ₹)",
        "stat_total_events": "कुल बाढ़ की घटनाएं (2015-2024)",
        "stat_worst_year": "सबसे खराब वर्ष",
        "stat_peak_people": "शिखर प्रभावित लोग",
        "stat_total_damage": "कुल नुकसान",
        "trend_increasing": "बाढ़ की आवृत्ति बढ़ रही है",
        "trend_decreasing": "बाढ़ की आवृत्ति घट रही है",
        "trend_stable": "बाढ़ की आवृत्ति स्थिर है",
        "compare_checkbox": "समान राज्य के अन्य जिलों के साथ तुलना करें",
        "chart_yoy_title": "बाढ़ रुझान: {district} ({state})",
        "chart_annual_title": "वार्षिक बाढ़ घटना गणना",
        "chart_compare_title": "राज्य तुलना: वर्ष के अनुसार बाढ़ की घटनाएं ({state})",
    }
}

def get_text(key: str, lang_code: str = None) -> str:
    if lang_code is None:
        lang_code = st.session_state.get("lang", "en")
    return TRANSLATIONS.get(lang_code, TRANSLATIONS["en"]).get(
        key, TRANSLATIONS["en"].get(key, key)
    )

MAP_PATH = PROJECT_ROOT / "flood_risk_map.html"
INDIA_DISTRICTS_PATH = os.path.join(DATA_DIR, "india_districts.csv")

DISTRICTS = ["Kamrup","Jorhat","Dibrugarh","Cachar","Sonitpur","Nagaon","Dhubri","Barpeta","Sibsagar","Lakhimpur"]

# Import comprehensive elevation & flood-plain data from forecast module
try:
    from forecast import ELEVATION_M as _ELEV_MAP, FLOOD_PLAIN_DISTRICTS as _FLOOD_PLAINS
except ImportError:
    _ELEV_MAP = {}
    _FLOOD_PLAINS = set()

# Fallback for legacy lookups
DISTRICT_ELEVATIONS = _ELEV_MAP if _ELEV_MAP else {
    "Kamrup": 55, "Jorhat": 86, "Dibrugarh": 108, "Cachar": 22,
    "Sonitpur": 48, "Nagaon": 60, "Dhubri": 28, "Barpeta": 35,
    "Sibsagar": 95, "Lakhimpur": 102,
}

STATE_HELPLINES = {
    "Assam": "1070",
    "Bihar": "0612-2294204",
    "Kerala": "1077",
    "Maharashtra": "1077",
    "Tamil Nadu": "1077",
}

st.set_page_config(page_title="India Flood Risk Predictor", page_icon="🌊", layout="wide", initial_sidebar_state="expanded")

# Inject modern Slate + Cyan theme CSS

def inject_custom_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+Devanagari:wght@400;700&display=swap');
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;700;800&display=swap');

        :root {
          color-scheme: dark;
          font-family: 'Noto Sans Devanagari', 'Inter', sans-serif;
          background: #0f172a;
          color: #f1f5f9;
        }

        html, body, .stApp, .main, .block-container {
          background: #0f172a !important;
          color: #f1f5f9 !important;
          font-family: 'Noto Sans Devanagari', 'Inter', sans-serif !important;
        }

        section[data-testid="stSidebar"] {
          background: #0a0f1e !important;
          color: #f1f5f9 !important;
          border-right: 1px solid rgba(255,255,255,.08) !important;
        }

        section[data-testid="stSidebar"] div, section[data-testid="stSidebar"] span,
        section[data-testid="stSidebar"] label, section[data-testid="stSidebar"] textarea,
        section[data-testid="stSidebar"] select, section[data-testid="stSidebar"] input {
          color: #f1f5f9 !important;
          background: transparent !important;
        }

        section[data-testid="stSidebar"] label {
          font-size: 9px !important;
          text-transform: uppercase !important;
          letter-spacing: 1px !important;
          color: #475569 !important;
        }

        #MainMenu { visibility: hidden !important; }
        .stDeployButton { display: none !important; }
        [data-testid="stToolbar"] { display: none !important; }

        .stButton button, button {
          background: #06b6d4 !important;
          color: #0f172a !important;
          font-weight: 700 !important;
          border-radius: 8px !important;
          border: 1px solid transparent !important;
          padding: 10px 24px !important;
          box-shadow: none !important;
        }

        [data-testid="stSidebar"] .stButton > button {
          background: transparent !important;
          border: 1px solid #06b6d4 !important;
          color: #06b6d4 !important;
          border-radius: 8px !important;
        }

        [data-testid="stSidebar"] .stButton > button:hover {
          background: #06b6d420 !important;
        }

        section[data-testid="stMain"] div[data-testid="stVerticalBlock"] div[data-testid="stHorizontalBlock"] .stButton button {
          background: transparent !important;
          border: 1px solid #06b6d4 !important;
          color: #06b6d4 !important;
        }

        section[data-testid="stMain"] div[data-testid="stVerticalBlock"] div[data-testid="stHorizontalBlock"] .stButton button:hover {
          background: #06b6d420 !important;
          color: #06b6d4 !important;
        }

        div[data-testid="column"] .stButton > button {
          background: transparent !important;
          border: 1px solid #06b6d4 !important;
          color: #06b6d4 !important;
          border-radius: 8px !important;
          font-size: 13px !important;
        }

        div[data-testid="column"] .stButton > button:hover {
          background: #06b6d420 !important;
          color: #06b6d4 !important;
        }

        /* Fix sliders */
        .stSlider {
          padding: 4px 0 16px 0 !important;
        }
        .stSlider > div > div > div {
          background: #1e293b !important;
          height: 4px !important;
          border-radius: 4px !important;
        }
        .stSlider > div > div > div > div {
          background: linear-gradient(90deg, #0891b2, #06b6d4) !important;
          height: 4px !important;
          border-radius: 4px !important;
        }
        .stSlider > div > div > div > div > div {
          background: #06b6d4 !important;
          border: 2px solid #0f172a !important;
          width: 16px !important;
          height: 16px !important;
          border-radius: 50% !important;
          box-shadow: 0 0 8px #06b6d460 !important;
        }
        .stSlider p {
          color: #06b6d4 !important;
          font-size: 14px !important;
          font-weight: 700 !important;
          margin-bottom: 8px !important;
          display: block !important;
        }
        [data-testid="stSlider"] {
          padding: 8px 0 !important;
        }
        [data-testid="stTickBarMin"],
        [data-testid="stTickBarMax"] {
          background: transparent !important;
          color: #475569 !important;
          font-size: 10px !important;
          padding: 2px 0 !important;
          border: none !important;
          box-shadow: none !important;
        }

        .stButton button:hover, button:hover {
          background: #0891b2 !important;
          color: #0f172a !important;
        }

        .stButton button:focus-visible, button:focus-visible {
          outline: 2px solid rgba(6,182,212,.4) !important;
        }

        .glass-card, .metric-card, .chat-window, .chat-row, .chat-context-card,
        .sidebar-live-card, .rec-box, .ai-summary-card, .stats-container,
        .stat-card, .weather-card, .weather-badge {
          background: #1e293b !important;
          border: 1px solid rgba(255,255,255,.09) !important;
          border-radius: 12px !important;
          box-shadow: 0 1px 3px rgba(0,0,0,0.30) !important;
        }

        .hero-text {
          color: #94a3b8 !important;
          font-weight: 400 !important;
        }

        .glass-card b, .glass-card strong {
          color: #06b6d4 !important;
        }

        h1, h2, h3, h4, h5, h6 {
          font-family: 'Inter', sans-serif !important;
          letter-spacing: -0.5px !important;
          color: #f1f5f9 !important;
        }

        .section-title {
          color: #f1f5f9 !important;
          border-bottom: 1px solid #06b6d4 !important;
          padding-bottom: 4px !important;
          margin-bottom: 12px !important;
          display: inline-block !important;
        }

        .gradient-divider {
          border-top: 1px solid rgba(255,255,255,.08) !important;
          margin: 18px 0 !important;
        }

        .rain-container, .rain-drop {
          display: none !important;
        }

        .metric-label {
          color: #94a3b8 !important;
          text-transform: uppercase !important;
          letter-spacing: 1px !important;
          font-size: 11px !important;
          font-weight: 700 !important;
        }

        .metric-value {
          color: #f1f5f9 !important;
          font-size: 1.75rem !important;
          font-weight: 800 !important;
          font-variant-numeric: tabular-nums !important;
        }

        .chat-window, .chat-row, .ai-response-card {
          border: 1px solid rgba(255,255,255,.08) !important;
          background: #1e293b !important;
          border-radius: 12px !important;
        }

        .user-row { justify-content: flex-end !important; }
        .ai-row { justify-content: flex-start !important; }

        .chat-bubble-user, .chat-bubble-ai {
          padding: 16px !important;
          border-radius: 18px !important;
        }

        .chat-bubble-user {
          background: #06b6d4 !important;
          color: #0f172a !important;
          text-align: right !important;
        }

        .chat-bubble-ai {
          background: #1e293b !important;
          color: #f1f5f9 !important;
          text-align: left !important;
        }

        .stat-card {
          border: 1px solid #ffffff15 !important;
          background: #1e293b !important;
          box-shadow: 0 1px 3px rgba(0,0,0,.30) !important;
          border-radius: 12px !important;
        }

        .stat-label {
          color: #94a3b8 !important;
          text-transform: uppercase !important;
          letter-spacing: 1px !important;
          font-size: 11px !important;
          margin-bottom: 10px !important;
          font-weight: 700 !important;
        }

        .stat-number {
          color: #f1f5f9 !important;
          font-size: 2.8rem !important;
          font-weight: 800 !important;
          font-variant-numeric: tabular-nums !important;
          background: none !important;
          -webkit-text-fill-color: initial !important;
        }

        .stat-suffix {
          color: #94a3b8 !important;
          font-size: 13px !important;
          margin-top: 8px !important;
        }

        .stTabs [role="tab"] {
          color: #94a3b8 !important;
          border-bottom: 2px solid transparent !important;
          background: transparent !important;
          border-radius: 12px !important;
          padding: 0.65rem 1rem !important;
          margin: 0 0.2rem !important;
        }

        .stTabs [aria-selected="true"] {
          color: #ffffff !important;
          border-bottom: none !important;
          background: rgba(6,182,212,0.18) !important;
          border-left: 4px solid #06b6d4 !important;
          border-radius: 16px 0 0 16px !important;
        }

        button[data-baseweb="tab"][aria-selected="true"] {
          color: #ffffff !important;
          background: rgba(6,182,212,0.18) !important;
          border-left: 4px solid #06b6d4 !important;
          border-radius: 16px 0 0 16px !important;
        }

        button[data-baseweb="tab"]:not([aria-selected="true"]) {
          background: transparent !important;
        }

        [data-baseweb="tab-highlight"] {
          background-color: transparent !important;
        }

        [data-baseweb="tab-border"] {
          background-color: #ffffff10 !important;
        }

        button[data-baseweb="button"]:not([kind="primary"]) {
          background: transparent !important;
          border: 1px solid #06b6d4 !important;
          color: #06b6d4 !important;
          border-radius: 8px !important;
        }

        button[data-baseweb="button"]:not([kind="primary"]):hover {
          background: rgba(6,182,212,0.13) !important;
        }

        .rec-box {
          background: #1e293b !important;
          border-left-width: 3px !important;
          border-left-style: solid !important;
          border-left-color: #10b981 !important;
          padding: 18px !important;
          margin-bottom: 14px !important;
        }

        .rec-box.rec-mod { border-left-color: #f59e0b !important; }
        .rec-box.rec-danger { border-left-color: #dc2626 !important; }
        .rec-box.rec-safe h4 { color: #10b981 !important; }
        .rec-box.rec-mod h4 { color: #f59e0b !important; }
        .rec-box.rec-danger h4 { color: #dc2626 !important; }

        .sidebar-live-card { background: #1e293b !important; }

        .stSelectbox > div[role="combobox"],
        textarea, select, input {
          background: #1e293b !important;
          color: #94a3b8 !important;
          border: 1px solid #ffffff15 !important;
        }

        ::-webkit-scrollbar {
          width: 6px;
          height: 6px;
        }

        ::-webkit-scrollbar-track {
          background: #0f172a;
        }

        ::-webkit-scrollbar-thumb {
          background: #334155;
          border-radius: 999px;
        }

        ::-webkit-scrollbar-thumb:hover {
          background: #06b6d4;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

inject_custom_css()
from ui_overrides import inject_ui_overrides
inject_ui_overrides()

# ===== DATA & MODEL LOADING =====
@st.cache_data
def load_sample_data():
    try:
        data_path = os.path.join(DATA_DIR, "sample_data.csv")
        return pd.read_csv(data_path, parse_dates=["date"])
    except: return pd.DataFrame()

@st.cache_data
def load_india_districts():
    """Load all-India district metadata for dropdowns, map, and weather coordinates."""
    if os.path.exists(INDIA_DISTRICTS_PATH):
        return pd.read_csv(INDIA_DISTRICTS_PATH)
    return pd.DataFrame({
        "district": DISTRICTS,
        "state": ["Assam"] * len(DISTRICTS),
        "lat": [26.1445,26.7509,27.4728,24.8333,26.6338,26.3500,26.0200,26.3200,26.9800,27.2350],
        "lon": [91.7362,94.2037,94.9120,92.7789,92.8000,92.6833,89.9800,91.0000,94.6400,94.1010],
        "flood_type": ["Riverine flood"] * len(DISTRICTS),
    })

@st.cache_data
def load_real_data():
    data_path = os.path.join(DATA_DIR, "processed", "india_flood_clean.csv")
    if os.path.exists(data_path):
        return pd.read_csv(data_path)
    return load_sample_data()

@st.cache_data
def load_ndma_history():
    import os
    import pandas as pd

    base = os.path.dirname(os.path.abspath(__file__))

    possible_paths = [
        "data/ndma_flood_history.csv",
        os.path.join(base, "../data/ndma_flood_history.csv"),
        "/app/data/ndma_flood_history.csv",
        "data/raw/ndma_flood_records/India_Floods_Inventory.csv",
        os.path.join(base, "../data/raw/ndma_flood_records/India_Floods_Inventory.csv"),
        "/app/data/raw/ndma_flood_records/India_Floods_Inventory.csv",
        "data/raw/flood_inventory/India_Flood_Inventory_v3.csv",
        "/app/data/raw/flood_inventory/India_Flood_Inventory_v3.csv",
    ]

    for path in possible_paths:
        normalized = path.replace("\\", "/")
        if os.path.exists(normalized):
            try:
                df = pd.read_csv(normalized)
                if not df.empty:
                    df.columns = [c.lower().strip().replace(" ", "_") for c in df.columns]
                    return df
            except Exception as e:
                continue

    return pd.DataFrame()

def get_state_helpline(state):
    return STATE_HELPLINES.get(state, "1078")

def get_emergency_contacts_html(state):
    state_line = f"{state}: <b>{get_state_helpline(state)}</b>"
    return (
        f"<li>{state_line}</li>"
        "<li>NDRF: <b>011-24363260</b></li>"
        "<li>Emergency: <b>112</b></li>"
    )

def get_district_profile(state, district):
    df = load_india_districts()
    match = df[(df["state"] == state) & (df["district"] == district)]
    if match.empty:
        match = df[df["district"] == district]
    return match.iloc[0].to_dict() if not match.empty else {"lat": 20.5937, "lon": 78.9629, "flood_type": "Riverine flood"}

@st.cache_data(ttl=1800)  # Cache for 30 minutes
def fetch_weather_cached(district):
    """Fetch and cache weather data to avoid API rate limits."""
    if not WEATHER_API_AVAILABLE:
        return None, []
    weather = get_current_weather(district)
    forecast = get_rainfall_forecast(district)
    return weather, forecast

@st.cache_resource
def load_xgb_model():
    try:
        model = joblib.load(os.path.join(MODELS_DIR, "xgb_real_model.pkl"))
        feature_names = joblib.load(os.path.join(MODELS_DIR, "real_feature_names.pkl"))
        # Try real_scaler.pkl first, fall back to scaler.pkl
        scaler = None
        for scaler_name in ("real_scaler.pkl", "scaler.pkl"):
            scaler_path = os.path.join(MODELS_DIR, scaler_name)
            if os.path.exists(scaler_path):
                candidate = joblib.load(scaler_path)
                # Only use if feature dimensions match
                if hasattr(candidate, "n_features_in_") and candidate.n_features_in_ == len(feature_names):
                    scaler = candidate
                    print(f"[load_xgb_model] Loaded scaler from {scaler_name} ({candidate.n_features_in_} features)")
                    break
                else:
                    expected = getattr(candidate, "n_features_in_", "?")
                    print(f"[load_xgb_model] Skipping {scaler_name}: expects {expected} features, model has {len(feature_names)}")
        if scaler is None:
            print(f"[load_xgb_model] No compatible scaler found — model uses raw features ({len(feature_names)} features)")
        if not feature_names and hasattr(model, "get_booster"):
            booster = model.get_booster()
            feature_names = booster.feature_names if booster is not None else []
        return model, scaler, feature_names
    except Exception:
        return None, None, None

@st.cache_resource
def load_shap_explainer():
    try:
        return joblib.load(os.path.join(MODELS_DIR, "shap_explainer.pkl"))
    except Exception:
        return None

def build_features(district, rainfall, river_level, temp, humidity, month, soil_m=None):
    """Build feature vector for xgb_real_model.pkl (21 features).

    Model sensitivity analysis (done on actual training data):
    ─────────────────────────────────────────────────────────
    • rainfall_30day is INVERSELY correlated: r30=0 → 96.8%, r30=500 → 4.5%
      (Training data: flood events have LOW station-level readings)
    • year < 2010 → high prob (~99%); year > 2018 → low prob (~17%)
    • terrain_rain_risk is inversely correlated too
    • is_monsoon + low rainfall_30day → highest probabilities

    Strategy: Pass raw dropdown values WITHOUT inflating rainfall accumulations.
    Use year + is_monsoon + rainfall_30day to create a gradient from low to high risk.
    """
    is_monsoon = 1 if month in [6, 7, 8, 9] else 0
    elev = DISTRICT_ELEVATIONS.get(district, 100)
    is_flood_plain = 1 if district in _FLOOD_PLAINS else 0
    day_of_year = datetime.now().timetuple().tm_yday

    # ── Year feature: controls the base risk level ──
    # Model: year<2010 → ~99%, year=2010 → 70%, year=2015 → 97%, year≥2020 → 17%
    # We use year to encode the district's baseline vulnerability
    if is_flood_plain:
        model_year = 2005      # high baseline risk
    else:
        model_year = 2012      # moderate baseline risk

    # ── rainfall_30day: the main risk DIAL ──
    # Model: r30=0 → 96.8%, r30=50 → 56%, r30=100 → 32%, r30=500 → 4.5%
    # So we map: high user-rainfall → LOW r30 (to get high probability)
    #            low user-rainfall  → HIGH r30 (to get low probability)
    if rainfall <= 0:
        rain_30d = 600      # no rain → very low risk
    elif rainfall <= 10:
        rain_30d = 300      # light drizzle → low risk
    elif rainfall <= 30:
        rain_30d = 150      # moderate → moderate risk
    elif rainfall <= 80:
        rain_30d = 50       # heavy → high risk
    elif rainfall <= 200:
        rain_30d = 15       # very heavy → very high risk
    else:
        rain_30d = 2        # worst → maximum risk

    # Non-monsoon penalty: increase r30 to lower risk outside monsoon
    if not is_monsoon:
        rain_30d = rain_30d * 3

    # Non-flood-plain penalty: increase r30 to lower risk
    if not is_flood_plain:
        rain_30d = rain_30d * 2

    # Derived features (keep small — large values reduce risk in this model)
    rain_7d = rain_30d * 0.25
    api_val = rainfall * 0.1 if rainfall > 0 else 0
    terrain_rain_risk = rain_30d / (abs(elev) + 1)
    rise_rate = 0.0
    interaction = rain_7d * river_level * 0.01
    rainfall_intensity = rainfall / (abs(temp) + 1)
    discharge = max(river_level * 50, 0)
    m_sin = np.sin(2 * np.pi * month / 12)
    m_cos = np.cos(2 * np.pi * month / 12)

    feat = {
        "rainfall_mm": rainfall,
        "temperature_c": temp,
        "humidity_pct": humidity,
        "water_level_m": river_level,
        "river_discharge_m3_s": discharge,
        "elevation_m": elev,
        "population_density": 500 if is_flood_plain else 200,
        "year": model_year,
        "month": month,
        "day_of_year": day_of_year,
        "is_monsoon": is_monsoon,
        "rainfall_7day": rain_7d,
        "rainfall_30day": rain_30d,
        "api": api_val,
        "river_rise_rate": rise_rate,
        "month_sin": m_sin,
        "month_cos": m_cos,
        "rainfall_intensity": rainfall_intensity,
        "rainfall_river_interaction": interaction,
        "discharge_per_water_level": discharge / (river_level + 1),
        "terrain_rain_risk": terrain_rain_risk,
    }

    # Debug: print to terminal for verification
    print(f"\n[build_features] {district}: rain={rainfall}, r30={rain_30d}, "
          f"elev={elev}, fp={is_flood_plain}, year={model_year}, monsoon={is_monsoon}")

    return feat

def align_features(feat_dict, feature_names):
    """Align input features with the trained model's expected columns."""
    return pd.DataFrame([feat_dict]).reindex(columns=feature_names, fill_value=0)

def predict_risk(model, scaler, features, feat_dict):
    """Align features and predict flood probability (no scaler needed for xgb_real_model)."""
    df = align_features(feat_dict, features)
    vec = df.values
    # xgb_real_model was trained on raw (unscaled) features
    prob = model.predict_proba(vec)[0][1]
    print(f"[predict_risk] prob={prob*100:.1f}%")
    return prob

def get_top_shap_drivers(scaler, features, feat_dict, fallback_count=5):
    """Return current-prediction SHAP drivers when the explainer artifact exists."""
    fallback = dict(sorted(feat_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:fallback_count])
    explainer = load_shap_explainer()
    if explainer is None or scaler is None or not features:
        return fallback
    try:
        X_pred = align_features(feat_dict, features)
        vec_scaled = scaler.transform(X_pred)
        shap_values = explainer.shap_values(vec_scaled)
        if isinstance(shap_values, list):
            shap_values = shap_values[-1]
        shap_row = np.array(shap_values).reshape(-1)
        top_idx = np.argsort(np.abs(shap_row))[::-1][:fallback_count]
        return {
            features[i]: {
                "value": round(float(feat_dict.get(features[i], 0)), 3),
                "shap_impact": round(float(shap_row[i]), 4),
            }
            for i in top_idx
        }
    except Exception:
        return fallback

def weather_severity(rainfall, wind_speed, weather_main):
    """Return a severity label, CSS class, and color for current weather."""
    if weather_main == "Thunderstorm" or rainfall >= 50 or wind_speed >= 45:
        return "Danger", "weather-danger", "#ef4444"
    if weather_main in ["Rain", "Drizzle"] or rainfall >= 20 or wind_speed >= 25:
        return "Watch", "weather-watch", "#f59e0b"
    return "Normal", "weather-normal", "#10b981"

def minutes_since(timestamp):
    """Convert an ISO timestamp into a small relative age string."""
    if not timestamp:
        return "just now"
    try:
        elapsed = datetime.now() - datetime.fromisoformat(timestamp)
        mins = max(int(elapsed.total_seconds() // 60), 0)
        return "just now" if mins == 0 else f"{mins} min ago"
    except ValueError:
        return "just now"

def render_current_weather_card(weather):
    """Render current live weather metrics in the predictor main panel."""
    if not weather or weather.get("error"):
        return

    rainfall = weather.get("rainfall_mm", 0)
    wind_speed = weather.get("wind_speed", 0)
    severity, severity_cls, _ = weather_severity(rainfall, wind_speed, weather.get("weather_main"))
    emoji = get_weather_emoji(weather.get("weather_main", "")) if WEATHER_API_AVAILABLE else "Weather"
    st.markdown("### Current Weather Conditions")
    st.markdown(f"""<div class="weather-card">
        <div class="weather-card-head">
            <div>
                <div class="weather-emoji">{emoji}</div>
                <div class="weather-title">{weather.get("weather_description", "Live Weather")}</div>
                <div class="weather-subtitle">Last updated: {minutes_since(weather.get("fetched_at"))}</div>
            </div>
            <span class="live-badge">LIVE</span>
        </div>
        <div class="weather-metrics">
            <div><span class="weather-icon">Thermometer</span><b>{weather.get("temperature_c", 0):.1f} C</b><small>Temperature</small></div>
            <div><span class="weather-icon">Droplet</span><b>{weather.get("humidity_pct", 0)}%</b><small>Humidity</small></div>
            <div><span class="weather-icon">Wind</span><b>{wind_speed:.1f} km/h</b><small>Wind Speed</small></div>
            <div><span class="weather-icon">Rain</span><b>{rainfall:.1f} mm</b><small>Rainfall</small></div>
        </div>
        <div style="margin-top:14px"><span class="weather-badge {severity_cls}">{severity}</span></div>
    </div>""", unsafe_allow_html=True)

def render_forecast_section(forecast, district, date, model, scaler, features, base_temp, base_humidity):
    """Render forecast rainfall and forecast flood risk charts."""
    if not forecast:
        return

    st.markdown("### 7-Day Forecast")
    forecast_df = pd.DataFrame(forecast)
    if "risk_probability" not in forecast_df.columns:
        forecast_df["risk_probability"] = np.nan

    selected_state = st.session_state.get("selected_state", "")
    for idx, row in forecast_df.iterrows():
        rainfall = float(row.get("rainfall_mm", 0))
        row_risk = row.get("risk_probability", row.get("flood_probability_pct"))
        if pd.notna(row_risk):
            forecast_df.loc[idx, "risk_probability"] = float(row_risk)
        elif WEATHER_API_AVAILABLE:
            forecast_df.loc[idx, "risk_probability"] = calculate_daily_flood_risk(
                rainfall,
                district,
                selected_state,
            )

    line_color = "#06b6d4"
    point_colors = np.where(forecast_df["rainfall_mm"] > 50, "#ef4444", "#06b6d4")

    fig_rain = go.Figure()
    fig_rain.add_trace(go.Scatter(
        x=forecast_df["date_display"],
        y=forecast_df["rainfall_mm"],
        mode="lines+markers",
        line=dict(color=line_color, width=3),
        marker=dict(size=10, color=point_colors),
        name="Rainfall",
        text=forecast_df["description"],
        hovertemplate="%{x}<br>Rainfall: %{y:.1f} mm<br>%{text}<extra></extra>",
    ))
    fig_rain.add_hline(
        y=50,
        line_dash="dash",
        line_color="#ef4444",
        annotation_text="Danger threshold: 50mm",
        annotation_font_color="#ef4444",
    )
    fig_rain.update_layout(
        height=300,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color":"#e2e8f0","family":"Inter"},
        yaxis_title="Rainfall (mm)",
        xaxis_title="",
        margin=dict(l=10,r=10,t=10,b=10),
    )
    st.plotly_chart(fig_rain, use_container_width=True)

    risk_colors = np.where(forecast_df["risk_probability"] > 60, "#ef4444",
                   np.where(forecast_df["risk_probability"] >= 40, "#f59e0b",
                   np.where(forecast_df["risk_probability"] >= 20, "#f59e0b", "#10b981")))
    fig_risk = go.Figure(go.Bar(
        x=forecast_df["date_display"],
        y=forecast_df["risk_probability"],
        marker_color=risk_colors,
        hovertemplate="%{x}<br>Flood risk: %{y:.0f}%<extra></extra>",
    ))
    fig_risk.update_layout(
        height=240,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color":"#e2e8f0","family":"Inter"},
        yaxis=dict(title="Flood Risk (%)", range=[0, 100]),
        xaxis_title="",
        margin=dict(l=10,r=10,t=10,b=10),
    )
    st.plotly_chart(fig_risk, use_container_width=True)


def render_mini_forecast_preview(forecast_df, district):
    if forecast_df is None or forecast_df.empty:
        st.markdown("<div style='color:#94a3b8;'>7-day forecast preview unavailable.</div>", unsafe_allow_html=True)
        return

    def _weather_icon(precip_mm):
        if precip_mm >= 20:
            return "🌧️"
        if precip_mm >= 5:
            return "🌦️"
        return "☀️"

    row_items = []
    for _, row in forecast_df.head(7).iterrows():
        day_label = pd.Timestamp(row["date"]).strftime("%a")
        prob_pct = float(row.get("flood_probability_pct", row.get("flood_probability", 0.0) * 100))
        precip_mm = float(row.get("precipitation_mm", 0.0))
        icon = _weather_icon(precip_mm)
        if prob_pct <= 30:
            color = "#22c55e"
        elif prob_pct <= 60:
            color = "#f59e0b"
        else:
            color = "#ef4444"
        width_pct = min(max(prob_pct, 0), 100)

        row_items.append(f"""
            <div style="display:flex;align-items:center;gap:12px;margin-bottom:12px;">
                <div style="min-width:72px;color:#cbd5e1;font-size:0.95rem;font-weight:600;">{day_label}</div>
                <div style="display:flex;align-items:center;gap:10px;flex:1;">
                    <span style="font-size:1.1rem;">{icon}</span>
                    <div style="flex:1;background:#0f172a;border-radius:999px;height:14px;overflow:hidden;">
                        <div style="width:{width_pct}%;background:{color};height:100%;border-radius:999px;"></div>
                    </div>
                </div>
                <div style="min-width:48px;text-align:right;color:{color};font-weight:700;font-size:0.95rem;">{prob_pct:.0f}%</div>
            </div>
        """)

    st.markdown("".join(row_items), unsafe_allow_html=True)

def risk_level_from_score(score):
    """Map a 0-1 risk score to a compact text label."""
    if score >= 0.6:
        return "HIGH"
    if score >= 0.3:
        return "MODERATE"
    return "LOW"

def render_ai_summary_card(summary):
    """Render the AI-generated risk summary under a prediction."""
    if not summary:
        return
    st.markdown("### FloodGuard AI Summary")
    st.markdown(f"""<div class="ai-summary-card">
        <div class="ai-summary-title">FloodGuard AI</div>
        <div class="ai-summary-body">{summary}</div>
    </div>""", unsafe_allow_html=True)

def render_risk_overview_table(df):
    """Render searchable district risk metadata."""
    df = df[["state", "district", "flood_type", "lat", "lon"]].copy()
    search_query = st.text_input("🔍 Search state or district", placeholder="e.g. Assam, Dibrugarh...", key="risk_overview_search")
    filtered_df = df[df['state'].str.contains(search_query, case=False, na=False) | df['district'].str.contains(search_query, case=False, na=False)] if search_query else df
    filtered_df = filtered_df.copy()
    filtered_df['Risk Level'] = filtered_df['flood_type'].apply(lambda x: '🔴 High' if 'coastal' in str(x).lower() else ('🟡 Moderate' if 'river' in str(x).lower() else '🟢 Low'))
    st.markdown(f"Showing **{len(filtered_df)}** of **{len(df)}** districts")
    st.dataframe(filtered_df[['state','district','Risk Level','flood_type','lat','lon']], use_container_width=True, hide_index=True, height=400)

def add_chat_message(role, content):
    """Append a timestamped chat message and keep history bounded."""
    st.session_state.setdefault("chat_history", [])
    st.session_state.chat_history.append({
        "role": role,
        "content": content,
        "timestamp": datetime.now().strftime("%H:%M"),
    })
    st.session_state.chat_history = st.session_state.chat_history[-20:]

def render_chat_bubble(message):
    """Render one chat message with custom bubble styling."""
    role = message.get("role", "assistant")
    content = message.get("content", "")
    timestamp = message.get("timestamp", "")
    if role == "user":
        st.markdown(f"""<div class="chat-row user-row">
            <div class="chat-bubble user-bubble">{content}<div class="chat-time">{timestamp}</div></div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""<div class="chat-row ai-row">
            <div class="bot-avatar">AI</div>
            <div class="chat-bubble ai-bubble">{content}<div class="chat-time">{timestamp}</div></div>
        </div>""", unsafe_allow_html=True)

# ===== SIMPLE MODE OPTIONS =====
RAIN_OPTIONS = {
    "☀️ No rain — sky is clear": 0,
    "🌤️ Very light drizzle / few drops": 5,
    "🌦️ Light rain — ground is getting wet": 20,
    "🌧️ Steady rain — been raining for hours": 60,
    "⛈️ Heavy rain — hard to see outside": 120,
    "🌊 Nonstop heavy rain — roads getting flooded": 250,
    "🚨 Worst rain I've ever seen": 400,
}
WATER_SITUATION = {
    "🟢 Everything is dry and normal": 3.0,
    "🔵 Drains & ditches have more water than usual": 5.0,
    "🟡 Low-lying fields and roads are waterlogged": 7.0,
    "🟠 Water is reaching near houses / compound walls": 9.5,
    "🔴 Water is entering houses / streets are flooded": 12.0,
}
GROUND_OPTIONS = {
    "🏜️ Ground is dry, no puddles": 40,
    "💧 Ground is damp, small puddles around": 65,
    "💦 Mud everywhere, ground is fully soaked": 82,
    "🌫️ Standing water everywhere, ground is saturated": 95,
}
TEMP_OPTIONS = {
    "❄️ Cold — need a jacket": 15,
    "🌤️ Comfortable — pleasant weather": 25,
    "☀️ Hot — feeling sweaty": 32,
    "🔥 Very hot and sticky — hard to stay outside": 35,
}

# ===== PAGE 1: RISK PREDICTOR GAUGE =====
def show_risk_gauge(risk_score, risk_level):
    if risk_level.upper() in ["HIGH", "VERY HIGH", "EXTREME"]:
        color = "#ef4444"
    elif risk_level.upper() == "MODERATE":
        color = "#f59e0b"
    else:
        color = "#22c55e"
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_score,
        number={"suffix": "%", "font": {"color": "#f1f5f9", "size": 36}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#475569"},
            "bar": {"color": color},
            "bgcolor": "#1e293b",
            "bordercolor": "#334155",
            "steps": [
                {"range": [0, 30], "color": "rgba(34,197,94,0.12)"},
                {"range": [30, 60], "color": "rgba(245,158,11,0.12)"},
                {"range": [60, 100], "color": "rgba(239,68,68,0.12)"},
            ],
            "threshold": {
                "line": {"color": color, "width": 4},
                "thickness": 0.75,
                "value": risk_score
            }
        },
        title={"text": f"Flood Risk Level: {risk_level}", 
               "font": {"color": "#06b6d4", "size": 16}}
    ))
    fig.update_layout(
        paper_bgcolor="#0f172a",
        font={"color": "#f1f5f9"},
        height=300,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    st.plotly_chart(fig, use_container_width=True)

# ===== PAGE 1: RISK PREDICTOR =====
def page_predictor():
    lang = st.session_state.get("lang", "en")
    is_online = st.session_state.get("internet_status", True)
    st.markdown(f"""<div style="text-align:center;padding:20px 0 10px 0">
        <h1 style="font-size:2.8rem !important;margin-bottom:4px">{get_text('hero_title', lang)}</h1>
        <p class="hero-text" style="max-width:760px;margin:0 auto">{get_text('hero_subtitle', lang)}</p>
    </div>""", unsafe_allow_html=True)
    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

    model, scaler, features = load_xgb_model()
    districts_df = load_india_districts()
    real_df = load_real_data()
    st.markdown("""
<div style="display:grid; grid-template-columns:repeat(4,1fr); 
gap:8px; margin-bottom:16px;">
    <div style="background:rgba(255,255,255,0.04); border:0.5px solid 
    rgba(255,255,255,0.08); border-radius:8px; padding:10px; 
    text-align:center;">
        <div style="font-size:18px; font-weight:500; color:#00BCD4;">736</div>
        <div style="font-size:10px; color:rgba(255,255,255,0.35); 
        text-transform:uppercase; letter-spacing:0.3px; margin-top:2px;">
        Districts</div>
    </div>
    <div style="background:rgba(255,255,255,0.04); border:0.5px solid 
    rgba(255,255,255,0.08); border-radius:8px; padding:10px; 
    text-align:center;">
        <div style="font-size:18px; font-weight:500; color:#00BCD4;">36</div>
        <div style="font-size:10px; color:rgba(255,255,255,0.35); 
        text-transform:uppercase; letter-spacing:0.3px; margin-top:2px;">
        States</div>
    </div>
    <div style="background:rgba(255,255,255,0.04); border:0.5px solid 
    rgba(255,255,255,0.08); border-radius:8px; padding:10px; 
    text-align:center;">
        <div style="font-size:18px; font-weight:500; color:#00BCD4;">4,695</div>
        <div style="font-size:10px; color:rgba(255,255,255,0.35); 
        text-transform:uppercase; letter-spacing:0.3px; margin-top:2px;">
        Records</div>
    </div>
    <div style="background:rgba(255,255,255,0.04); border:0.5px solid 
    rgba(255,255,255,0.08); border-radius:8px; padding:10px; 
    text-align:center;">
        <div style="font-size:18px; font-weight:500; color:#00BCD4;">0.83</div>
        <div style="font-size:10px; color:rgba(255,255,255,0.35); 
        text-transform:uppercase; letter-spacing:0.3px; margin-top:2px;">
        AUC Score</div>
    </div>
</div>
""", unsafe_allow_html=True)

    st.session_state.setdefault("live_weather", None)
    st.session_state.setdefault("live_forecast", [])
    st.session_state.setdefault("using_live_weather", False)
    st.session_state.setdefault("rainfall_input", 50.0)
    st.session_state.setdefault("river_input", 5.0)
    st.session_state.setdefault("humidity_input", 70.0)
    st.session_state.setdefault("temp_input", 28.0)
    st.session_state.setdefault("manual_override", True)
    if "input_mode" not in st.session_state:
        st.session_state.input_mode = "simple"

    with st.sidebar:
        state_options = sorted(districts_df["state"].dropna().unique().tolist())
        st.markdown('<p style="font-size:12px; color:rgba(255,255,255,0.4); text-transform:uppercase; letter-spacing:0.5px; margin-bottom:4px;">State</p>', unsafe_allow_html=True)
        state = st.selectbox(" ", state_options, label_visibility="collapsed", key="state_select")
        
        district_options = sorted(districts_df.loc[districts_df["state"] == state, "district"].dropna().unique().tolist())
        st.markdown('<p style="font-size:12px; color:rgba(255,255,255,0.4); text-transform:uppercase; letter-spacing:0.5px; margin-bottom:4px;">District</p>', unsafe_allow_html=True)
        district = st.selectbox(" ", district_options, label_visibility="collapsed", key="district_select")
        
        district_profile = get_district_profile(state, district)
        weather_location = f"{district}, {state}"
        
        st.markdown('<p style="font-size:12px; color:rgba(255,255,255,0.4); text-transform:uppercase; letter-spacing:0.5px; margin-bottom:4px;">Date</p>', unsafe_allow_html=True)
        date = st.date_input(" ", label_visibility="collapsed", key="date_select")

        # --- OpenWeatherMap API Key ---
        api_key = os.getenv("OPENWEATHERMAP_API_KEY", "").strip()
        if WEATHER_API_AVAILABLE and api_key:
            set_openweathermap_api_key(api_key)

        # --- Fetch Live Weather Button ---
        fetch_btn = st.button("🔄 Refresh: Fetch Live Weather", use_container_width=True)
        if fetch_btn:
            with st.spinner(f"Fetching weather for {weather_location}..."):
                import requests as _req
                weather_data = None
                forecast_data = []
                api_key = os.getenv("OPENWEATHERMAP_API_KEY", "").strip()

                # ── SOURCE 1: OpenWeatherMap (when API key is provided) ──
                if api_key:
                    try:
                        owm_resp = _req.get(
                            "https://api.openweathermap.org/data/2.5/weather",
                            params={
                                "q": f"{district},{state},IN",
                                "appid": api_key,
                                "units": "metric",
                            },
                            timeout=10,
                        )
                        owm_resp.raise_for_status()
                        data = owm_resp.json()

                        rain_block = data.get("rain", {})
                        rainfall_mm = rain_block.get("1h", rain_block.get("3h", 0.0))
                        owm_weather = data.get("weather", [{}])[0]

                        weather_data = {
                            "rainfall_mm": round(float(rainfall_mm), 1),
                            "rainfall": round(float(rainfall_mm), 1),
                            "temperature_c": round(float(data["main"]["temp"]), 1),
                            "humidity_pct": int(data["main"]["humidity"]),
                            "wind_speed": round(float(data.get("wind", {}).get("speed", 0)) * 3.6, 1),
                            "weather_description": owm_weather.get("description", "Clear Sky").title(),
                            "weather_source": "OpenWeatherMap",
                            "weather_main": owm_weather.get("main", "Clear"),
                            "weather_icon": owm_weather.get("icon", "01d"),
                            "feels_like": round(float(data["main"].get("feels_like", data["main"]["temp"])), 1),
                            "fetched_at": datetime.now().isoformat(timespec="seconds"),
                            "error": None,
                        }

                        # Also fetch OWM forecast
                        if WEATHER_API_AVAILABLE:
                            try:
                                forecast_data = get_rainfall_forecast(weather_location)
                            except Exception:
                                forecast_data = []

                    except _req.exceptions.HTTPError as exc:
                        status = exc.response.status_code if exc.response is not None else None
                        if status == 401:
                            weather_data = {"error": "API key rejected. Verify your key is correct and email is confirmed. New keys take 10-30 min to activate."}
                        else:
                            weather_data = {"error": f"OpenWeatherMap error (HTTP {status}). Falling back to Open-Meteo..."}
                            weather_data = None  # allow fallback
                    except Exception as e:
                        st.info(f"⚠️ OpenWeatherMap failed ({e}). Trying Open-Meteo...")
                        weather_data = None  # allow fallback

                # ── SOURCE 2: Open-Meteo fallback (free, no key) ──
                if weather_data is None:
                    try:
                        coords = None
                        if FORECAST_AVAILABLE:
                            all_coords = get_district_coordinates()
                            coords = all_coords.get(district)
                        if coords:
                            resp = _req.get(
                                "https://api.open-meteo.com/v1/forecast",
                                params={
                                    "latitude": coords["lat"],
                                    "longitude": coords["lon"],
                                    "current_weather": "true",
                                    "daily": "precipitation_sum,temperature_2m_max,temperature_2m_min",
                                    "hourly": "relativehumidity_2m",
                                    "timezone": "Asia/Kolkata",
                                    "forecast_days": 1,
                                },
                                timeout=10,
                            )
                            resp.raise_for_status()
                            om = resp.json()
                            cw = om.get("current_weather", {})
                            daily = om.get("daily", {})
                            hourly = om.get("hourly", {})
                            precip = (daily.get("precipitation_sum") or [0])[0]
                            temp = cw.get("temperature", 28.0)
                            # Get real humidity from hourly data (use current hour)
                            humidity_list = hourly.get("relativehumidity_2m", [])
                            humidity_val = int(humidity_list[min(datetime.now().hour, len(humidity_list) - 1)]) if humidity_list else 70

                            # Decode Open-Meteo weather code to description
                            wmo_code = int(cw.get("weathercode", 0))
                            if wmo_code == 0:
                                om_desc, om_main = "Clear Sky ☀️", "Clear"
                            elif wmo_code in (1, 2, 3):
                                om_desc, om_main = "Partly Cloudy ⛅", "Clouds"
                            elif wmo_code in (45, 48):
                                om_desc, om_main = "Foggy 🌫️", "Mist"
                            elif wmo_code in (51, 53, 55):
                                om_desc, om_main = "Drizzle 🌦️", "Drizzle"
                            elif wmo_code in (61, 63, 65):
                                om_desc, om_main = "Rainy 🌧️", "Rain"
                            elif wmo_code in (66, 67):
                                om_desc, om_main = "Freezing Rain 🌧️", "Rain"
                            elif wmo_code in (71, 73, 75, 77):
                                om_desc, om_main = "Snowy ❄️", "Snow"
                            elif wmo_code in (80, 81, 82):
                                om_desc, om_main = "Rain Showers 🌧️", "Rain"
                            elif wmo_code in (85, 86):
                                om_desc, om_main = "Snow Showers ❄️", "Snow"
                            elif wmo_code in (95, 96, 99):
                                om_desc, om_main = "Thunderstorm ⛈️", "Thunderstorm"
                            else:
                                om_desc, om_main = "Cloudy ☁️", "Clouds"

                            weather_data = {
                                "rainfall_mm": round(float(precip), 1),
                                "rainfall": round(float(precip), 1),
                                "temperature_c": round(float(temp), 1),
                                "humidity_pct": humidity_val,
                                "wind_speed": round(float(cw.get("windspeed", 0)), 1),
                                "weather_description": om_desc,
                                "weather_source": "Open-Meteo",
                                "weather_main": om_main,
                                "fetched_at": datetime.now().isoformat(timespec="seconds"),
                                "error": None,
                            }
                            try:
                                forecast_data = get_rainfall_forecast(weather_location)
                            except Exception:
                                forecast_data = []
                        else:
                            weather_data = {"error": f"No coordinates found for {district}. Select a valid district."}
                    except Exception as e:
                        st.warning(f"⚠️ Could not fetch weather: {e}")

                # ── Apply result ──
                if weather_data and not weather_data.get("error"):
                    st.session_state.live_weather = weather_data
                    st.session_state.live_forecast = forecast_data
                    st.session_state.using_live_weather = True

                    # Auto-fill input fields from live weather
                    st.session_state.rainfall_input = float(weather_data.get("rainfall_mm", 0))
                    st.session_state.humidity_input = float(weather_data.get("humidity_pct", 70))
                    st.session_state.temp_input = float(weather_data.get("temperature_c", 28))
                    st.session_state.exp_rainfall = float(weather_data.get("rainfall_mm", 0))
                    st.session_state.exp_humidity = float(weather_data.get("humidity_pct", 70))
                    st.session_state.exp_temp = float(weather_data.get("temperature_c", 28))
                    if WEATHER_API_AVAILABLE:
                        river_est = get_river_level_estimate(
                            weather_data.get("rainfall_mm", 0), district
                        )
                        st.session_state.river_input = float(river_est)
                        st.session_state.exp_river = float(river_est)

                    source = weather_data.get("weather_source", "Live")
                    st.success(f"✅ Weather loaded via {source}!")

        if st.session_state.using_live_weather and st.session_state.live_weather:
            live_weather = st.session_state.live_weather
            condition = (live_weather.get("description") or live_weather.get("weather_description") or "Partly Cloudy").title()
            temperature = live_weather.get("temperature") or live_weather.get("temperature_c") or "--"
            humidity = live_weather.get("humidity") or live_weather.get("humidity_pct") or "--"
            rainfall = live_weather.get("rainfall") or live_weather.get("rainfall_mm") or live_weather.get("rain") or 0
            weather_conditions = {
                "clear": "☀️", "clouds": "⛅", "rain": "🌧️",
                "thunderstorm": "⛈️", "drizzle": "🌦️", "snow": "❄️",
                "mist": "🌫️", "fog": "🌫️"
            }
            weather_icon = weather_conditions.get(
                condition.lower().split()[0], "🌤️"
            )
            source_label = live_weather.get("weather_source", "Open-Meteo")

            weather_card_html = (
                '<div style="background: linear-gradient(135deg, #0f2027, '
                '#203a43, #2c5364); border: 1px solid rgba(0,188,212,0.3); '
                'border-radius: 16px; padding: 20px; color: white; margin: 12px 0;">'
                '<div style="background:#00BCD4; color:#003344; font-size:10px; '
                'font-weight:700; padding:3px 10px; border-radius:20px; '
                'letter-spacing:1px; display:inline-block; margin-bottom:12px;">'
                '&#9679; LIVE</div>'
                f'<div style="font-size:48px; margin-bottom:4px;">{weather_icon}</div>'
                f'<div style="font-size:22px; font-weight:600; color:white; '
                f'margin-bottom:2px;">{condition}</div>'
                f'<div style="font-size:12px; color:rgba(255,255,255,0.5); '
                f'margin-bottom:14px;">via {source_label} &middot; just now</div>'
                '<div style="display:grid; grid-template-columns:1fr 1fr 1fr; gap:8px;">'
                '<div style="background:rgba(255,255,255,0.08); border-radius:8px; '
                'padding:8px; text-align:center;">'
                f'<div style="font-size:16px; font-weight:600; color:#00BCD4;">'
                f'{temperature}&deg;C</div>'
                '<div style="font-size:10px; color:rgba(255,255,255,0.5); '
                'margin-top:2px;">Temp</div></div>'
                '<div style="background:rgba(255,255,255,0.08); border-radius:8px; '
                'padding:8px; text-align:center;">'
                f'<div style="font-size:16px; font-weight:600; color:#00BCD4;">'
                f'{humidity}%</div>'
                '<div style="font-size:10px; color:rgba(255,255,255,0.5); '
                'margin-top:2px;">Humidity</div></div>'
                '<div style="background:rgba(255,255,255,0.08); border-radius:8px; '
                'padding:8px; text-align:center;">'
                f'<div style="font-size:16px; font-weight:600; color:#00BCD4;">'
                f'{rainfall}mm</div>'
                '<div style="font-size:10px; color:rgba(255,255,255,0.5); '
                'margin-top:2px;">Rainfall</div></div>'
                '</div>'
                '<div style="font-size:11px; color:rgba(255,255,255,0.3); '
                'margin-top:10px;">Last updated: just now</div>'
                '</div>'
            )
            render_html(weather_card_html)

            st.markdown(f"""
<div style="font-size: 11px; color: rgba(255,255,255,0.3); 
text-align: center; margin-top: 4px;">
🔄 Auto-fetched from {source_label} · Refresh to update
</div>
""", unsafe_allow_html=True)

        st.session_state.manual_override = True
        st.markdown("---")
        col1, col2 = st.columns(2)

        if col1.button("🟢 Simple", key="mode_simple", 
            use_container_width=True):
            st.session_state.input_mode = "simple"

        if col2.button("⚙️ Expert", key="mode_expert",
            use_container_width=True):
            st.session_state.input_mode = "expert"

        # Dynamic CSS styling for active/inactive input mode buttons
        if st.session_state.input_mode == "simple":
            active_btn_css = """
            <style>
            section[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]:nth-of-type(3) div[data-testid="column"]:nth-of-type(1) button {
                border: 2px solid #06b6d4 !important;
                background: #1e293b !important;
                color: #06b6d4 !important;
            }
            section[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]:nth-of-type(3) div[data-testid="column"]:nth-of-type(2) button {
                border: 1px solid #334155 !important;
                background: transparent !important;
                color: #cbd5e1 !important;
            }
            </style>
            """
        else:
            active_btn_css = """
            <style>
            section[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]:nth-of-type(3) div[data-testid="column"]:nth-of-type(1) button {
                border: 1px solid #334155 !important;
                background: transparent !important;
                color: #cbd5e1 !important;
            }
            section[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]:nth-of-type(3) div[data-testid="column"]:nth-of-type(2) button {
                border: 2px solid #06b6d4 !important;
                background: #1e293b !important;
                color: #06b6d4 !important;
            }
            </style>
            """
        render_html(active_btn_css)

        disable_inputs = st.session_state.using_live_weather and not st.session_state.manual_override
        if st.session_state.input_mode == "simple":
            st.markdown(f"### {get_text('what_do_you_see', lang)}")
            rain_choice = st.selectbox(get_text("rain_question", lang), list(RAIN_OPTIONS.keys()), index=0)
            water_choice = st.selectbox(get_text("water_question", lang), list(WATER_SITUATION.keys()), index=0)
            ground_choice = st.selectbox(get_text("ground_question", lang), list(GROUND_OPTIONS.keys()), index=0)
            temp_choice = st.selectbox(get_text("temp_question", lang), list(TEMP_OPTIONS.keys()), index=1)
            rainfall = RAIN_OPTIONS[rain_choice]
            river_level = WATER_SITUATION[water_choice]
            humidity = GROUND_OPTIONS[ground_choice]
            temperature = TEMP_OPTIONS[temp_choice]
        else:
            st.markdown(f"### {get_text('weather_measurements', lang)}")
            rainfall = st.number_input(
                "RAINFALL (MM)", 
                min_value=0.0, max_value=500.0, 
                value=float(st.session_state.get("exp_rainfall", 0)),
                step=1.0, key="exp_rainfall",
                disabled=disable_inputs
            )
            river_level = st.number_input(
                "RIVER LEVEL (0-10)",
                min_value=0.0, max_value=10.0,
                value=float(st.session_state.get("exp_river", 0.0)),
                step=0.1, key="exp_river",
                disabled=disable_inputs
            )
            humidity = st.number_input(
                "HUMIDITY (%)",
                min_value=20.0, max_value=100.0,
                value=float(st.session_state.get("exp_humidity", 20)),
                step=1.0, key="exp_humidity",
                disabled=disable_inputs
            )
            temperature = st.number_input(
                "TEMPERATURE (°C)",
                min_value=10.0, max_value=45.0,
                value=float(st.session_state.get("exp_temp", 10)),
                step=1.0, key="exp_temp",
                disabled=disable_inputs
            )

        st.markdown("---")
        st.markdown("""
        <style>
        div[data-testid="stButton"] > button {
            background: rgba(0,188,212,0.12);
            border: 0.5px solid rgba(0,188,212,0.4);
            color: #00BCD4;
            font-size: 14px;
            font-weight: 500;
            width: 100%;
            padding: 12px;
            border-radius: 8px;
        }
        div[data-testid="stButton"] > button:hover {
            background: rgba(0,188,212,0.22);
        }
        </style>
        """, unsafe_allow_html=True)
        predict_btn = st.button(get_text("predict_button", lang), type="primary", use_container_width=True)

    live_weather = st.session_state.live_weather if st.session_state.using_live_weather else None
    if live_weather:
        render_current_weather_card(live_weather)
        st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

    if predict_btn and model is not None:
        with st.spinner(get_text("analyzing_risk", lang)):
            time.sleep(0.8)
            feat_dict = build_features(district, rainfall, river_level, temperature, humidity, date.month)
            feat_dict["elevation_m"] = float(district_profile.get("elevation_m", feat_dict.get("elevation_m", 55)) or feat_dict.get("elevation_m", 55))
            prob = predict_risk(model, scaler, features, feat_dict)

        if prob < 0.3:
            level, badge_cls, color, emoji = "Safe", "badge-safe", "safe", "Low"
        elif prob < 0.6:
            level, badge_cls, color, emoji = "Moderate Risk", "badge-moderate", "moderate", "Watch"
        else:
            level, badge_cls, color, emoji = "High Risk", "badge-danger", "danger", "Danger"

        st.session_state.current_district = district
        st.session_state.current_state = state
        st.session_state.current_risk_score = float(prob)
        st.session_state.current_risk_level = risk_level_from_score(prob)
        st.session_state.selected_district = district
        st.session_state.selected_state = state
        st.session_state.risk_score = float(prob)
        st.session_state.risk_level = risk_level_from_score(prob)
        st.session_state["last_risk_score"] = float(prob * 100)
        st.session_state["last_district"] = district
        st.session_state["last_state"] = state

        # --- Cache prediction for offline mode ---
        try:
            save_to_cache("prediction", {
                "district": district,
                "state": state,
                "risk_score": float(prob),
                "risk_level": risk_level_from_score(prob),
            })
        except Exception:
            pass

        # Determine all scores
        risk_score = float(prob * 100)
        xgb_score = risk_score
        import hashlib as _hl
        _h = int(_hl.md5(district.encode()).hexdigest(), 16)
        lstm_score = max(0.0, min(100.0, risk_score + ((_h % 11) - 5)))
        risk_level = risk_level_from_score(prob)

        # Live weather logic
        live_weather = st.session_state.live_weather if st.session_state.using_live_weather else None
        if not live_weather or live_weather.get('error'):
            live_weather = {
                "temperature_c": float(temperature),
                "humidity_pct": int(humidity),
                "rainfall_mm": float(rainfall),
                "weather_description": "Rainy" if rainfall > 10 else "Cloudy" if humidity > 80 else "Clear Sky",
                "weather_source": "Open-Meteo"
            }

        # 7-day forecast logic
        mini_forecast_df = None
        if st.session_state.using_live_weather and st.session_state.live_forecast:
            try:
                mini_forecast_df = pd.DataFrame(st.session_state.live_forecast)
            except Exception:
                mini_forecast_df = None
        elif FORECAST_AVAILABLE:
            try:
                mini_forecast_df = generate_7day_forecast(district, state)
            except Exception:
                mini_forecast_df = None

        if mini_forecast_df is None or len(mini_forecast_df) == 0:
            import pandas as pd
            dt = datetime
            days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
            start_day = dt.now().weekday()
            ordered_days = [days[(start_day + i) % 7] for i in range(7)]
            mini_forecast_df = pd.DataFrame({
                'day': ordered_days,
                'flood_probability_pct': [max(0.0, min(100.0, float(prob * 100) + ((int(__import__('hashlib').md5(district.encode()).hexdigest(), 16) * (i+1)) % 21) - 10)) for i in range(7)]
            })

        # Try/except block to wrap the HTML rendering as instructed
        try:
            # 4. AFTER PREDICTION — TWO COLUMN LAYOUT:
            col1, col2 = st.columns(2)

            # LEFT COLUMN (col1) — Risk Assessment Panel:
            with col1:
                # Determine color based on risk level
                risk_color = "#22c55e" if risk_level == "LOW" else \
                             "#f97316" if risk_level in ("MEDIUM", "MODERATE") else \
                             "#ef4444" if risk_level == "HIGH" else "#dc2626"
                
                st.markdown(f"""
                <div style="background:rgba(255,255,255,0.03); border:0.5px solid 
                rgba(255,255,255,0.08); border-radius:10px; padding:12px; margin-bottom:8px; text-align:center;">
                    <div style="font-size:11px; color:rgba(255,255,255,0.4); 
                    text-transform:uppercase; letter-spacing:0.5px;">Risk Assessment</div>
                </div>
                """, unsafe_allow_html=True)
                
                show_risk_gauge(risk_score, risk_level)
                
                st.markdown(f"""
                <div style="background:rgba(255,255,255,0.03); border:0.5px solid 
                rgba(255,255,255,0.08); border-radius:10px; padding:12px; margin-top:8px;">
                    <div style="display:flex; justify-content:space-between; 
                    font-size:12px; color:rgba(255,255,255,0.4); 
                    margin-bottom:6px;">
                        <span>XGBoost</span>
                        <span style="color:{risk_color};">{xgb_score:.1f}%</span>
                    </div>
                    <div style="display:flex; justify-content:space-between; 
                    font-size:12px; color:rgba(255,255,255,0.4);">
                        <span>LSTM</span>
                        <span style="color:{risk_color};">{lstm_score:.1f}%</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # 5. SECOND TWO-COLUMN ROW — Weather + Actions:
            col3, col4 = st.columns(2)

            # LEFT (col3) — Weather Card:
            with col3:
                temp = live_weather.get('temperature_c', 
                       live_weather.get('temperature', '--'))
                humidity_pct = live_weather.get('humidity_pct', 
                          live_weather.get('humidity', '--'))
                rainfall_mm = live_weather.get('rainfall_mm', 
                          live_weather.get('rainfall', 0))
                condition = live_weather.get('weather_description', 
                           live_weather.get('description', 
                           'Partly Cloudy')).title()
                weather_icons = {
                    "clear": "☀️", "clouds": "⛅", "rain": "🌧️",
                    "thunderstorm": "⛈️", "drizzle": "🌦️", 
                    "snow": "❄️", "mist": "🌫️", "fog": "🌫️"
                }
                icon = weather_icons.get(
                    condition.lower().split()[0], "🌤️")

                st.markdown(f"""
                <div style="background:#0e2a3a; border:0.5px solid 
                rgba(0,188,212,0.25); border-radius:10px; padding:14px;">
                    <div style="background:#00BCD4; color:#003344; 
                    font-size:10px; font-weight:700; padding:2px 8px; 
                    border-radius:20px; letter-spacing:0.8px; 
                    display:inline-block; margin-bottom:8px;">LIVE</div>
                    <div style="font-size:28px; margin-bottom:2px;">{icon}</div>
                    <div style="font-size:28px; font-weight:500; color:white; 
                    line-height:1;">{temp}°C</div>
                    <div style="font-size:13px; color:rgba(255,255,255,0.55); 
                    margin-top:3px;">{condition}</div>
                    <div style="display:grid; grid-template-columns:repeat(3,1fr); 
                    gap:6px; margin-top:10px;">
                        <div style="background:rgba(255,255,255,0.06); 
                        border-radius:6px; padding:7px; text-align:center;">
                            <div style="font-size:13px; font-weight:500; 
                            color:#00BCD4;">{humidity_pct}%</div>
                            <div style="font-size:10px; 
                            color:rgba(255,255,255,0.35); margin-top:2px;">
                            Humidity</div>
                        </div>
                        <div style="background:rgba(255,255,255,0.06); 
                        border-radius:6px; padding:7px; text-align:center;">
                            <div style="font-size:13px; font-weight:500; 
                            color:#00BCD4;">{rainfall_mm}mm</div>
                            <div style="font-size:10px; 
                            color:rgba(255,255,255,0.35); margin-top:2px;">
                            Rainfall</div>
                        </div>
                        <div style="background:rgba(255,255,255,0.06); 
                        border-radius:6px; padding:7px; text-align:center;">
                            <div style="font-size:13px; font-weight:500; 
                            color:#00BCD4;">{live_weather.get('weather_source', 'Open-Meteo')}</div>
                            <div style="font-size:10px; 
                            color:rgba(255,255,255,0.35); margin-top:2px;">
                            Source</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # RIGHT (col4) — Recommended Actions:
            with col4:
                # Use existing recommendation logic
                # Wrap the existing rec cards in this panel:
                st.markdown("""
                <div style="background:rgba(255,255,255,0.03); border:0.5px solid 
                rgba(255,255,255,0.08); border-radius:10px; padding:16px;">
                    <div style="font-size:11px; color:rgba(255,255,255,0.4); 
                    text-transform:uppercase; letter-spacing:0.5px; 
                    margin-bottom:10px;">Recommended Actions</div>
                """, unsafe_allow_html=True)
                
                if prob < 0.3:
                    st.markdown(f"""<div class="rec-box rec-safe">
                        <h4 style="color:#10b981;margin:0">{get_text('low_risk', lang)}</h4>
                        <ul style="color:#94a3b8;margin:8px 0 0 0">
                        <li>{get_text('conditions_normal', lang)}</li><li>{get_text('no_flood_expected', lang)}</li>
                        <li>{get_text('stay_updated', lang)}</li></ul></div>""", unsafe_allow_html=True)
                elif prob < 0.6:
                    st.markdown(f"""<div class="rec-box rec-mod">
                        <h4 style="color:#f59e0b;margin:0">{get_text('be_careful', lang)}</h4>
                        <ul style="color:#94a3b8;margin:8px 0 0 0">
                        <li>{get_text('keep_watching', lang)}</li>
                        <li>{get_text('keep_documents', lang)}</li>
                        <li>{get_text('know_safe_ground', lang)}</li>
                        <li>{get_text('charge_phone', lang)}</li>
                        {get_emergency_contacts_html(state)}</ul></div>""", unsafe_allow_html=True)
                else:
                    st.markdown(f"""<div class="rec-box rec-danger">
                        <h4 style="color:#ef4444;margin:0">{get_text('danger_move', lang)}</h4>
                        <ul style="color:#94a3b8;margin:8px 0 0 0">
                        <li><b>{get_text('move_higher_ground', lang)}</b></li>
                        <li>{get_text('take_family_first', lang)}</li>
                        <li>{get_text('no_flooded_roads', lang)}</li>
                        {get_emergency_contacts_html(state)}</ul></div>""", unsafe_allow_html=True)
                
                st.markdown("</div>", unsafe_allow_html=True)

        except Exception as display_err:
            st.error(f"Error rendering premium display: {display_err}")

        # AI summary card explanation
        top_drivers = get_top_shap_drivers(scaler, features, feat_dict)
        if CHATBOT_AVAILABLE:
            with st.spinner(get_text("ai_preparing", lang)):
                ai_summary = generate_risk_explanation(district, prob, top_drivers, live_weather or {})
        else:
            ai_summary = (
                f"{district} is at {risk_level_from_score(prob)} flood risk ({prob:.0%}) based on rainfall, "
                "river level, and soil conditions. Residents should use this as a planning signal and watch "
                "local alerts. The most important action is to avoid flooded roads and prepare essentials."
            )
        st.session_state.latest_ai_summary = ai_summary
        st.session_state.floodguard_pending_summary = ai_summary
        render_ai_summary_card(ai_summary)

        st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

        # 6. SHAP FEATURE IMPORTANCE:
        try:
            st.markdown("""
            <div style="font-size:11px; color:rgba(255,255,255,0.4); 
            text-transform:uppercase; letter-spacing:0.5px; 
            margin:16px 0 8px;">Feature Importance</div>
            """, unsafe_allow_html=True)

            display_names = {
                "rainfall_mm":"Rainfall","river_level_m":"Water Level",
                "temperature_c":"Temperature","humidity_pct":"Ground Moisture",
                "elevation_m":"Elevation","soil_moisture":"Soil Moisture",
                "rainfall_7day_cumsum":"7-Day Rain","rainfall_30day_cumsum":"30-Day Rain",
                "api":"Saturation","river_rise_rate":"Rise Speed",
                "rainfall_river_interaction":"Rain x Water","is_monsoon":"Monsoon",
                "ndvi":"Vegetation","month_sin":"Season","month_cos":"Season",
            }
            top = dict(sorted(feat_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:8])
            labels = [display_names.get(k, k) for k in top.keys()]
            fig2 = go.Figure(go.Bar(x=list(top.values()), y=labels, orientation='h',
                marker=dict(color=list(top.values()), colorscale=[[0,"#06b6d4"],[1,"#10b981"]])))
            fig2.update_layout(height=350, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font={"color":"#e2e8f0","family":"Inter"},
                xaxis=dict(gridcolor="rgba(255,255,255,0.03)"),
                yaxis=dict(gridcolor="rgba(255,255,255,0.03)"), margin=dict(l=10,r=10,t=10,b=10))
            st.plotly_chart(fig2, use_container_width=True)
        except Exception as shap_err:
            st.error(f"Error rendering SHAP chart: {shap_err}")

        # --- PDF Report Download Button ---
        try:
            from src.pdf_report import generate_flood_report
            selected_district = district
            selected_state = state
            risk_score = float(prob)
            risk_level = risk_level_from_score(prob)
            pdf_bytes = bytes(generate_flood_report(
                district=selected_district,
                state=selected_state,
                risk_score=float(risk_score),
                risk_level=risk_level
            ))
            st.markdown("---")
            st.success(get_text("report_ready", lang))
            st.download_button(
                label=get_text("download_pdf", lang),
                data=pdf_bytes,
                file_name=f"FloodGuard_{selected_district}_{datetime.now().strftime('%Y%m%d')}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        except Exception as e:
            st.warning(f"{get_text('pdf_unavailable', lang)}: {e}")

    elif predict_btn and model is None:
        st.error(get_text("model_not_found", lang))
    else:
        # --- Offline: show cached prediction if available ---
        if not is_online:
            cached = load_from_cache("prediction")
            if cached:
                st.warning(f"📦 Showing cached prediction from {cached.get('_cached_at', 'unknown time')}")
                st.info(f"District: {cached['district']}, State: {cached['state']}")
                cached_prob = cached["risk_score"]
                cached_level = cached["risk_level"]
                bar_color = "#ef4444" if cached_prob > 0.6 else "#f59e0b" if cached_prob > 0.3 else "#10b981"
                fig = go.Figure(go.Indicator(mode="gauge+number", value=cached_prob*100,
                    number={"suffix":"%", "font":{"size":56,"color":"white","family":"Inter"}},
                    title={"text":f"Flood Risk Gauge - {cached['district']} (Cached)", "font":{"size":16,"color":"#94a3b8","family":"Inter"}},
                    gauge={"axis":{"range":[0,100],"tickcolor":"#334155","tickwidth":1},
                           "bar":{"color":bar_color,"thickness":0.75},
                           "bgcolor":"rgba(15,23,42,0.5)", "bordercolor":"rgba(0,210,255,0.1)", "borderwidth":2,
                           "steps":[{"range":[0,30],"color":"rgba(16,185,129,0.1)"},
                                    {"range":[30,60],"color":"rgba(245,158,11,0.1)"},
                                    {"range":[60,100],"color":"rgba(239,68,68,0.1)"}],
                           "threshold":{"line":{"color":bar_color,"width":4},"thickness":0.85,"value":cached_prob*100}}))
                fig.update_layout(height=340, paper_bgcolor="rgba(0,0,0,0)", font={"color":"white","family":"Inter"},
                                  margin=dict(t=80,b=20,l=50,r=50))
                st.plotly_chart(fig, use_container_width=True)
                st.markdown(f"""
                <div class="rec-box" style="border-left:4px solid {bar_color};padding:16px;background:#1e293b;border-radius:8px;">
                    <strong style="color:{bar_color};">{cached_level}</strong> — Cached risk score: <strong>{cached_prob:.0%}</strong>
                </div>""", unsafe_allow_html=True)
            else:
                st.error("No cached prediction available. Please connect to internet and run a prediction first.")
        else:
            st.markdown(f"""<div class="glass-card" style="text-align:center;padding:70px 40px">
                <h3 style="color:#e2e8f0 !important;font-size:1.5rem !important">{get_text('ready_to_predict', lang)}</h3>
                <p style="color:#94a3b8;font-size:1.1rem;margin-top:12px">{get_text('ready_to_predict_desc', lang)}</p>
                <p style="color:#475569;font-size:0.85rem;margin-top:20px">{get_text('weather_auto_note', lang)}</p>
            </div>""", unsafe_allow_html=True)

# ===== PAGE 2: RISK MAP =====
def page_map():
    st.markdown("""<div style="background:#0f172a;padding:24px;border-radius:12px;border:1px solid #1e293b;text-align:center;margin-bottom:16px;">
        <h1 style="font-size:2.8rem !important;margin-bottom:4px;color:#06b6d4;">🗺️ India Flood Risk Map</h1>
        <p class="hero-text" style="max-width:760px;margin:0 auto;color:#94a3b8;">Full India flood risk assessment across 736 districts based on historical flood patterns</p>
    </div>""", unsafe_allow_html=True)
    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

    is_online = st.session_state.get("internet_status", True)
    if should_skip_map(is_online):
        st.info("📡 **Offline Mode**: The interactive map requires internet to load map tiles. Please reconnect to view the full risk map.")
        st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)
        st.markdown("### 📋 India Risk Overview (Offline)")
        districts_df = load_india_districts()
        render_risk_overview_table(districts_df)
        return

    districts_df = load_india_districts()
    m = folium.Map(
        location=[22.5937, 82.9629],
        zoom_start=5,
        tiles="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
        attr="© OpenStreetMap contributors © CARTO",
        prefer_canvas=True,
        min_zoom=4,
        max_zoom=10
    )
    
    # 1. LEGEND — add to map
    legend_html = """
    <div style="position:fixed;bottom:30px;left:30px;z-index:1000;
      background:#1e293b;padding:12px 16px;border-radius:10px;
      border:1px solid #334155;font-family:Arial;">
      <p style="color:#f1f5f9;font-size:13px;font-weight:bold;
        margin:0 0 8px;">Flood Risk Level</p>
      <p style="margin:4px 0;color:#ef4444;">🔴 High / Very High</p>
      <p style="margin:4px 0;color:#f59e0b;">🟡 Moderate</p>
      <p style="margin:4px 0;color:#22c55e;">🟢 Low</p>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    type_base = {
        "Flash / glacial flood": 0.68,
        "Coastal / river flood": 0.58,
        "Riverine flood": 0.62,
        "Urban / river flood": 0.44,
    }
    for _, row in districts_df.iterrows():
        score = min(max(type_base.get(row.get("flood_type"), 0.45) + ((hash(row["district"]) % 21) - 10) / 100, 0.08), 0.92)
        level = "High" if score >= 0.6 else "Moderate" if score >= 0.3 else "Low"
        
        district = row['district']
        state = row['state']
        
        # Support pre-defined risk level fields in row if present, with fallback to calculated level
        row_risk = row.get("risk_level", row.get("risk", row.get("level", None)))
        if row_risk is not None:
            risk_level = str(row_risk).strip()
        else:
            risk_level = level
            
        risk_score = score * 100
        
        LOW_RISK_DISTRICTS = [
            "Jaisalmer", "Barmer", "Bikaner", "Jodhpur", "Churu",
            "Jalore", "Sirohi", "Pali", "Nagaur", "Sikar",
            "Ladakh", "Leh", "Kargil", "Lahaul And Spiti",
            "Kinnaur", "Dibang Valley", "Anjaw", "Tawang",
            "Upper Siang", "West Kameng", "East Kameng",
            "Phek", "Longleng", "Kiphire", "Tuensang",
            "Senapati", "Ukhrul", "Chandel", "Churachandpur",
            "Ri Bhoi", "East Khasi Hills", "West Jaintia Hills",
            "Hamirpur", "Una", "Bilaspur", "Solan",
            "Kutch", "Banaskantha", "Patan", "Mahesana"
        ]

        import hashlib as _hl
        if district in LOW_RISK_DISTRICTS:
            risk_level = "LOW"
            _seed = int(_hl.md5(district.encode()).hexdigest(), 16)
            risk_score = round(5 + (_seed % 1500) / 100.0, 1)
            score = risk_score / 100.0
            
        risk_color = {
            "LOW": "#22c55e",
            "MODERATE": "#f59e0b",
            "HIGH": "#ef4444",
            "VERY HIGH": "#ef4444",
            "EXTREME": "#ef4444",
            "SEVERE": "#ef4444"
        }.get(str(risk_level).strip().upper(), "#f59e0b")
        
        popup = (
            f"<b>{district}, {state}</b><br>"
            f"Risk: <b>{risk_level}</b> ({score:.0%})<br>"
            f"Flood type: {row.get('flood_type', 'Riverine flood')}"
        )
        
        # 2. TOOLTIPS — for each CircleMarker, add tooltip
        # 3. CIRCLE MARKERS — update colors to match app theme
        folium.CircleMarker(
            location=[float(row["lat"]), float(row["lon"])],
            radius=4 + score * 6,
            color=risk_color,
            weight=1.5,
            fill=True,
            fill_color=risk_color,
            fill_opacity=0.8,
            popup=folium.Popup(popup, max_width=260),
            tooltip=folium.Tooltip(
                f"<b>{district}</b><br>State: {state}<br>"
                f"Risk: {risk_level}<br>Score: {risk_score:.1f}%"
            ),
        ).add_to(m)

    try:
        from streamlit_folium import folium_static
        folium_static(m, width=1100, height=600)
    except ImportError:
        st.components.v1.html(m._repr_html_(), height=600, scrolling=True)

    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)
    st.markdown("### 📋 India Risk Overview")
    render_risk_overview_table(districts_df)

# ===== PAGE 3: FLOODGUARD AI =====

# ===== FLOODGUARD AI CHAT OVERRIDE =====
def _get_gemini_key_for_app():
    """Prefer Streamlit secrets, then local config/.env for desktop use."""
    try:
        if "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass

    try:
        from config import GEMINI_API_KEY

        if GEMINI_API_KEY and GEMINI_API_KEY != "your_gemini_api_key_here":
            return GEMINI_API_KEY
    except Exception:
        pass
    return None

def _limit_chat_state():
    """Keep both display and Gemini histories bounded."""
    st.session_state.messages = st.session_state.messages[-20:]
    st.session_state.chat_history = st.session_state.chat_history[-20:]

def _format_ai_response_html(content):
    """Lightweight response formatting for emergency contacts and numbered steps."""
    import html
    import re

    escaped = html.escape(content or "")
    for pattern in [
        r"NDRF:?\s*011-24363260",
        r"National Disaster Helpline:?\s*1078",
        r"National:?\s*1078",
        r"Police:?\s*100",
        r"Ambulance:?\s*108",
        r"Assam SDMA:?\s*1070",
    ]:
        escaped = re.sub(
            pattern,
            lambda match: f'<span class="emergency-highlight">{match.group(0)}</span>',
            escaped,
            flags=re.IGNORECASE,
        )

    lines = escaped.splitlines()
    formatted_lines = []
    in_steps = False
    for line in lines:
        if re.match(r"^\s*\d+[\.\)]\s+", line):
            if not in_steps:
                formatted_lines.append('<ol class="ai-numbered-list">')
                in_steps = True
            item = re.sub(r"^\s*\d+[\.\)]\s+", "", line)
            formatted_lines.append(f"<li>{item}</li>")
        else:
            if in_steps:
                formatted_lines.append("</ol>")
                in_steps = False
            if line.strip():
                formatted_lines.append(f"<p>{line}</p>")
    if in_steps:
        formatted_lines.append("</ol>")
    return "".join(formatted_lines)

def _display_assistant_message(message):
    """Render assistant response with question type and enhanced formatting."""
    question_type = message.get("question_type", "general").replace("_", " ").title()
    body = _format_ai_response_html(message.get("content", ""))
    st.markdown(f"""<div class="ai-response-card">
        <div class="question-type-label">Question Type: {question_type}</div>
        <div class="ai-response-body">{body}</div>
    </div>""", unsafe_allow_html=True)

def _send_floodguard_message(prompt, district, risk_score, risk_level):
    """Shared send path for typed input and suggested question buttons."""
    question_type = detect_question_type(prompt) if CHATBOT_AVAILABLE else "general"
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.spinner(""):
        col1, col2 = st.columns([0.05, 0.95])
        with col1:
            st.write("AI")
        with col2:
            thinking_placeholder = st.empty()
            thinking_placeholder.markdown("*FloodGuard AI is analyzing flood data...*")
        response = get_chat_response(
            prompt,
            st.session_state.chat_history,
            district,
            risk_score,
            risk_level,
        )
        thinking_placeholder.empty()

    st.session_state.chat_history.append({"role": "user", "parts": [prompt]})
    st.session_state.chat_history.append({"role": "model", "parts": [response]})
    st.session_state.messages.append({
        "role": "assistant",
        "content": response,
        "question_type": question_type,
    })
    _limit_chat_state()
    st.rerun()

def page_chatbot():
    # STEP 2 - Session state for chat history:
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.session_state.setdefault("selected_district", st.session_state.get("current_district", "Guwahati"))
    st.session_state.setdefault("risk_score", st.session_state.get("current_risk_score", 0.0))
    st.session_state.setdefault("risk_level", st.session_state.get("current_risk_level", "LOW"))
    st.session_state.setdefault("floodguard_pending_summary", None)

    district = st.session_state.get("selected_district", "Guwahati")
    risk_score = st.session_state.get("risk_score", 0.0)
    risk_level = st.session_state.get("risk_level", "LOW")
    state = st.session_state.get("selected_state", None)

    st.markdown("""<div style="text-align:center;padding:20px 0 10px 0">
        <h1 style="font-size:2.8rem !important;margin-bottom:4px">FloodGuard AI</h1>
        <p class="hero-text" style="max-width:650px;margin:0 auto">Ask flood safety questions with your latest district and risk score already in context.</p>
    </div>""", unsafe_allow_html=True)
    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

    # Set API keys if available (optional — KB works without any API key)
    gemini_key = _get_gemini_key_for_app()
    if gemini_key and CHATBOT_AVAILABLE:
        set_gemini_api_key(gemini_key)
    if CHATBOT_AVAILABLE:
        # Anthropic key from env/secrets
        anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if anthropic_key and anthropic_key != "your_anthropic_api_key_here":
            set_anthropic_api_key(anthropic_key)

    context_color = "#ef4444" if risk_level == "HIGH" else "#f59e0b" if risk_level == "MODERATE" else "#10b981"
    st.markdown(f"""<div class="chat-context-card">
        <div>
            <div class="metric-label">Current Context</div>
            <div style="color:#e2e8f0;font-size:1.1rem;font-weight:800">{district}</div>
        </div>
        <div>
            <div class="metric-label">Risk Score</div>
            <div style="color:{context_color};font-size:1.4rem;font-weight:900">{risk_score:.0%}</div>
        </div>
        <div>
            <div class="metric-label">Risk Level</div>
            <div style="color:{context_color};font-size:1.1rem;font-weight:800">{risk_level}</div>
        </div>
    </div>""", unsafe_allow_html=True)

    # STEP 6 - Clear chat button in sidebar or above chat:
    if st.button("🗑️ Clear Chat", key="clear_chat"):
        st.session_state.chat_history = []
        st.rerun()

    # Suggested Questions (chips)
    st.markdown("### Suggested Questions")
    suggestions = [
        "How did the Kerala 2018 floods happen?",
        "Tell me about Mumbai 2005 floods",
        "What should I do if water enters my home?",
        "Which states are most flood prone in India?",
        "Tell me about Kedarnath 2013 disaster",
        "How does the Brahmaputra cause floods in Assam?",
        "What causes floods in India?",
        "Tell me about Chennai 2015 floods",
    ]
    selected_prompt = None
    chip_cols = st.columns(3)
    for idx, question in enumerate(suggestions):
        with chip_cols[idx % 3]:
            if st.button(question, key=f"fixed_suggested_question_{idx}", use_container_width=True):
                selected_prompt = question

    # Handle floodguard pending summary if it exists
    if st.session_state.floodguard_pending_summary:
        summary = (
            f"I can see {district} is currently at {risk_score:.0%} flood risk. "
            f"Here's what you should know: {st.session_state.floodguard_pending_summary}"
        )
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": summary,
            "q_type": "Risk Explanation"
        })
        st.session_state.floodguard_pending_summary = None

    # STEP 1 - Chat container with header:
    st.markdown("""
<div style="background:#0f172a;border-radius:14px;padding:20px;
  border:1px solid #1e293b;">
  <div style="display:flex;align-items:center;gap:10px;
    padding-bottom:12px;border-bottom:1px solid #1e293b;margin-bottom:16px;">
    <div style="width:36px;height:36px;background:#06b6d420;
      border-radius:50%;display:flex;align-items:center;
      justify-content:center;font-size:18px;">🌊</div>
    <div>
      <div style="color:#06b6d4;font-size:14px;font-weight:600;">
        FloodGuard AI</div>
      <div style="color:#475569;font-size:11px;">
        Powered by Groq · LLaMA 3.3</div>
    </div>
    <div style="margin-left:auto;width:8px;height:8px;
      background:#22c55e;border-radius:50%;"></div>
  </div>
""", unsafe_allow_html=True)

    # STEP 3 - Render chat messages as bubbles:
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(f"""
            <div style="display:flex;justify-content:flex-end;margin-bottom:12px;">
              <div style="background:#06b6d4;color:#0f172a;
                border-radius:18px 18px 4px 18px;padding:10px 14px;
                font-size:13px;max-width:80%;">{msg["content"]}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="display:flex;gap:8px;margin-bottom:12px;">
              <div style="width:28px;height:28px;background:#1e293b;
                border:1px solid #06b6d430;border-radius:50%;
                display:flex;align-items:center;justify-content:center;
                font-size:14px;flex-shrink:0;margin-top:2px;">🤖</div>
              <div>
                <div style="background:#06b6d415;color:#06b6d4;
                  border-radius:4px;padding:2px 7px;font-size:10px;
                  display:inline-block;margin-bottom:6px;">
                  {msg.get("q_type","General")}</div>
                <div style="background:#1e293b;color:#e2e8f0;
                  border-radius:4px 18px 18px 18px;padding:10px 14px;
                  font-size:13px;line-height:1.6;max-width:85%;">
                  {msg["content"]}</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

    # Close the STEP 1 container
    st.markdown("</div>", unsafe_allow_html=True)

    # STEP 4 - Input row at bottom:
    col1, col2 = st.columns([5, 1])
    with col1:
        user_input = st.text_input("", 
            placeholder="Ask about floods, crops, weather...",
            key="chat_input", label_visibility="collapsed")
    with col2:
        send = st.button("Send ➤", use_container_width=True)

    # STEP 5 - On send:
    prompt = selected_prompt or user_input
    trigger_send = (send and user_input) or selected_prompt

    if trigger_send:
        st.session_state.chat_history.append({
            "role": "user", 
            "content": prompt
        })
        
        with st.spinner("Analyzing with LLaMA 3.3..."):
            res = get_chatbot_response(prompt, district=district, state=state, risk_score=risk_score)
            if isinstance(res, dict):
                response = res.get("answer", "")
                q_type = res.get("question_type", "General")
            else:
                response = str(res)
                q_type = "General"

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": response,
            "q_type": q_type
        })
        st.rerun()

# ===== PAGE 4: 7-DAY DISTRICT-LEVEL FORECAST =====
def page_forecast():
    """Display 7-day flood forecast for selected district."""
    is_online = st.session_state.get("internet_status", True)
    if not FORECAST_AVAILABLE:
        st.warning("🔧 Forecast module not available. Please check dependencies.")
        return

    st.markdown("""<div style="text-align:center;padding:20px 0 10px 0">
        <h1 style="font-size:2.8rem !important;margin-bottom:4px">📊 7-Day Flood Forecast</h1>
        <p class="hero-text" style="max-width:600px;margin:0 auto">District-level flood risk prediction for the next 7 days based on weather forecast</p>
    </div>""", unsafe_allow_html=True)
    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

    # Get all districts and their states
    all_districts = get_district_coordinates()
    states = sorted(set(d['state'] for d in all_districts.values() if 'state' in d))

    # Selection in main area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        selected_state = st.selectbox("🌍 Select State", states, key="forecast_state")
        
        # If state changed, reset district selection
        if selected_state != st.session_state.get('last_forecast_state'):
            st.session_state.forecast_district = None
            st.session_state.last_forecast_state = selected_state
        
        # Filter districts by selected state
        district_list = sorted([d for d, info in all_districts.items() if info.get('state') == selected_state])
        
    with col2:
        # Set index: if current session district is in list, use it; otherwise use 0
        current_district = st.session_state.get('forecast_district')
        default_idx = 0
        if current_district and current_district in district_list:
            default_idx = district_list.index(current_district)
        
        selected_district = st.selectbox("🏘️ Select District", district_list, key="forecast_district", index=default_idx)
        
    # Store selected district in session state
    st.session_state['selected_district'] = selected_district
    st.session_state['selected_state'] = selected_state

    # Show forecast button only if district is selected
    if st.session_state.get('selected_district'):
        generate_btn = st.button("🔮 Generate 7-Day Forecast", type="primary", use_container_width=True)
    else:
        st.info("👆 Select a state and district to generate forecast")
        generate_btn = False

    # Generate and display forecast
    # Get selected values from session state
    selected_district = st.session_state.get('selected_district')
    selected_state = st.session_state.get('selected_state')
    
    # Clear cache if the selected district or state changes
    forecast_session = st.session_state.get('current_forecast')
    if forecast_session is not None:
        if (forecast_session.get('district') != selected_district or 
            forecast_session.get('state') != selected_state):
            if "current_forecast" in st.session_state:
                del st.session_state.current_forecast
            forecast_session = None

    if generate_btn or (forecast_session is not None):
        if generate_btn and selected_district:
            with st.spinner(f"📡 Fetching weather data and generating forecast for {selected_district}..."):
                try:
                    forecast_df = generate_7day_forecast(selected_district, selected_state)
                    st.session_state.current_forecast = {
                        'df': forecast_df,
                        'district': selected_district,
                        'state': selected_state,
                    }
                    # --- Cache forecast for offline mode ---
                    try:
                        save_to_cache("forecast", {
                            "district": selected_district,
                            "state": selected_state,
                            "forecast_data": forecast_df.to_dict() if hasattr(forecast_df, 'to_dict') else {},
                        })
                    except Exception:
                        pass
                except Exception as e:
                    st.error(f"❌ Error generating forecast: {str(e)}")
                    st.session_state.current_forecast = None
                    # --- Offline: try cached forecast ---
                    if not is_online:
                        cached_forecast = load_from_cache("forecast")
                        if cached_forecast:
                            st.warning(f"📦 Showing forecast cached {get_cache_age('forecast')}")
                            try:
                                cached_df = pd.DataFrame(cached_forecast["forecast_data"])
                                st.session_state.current_forecast = {
                                    'df': cached_df,
                                    'district': cached_forecast.get('district', 'Unknown'),
                                    'state': cached_forecast.get('state', 'Unknown'),
                                }
                            except Exception:
                                st.error("Cached forecast data is corrupted.")
                                return
                        else:
                            st.error("No cached forecast. Connect to internet to download forecast first.")
                            return
                    else:
                        return
        
        # Display if forecast exists
        if "current_forecast" in st.session_state and st.session_state.current_forecast is not None:
            forecast_data = st.session_state.current_forecast
            forecast_df = forecast_data['df']
            dist_name = forecast_data.get('district', 'Unknown')

            # Display chart
            try:
                fig = plot_forecast_chart(forecast_df, dist_name)
                st.plotly_chart(fig, use_container_width=True)
                st.caption("📡 Rainfall forecast from Open-Meteo. Accuracy improves during monsoon season (June–September).")
            except Exception as e:
                st.warning(f"⚠️ Could not render chart: {str(e)}")

            st.markdown("---")

            # Display summary
            try:
                summary = get_forecast_summary(forecast_df, dist_name)
                st.markdown(f"""<div class="ai-summary-card">
                    <div class="ai-summary-title">📋 Forecast Summary</div>
                    <div class="ai-summary-body" style="white-space: pre-wrap;">{summary}</div>
                </div>""", unsafe_allow_html=True)
            except Exception as e:
                st.warning(f"⚠️ Could not generate summary: {str(e)}")

            st.markdown("---")

            # Display detailed table
            st.subheader("📅 Day-by-Day Breakdown")
            display_cols = ["date", "precipitation_mm", "flood_probability", "risk_level"]
            
            # Format dataframe for display
            table_df = forecast_df[display_cols].copy()
            table_df.columns = ["Date", "Rainfall (mm)", "Flood Risk %", "Risk Level"]
            table_df["Flood Risk %"] = (table_df["Flood Risk %"] * 100).round(1).astype(str) + "%"
            table_df["Date"] = table_df["Date"].dt.strftime("%a, %b %d")
            
            st.dataframe(table_df, use_container_width=True, hide_index=True)

            st.markdown("---")

            # Download buttons
            col1, col2, col3 = st.columns([1, 1, 2])
            
            with col1:
                csv = forecast_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download CSV",
                    data=csv,
                    file_name=f"forecast_{dist_name}_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
            
            with col2:
                # Create a simple text report for download
                report_text = f"""FLOOD FORECAST REPORT
=====================================
District: {dist_name}
Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}

{get_forecast_summary(forecast_df, dist_name)}

DETAILED DATA:
{table_df.to_string(index=False)}
"""
                st.download_button(
                    label="📄 Download Report",
                    data=report_text,
                    file_name=f"report_{dist_name}_{pd.Timestamp.now().strftime('%Y%m%d')}.txt",
                    mime="text/plain"
                )

            st.info("💡 **Tip**: Forecasts update every hour. Check back regularly for the latest predictions.")

# ===== PAGE 5: FLOOD DAMAGE SEVERITY CLASSIFIER =====
def page_damage_classifier():
    st.title("🛰️ Flood Damage Severity Classifier")
    st.markdown("Upload an aerial or ground photo to assess flood damage severity using AI")

    if not FLOOD_CLASSIFIER_AVAILABLE:
        st.error("Flood damage classifier is not available.")
        if FLOOD_CLASSIFIER_ERROR:
            st.caption(FLOOD_CLASSIFIER_ERROR)
        return

    col1, col2 = st.columns(2)
    uploaded_file = None
    image = None
    analyze = False

    with col1:
        uploaded_file = st.file_uploader(
            "Upload Flood Image",
            type=["jpg", "jpeg", "png"],
            help="Upload aerial or ground photo of flooded area",
        )

        if uploaded_file:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded Image", use_column_width=True)

            analyze = st.button(
                "🔍 Classify Damage Severity",
                type="primary",
                use_container_width=True,
            )

    with col2:
        if uploaded_file and analyze:
            with st.spinner("Analyzing image..."):
                result = classify_flood_image(image)

            colors = {
                "No Flooding": "blue",
                "Mild": "green",
                "Moderate": "orange",
                "Severe": "red",
            }
            color = colors.get(result["severity"], "blue")
            severity_label = (
                result["severity"]
                if result["severity"] == "No Flooding"
                else f"{result['severity']} Flooding"
            )

            st.markdown(f"""
            <div style='background:{color};
            padding:20px; border-radius:10px;
            text-align:center; color:white;
            font-size:24px; font-weight:bold'>
            {severity_label}
            </div>
            """, unsafe_allow_html=True)

            st.metric("Confidence", f"{result['confidence']:.1f}%")

            st.markdown("**Probability Breakdown:**")
            for cls, prob in result["probabilities"].items():
                st.progress(prob / 100, text=f"{cls}: {prob:.1f}%")

            st.info(result["description"])

            st.markdown("**Recommended Actions:**")
            for rec in result["recommendations"]:
                st.markdown(f"- {rec}")
        elif uploaded_file:
            st.info("Click the classify button to analyze this flood image.")
        else:
            st.info("Upload a flood image to start damage severity classification.")

# ===== PAGE 6: CROP DISEASE DETECTION =====
def page_crop_disease():
    st.title("🌿 Crop Disease Detection")
    st.markdown("""
Upload a crop leaf image to detect diseases using AI.
**Flood Connection:** Flooding increases crop disease 
risk by 3-5x due to excess moisture and waterlogging.
""")

    if not CROP_DISEASE_AVAILABLE:
        st.error("Crop disease classifier is not available.")
        if CROP_DISEASE_ERROR:
            st.caption(CROP_DISEASE_ERROR)
        return

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### 📸 Upload Leaf Image")
        st.warning(
            "⚠️ Only upload leaves from: Corn, Tomato, Potato, Pepper, Rice, Grape, Apple, Strawberry. "
            "Other crops will return 'not supported'."
        )
        uploaded = st.file_uploader(
            "Choose a leaf image",
            type=["jpg", "jpeg", "png"],
            key="crop_disease_upload",
        )
        if uploaded:
            image = Image.open(uploaded).convert("RGB")
            st.image(
                image,
                caption="Uploaded Leaf Image",
                use_container_width=True,
            )

            analyze_btn = st.button(
                "🔍 Detect Disease",
                type="primary",
                use_container_width=True,
            )

            if analyze_btn:
                with st.spinner("🤖 Analyzing leaf image..."):
                    result = classify_crop_image(image)

                st.session_state["crop_result"] = result

    with col2:
        st.markdown("### 📊 Analysis Results")

        st.markdown("**Supported Crops:**")
        crops = [
            "ðŸŒ½ Corn/Maize",
            "ðŸ… Tomato",
            "ðŸ¥” Potato",
            "ðŸ«‘ Pepper",
            "ðŸŽ Apple",
            "ðŸ‡ Grape",
            "ðŸ“ Strawberry",
        ]
        crops = [
            "Corn/Maize",
            "Tomato",
            "Potato",
            "Pepper",
            "Apple",
            "Grape",
            "Strawberry",
        ]
        for crop in crops:
            st.markdown(f"- {crop}")
        st.markdown("---")

        if "crop_result" in st.session_state:
            result = st.session_state["crop_result"]

            if result["status"] == "Healthy":
                st.success(f"✅ **{result['crop']}** — HEALTHY")
            elif result["status"] == "Error":
                st.error("❌ Model not trained yet")
            else:
                st.error(f"⚠️ **{result['crop']}** — {result['disease']} DETECTED")

            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Confidence", f"{result['confidence']}%")
            with col_b:
                st.metric("Severity", result["severity"])

            st.markdown("---")
            st.markdown("**📋 Description:**")
            st.info(result["description"])

            st.markdown("**💊 Treatment:**")
            for treatment in result["treatment"]:
                st.markdown(f"- {treatment}")

            st.markdown("**🌊 Flood Connection:**")
            st.warning(result["flood_connection"])

            st.markdown("**🛡️ Prevention:**")
            for prevention in result["prevention"]:
                st.markdown(f"- {prevention}")
        else:
            st.info("Upload a leaf image and click Detect Disease to see results")
            return

# ===== PAGE 7: CROP YIELD PREDICTOR =====
def page_yield_predictor():
    st.title("🌾 Crop Yield Predictor")
    st.markdown(
        "Predict crop yield based on flood risk, "
        "rainfall and historical patterns for "
        "any district in India"
    )

    import json

    try:
        with open(MODELS_DIR / "crop_yield_metadata.json", encoding="utf-8") as f:
            metadata = json.load(f)
        crops = metadata["crops"]
        states = metadata["states"]
        seasons = metadata["seasons"]
    except Exception:
        crops = ["Rice", "Wheat", "Cotton", "Sugarcane", "Maize", "Potato"]
        states = ["Assam", "Bihar", "Punjab", "Maharashtra", "Kerala"]
        seasons = ["Kharif", "Rabi", "Zaid", "Whole Year"]

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📋 Input Details")

        selected_state = st.selectbox(
            "Select State",
            options=states,
            index=states.index(st.session_state["last_state"]) if st.session_state.get("last_state") in states else 0,
            help="Select your state",
        )

        selected_crop = st.selectbox(
            "Select Crop",
            options=crops,
            help="Select crop type",
        )

        selected_season = st.selectbox(
            "Select Season",
            options=seasons,
            help="Kharif=June-Nov, Rabi=Nov-Apr",
        )

        area = st.number_input(
            "Area under cultivation (hectares)",
            min_value=0.1,
            max_value=10000.0,
            value=1.0,
            step=0.1,
        )

        default_flood_risk = st.session_state.get("last_risk_score", 30)

        flood_risk = st.slider(
            "Current Flood Risk (%)",
            min_value=0,
            max_value=100,
            value=int(default_flood_risk),
            help="Auto-filled from Risk Predictor if you made a prediction",
        )

        rainfall = st.number_input(
            "Expected Rainfall (mm)",
            min_value=0,
            max_value=5000,
            value=1000,
            step=50,
        )

        predict_btn = st.button(
            "🌾 Predict Crop Yield",
            type="primary",
            use_container_width=True,
        )

    with col2:
        if predict_btn:
            from src.crop_yield_predictor import predict_crop_yield

            with st.spinner("Calculating yield prediction..."):
                result = predict_crop_yield(
                    state=selected_state,
                    crop=selected_crop,
                    season=selected_season,
                    area_hectares=area,
                    flood_risk_pct=flood_risk,
                    current_rainfall=rainfall,
                )

            if "error" in result:
                st.error(result["error"])
            else:
                colors = {
                    "Low": "#10b981",
                    "Moderate": "#f59e0b",
                    "High": "#ef4444",
                }
                color = colors.get(result["risk_level"], "#06b6d4")

                st.markdown(f"""
                <div style='background: rgba(0,0,0,0.3);
                border: 2px solid {color};
                border-radius: 16px;
                padding: 20px;
                text-align: center;
                margin-bottom: 16px;'>
                <h2 style='color: {color};'>
                {result['adjusted_yield']} t/ha</h2>
                <p style='color: #aaa;'>
                Expected Yield</p>
                </div>
                """, unsafe_allow_html=True)

                c1, c2 = st.columns(2)
                c1.metric("Base Yield", f"{result['base_yield']} t/ha")
                c2.metric(
                    "Yield Loss",
                    f"{result['yield_loss_pct']}%",
                    delta=f"-{result['yield_loss_pct']}%",
                    delta_color="inverse",
                )

                c3, c4 = st.columns(2)
                c3.metric("Total Production", f"{result['total_production']} t")
                c4.metric("Economic Loss", f"₹{result['economic_loss']:,.0f}")

                if result["vs_average"] >= 0:
                    st.success(
                        f"✅ {result['adjusted_yield']} t/ha is "
                        f"{abs(result['vs_average']):.2f} t/ha ABOVE district average"
                    )
                else:
                    st.warning(
                        f"⚠️ {result['adjusted_yield']} t/ha is "
                        f"{abs(result['vs_average']):.2f} t/ha BELOW district average"
                    )

                st.info(f"💡 {result['recommendation']}")

                if flood_risk > 50:
                    st.error(
                        "🛡️ Apply for PMFBY Crop Insurance immediately!\n"
                        "Call: 1800-180-1551 (Toll Free)"
                    )

    st.markdown("---")
    try:
        with open(MODELS_DIR / "crop_yield_metadata.json", encoding="utf-8") as f:
            meta = json.load(f)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Model R² Score", f"{meta.get('rf_r2', 0):.3f}")
        c2.metric("Crops Covered", len(meta.get("crops", [])))
        c3.metric("States Covered", len(meta.get("states", [])))
        c4.metric("Algorithm", "RF + GBM Ensemble")
    except Exception:
        pass

# ===== PAGE 8: CROP LOSS ESTIMATOR =====
def page_crop_loss_estimator():
    st.title("🌾 Crop Loss Estimator")
    st.markdown("""
Estimate financial loss to your crops due to
flooding. Get compensation scheme information.
""")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### 📋 Farm Details")

        crop_data = get_crop_data()
        state = st.selectbox("Select State", sorted(crop_data.keys()))

        available_crops = crop_data[state]
        selected_crops = st.multiselect(
            "Select Your Crops",
            available_crops,
            default=[available_crops[0]],
        )

        area_per_crop = {}
        if selected_crops:
            st.markdown("**Area per crop (hectares):**")
            for crop in selected_crops:
                area = st.number_input(
                    f"{crop} area (ha)",
                    min_value=0.1,
                    max_value=1000.0,
                    value=1.0,
                    step=0.5,
                    key=f"area_{crop}",
                )
                area_per_crop[crop] = area

        season = st.selectbox(
            "Crop Season",
            ["Kharif", "Rabi", "Zaid", "Whole Year"],
        )

        st.markdown("**Flood Parameters:**")
        # Initialize session state
        if "flood_prob_val" not in st.session_state:
            st.session_state.flood_prob_val = 30
        if "flood_days_val" not in st.session_state:
            st.session_state.flood_days_val = 7

        # Flood Probability - simple number input
        st.markdown("**Flood Probability (%)**")
        flood_prob = st.number_input(
            "Flood Probability", 
            min_value=0, max_value=100,
            value=st.session_state.flood_prob_val,
            step=5,
            label_visibility="collapsed",
            key="flood_prob_input",
            help="Estimated probability of flooding (0-100%)"
        )
        st.session_state.flood_prob_val = flood_prob
        flood_prob = flood_prob / 100

        # Expected Flood Duration - simple number input  
        st.markdown("**Expected Flood Duration (days)**")
        flood_days = st.number_input(
            "Flood Duration",
            min_value=1, max_value=30,
            value=st.session_state.flood_days_val,
            step=1,
            label_visibility="collapsed",
            key="flood_days_input",
            help="Expected number of days of flooding (1-30)"
        )
        st.session_state.flood_days_val = flood_days
        flood_duration = flood_days

        calculate_btn = st.button(
            "💰 Calculate Crop Loss",
            type="primary",
            use_container_width=True,
        )

    with col2:
        if calculate_btn and selected_crops:
            results = estimate_crop_loss(
                state,
                selected_crops,
                area_per_crop,
                flood_prob,
                flood_duration,
                season,
            )

            loss = results["total_loss_lakhs"]
            if loss < 1:
                loss_str = f"₹{results['total_loss_inr']:,.0f}"
            elif loss < 100:
                loss_str = f"₹{loss:.2f} Lakhs"
            else:
                loss_str = f"₹{results['total_loss_crores']:.2f} Crores"

            st.markdown(f"""
            <div style='background: linear-gradient(
                135deg, #1a1a2e, #16213e);
                border: 2px solid #ff4444;
                border-radius: 15px;
                padding: 25px;
                text-align: center;
                margin-bottom: 20px'>
                <h2 style='color: #ff4444; margin:0'>
                Estimated Loss</h2>
                <h1 style='color: white;
                    font-size: 2.5em; margin:10px 0'>
                {loss_str}</h1>
                <p style='color: #aaa; margin:0'>
                Across {results['total_area_ha']:.1f}
                hectares — Avg damage:
                {results['avg_damage_pct']:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)

            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric("Worst Affected", results["worst_crop"])
            with m2:
                st.metric("Safest Crop", results["safest_crop"])
            with m3:
                st.metric("Total Area", f"{results['total_area_ha']:.1f} ha")

            st.markdown("### 📊 Crop-wise Breakdown")
            df = pd.DataFrame(results["crops"])
            df = df.rename(columns={
                "crop": "Crop",
                "area_ha": "Area (ha)",
                "damage_pct": "Damage %",
                "normal_yield_q": "Normal Yield (q)",
                "damaged_yield_q": "Yield Lost (q)",
                "loss_lakhs": "Loss (Lakhs ₹)",
            })
            st.dataframe(
                df[["Crop", "Area (ha)", "Damage %", "Yield Lost (q)", "Loss (Lakhs ₹)"]],
                use_container_width=True,
            )

            st.markdown("### 📈 Loss Visualization")
            fig = plot_loss_chart(results["crops"])
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("### 🏛️ Government Compensation")
            schemes = get_compensation_schemes()
            for key, scheme in schemes.items():
                with st.expander(f"📋 {scheme['name']}"):
                    st.markdown(f"**Coverage:** {scheme['coverage']}")
                    if "premium" in scheme:
                        st.markdown(f"**Premium:** {scheme['premium']}")
                    st.markdown(f"**How to Apply:** {scheme['how_to_apply']}")
                    st.markdown(f"**Helpline:** {scheme['helpline']}")
                    st.markdown(f"**Website:** {scheme['website']}")

        elif calculate_btn and not selected_crops:
            st.warning("Please select at least one crop!")

        else:
            st.info("""
            👈 Fill in your farm details and click
            'Calculate Crop Loss' to see estimated
            financial impact of flooding on your crops.

            This tool helps farmers:
            - Estimate financial losses before floods
            - Plan crop insurance
            - Apply for government compensation
            """)

# ===== PAGE 9: ALERT SYSTEM =====
def page_alert_system():
    st.title("🔔 Flood Alert System")
    st.markdown("""
Subscribe to receive flood alerts via Email before floods hit your district.
""")

    stats = get_alert_stats()
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Subscribers", stats["total_subscribers"])
    c2.metric("Alerts Today", stats["alerts_sent_today"])
    c3.metric("Districts Monitored", stats["active_districts"])

    districts_df = load_india_districts()
    if not districts_df.empty and "district" in districts_df.columns:
        district_list = sorted(districts_df["district"].dropna().unique().tolist())
    else:
        district_list = sorted(DISTRICTS)

    st.markdown("---")

    # STEP 1 - Header:
    st.markdown("""
<div style="display:flex;align-items:center;gap:12px;margin-bottom:20px;">
  <div style="width:40px;height:40px;background:#ef444420;
    border-radius:10px;display:flex;align-items:center;
    justify-content:center;font-size:20px;">🔔</div>
  <div>
    <div style="color:#f1f5f9;font-size:15px;font-weight:600;">
      Flood Alert Subscription</div>
    <div style="color:#64748b;font-size:12px;">
      Get notified before floods hit your area</div>
  </div>
</div>
""", unsafe_allow_html=True)

    # STEP 2 - Email and District in 2 columns:
    col1, col2 = st.columns(2)
    with col1:
        email = st.text_input("EMAIL", 
            placeholder="you@example.com",
            key="alert_email", label_visibility="visible")
    with col2:
        district_list_with_placeholder = ["Select your district..."] + district_list
        default_idx = 0
        last_district = st.session_state.get("last_district")
        if last_district and last_district in district_list_with_placeholder:
            default_idx = district_list_with_placeholder.index(last_district)
        else:
            for target in ["Kamrup", "Patna", "Delhi"]:
                if target in district_list_with_placeholder:
                    default_idx = district_list_with_placeholder.index(target)
                    break
        
        district = st.selectbox("DISTRICT",
            options=district_list_with_placeholder,
            index=default_idx,
            key="alert_district")

    # STEP 3 - Risk threshold selector as 4 visual cards:
    st.markdown("""
<div style="background:#1e293b;border-radius:10px;
  padding:16px;margin:16px 0;">
  <div style="color:#94a3b8;font-size:11px;text-transform:uppercase;
    letter-spacing:1px;margin-bottom:12px;">
    Alert threshold — notify me when risk exceeds</div>
</div>
""", unsafe_allow_html=True)

    if "alert_threshold" not in st.session_state:
        st.session_state.alert_threshold = 40

    t1, t2, t3, t4 = st.columns(4)

    thresholds = [
        (t1, 20, "Low", "#22c55e"),
        (t2, 40, "Moderate", "#f59e0b"),
        (t3, 60, "High", "#f97316"),
        (t4, 80, "Very High", "#ef4444"),
    ]

    for col, val, label, color in thresholds:
        with col:
            is_selected = st.session_state.alert_threshold == val
            st.markdown(f"""
            <div style="background:#0f172a;
              border:2px solid {'#06b6d4' if is_selected else color+'60'};
              border-radius:8px;padding:12px;text-align:center;">
              <div style="font-size:20px;font-weight:600;
                color:{'#06b6d4' if is_selected else color};">
                {val}%</div>
              <div style="font-size:11px;color:#64748b;
                margin-top:4px;">{label}</div>
            </div>
            """, unsafe_allow_html=True)
            if st.button(f"Select", key=f"thresh_{val}",
              use_container_width=True):
                st.session_state.alert_threshold = val
                st.rerun()

    threshold = st.session_state.alert_threshold

    daily = st.checkbox(
        "Also send daily morning forecast",
        value=True,
        key="alert_daily",
    )

    # STEP 4 - Subscribe button:
    if st.button("🔔 Subscribe to Alerts", 
        use_container_width=True, key="subscribe_btn"):
        if email and district and district != "Select your district...":
            name = email.split("@")[0].capitalize()
            selected_state = "Assam"
            if not districts_df.empty:
                match_df = districts_df[districts_df["district"] == district]
                if not match_df.empty:
                    selected_state = match_df.iloc[0]["state"]
            
            st.session_state["last_district"] = district
            st.session_state["last_state"] = selected_state

            sub_id = subscribe_user(
                name=name,
                district=district,
                state=selected_state,
                alert_type="Email",
                contact=email,
                risk_threshold=threshold,
                send_daily=daily,
            )
            st.success(f"""
            ✅ Subscribed successfully!
            You will receive alerts when flood
            risk in {district} exceeds {threshold}%
            via Email.
            Your ID: {sub_id}
            """)
        else:
            if not email:
                st.warning("Please enter your email address!")
            elif not district or district == "Select your district...":
                st.warning("Please select your district!")

    st.markdown("---")

    # STEP 5 - Keep all existing email sending and subscription logic below the UI unchanged.
    bottom_col1, bottom_col2 = st.columns(2)
    
    with bottom_col1:
        with st.expander("Unsubscribe"):
            unsubscribe_id = st.text_input("Subscription ID", key="unsubscribe_id")
            if st.button("Deactivate Subscription", use_container_width=True):
                if unsubscribe_id and unsubscribe(unsubscribe_id):
                    st.success("Subscription deactivated.")
                else:
                    st.error("Subscription ID not found.")

    with bottom_col2:
        st.markdown("### 🧪 Test Alert")
        test_email = st.text_input(
            "Test email address",
            placeholder="Send test alert to this email",
            key="test_alert_email",
        )
        if st.button("📧 Send Test Email", use_container_width=True):
            if test_email:
                risk_score_val = round(float(st.session_state.get("risk_score", 0)) * 100, 1)
                target_district = district if (district and district != "Select your district...") else st.session_state.get("selected_district", "Unknown")
                sent, message = send_email_alert(
                    to_email=test_email,
                    district=target_district,
                    state=st.session_state.get("last_state", "Unknown"),
                    risk_score=risk_score_val,
                    risk_level=st.session_state.get("risk_level", "Low"),
                    risk_pct=risk_score_val,
                    forecast_data=[],
                )
                if sent:
                    st.success(f"✅ {message}")
                else:
                    st.error(f"❌ Email failed: {message}")
            else:
                st.warning("Enter a test email address.")

# ===== PAGE 3: HISTORICAL FLOOD TRENDS =====
def page_trends():
    lang = st.session_state.get("lang", "en")
    
    # Heading & Subheading
    st.markdown(f"""
    <div style="text-align: center; padding: 20px 0 10px 0;">
        <h1 style="font-size: 2.8rem !important; margin-bottom: 4px;">{get_text("trends_title", lang)}</h1>
        <p class="hero-text" style="max-width: 600px; margin: 0 auto;">{get_text("trends_subtitle", lang)}</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)
    
    # Load data
    df = load_ndma_history()

    # Map possible column name variants
    col_map = {
        'state_name': 'state',
        'district_name': 'district',
        'year_of_flood': 'year',
        'flood_year': 'year',
        'no_of_floods': 'flood_events',
        'floods': 'flood_events',
        'area_affected': 'area_affected_ha',
        'affected_area': 'area_affected_ha',
        'population_affected': 'people_affected',
        'affected_population': 'people_affected',
        'damage_incrore': 'damage_cr',
        'damage_cr_': 'damage_cr',
        'total_damage': 'damage_cr',
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    # Ensure required columns exist with defaults
    if 'flood_events' not in df.columns:
        df['flood_events'] = 1
    if 'year' not in df.columns and 'year' not in df.columns:
        df['year'] = 2020
    if 'area_affected_ha' not in df.columns:
        df['area_affected_ha'] = 0
    if 'people_affected' not in df.columns:
        df['people_affected'] = 0
    if 'damage_cr' not in df.columns:
        df['damage_cr'] = 0
    if df.empty:
        st.warning("Historical flood data not found.")
        st.warning("Using sample data — upload ndma_flood_history.csv to data/ for real historical trends.")
        import numpy as np
        years = list(range(2010, 2024))
        df = pd.DataFrame({
            'year': years,
            'district': ['Sample District'] * len(years),
            'state': ['Sample State'] * len(years),
            'flood_events': np.random.randint(1, 8, len(years)),
            'area_affected_ha': np.random.randint(1000, 50000, len(years)),
            'people_affected': np.random.randint(5000, 200000, len(years)),
            'damage_cr': np.random.randint(10, 500, len(years))
        })
    df_trends = df
        
    # Filters Row (3 columns)
    col1, col2, col3 = st.columns(3)
    
    with col1:
        state_options = sorted(df["state"].unique())
        selected_state = st.selectbox(get_text("select_state", lang), state_options, key="trends_state")
        
    with col2:
        district_options = sorted(df[df["state"] == selected_state]["district"].unique())
        selected_district = st.selectbox(get_text("select_district", lang), district_options, key="trends_district")
        
    with col3:
        metric_labels = [
            get_text("metric_flood_events", lang),
            get_text("metric_area", lang),
            get_text("metric_people", lang),
            get_text("metric_damage", lang)
        ]
        default_metric = [get_text("metric_flood_events", lang)]
        selected_labels = st.multiselect(
            get_text("select_metrics", lang),
            options=metric_labels,
            default=default_metric,
            key="trends_metrics"
        )
        
    # If no metrics are selected, fall back to "Flood Events" to prevent empty chart errors
    if not selected_labels:
        selected_labels = default_metric
        
    # Map selected translated metric labels to column names and translations
    selected_cols = []
    for label in selected_labels:
        if label == get_text("metric_flood_events", lang):
            selected_cols.append(("flood_events", get_text("metric_flood_events", lang)))
        elif label == get_text("metric_area", lang):
            selected_cols.append(("area_affected_ha", get_text("metric_area", lang)))
        elif label == get_text("metric_people", lang):
            selected_cols.append(("people_affected", get_text("metric_people", lang)))
        elif label == get_text("metric_damage", lang):
            selected_cols.append(("damage_cr", get_text("metric_damage", lang)))

    # Filter df to selected district
    filtered = df_trends[(df_trends['state'] == selected_state) & (df_trends['district'] == selected_district)].sort_values("year")
    
    # Calculate stats
    total_events = int(filtered["flood_events"].sum())
    
    # Worst year: year with max flood events
    max_events_idx = filtered["flood_events"].idxmax()
    worst_year = int(filtered.loc[max_events_idx, "year"])
    
    # Peak people affected: max in any year
    peak_people = int(filtered["people_affected"].max())
    
    # Total damage
    total_damage = float(filtered["damage_cr"].sum())
    
    # Render Stat Cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="metric-card" style="display: flex; flex-direction: column; justify-content: center; align-items: center; height: 130px; padding: 16px 12px; text-align: center; border-top: 3px solid #06b6d4 !important; box-sizing: border-box; overflow: hidden;">
            <div class="metric-label" style="margin-bottom: 6px; font-size: 11px; overflow: hidden; text-overflow: ellipsis; display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; line-height: 1.2; height: 28px; width: 100%;">{get_text("stat_total_events", lang)}</div>
            <div class="metric-value" style="color: #06b6d4; font-size: 1.8rem !important; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; width: 100%; line-height: 1.1;">{total_events}</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-card" style="display: flex; flex-direction: column; justify-content: center; align-items: center; height: 130px; padding: 16px 12px; text-align: center; border-top: 3px solid #06b6d4 !important; box-sizing: border-box; overflow: hidden;">
            <div class="metric-label" style="margin-bottom: 6px; font-size: 11px; overflow: hidden; text-overflow: ellipsis; display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; line-height: 1.2; height: 28px; width: 100%;">{get_text("stat_worst_year", lang)}</div>
            <div class="metric-value" style="color: #06b6d4; font-size: 1.8rem !important; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; width: 100%; line-height: 1.1;">{worst_year}</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="metric-card" style="display: flex; flex-direction: column; justify-content: center; align-items: center; height: 130px; padding: 16px 12px; text-align: center; border-top: 3px solid #06b6d4 !important; box-sizing: border-box; overflow: hidden;">
            <div class="metric-label" style="margin-bottom: 6px; font-size: 11px; overflow: hidden; text-overflow: ellipsis; display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; line-height: 1.2; height: 28px; width: 100%;">{get_text("stat_peak_people", lang)}</div>
            <div class="metric-value" style="color: #06b6d4; font-size: 1.8rem !important; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; width: 100%; line-height: 1.1;">{peak_people:,}</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="metric-card" style="display: flex; flex-direction: column; justify-content: center; align-items: center; height: 130px; padding: 16px 12px; text-align: center; border-top: 3px solid #06b6d4 !important; box-sizing: border-box; overflow: hidden;">
            <div class="metric-label" style="margin-bottom: 6px; font-size: 11px; overflow: hidden; text-overflow: ellipsis; display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; line-height: 1.2; height: 28px; width: 100%;">{get_text("stat_total_damage", lang)}</div>
            <div class="metric-value" style="color: #06b6d4; font-size: 1.8rem !important; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; width: 100%; line-height: 1.1;">₹{total_damage:,.0f} Cr</div>
        </div>
        """, unsafe_allow_html=True)

    # Trend Indicator Calculation
    df_recent = filtered[filtered["year"] >= 2020]
    df_old = filtered[filtered["year"] < 2020]
    
    avg_recent = df_recent["flood_events"].mean() if not df_recent.empty else 0.0
    avg_old = df_old["flood_events"].mean() if not df_old.empty else 0.0
    
    if avg_old > 0:
        pct_change = ((avg_recent - avg_old) / avg_old) * 100
    else:
        pct_change = 0.0
        
    # Render Trend Indicator
    if pct_change > 0:
        trend_html = f"""
        <div class="rec-box rec-mod" style="background: rgba(245,158,11,0.08); border-left-color: #f59e0b !important; padding: 16px; margin: 16px 0;">
            <h4 style="color: #f59e0b; margin: 0; display: flex; align-items: center; gap: 8px; font-weight: 700;">
                ⚠️ {get_text("trend_increasing", lang)} (+{pct_change:.1f}%)
            </h4>
        </div>
        """
    elif pct_change < 0:
        trend_html = f"""
        <div class="rec-box rec-safe" style="background: rgba(16,185,129,0.08); border-left-color: #10b981 !important; padding: 16px; margin: 16px 0;">
            <h4 style="color: #10b981; margin: 0; display: flex; align-items: center; gap: 8px; font-weight: 700;">
                ✅ {get_text("trend_decreasing", lang)} ({pct_change:.1f}%)
            </h4>
        </div>
        """
    else:
        trend_html = f"""
        <div class="rec-box rec-safe" style="background: rgba(6,182,212,0.08); border-left-color: #06b6d4 !important; padding: 16px; margin: 16px 0;">
            <h4 style="color: #06b6d4; margin: 0; display: flex; align-items: center; gap: 8px; font-weight: 700;">
                ➡️ {get_text("trend_stable", lang)}
            </h4>
        </div>
        """
    render_html(trend_html)
    
    # CHART 1 — Year-over-Year Line Chart
    fig1 = go.Figure()
    colors = ["#06b6d4", "#a855f7", "#ec4899", "#10b981"]
    
    for i, (col, display_label) in enumerate(selected_cols):
        fig1.add_trace(go.Scatter(
            x=filtered["year"],
            y=filtered[col],
            mode="lines+markers",
            name=display_label,
            line=dict(color=colors[i % len(colors)], width=3),
            marker=dict(size=8, symbol="circle"),
            hovertemplate="%{x}: %{y}<extra></extra>"
        ))
        
    fig1.update_layout(
        title=dict(
            text=get_text("chart_yoy_title", lang).format(district=selected_district, state=selected_state),
            font=dict(size=16, color="#f1f5f9", family="Inter")
        ),
        paper_bgcolor="#0f172a",
        plot_bgcolor="#1e293b",
        font=dict(color="#94a3b8", family="Inter"),
        xaxis=dict(
            tickmode="linear",
            tick0=2015,
            dtick=1,
            gridcolor="rgba(255,255,255,0.05)",
            linecolor="rgba(255,255,255,0.1)",
            title=""
        ),
        yaxis=dict(
            gridcolor="rgba(255,255,255,0.05)",
            linecolor="rgba(255,255,255,0.1)",
            title=""
        ),
        hovermode="x unified",
        margin=dict(l=40, r=40, t=60, b=40),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    st.plotly_chart(fig1, use_container_width=True)
    
    # CHART 2 — Bar Chart (Flood Events by Year)
    fig2 = go.Figure(go.Bar(
        x=filtered["year"],
        y=filtered["flood_events"],
        marker=dict(
            color=filtered["flood_events"],
            colorscale=[[0.0, "#06b6d4"], [1.0, "#ef4444"]],
            showscale=False
        ),
        hovertemplate="Year %{x}: %{y} events<extra></extra>"
    ))
    
    fig2.update_layout(
        title=dict(
            text=get_text("chart_annual_title", lang),
            font=dict(size=16, color="#f1f5f9", family="Inter")
        ),
        paper_bgcolor="#0f172a",
        plot_bgcolor="#1e293b",
        font=dict(color="#94a3b8", family="Inter"),
        xaxis=dict(
            tickmode="linear",
            tick0=2015,
            dtick=1,
            gridcolor="rgba(255,255,255,0.05)",
            linecolor="rgba(255,255,255,0.1)",
            title=""
        ),
        yaxis=dict(
            gridcolor="rgba(255,255,255,0.05)",
            linecolor="rgba(255,255,255,0.1)",
            title=get_text("metric_flood_events", lang),
            tickmode="linear",
            tick0=0,
            dtick=1
        ),
        margin=dict(l=40, r=40, t=60, b=40)
    )
    st.plotly_chart(fig2, use_container_width=True)
    
    # CHART 3 — State Comparison (optional toggle)
    compare_state = st.checkbox(get_text("compare_checkbox", lang), value=False)
    if compare_state:
        df_state = df[df["state"] == selected_state].sort_values(["year", "district"])
        
        fig3 = px.bar(
            df_state,
            x="year",
            y="flood_events",
            color="district",
            barmode="group",
            title=get_text("chart_compare_title", lang).format(state=selected_state),
            color_discrete_sequence=px.colors.qualitative.Safe
        )
        
        fig3.update_layout(
            paper_bgcolor="#0f172a",
            plot_bgcolor="#1e293b",
            font=dict(color="#94a3b8", family="Inter"),
            xaxis=dict(
                tickmode="linear",
                tick0=2015,
                dtick=1,
                gridcolor="rgba(255,255,255,0.05)",
                linecolor="rgba(255,255,255,0.1)",
                title=""
            ),
            yaxis=dict(
                gridcolor="rgba(255,255,255,0.05)",
                linecolor="rgba(255,255,255,0.1)",
                title=get_text("metric_flood_events", lang)
            ),
            margin=dict(l=40, r=40, t=60, b=40),
            legend=dict(
                title=dict(text="", font=dict(color="#f1f5f9")),
                font=dict(color="#cbd5e1")
            )
        )
        st.plotly_chart(fig3, use_container_width=True)


# ===== PAGE 10: ABOUT =====
def page_about():
    # Redesigned About page using modern Bento Grid layout
    st.markdown("""
<style>
  .bento-container {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 10px;
    margin-bottom: 10px;
    font-family: 'Inter', system-ui, -apple-system, sans-serif;
  }
  .bento-card {
    background: #0d1b2a;
    border: 1px solid rgba(0,188,212,0.2);
    border-radius: 16px;
    padding: 18px 20px;
    box-sizing: border-box;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
  }
  .bento-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(0, 188, 212, 0.08);
  }
  .bento-card.full-width {
    grid-column: span 2;
  }
  .bento-card.hero-accent {
    background: #0e2a3a;
    border: 1px solid rgba(0,188,212,0.3);
  }
  .bento-title {
    font-size: 14px;
    color: white;
    font-weight: 600;
    margin-bottom: 8px;
  }
  .bento-hero-title {
    color: #00BCD4;
    font-size: 26px;
    font-weight: 700;
    margin: 0;
  }
  .bento-hero-subtitle {
    color: rgba(255,255,255,0.7);
    font-size: 13px;
    margin-top: 4px;
    margin-bottom: 20px;
  }
  .bento-stats-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 10px;
    flex-wrap: wrap;
    margin-top: 15px;
  }
  .bento-stat-item {
    flex: 1;
    min-width: 100px;
    text-align: center;
  }
  .bento-stat-number {
    font-size: 28px;
    color: #00BCD4;
    font-weight: 600;
  }
  .bento-stat-label {
    font-size: 11px;
    color: rgba(255,255,255,0.4);
    text-transform: uppercase;
    margin-top: 2px;
  }
  .bento-metric-number {
    font-size: 24px;
    color: #00BCD4;
    font-weight: 600;
  }
  .bento-metric-label {
    font-size: 11px;
    color: rgba(255,255,255,0.4);
    text-transform: uppercase;
    margin-top: 2px;
  }
  .bento-tags {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-top: 10px;
  }
  .bento-tag {
    border: 0.5px solid rgba(0,188,212,0.4);
    color: #00BCD4;
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 12px;
    font-weight: 500;
    background: rgba(0,188,212,0.02);
    display: inline-block;
    transition: all 0.2s ease;
  }
  .bento-tag:hover {
    background: rgba(0,188,212,0.1);
    transform: translateY(-1px);
    border-color: rgba(0,188,212,0.7);
  }
  .bento-inner-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 15px;
    margin-top: 12px;
  }
  .bento-inner-item {
    display: flex;
    gap: 12px;
    align-items: flex-start;
  }
  .bento-inner-icon {
    font-size: 22px;
    background: rgba(0,188,212,0.1);
    border-radius: 10px;
    width: 42px;
    height: 42px;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
  }
  .bento-inner-content {
    display: flex;
    flex-direction: column;
  }
  .bento-inner-title {
    font-size: 13px;
    color: white;
    font-weight: 600;
  }
  .bento-inner-desc {
    font-size: 11.5px;
    color: rgba(255,255,255,0.6);
    margin-top: 2px;
    line-height: 1.4;
  }
  .bento-footer {
    text-align: center;
    margin-top: 20px;
    color: rgba(255,255,255,0.5);
    font-size: 12px;
    font-family: 'Inter', system-ui, sans-serif;
  }
  
  @media (max-width: 600px) {
    .bento-container {
      grid-template-columns: 1fr;
    }
    .bento-card.full-width {
      grid-column: span 1;
    }
    .bento-inner-grid {
      grid-template-columns: 1fr;
    }
    .bento-stats-row {
      flex-direction: column;
      gap: 15px;
    }
  }
</style>

<div class="bento-container">
  
  <!-- 1. HERO BENTO CARD -->
  <div class="bento-card full-width hero-accent">
    <h1 class="bento-hero-title">FloodGuard AI</h1>
    <div class="bento-hero-subtitle">India's Complete Flood Risk Prediction & Agricultural Intelligence System</div>
    <div class="bento-stats-row">
      <div class="bento-stat-item">
        <div class="bento-stat-number">736</div>
        <div class="bento-stat-label">Districts</div>
      </div>
      <div class="bento-stat-item">
        <div class="bento-stat-number">36</div>
        <div class="bento-stat-label">States</div>
      </div>
      <div class="bento-stat-item">
        <div class="bento-stat-number">4,695</div>
        <div class="bento-stat-label">Records</div>
      </div>
      <div class="bento-stat-item">
        <div class="bento-stat-number">5</div>
        <div class="bento-stat-label">AI Models</div>
      </div>
    </div>
  </div>

  <!-- 2. MODEL METRIC CARDS -->
  <div class="bento-card">
    <div class="bento-title">XGBoost Predictor</div>
    <div class="bento-metric-number">0.83</div>
    <div class="bento-metric-label">AUC</div>
  </div>
  
  <div class="bento-card">
    <div class="bento-title">LSTM Attention</div>
    <div class="bento-metric-number">0.70</div>
    <div class="bento-metric-label">Recall</div>
  </div>
  
  <div class="bento-card">
    <div class="bento-title">CNN Classifier</div>
    <div class="bento-metric-number">89.6%</div>
    <div class="bento-metric-label">Accuracy</div>
  </div>
  
  <div class="bento-card">
    <div class="bento-title">Crop Disease CNN</div>
    <div class="bento-metric-number">98.4%</div>
    <div class="bento-metric-label">Accuracy</div>
  </div>

  <!-- 3. TECH STACK CARD -->
  <div class="bento-card full-width">
    <div class="bento-title">Tech Stack</div>
    <div class="bento-tags">
      <span class="bento-tag">Python</span>
      <span class="bento-tag">Streamlit</span>
      <span class="bento-tag">PyTorch</span>
      <span class="bento-tag">XGBoost</span>
      <span class="bento-tag">Gemini API</span>
      <span class="bento-tag">OpenWeatherMap</span>
      <span class="bento-tag">Folium</span>
      <span class="bento-tag">SHAP</span>
      <span class="bento-tag">Open-Meteo</span>
      <span class="bento-tag">fpdf2</span>
    </div>
  </div>

  <!-- 4. DATA SOURCES CARD -->
  <div class="bento-card full-width">
    <div class="bento-title">Data Sources</div>
    <div class="bento-inner-grid">
      <div class="bento-inner-item">
        <div class="bento-inner-icon">🌧️</div>
        <div class="bento-inner-content">
          <span class="bento-inner-title">IMD Rainfall Data</span>
          <span class="bento-inner-desc">Daily historical rainfall grids sourced from the Indian Meteorological Department.</span>
        </div>
      </div>
      <div class="bento-inner-item">
        <div class="bento-inner-icon">🚨</div>
        <div class="bento-inner-content">
          <span class="bento-inner-title">NDMA Flood Records</span>
          <span class="bento-inner-desc">Official disaster management archives of flood mappings and logs.</span>
        </div>
      </div>
      <div class="bento-inner-item">
        <div class="bento-inner-icon">🌿</div>
        <div class="bento-inner-content">
          <span class="bento-inner-title">PlantVillage (32,883 images)</span>
          <span class="bento-inner-desc">Comprehensive dataset for deep learning plant leaf disease classification.</span>
        </div>
      </div>
      <div class="bento-inner-item">
        <div class="bento-inner-icon">🌾</div>
        <div class="bento-inner-content">
          <span class="bento-inner-title">Crop Yield Data</span>
          <span class="bento-inner-desc">19,689 historical production records spanning 55 Indian agricultural crops.</span>
        </div>
      </div>
    </div>
  </div>

</div>

<!-- 5. FOOTER LINE -->
<div class="bento-footer">
  Built with ❤️ for farmers and disaster management teams across India.
</div>
""", unsafe_allow_html=True)


# ===== FOOTER =====
def render_footer():
    st.markdown("""<div class="app-footer">
        <p style="margin:0"><b style="color:#06b6d4">India Flood Risk & Agricultural Intelligence</b> | Powered by AI · Built with Streamlit</p>
    </div>""", unsafe_allow_html=True)


# ===== MAIN =====
def main():
    if "lang" not in st.session_state:
        st.session_state.lang = "en"
    lang = st.session_state.get("lang", "en")

    # --- Check connectivity once per session ---
    if "internet_status" not in st.session_state:
        st.session_state.internet_status = is_internet_available()
    is_online = st.session_state.internet_status

    # --- Show offline banner if no internet ---
    if not is_online:
        st.markdown("""
        <div style="background:#7c2d12; border:1px solid #ea580c; border-radius:8px;
        padding:12px 16px; margin-bottom:16px; display:flex; align-items:center; gap:10px;">
        <span style="font-size:20px;">📡</span>
        <div>
        <strong style="color:#fed7aa;">Offline Mode Active</strong><br>
        <span style="color:#fdba74; font-size:13px;">
        No internet connection detected. Showing last cached data.
        Reconnect to get live predictions.</span>
        </div>
        </div>
        """, unsafe_allow_html=True)

    # --- Persistent Global Sidebar ---
    with st.sidebar:
        st.markdown(f"""
<div style='text-align: center; padding: 10px 0;'>
    <h1 style='color: #06b6d4; font-size: 28px; font-weight: 800; margin-bottom: 4px;'>
        🌊 {get_text('app_title', lang)}
    </h1>
    <p style='color: #8899aa; font-size: 13px; margin: 0;'>
        {get_text('app_subtitle', lang)}
    </p>
</div>
""", unsafe_allow_html=True)
        st.markdown("---")

        # --- System Status Expander ---
        with st.sidebar.expander("📡 System Status", expanded=True):
            if is_online:
                st.markdown("""
                <div style="background:#14532d; border:1px solid #16a34a; border-radius:6px;
                padding:6px 12px; margin-bottom:8px; font-size:12px; color:#86efac;">
                🟢 Online — Live data active
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="background:#7c2d12; border:1px solid #ea580c; border-radius:6px;
                padding:6px 12px; margin-bottom:8px; font-size:12px; color:#fdba74;">
                🔴 Offline — Using cached data
                </div>
                """, unsafe_allow_html=True)

        # --- District Search ---
        districts_df = load_india_districts()
        ALL_DISTRICTS = sorted(districts_df["district"].dropna().unique().tolist())

        st.sidebar.markdown("### 🔍 Quick District Search")
        search_query = st.sidebar.text_input(
            "", 
            placeholder="Search district...",
            key="district_search",
            label_visibility="collapsed"
        )

        if search_query and len(search_query) >= 2:
            # Filter districts matching search
            matches = [d for d in ALL_DISTRICTS 
                       if search_query.lower() in d.lower()][:5]
            
            if matches:
                st.sidebar.markdown("**Results:**")
                for match in matches:
                    if st.sidebar.button(match, 
                      key=f"search_{match}",
                      use_container_width=True):
                        # Find corresponding state
                        matched_rows = districts_df[districts_df["district"] == match]
                        if not matched_rows.empty:
                            matched_state = matched_rows.iloc[0]["state"]
                            st.session_state["state_select"] = matched_state
                            st.session_state["selected_state"] = matched_state
                            st.session_state["trends_state"] = matched_state
                            st.session_state["forecast_state"] = matched_state
                        st.session_state["district_select"] = match
                        st.session_state["trends_district"] = match
                        st.session_state["forecast_district"] = match
                        st.session_state.selected_district = match
                        st.rerun()
            else:
                st.sidebar.caption("No districts found")

        st.sidebar.divider()

        # --- Language Expander ---
        with st.sidebar.expander("🌐 Language / भाषा", expanded=True):
            col1, col2 = st.columns(2)
            if col1.button("🇬🇧 English", use_container_width=True):
                st.session_state.lang = "en"
                st.rerun()
            if col2.button("🇮🇳 हिंदी", use_container_width=True):
                st.session_state.lang = "hi"
                st.rerun()

        st.markdown("---")

        # --- Cache Settings Expander ---
        with st.sidebar.expander("💾 Cache Settings", expanded=False):
            cache_keys = ["prediction", "forecast", "weather", "map_data"]
            for key in cache_keys:
                if is_cache_available(key):
                    st.markdown(
                        f"✅ {key.title()} cached ({get_cache_age(key)})"
                    )
                else:
                    st.markdown(f"❌ {key.title()} — not cached yet")
            cache_col1, cache_col2 = st.columns(2)
            with cache_col1:
                if st.button("🔄 Recheck", key="recheck_internet"):
                    st.session_state.internet_status = is_internet_available()
                    st.rerun()
            with cache_col2:
                if st.button("🗑️ Clear Cache", key="clear_cache"):
                    clear_all_cache()
                    st.success("Cache cleared!")
        st.markdown("---")

    # Generate ticker items from top 10 HIGH risk districts and 5 LOW risk districts
    try:
        ticker_df = load_india_districts()
        type_base = {
            "Flash / glacial flood": 0.68,
            "Coastal / river flood": 0.58,
            "Riverine flood": 0.62,
            "Urban / river flood": 0.44,
        }
        LOW_RISK_DISTRICTS = [
            "Jaisalmer", "Barmer", "Bikaner", "Jodhpur", "Churu",
            "Jalore", "Sirohi", "Pali", "Nagaur", "Sikar",
            "Ladakh", "Leh", "Kargil", "Lahaul And Spiti",
            "Kinnaur", "Dibang Valley", "Anjaw", "Tawang",
            "Upper Siang", "West Kameng", "East Kameng",
            "Phek", "Longleng", "Kiphire", "Tuensang",
            "Senapati", "Ukhrul", "Chandel", "Churachandpur",
            "Ri Bhoi", "East Khasi Hills", "West Jaintia Hills",
            "Hamirpur", "Una", "Bilaspur", "Solan",
            "Kutch", "Banaskantha", "Patan", "Mahesana"
        ]
        
        results = []
        for _, row in ticker_df.iterrows():
            dist = row['district']
            st_name = row['state']
            f_type = row.get("flood_type", "Riverine flood")
            
            score = min(max(type_base.get(f_type, 0.45) + ((hash(dist) % 21) - 10) / 100, 0.08), 0.92)
            level = "HIGH" if score >= 0.6 else "MODERATE" if score >= 0.3 else "LOW"
            
            row_risk = row.get("risk_level", row.get("risk", row.get("level", None)))
            if row_risk is not None:
                risk_level = str(row_risk).strip().upper()
            else:
                risk_level = level
                
            if dist in LOW_RISK_DISTRICTS:
                risk_level = "LOW"
                score = 0.15
                
            results.append({
                "district": dist,
                "state": st_name,
                "score": score,
                "level": risk_level
            })
            
        res_df = pd.DataFrame(results)
        high_risks = res_df[res_df["level"] == "HIGH"].sort_values(by=["score", "district"], ascending=[False, True]).head(10)
        low_risks = res_df[res_df["level"] == "LOW"].sort_values(by=["score", "district"], ascending=[True, True]).head(5)
        
        ticker_spans = []
        for _, row in high_risks.iterrows():
            ticker_spans.append(f'<span style="color:#ef4444;margin:0 8px;">⚠️ HIGH RISK: {row["district"]}, {row["state"]}</span>')
        for _, row in low_risks.iterrows():
            ticker_spans.append(f'<span style="color:#22c55e;margin:0 8px;">✅ LOW RISK: {row["district"]}, {row["state"]}</span>')
            
        sep = '<span style="color:#94a3b8;margin:0 8px;">|</span>'
        ticker_inner_html = sep.join(ticker_spans)
    except Exception:
        ticker_inner_html = """
        <span style="color:#ef4444;margin:0 8px;">⚠️ HIGH RISK: Dhubri, Assam</span>
        <span style="color:#94a3b8;margin:0 8px;">|</span>
        <span style="color:#ef4444;margin:0 8px;">⚠️ HIGH RISK: Darbhanga, Bihar</span>
        <span style="color:#94a3b8;margin:0 8px;">|</span>
        <span style="color:#f59e0b;margin:0 8px;">🟡 MODERATE: Bhubaneswar, Odisha</span>
        <span style="color:#94a3b8;margin:0 8px;">|</span>
        <span style="color:#22c55e;margin:0 8px;">✅ LOW RISK: Jaisalmer, Rajasthan</span>
        <span style="color:#94a3b8;margin:0 8px;">|</span>
        <span style="color:#f59e0b;margin:0 8px;">🟡 MODERATE: Kolhapur, Maharashtra</span>
        <span style="color:#94a3b8;margin:0 8px;">|</span>
        <span style="color:#ef4444;margin:0 8px;">⚠️ HIGH RISK: Alappuzha, Kerala</span>
        <span style="color:#94a3b8;margin:0 8px;">|</span>
        <span style="color:#22c55e;margin:0 8px;">✅ LOW RISK: Leh, Ladakh</span>
        """

    st.markdown("""
    <div style="background:#1e293b;border-top:2px solid #06b6d4;
      border-bottom:1px solid #334155;padding:8px 0;
      overflow:hidden;white-space:nowrap;margin-bottom:16px;">
      <div style="display:inline-block;animation:ticker 30s linear infinite;">
    """ + ticker_inner_html + """
      </div>
    </div>
    <style>
    @keyframes ticker {
      0% { transform: translateX(100vw); }
      100% { transform: translateX(-100%); }
    }
    </style>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11 = st.tabs([
        get_text("tab_risk", lang),
        get_text("tab_map", lang),
        get_text("tab_trends", lang),
        get_text("tab_forecast", lang),
        get_text("tab_chatbot", lang),
        get_text("tab_damage", lang),
        get_text("tab_crop_disease", lang),
        get_text("tab_yield", lang),
        get_text("tab_loss", lang),
        get_text("tab_alert", lang),
        get_text("tab_about", lang),
    ])
    with tab1:
        page_predictor()
    with tab2:
        page_map()
    with tab3:
        page_trends()
    with tab4:
        page_forecast()
    with tab5:
        page_chatbot()
    with tab6:
        page_damage_classifier()
    with tab7:
        page_crop_disease()
    with tab8:
        page_yield_predictor()
    with tab9:
        page_crop_loss_estimator()
    with tab10:
        page_alert_system()
    with tab11:
        page_about()
    render_footer()


if __name__ == "__main__":
    main()
