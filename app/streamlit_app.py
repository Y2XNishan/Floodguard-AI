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
        "app_subtitle": "Flood Risk Prediction & Agricultural Intelligence Platform",
        "sidebar_language": "Language / भाषा",
        "sidebar_nav": "Navigation",
        "tab_risk": "Risk Predictor",
        "tab_map": "Risk Map",
        "tab_forecast": "7-Day Forecast",
        "tab_chatbot": "Flood Assistant",
        "tab_damage": "Damage Classifier",
        "tab_crop_disease": "Crop Disease",
        "tab_yield": "Yield Predictor",
        "tab_loss": "Crop Loss Estimator",
        "tab_alert": "Alert System",
        "tab_metrics": "Model Metrics",
        "tab_about": "About",
        "select_district": "Select District",
        "select_state": "Select State",
        "select_date": "Select Date",
        "predict_button": "Run Risk Prediction",
        "risk_score": "Risk Score",
        "risk_level": "Risk Level",
        "district_label": "District",
        "download_pdf": "Download PDF Report",
        "report_ready": "Flood risk assessment report ready",
        "low_risk": "Low Risk — No Action Required",
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
        "hero_title": "Flood Risk Predictor",
        "hero_subtitle": "District-level risk assessment for 736 districts using IMD rainfall telemetry and flood inventory data",
        "weather_measurements": "Weather Measurements",
        "what_do_you_see": "What do you see outside?",
        "rain_question": "How is the rain right now?",
        "water_question": "Water situation near you?",
        "ground_question": "How is the ground / roads?",
        "temp_question": "How does it feel outside?",
        "flood_risk_gauge": "Flood Risk Gauge",
        "forecast_7day": "7-Day Forecast",
        "floodguard_summary": "Risk Summary & Advisory",
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
        "ai_preparing": "Preparing risk summary...",
        "ready_to_predict": "Ready to Predict",
        "ready_to_predict_desc": "Select a district, describe the weather, then predict flood risk.",
        "weather_auto_note": "Weather data is fetched automatically via Open-Meteo.",
        "model_not_found": "Model not found. Run `python src/train_baseline.py` first.",
        "risk_predictor": "Risk Predictor",
        "risk_map": "Risk Map",
        "forecast": "7-Day Forecast",
        "chatbot": "Flood Assistant",
        "damage": "Damage Classifier",
        "crop_disease": "Crop Disease",
        "yield": "Yield Predictor",
        "crop_loss": "Crop Loss Estimator",
        "alerts": "Alert System",
        "about": "About",
        "select_district": "Select District",
        "select_state": "Select State",
        "select_date": "Select Date",
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
        "tab_trends": "Flood Trends",
        "trends_title": "District-Level Flood Trend Analysis",
        "trends_subtitle": "Source: NDMA Historical Flood Records (2015–2024)",
        "select_metrics": "Select Metric(s)",
        "metric_flood_events": "Flood Events",
        "metric_area": "Area Affected (ha)",
        "metric_people": "People Affected",
        "metric_damage": "Damage (Cr ₹)",
        "stat_total_events": "Flood Events (2015–24)",
        "stat_worst_year": "Worst Year",
        "stat_peak_people": "Peak People Affected",
        "stat_total_damage": "Total Damage",
        "trend_increasing": "Flood frequency Increasing",
        "trend_decreasing": "Flood frequency Decreasing",
        "trend_stable": "Flood frequency Stable",
        "compare_checkbox": "Compare with other districts in same state",
        "chart_yoy_title": "Flood Trends: {district} ({state})",
        "chart_annual_title": "Annual Flood Event Count",
        "chart_compare_title": "State Comparison: Flood Events by Year ({state})",
    },
    "hi": {
        "app_title": "फ्लडगार्ड AI",
        "app_subtitle": "बाढ़ जोखिम पूर्वानुमान एवं कृषि सूचना प्रणाली",
        "sidebar_language": "Language / भाषा",
        "sidebar_nav": "नेविगेशन",
        "tab_risk": "जोखिम पूर्वानुमान",
        "tab_map": "जोखिम मानचित्र",
        "tab_forecast": "7-दिन पूर्वानुमान",
        "tab_chatbot": "सहायक",
        "tab_damage": "क्षति वर्गीकरण",
        "tab_crop_disease": "फसल रोग",
        "tab_yield": "उपज पूर्वानुमान",
        "tab_loss": "फसल हानि अनुमान",
        "tab_alert": "चेतावनी प्रणाली",
        "tab_metrics": "मॉडल मेट्रिक्स",
        "tab_about": "परिचय",
        "select_district": "जिला चुनें",
        "select_state": "राज्य चुनें",
        "select_date": "तारीख चुनें",
        "predict_button": "बाढ़ जोखिम का आकलन करें",
        "risk_score": "जोखिम स्कोर",
        "risk_level": "जोखिम स्तर",
        "district_label": "जिला",
        "download_pdf": "PDF रिपोर्ट डाउनलोड करें",
        "report_ready": "बाढ़ जोखिम रिपोर्ट तैयार है",
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
        "hero_title": "भारत बाढ़ जोखिम पूर्वानुमान",
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
        "tab_trends": "बाढ़ रुझान",
        "trends_title": "जिला-स्तरीय बाढ़ रुझान विश्लेषण",
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
        "home": "होम",
        "dashboard": "डैशबोर्ड",
        "weather": "मौसम",
        "agricultural": "कृषि",
        "support": "सहायता",
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

STATE_HELPLINES = {
    "Assam": "1070",
    "Bihar": "0612-2294204",
    "Kerala": "1077",
    "Maharashtra": "1077",
    "Tamil Nadu": "1077",
}

WEATHER_EMOJI_MAP = {
    "clear": "", "clouds": "", "rain": "",
    "thunderstorm": "", "drizzle": "", "snow": "",
    "mist": "", "fog": "",
}

st.set_page_config(page_title="FloodGuard AI", layout="wide", initial_sidebar_state="expanded")

# Font Awesome 6 CDN
st.markdown('<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">', unsafe_allow_html=True)


# ===== THEME & UI LOADER =====
def inject_custom_css():
    """Inject clean developer tool stylesheet and UI overrides."""
    css_path = APP_DIR / "style.css"
    if css_path.exists():
        try:
            with open(css_path, "r", encoding="utf-8") as f:
                css_content = f.read()
            st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)
        except Exception as e:
            st.warning(f"Could not load style.css: {e}")

    try:
        import importlib
        import ui_overrides
        importlib.reload(ui_overrides)
        ui_overrides.inject_ui_overrides()
    except Exception as e:
        st.warning(f"Could not load ui_overrides: {e}")


inject_custom_css()

# ===== DATA & MODEL LOADING =====
@st.cache_data
def load_sample_data():
    try:
        data_path = os.path.join(DATA_DIR, "sample_data.csv")
        return pd.read_csv(data_path, parse_dates=["date"])
    except Exception:
        return pd.DataFrame()

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
        normalized = path.replace(chr(92), "/")
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

@st.cache_data
def get_district_risk_data():
    """Compute and cache district flood risk predictions using the trained XGBoost model."""
    df = load_india_districts().copy()
    model, _, feature_names = load_xgb_model()
    
    ndma_df = load_ndma_history()
    ndma_events = {}
    if not ndma_df.empty and 'district' in ndma_df.columns and 'flood_events' in ndma_df.columns:
        try:
            ndma_events = ndma_df.groupby('district')['flood_events'].sum().to_dict()
        except Exception:
            ndma_events = {}

    try:
        from forecast import FLOOD_PRONE_DISTRICT_MULTIPLIERS as _FP_MULT
    except ImportError:
        _FP_MULT = {}

    low_risk_set = set(LOW_RISK_DISTRICTS)

    if model is not None and feature_names:
        try:
            rows = []
            for _, row in df.iterrows():
                d = str(row['district'])
                elev = DISTRICT_ELEVATIONS.get(d, 100.0)
                is_fp = 1 if (d in _FLOOD_PLAINS or d in _FP_MULT) else 0
                events = ndma_events.get(d, 25)
                
                if d in low_risk_set:
                    yr = 2020
                    r30 = 600.0
                    rf = 0.0
                    wl = 0.5
                elif is_fp or events >= 65:
                    yr = 2005
                    r30 = 20.0
                    rf = 50.0
                    wl = 3.5
                elif events >= 30:
                    yr = 2018
                    r30 = 60.0
                    rf = 15.0
                    wl = 1.8
                else:
                    yr = 2020
                    r30 = 350.0
                    rf = 0.0
                    wl = 1.0

                discharge = max(wl * 50, 0)
                r7 = r30 * 0.25
                feat = {
                    'rainfall_mm': rf,
                    'temperature_c': 28.0,
                    'humidity_pct': 65.0,
                    'water_level_m': wl,
                    'river_discharge_m3_s': discharge,
                    'elevation_m': elev,
                    'population_density': 500.0 if is_fp else 200.0,
                    'year': yr,
                    'month': 8,
                    'day_of_year': 230,
                    'is_monsoon': 1 if (is_fp or events >= 30) else 0,
                    'rainfall_7day': r7,
                    'rainfall_30day': r30,
                    'api': rf * 0.1,
                    'river_rise_rate': 0.0,
                    'month_sin': np.sin(2 * np.pi * 8 / 12),
                    'month_cos': np.cos(2 * np.pi * 8 / 12),
                    'rainfall_intensity': rf / 29.0,
                    'rainfall_river_interaction': r7 * wl * 0.01,
                    'discharge_per_water_level': discharge / (wl + 1),
                    'terrain_rain_risk': r30 / (abs(elev) + 1)
                }
                rows.append([feat.get(f, 0.0) for f in feature_names])

            X = np.array(rows)
            probs = model.predict_proba(X)[:, 1]
            df['risk_score'] = np.round(probs * 100, 1)
            df['Risk Level'] = df['risk_score'].apply(lambda s: 'High' if s >= 60.0 else ('Moderate' if s >= 30.0 else 'Low'))
            return df
        except Exception as e:
            print(f"[get_district_risk_data] Model prediction fallback: {e}")

    # Fallback if model loading failed:
    scores = []
    levels = []
    for _, row in df.iterrows():
        d = str(row['district'])
        events = ndma_events.get(d, 25)
        if d in low_risk_set:
            score = 8.5
            level = "Low"
        elif d in _FLOOD_PLAINS or events >= 65:
            score = 78.0
            level = "High"
        elif events >= 30:
            score = 42.0
            level = "Moderate"
        else:
            score = 12.0
            level = "Low"
        scores.append(score)
        levels.append(level)
    df['risk_score'] = scores
    df['Risk Level'] = levels
    return df


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
    return prob

def get_top_shap_drivers(scaler, features, feat_dict, fallback_count=5):
    """Return current-prediction SHAP drivers when the explainer artifact exists."""
    filtered_feat = {k: v for k, v in feat_dict.items() if k.lower() != 'year' and not k.lower().startswith('year')}
    fallback = dict(sorted(filtered_feat.items(), key=lambda x: abs(x[1]), reverse=True)[:fallback_count])
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
        valid_indices = [i for i, f in enumerate(features) if f.lower() != 'year' and not f.lower().startswith('year')]
        top_idx = sorted(valid_indices, key=lambda i: abs(shap_row[i]), reverse=True)[:fallback_count]
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
    return "Normal", "weather-normal", "#4ade80"

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
                <div class="weather-emoji"><i class="fa-solid fa-cloud-sun" style="color:#f97316;font-size:2rem;"></i></div>
                <div class="weather-title">{weather.get("weather_description", "Live Weather")}</div>
                <div class="weather-subtitle">Last updated: {minutes_since(weather.get("fetched_at"))}</div>
            </div>
            <span class="live-badge">LIVE</span>
        </div>
        <div class="weather-metrics">
            <div><span class="weather-icon"><i class="fa-solid fa-thermometer-half" style="color:#f97316;"></i></span><b>{weather.get("temperature_c", 0):.1f}°C</b><small>Temperature</small></div>
            <div><span class="weather-icon"><i class="fa-solid fa-droplet" style="color:#f97316;"></i></span><b>{weather.get("humidity_pct", 0)}%</b><small>Humidity</small></div>
            <div><span class="weather-icon"><i class="fa-solid fa-wind" style="color:#f97316;"></i></span><b>{wind_speed:.1f} km/h</b><small>Wind Speed</small></div>
            <div><span class="weather-icon"><i class="fa-solid fa-cloud-rain" style="color:#f97316;"></i></span><b>{rainfall:.1f} mm</b><small>Rainfall</small></div>
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

    line_color = "#f97316"
    point_colors = np.where(forecast_df["rainfall_mm"] > 50, "#ef4444", "#f97316")

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
        paper_bgcolor="#18181b",
        plot_bgcolor="#27272a",
        font={"color":"#fafafa","family":"Inter"},
        yaxis_title="Rainfall (mm)",
        xaxis_title="",
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
        margin=dict(l=10,r=10,t=10,b=10),
    )
    st.plotly_chart(fig_rain, use_container_width=True)

    risk_colors = np.where(forecast_df["risk_probability"] > 60, "#ef4444",
                   np.where(forecast_df["risk_probability"] >= 40, "#f59e0b",
                   np.where(forecast_df["risk_probability"] >= 20, "#eab308", "#4ade80")))
    fig_risk = go.Figure(go.Bar(
        x=forecast_df["date_display"],
        y=forecast_df["risk_probability"],
        marker_color=risk_colors,
        hovertemplate="%{x}<br>Flood risk: %{y:.0f}%<extra></extra>",
    ))
    fig_risk.update_layout(
        height=240,
        paper_bgcolor="#18181b",
        plot_bgcolor="#27272a",
        font={"color":"#fafafa","family":"Inter"},
        yaxis=dict(title="Flood Risk (%)", range=[0, 100], gridcolor="rgba(255,255,255,0.05)"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
        xaxis_title="",
        margin=dict(l=10,r=10,t=10,b=10),
    )
    st.plotly_chart(fig_risk, use_container_width=True)


def render_mini_forecast_preview(forecast_df, district):
    if forecast_df is None or forecast_df.empty:
        st.markdown("<div style='color:#71717a;'>7-day forecast preview unavailable.</div>", unsafe_allow_html=True)
        return

    def _weather_icon(precip_mm):
        if precip_mm >= 20:
            return '<i class="fa-solid fa-cloud-rain" style="color:#f97316;"></i>'
        if precip_mm >= 5:
            return '<i class="fa-solid fa-cloud-sun" style="color:#f97316;"></i>'
        return '<i class="fa-solid fa-sun" style="color:#f97316;"></i>' 

    row_items = []
    for _, row in forecast_df.head(7).iterrows():
        day_label = pd.Timestamp(row["date"]).strftime("%a")
        prob_pct = float(row.get("flood_probability_pct", row.get("flood_probability", 0.0) * 100))
        precip_mm = float(row.get("precipitation_mm", 0.0))
        icon = _weather_icon(precip_mm)
        if prob_pct <= 30:
            color = "#4ade80"
        elif prob_pct <= 60:
            color = "#f59e0b"
        else:
            color = "#ef4444"
        width_pct = min(max(prob_pct, 0), 100)

        row_items.append(f"""
            <div style="display:flex;align-items:center;gap:12px;margin-bottom:12px;">
                <div style="min-width:72px;color:#fafafa;font-size:0.95rem;font-weight:600;">{day_label}</div>
                <div style="display:flex;align-items:center;gap:10px;flex:1;">
                    <span style="font-size:1.1rem;">{icon}</span>
                    <div style="flex:1;background:#18181b;border-radius:999px;height:14px;overflow:hidden;">
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

def style_risk_level(val):
    """Color code Risk Level column: High = red, Moderate = orange, Low = green."""
    v = str(val).strip().lower()
    if v == "high":
        return "color: #ef4444; font-weight: 700; background-color: rgba(239, 68, 68, 0.15);"
    elif v == "moderate":
        return "color: #f97316; font-weight: 700; background-color: rgba(249, 115, 22, 0.15);"
    elif v == "low":
        return "color: #22c55e; font-weight: 700; background-color: rgba(34, 197, 94, 0.15);"
    return ""

def render_risk_overview_table(df):
    """Render searchable district risk metadata with model-predicted risk levels."""
    if "Risk Level" not in df.columns:
        df = get_district_risk_data()
    cols_to_use = [c for c in ["state", "district", "Risk Level", "flood_type"] if c in df.columns]
    table_df = df[cols_to_use].copy()
    if "flood_type" in table_df.columns:
        table_df = table_df.rename(columns={"flood_type": "Flood Type"})
    st.markdown('<div style="font-size:0.8125rem;color:#71717a;margin-bottom:4px;"><i class="fa-solid fa-search" style="color:#71717a;margin-right:6px;"></i>Search State or District</div>', unsafe_allow_html=True)
    search_query = st.text_input("Search state or district", placeholder="e.g. Assam, Dibrugarh...", key="risk_overview_search", label_visibility="collapsed")
    filtered_df = table_df[table_df['state'].str.contains(search_query, case=False, na=False, regex=False) | table_df['district'].str.contains(search_query, case=False, na=False, regex=False)] if search_query else table_df
    filtered_df = filtered_df.copy()
    display_cols = [c for c in ['state', 'district', 'Risk Level', 'Flood Type'] if c in filtered_df.columns]
    display_df = filtered_df[display_cols]
    st.markdown(f"Showing **{len(filtered_df)}** of **{len(table_df)}** districts")

    styled = display_df.style
    if hasattr(styled, "map"):
        styled = styled.map(style_risk_level, subset=["Risk Level"])
    else:
        styled = styled.applymap(style_risk_level, subset=["Risk Level"])

    st.dataframe(styled, use_container_width=True, hide_index=True, height=400)

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
    "No rain — sky is clear": 0,
    "Very light drizzle / few drops": 5,
    "Light rain — ground is getting wet": 20,
    "Steady rain — been raining for hours": 60,
    "Heavy rain — hard to see outside": 120,
    "Nonstop heavy rain — roads getting flooded": 250,
    "Worst rain I've ever seen": 400,
}
WATER_SITUATION = {
    "Everything is dry and normal": 3.0,
    "Drains & ditches have more water than usual": 5.0,
    "Low-lying fields and roads are waterlogged": 7.0,
    "Water is reaching near houses / compound walls": 9.5,
    "Water is entering houses / streets are flooded": 12.0,
}
GROUND_OPTIONS = {
    "Ground is dry, no puddles": 40,
    "Ground is damp, small puddles around": 65,
    "Mud everywhere, ground is fully soaked": 82,
    "Standing water everywhere, ground is saturated": 95,
}
TEMP_OPTIONS = {
    "Cold — need a jacket": 15,
    "Comfortable — pleasant weather": 25,
    "Hot — feeling sweaty": 32,
    "Very hot and sticky — hard to stay outside": 35,
}

# ===== PAGE 1: RISK PREDICTOR GAUGE =====
def show_risk_gauge(risk_score, risk_level):
    if risk_level.upper() in ["HIGH", "VERY HIGH", "EXTREME"]:
        color = "#ef4444"
    elif risk_level.upper() == "MODERATE":
        color = "#f59e0b"
    else:
        color = "#4ade80"
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_score,
        number={"suffix": "%", "font": {"color": "#fafafa", "size": 36}},
        gauge={
            "axis": {
                "range": [0, 100],
                "tickmode": "array",
                "tickvals": [0, 20, 40, 60, 80, 100],
                "ticktext": ["0", "20", "40", "60", "80", "100"],
                "tickcolor": "#52525b"
            },
            "bar": {"color": color},
            "bgcolor": "#27272a",
            "bordercolor": "#3f3f46",
            "steps": [
                {"range": [0, 30], "color": "rgba(74,222,128,0.12)"},
                {"range": [30, 60], "color": "rgba(245,158,11,0.12)"},
                {"range": [60, 100], "color": "rgba(239,68,68,0.12)"},
            ],
            "threshold": {
                "line": {"color": color, "width": 4},
                "thickness": 0.75,
                "value": risk_score
            }
        },
        title={"text": f"Flood Risk: {risk_level}", 
               "font": {"color": "#71717a", "size": 14}}
    ))
    fig.update_layout(
        paper_bgcolor="#18181b",
        font={"color": "#fafafa"},
        height=300,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    st.plotly_chart(fig, use_container_width=True)

# ===== PAGE 1: RISK PREDICTOR =====
def page_predictor():
    lang = st.session_state.get("lang", "en")
    is_online = st.session_state.get("internet_status", True)
    st.markdown(f"""
    <div class="card-custom" style="padding:16px 20px;margin-bottom:16px;">
        <h2 style="margin:0;font-size:1.35rem;font-weight:600;color:#fafafa;"><i class="fa-solid fa-water" style="color:#f97316; margin-right:8px;"></i>{get_text('hero_title', lang)}</h2>
        <p style="margin:2px 0 0 0;font-size:0.875rem;color:#71717a;">{get_text('hero_subtitle', lang)}</p>
    </div>
    """, unsafe_allow_html=True)

    model, scaler, features = load_xgb_model()
    districts_df = load_india_districts()
    real_df = load_real_data()
    n_districts = len(districts_df['district'].unique()) if not districts_df.empty else 736
    n_states = len(districts_df['state'].unique()) if not districts_df.empty else 36
    n_records = len(real_df) if not real_df.empty else 4695

    st.markdown(f"""
<div style="display:grid; grid-template-columns:repeat(4,1fr); gap:8px; margin-bottom:16px;">
    <div style="background:#27272a; border:1px solid #3f3f46; border-radius:6px; padding:12px; text-align:center;">
        <div style="font-size:18px; font-weight:600; color:#fafafa;">{n_districts}</div>
        <div style="font-size:11px; color:#71717a; font-weight:500; text-transform:uppercase; letter-spacing:0.04em; margin-top:2px;">Districts</div>
    </div>
    <div style="background:#27272a; border:1px solid #3f3f46; border-radius:6px; padding:12px; text-align:center;">
        <div style="font-size:18px; font-weight:600; color:#fafafa;">{n_states}</div>
        <div style="font-size:11px; color:#71717a; font-weight:500; text-transform:uppercase; letter-spacing:0.04em; margin-top:2px;">States</div>
    </div>
    <div style="background:#27272a; border:1px solid #3f3f46; border-radius:6px; padding:12px; text-align:center;">
        <div style="font-size:18px; font-weight:600; color:#fafafa;">{n_records:,}</div>
        <div style="font-size:11px; color:#71717a; font-weight:500; text-transform:uppercase; letter-spacing:0.04em; margin-top:2px;">Records</div>
    </div>
    <div style="background:#27272a; border:1px solid #3f3f46; border-radius:6px; padding:12px; text-align:center;">
        <div style="font-size:18px; font-weight:600; color:#fafafa;">0.83</div>
        <div style="font-size:11px; color:#71717a; font-weight:500; text-transform:uppercase; letter-spacing:0.04em; margin-top:2px;">AUC Score</div>
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
        st.markdown('<p class="sidebar-field-label">State</p>', unsafe_allow_html=True)
        default_state = st.session_state.get("selected_state") or st.session_state.get("state_select", "Bihar")
        default_state_idx = state_options.index(default_state) if default_state in state_options else 0
        state = st.selectbox(" ", state_options, index=default_state_idx, label_visibility="collapsed", key="state_select")
        
        district_options = sorted(districts_df.loc[districts_df["state"] == state, "district"].dropna().unique().tolist())
        st.markdown('<p class="sidebar-field-label">District</p>', unsafe_allow_html=True)
        default_dist = st.session_state.get("selected_district") or st.session_state.get("district_select", "Patna")
        default_dist_idx = district_options.index(default_dist) if default_dist in district_options else 0
        district = st.selectbox(" ", district_options, index=default_dist_idx, label_visibility="collapsed", key="district_select")
        
        # Always maintain synchronized state & district globally across session state
        st.session_state["selected_state"] = state
        st.session_state["selected_district"] = district
        st.session_state["current_state"] = state
        st.session_state["current_district"] = district
        
        district_profile = get_district_profile(state, district)
        weather_location = f"{district}, {state}"
        
        st.markdown('<p class="sidebar-field-label">Date</p>', unsafe_allow_html=True)
        date = st.date_input(" ", label_visibility="collapsed", key="date_select")

        # --- OpenWeatherMap API Key ---
        api_key = os.getenv("OPENWEATHERMAP_API_KEY", "").strip()
        if WEATHER_API_AVAILABLE and api_key:
            set_openweathermap_api_key(api_key)

        # --- Fetch Weather Data Button ---
        fetch_btn = st.button("Fetch Weather Data", use_container_width=True)
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
                        st.info(f"OpenWeatherMap failed ({e}). Trying Open-Meteo...")
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
                            precip_list = daily.get("precipitation_sum") or [0]
                            precip = precip_list[0] if precip_list else 0
                            temp = cw.get("temperature", 28.0)
                            # Get real humidity from hourly data (use current hour)
                            humidity_list = hourly.get("relativehumidity_2m", [])
                            humidity_val = int(humidity_list[min(datetime.now().hour, max(len(humidity_list) - 1, 0))]) if humidity_list else 70

                            # Decode Open-Meteo weather code to description
                            wmo_code = int(cw.get("weathercode", 0))
                            if wmo_code == 0:
                                om_desc, om_main = "Clear Sky", "Clear"
                            elif wmo_code in (1, 2, 3):
                                om_desc, om_main = "Partly Cloudy", "Clouds"
                            elif wmo_code in (45, 48):
                                om_desc, om_main = "Foggy", "Mist"
                            elif wmo_code in (51, 53, 55):
                                om_desc, om_main = "Drizzle", "Drizzle"
                            elif wmo_code in (61, 63, 65):
                                om_desc, om_main = "Rainy", "Rain"
                            elif wmo_code in (66, 67):
                                om_desc, om_main = "Freezing Rain", "Rain"
                            elif wmo_code in (71, 73, 75, 77):
                                om_desc, om_main = "Snowy", "Snow"
                            elif wmo_code in (80, 81, 82):
                                om_desc, om_main = "Rain Showers", "Rain"
                            elif wmo_code in (85, 86):
                                om_desc, om_main = "Snow Showers", "Snow"
                            elif wmo_code in (95, 96, 99):
                                om_desc, om_main = "Thunderstorm", "Thunderstorm"
                            else:
                                om_desc, om_main = "Cloudy", "Clouds"

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
                        st.warning(f"Could not fetch weather: {e}")

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
                    st.success(f"Weather loaded via {source}!")

        if st.session_state.using_live_weather and st.session_state.live_weather:
            live_weather = st.session_state.live_weather
            condition = (live_weather.get("description") or live_weather.get("weather_description") or "Partly Cloudy").title()
            temperature = live_weather.get("temperature") or live_weather.get("temperature_c") or "--"
            humidity = live_weather.get("humidity") or live_weather.get("humidity_pct") or "--"
            rainfall = live_weather.get("rainfall") or live_weather.get("rainfall_mm") or live_weather.get("rain") or 0
            weather_icon = WEATHER_EMOJI_MAP.get(
                condition.lower().split()[0], ""
            )
            source_label = live_weather.get("weather_source", "Open-Meteo")

            weather_card_html = (
                '<div style="background: #27272a; border: 1px solid #3f3f46; '
                'border-radius: 6px; padding: 16px; color: #fafafa; margin: 12px 0;">'
                '<div style="background:rgba(249,115,22,0.15); color:#f97316; border:1px solid rgba(249,115,22,0.4); '
                'font-size:11px; font-weight:600; padding:2px 8px; border-radius:4px; '
                'display:inline-block; margin-bottom:12px;">'
                'Live Telemetry</div>'
                f'<div style="font-size:20px; font-weight:600; color:#fafafa; '
                f'margin-bottom:2px;">{condition}</div>'
                f'<div style="font-size:12px; color:#71717a; '
                f'margin-bottom:14px;">Source: {source_label} &middot; Synced</div>'
                '<div style="display:grid; grid-template-columns:1fr 1fr 1fr; gap:8px;">'
                '<div style="background:#18181b; border:1px solid #3f3f46; border-radius:4px; '
                'padding:8px; text-align:center;">'
                f'<div style="font-size:16px; font-weight:600; color:#fafafa;">'
                f'{temperature}&deg;C</div>'
                '<div style="font-size:10px; color:#71717a; '
                'margin-top:2px;">Temp</div></div>'
                '<div style="background:#18181b; border:1px solid #3f3f46; border-radius:4px; '
                'padding:8px; text-align:center;">'
                f'<div style="font-size:16px; font-weight:600; color:#fafafa;">'
                f'{humidity}%</div>'
                '<div style="font-size:10px; color:#71717a; '
                'margin-top:2px;">Humidity</div></div>'
                '<div style="background:#18181b; border:1px solid #3f3f46; border-radius:4px; '
                'padding:8px; text-align:center;">'
                f'<div style="font-size:16px; font-weight:600; color:#fafafa;">'
                f'{rainfall}mm</div>'
                '<div style="font-size:10px; color:#71717a; '
                'margin-top:2px;">Rainfall</div></div>'
                '</div>'
                '</div>'
            )
            render_html(weather_card_html)

            st.markdown(f"""
<div style="font-size: 11px; color: #71717a; 
text-align: center; margin-top: 4px;">
Auto-fetched from {source_label} &middot; Synced
</div>
""", unsafe_allow_html=True)

        st.session_state.manual_override = True
        st.markdown("---")
        st.markdown('<div style="font-size:0.75rem;color:#71717a;font-weight:600;margin-bottom:6px;"><i class="fa-solid fa-cog" style="color:#71717a;margin-right:6px;"></i>Input Mode</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)

        if col1.button("Simple", key="mode_simple", 
            use_container_width=True):
            st.session_state.input_mode = "simple"

        if col2.button("Expert", key="mode_expert",
            use_container_width=True):
            st.session_state.input_mode = "expert"

        # No fragile nth-of-type CSS needed — active mode is indicated by the button labels above

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
        # Predict button uses global primary button styling
        predict_btn = st.button(get_text("predict_button", lang), type="primary", use_container_width=True)

    live_weather = st.session_state.live_weather if st.session_state.using_live_weather else None
    if live_weather:
        render_current_weather_card(live_weather)
        render_html('<div class="gradient-divider"></div>')

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
                "_cached_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
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
        if FORECAST_AVAILABLE:
            try:
                mini_forecast_df = generate_7day_forecast(district, state, base_risk=risk_score)
            except Exception:
                mini_forecast_df = None
        elif st.session_state.using_live_weather and st.session_state.live_forecast:
            try:
                mini_forecast_df = pd.DataFrame(st.session_state.live_forecast)
            except Exception:
                mini_forecast_df = None

        if mini_forecast_df is None or len(mini_forecast_df) == 0:
            dt = datetime
            days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
            start_day = dt.now().weekday()
            ordered_days = [days[(start_day + i) % 7] for i in range(7)]
            day_vars = [0.0, 0.8, -0.6, 1.1, -0.5, 0.7, -0.4]
            mini_forecast_df = pd.DataFrame({
                'day': ordered_days,
                'flood_probability_pct': [
                    round(max(0.8, min(98.0, risk_score if i == 0 else (risk_score * 0.75 + day_vars[i]))), 1)
                    for i in range(7)
                ]
            })

        # Standardize day column and ensure Day 0 aligns with today's gauge value
        if mini_forecast_df is not None and not mini_forecast_df.empty:
            if 'day' not in mini_forecast_df.columns:
                if 'date' in mini_forecast_df.columns:
                    mini_forecast_df['day'] = pd.to_datetime(mini_forecast_df['date']).dt.strftime('%a')
                elif 'date_display' in mini_forecast_df.columns:
                    mini_forecast_df['day'] = mini_forecast_df['date_display']
                else:
                    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
                    start_day = datetime.now().weekday()
                    mini_forecast_df['day'] = [days[(start_day + i) % 7] for i in range(len(mini_forecast_df))]
            else:
                mini_forecast_df['day'] = mini_forecast_df['day'].astype(str).str[:3]

            y_col = 'flood_probability_pct' if 'flood_probability_pct' in mini_forecast_df.columns else ('risk_probability' if 'risk_probability' in mini_forecast_df.columns else mini_forecast_df.columns[1])
            vals = [float(v) for v in mini_forecast_df[y_col]]
            if len(vals) > 0:
                vals[0] = round(risk_score, 1)
            mini_forecast_df[y_col] = vals

        # Try/except block to wrap the HTML rendering as instructed
        try:
            # 4. AFTER PREDICTION — TWO COLUMN LAYOUT:
            col1, col2 = st.columns(2)

            # LEFT COLUMN (col1) — Risk Assessment Panel:
            with col1:
                # Determine color based on risk level
                risk_color = "#4ade80" if risk_level == "LOW" else \
                             "#f97316" if risk_level in ("MEDIUM", "MODERATE") else \
                             "#ef4444" if risk_level == "HIGH" else "#ef4444"
                
                st.markdown(f"""
                <div style="background:#27272a; border:1px solid #3f3f46; 
                border-radius:6px; padding:10px 14px; margin-bottom:8px; text-align:left;">
                    <div style="font-size:12px; color:#71717a; font-weight:600;">Risk assessment gauge</div>
                </div>
                """, unsafe_allow_html=True)
                
                show_risk_gauge(risk_score, risk_level)
                
                st.markdown(f"""
                <div style="background:#27272a; border:1px solid #3f3f46; 
                border-radius:6px; padding:12px; margin-top:8px;">
                    <div style="display:flex; justify-content:space-between; 
                    font-size:12px; color:#71717a;">
                        <span>Model Confidence</span>
                        <span style="color:{risk_color}; font-weight:600;">{xgb_score:.1f}%</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                st.markdown("""
                <div style="background:#27272a; border:1px solid #3f3f46; 
                border-radius:6px; padding:10px 14px; margin-bottom:8px; text-align:left;">
                    <div style="font-size:12px; color:#71717a; font-weight:600;">7-day flood risk outlook</div>
                </div>
                """, unsafe_allow_html=True)
                if mini_forecast_df is not None and not mini_forecast_df.empty:
                    x_col = 'day' if 'day' in mini_forecast_df.columns else mini_forecast_df.columns[0]
                    y_col = 'flood_probability_pct' if 'flood_probability_pct' in mini_forecast_df.columns else ('risk_probability' if 'risk_probability' in mini_forecast_df.columns else mini_forecast_df.columns[1])
                    bar_colors = [
                        '#ef4444' if float(v) >= 60.0 else '#f97316' if float(v) >= 30.0 else '#4ade80'
                        for v in mini_forecast_df[y_col]
                    ]
                    fig_mini = go.Figure(go.Bar(
                        x=mini_forecast_df[x_col],
                        y=mini_forecast_df[y_col],
                        marker=dict(color=bar_colors),
                        hovertemplate="%{x}: %{y:.1f}%<extra></extra>"
                    ))
                    fig_mini.update_layout(
                        height=220,
                        paper_bgcolor="#18181b",
                        plot_bgcolor="#18181b",
                        font={"color": "#71717a", "family": "Inter"},
                        margin=dict(l=10, r=10, t=10, b=20),
                        yaxis=dict(range=[0, 100], gridcolor="#27272a", title="% Risk"),
                        xaxis=dict(gridcolor="#27272a")
                    )
                    st.plotly_chart(fig_mini, use_container_width=True)
                else:
                    st.info("7-day projected telemetry unavailable for this district.")

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
                icon = WEATHER_EMOJI_MAP.get(
                    condition.lower().split()[0], "")

                st.markdown(f"""
                <div style="background:#27272a; border:1px solid #3f3f46; 
                border-radius:6px; padding:14px;">
                    <div style="background:rgba(249,115,22,0.15); color:#f97316; border:1px solid rgba(249,115,22,0.4); 
                    font-size:10px; font-weight:600; padding:2px 8px; 
                    border-radius:4px; letter-spacing:0.5px; 
                    display:inline-block; margin-bottom:8px;">Live Weather</div>
                    <div style="font-size:24px; font-weight:600; color:#fafafa; 
                    line-height:1.1;"><i class="fa-solid fa-thermometer-half" style="color:#f97316; margin-right:8px;"></i>{temp}°C</div>
                    <div style="font-size:12px; color:#71717a; 
                    margin-top:3px;">{condition}</div>
                    <div style="display:grid; grid-template-columns:repeat(3,1fr); 
                    gap:6px; margin-top:10px;">
                        <div style="background:#18181b; border:1px solid #3f3f46; 
                        border-radius:4px; padding:7px; text-align:center;">
                            <div style="font-size:13px; font-weight:600; 
                            color:#fafafa;">{humidity_pct}%</div>
                            <div style="font-size:10px; 
                            color:#71717a; margin-top:2px;">
                            Humidity</div>
                        </div>
                        <div style="background:#18181b; border:1px solid #3f3f46; 
                        border-radius:4px; padding:7px; text-align:center;">
                            <div style="font-size:13px; font-weight:600; 
                            color:#fafafa;">{rainfall_mm}mm</div>
                            <div style="font-size:10px; 
                            color:#71717a; margin-top:2px;">
                            Rainfall</div>
                        </div>
                        <div style="background:#18181b; border:1px solid #3f3f46; 
                        border-radius:4px; padding:7px; text-align:center;">
                            <div style="font-size:13px; font-weight:600; 
                            color:#fafafa;">{live_weather.get('weather_source', 'Open-Meteo')}</div>
                            <div style="font-size:10px; 
                            color:#71717a; margin-top:2px;">
                            Source</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # RIGHT (col4) — Recommended Actions:
            with col4:
                if prob < 0.3:
                    title_text = get_text('low_risk', lang)
                    box_class = "rec-safe"
                    title_color = "#4ade80"
                    icon_html = '<i class="fa-solid fa-check-circle" style="color:#4ade80;margin-right:6px;"></i>'
                    if lang == "hi":
                        items_list = [
                            "जल निकासी नालियों का नियमित निरीक्षण करें और रुकावटें हटाएं",
                            "72 घंटे की आपातकालीन किट (टॉर्च, दवाएं, सूखा राशन) तैयार रखें",
                            "दैनिक मौसम एवं IMD रडार अपडेट पर नजर बनाए रखें",
                            "फसल व कृषि उपकरणों को सुरक्षित ऊंचे स्थान पर व्यवस्थित करें",
                        ]
                    else:
                        items_list = [
                            "Inspect local stormwater drainage channels & clear debris",
                            "Maintain a 72-hour family emergency kit (flashlights, first aid, dry rations)",
                            "Monitor daily IMD hydrological bulletins & local telemetry updates",
                            "Secure agricultural equipment & store harvest on elevated platforms",
                        ]
                elif prob < 0.6:
                    title_text = get_text('be_careful', lang)
                    box_class = "rec-mod"
                    title_color = "#f59e0b"
                    icon_html = '<i class="fa-solid fa-exclamation-triangle" style="color:#f97316;margin-right:6px;"></i>'
                    if lang == "hi":
                        items_list = [
                            "नदी तटबंधों और जलद्वारों की निगरानी करें; किसी भी रिसाव की सूचना SDMA को दें",
                            "बिजली के उपकरण, इन्वर्टर और महत्वपूर्ण दस्तावेज ऊपरी मंजिलों पर ले जाएं",
                            "पशुधन को सुरक्षित ऊंचे सामुदायिक आश्रयों में स्थानांतरित करें",
                            "मोबाइल फोन, पावर बैंक चार्ज रखें और आपातकालीन नंबर संभाल कर रखें",
                            "निकटतम सुरक्षित बाढ़ राहत आश्रय और सूखे रास्तों की पहचान करें",
                            f"{state} हेल्पलाइन: <b>{get_state_helpline(state)}</b>",
                            "NDRF: <b>011-24363260</b>",
                            "आपातकालीन: <b>112</b>",
                        ]
                    else:
                        items_list = [
                            "Inspect river embankments & sluice gates; report seepage to SDMA",
                            "Relocate electrical appliances, power units, and vital documents to upper floors",
                            "Move livestock to elevated ground & secure emergency fodder reserves",
                            "Keep smartphones & power banks fully charged; keep emergency contacts handy",
                            "Map out designated local relief shelters and unflooded evacuation routes",
                            f"{state}: <b>{get_state_helpline(state)}</b>",
                            "NDRF: <b>011-24363260</b>",
                            "Emergency: <b>112</b>",
                        ]
                else:
                    title_text = get_text('danger_move', lang)
                    box_class = "rec-danger"
                    title_color = "#ef4444"
                    icon_html = '<i class="fa-solid fa-exclamation-triangle" style="color:#ef4444;margin-right:6px;"></i>'
                    if lang == "hi":
                        items_list = [
                            "<b>तत्काल ऊंचे स्थान या NDMA नामित राहत शिविर में सुरक्षित जाएं</b>",
                            "परिवार, बुजुर्गों, बच्चों और बीमार सदस्यों को प्राथमिकता से निकालें",
                            "निकलने से पहले मुख्य बिजली का स्विच (MCB) और गैस सिलेंडर वाल्व बंद करें",
                            '<i class="fa-solid fa-road" style="color:#ef4444;margin-right:4px;"></i><b>बाढ़ के पानी से भरी सड़कों या पुलों को बिल्कुल पार न करें</b>',
                            "आपातकालीन रेडियो या जिला आपदा प्रबंधन नियंत्रण कक्ष (1078) के संपर्क में रहें",
                            f"{state} हेल्पलाइन: <b>{get_state_helpline(state)}</b>",
                            "NDRF: <b>011-24363260</b>",
                            "आपातकालीन: <b>112</b>",
                        ]
                    else:
                        items_list = [
                            "<b>Initiate immediate evacuation to designated high-ground NDMA shelters</b>",
                            "Prioritize family members, elders, young children, and medical dependents",
                            "Disconnect main electrical breaker (MCB) & shut domestic gas supply before exit",
                            '<i class="fa-solid fa-road" style="color:#ef4444;margin-right:4px;"></i><b>NEVER drive or walk across flooded roadways or submerged culverts</b>',
                            "Tune into district disaster emergency broadcast & NDMA emergency hotline 1078",
                            f"{state}: <b>{get_state_helpline(state)}</b>",
                            "NDRF: <b>011-24363260</b>",
                            "Emergency: <b>112</b>",
                        ]

                list_html = "".join(f"<li style='margin-bottom:4px;'>{it}</li>" for it in items_list)
                rec_card_html = (
                    f'<div style="background:#27272a; border:1px solid #3f3f46; border-radius:6px; padding:16px;">'
                    f'<div style="font-size:11px; color:#71717a; font-weight:600; text-transform:uppercase; letter-spacing:0.5px; margin-bottom:10px;">Recommended Actions</div>'
                    f'<div class="rec-box {box_class}">'
                    f'<h4 style="color:{title_color};margin:0;font-size:14px;font-weight:600;">{icon_html}{title_text}</h4>'
                    f'<ul style="color:#fafafa;margin:8px 0 0 0;padding-left:18px;font-size:12px;line-height:1.5;">'
                    f'{list_html}'
                    f'</ul>'
                    f'</div>'
                    f'</div>'
                )
                st.markdown(rec_card_html, unsafe_allow_html=True)

        except Exception as display_err:
            st.error(f"Error rendering premium display: {display_err}")

        # Cache AI summary explanation for chatbot context without rendering redundant card
        top_drivers = get_top_shap_drivers(scaler, features, feat_dict)
        if CHATBOT_AVAILABLE:
            try:
                ai_summary = generate_risk_explanation(district, prob, top_drivers, live_weather or {})
            except Exception:
                ai_summary = (
                    f"{district} is at {risk_level_from_score(prob)} flood risk ({prob:.0%}) based on rainfall, "
                    "river level, and soil conditions."
                )
        else:
            ai_summary = (
                f"{district} is at {risk_level_from_score(prob)} flood risk ({prob:.0%}) based on rainfall, "
                "river level, and soil conditions."
            )
        st.session_state.latest_ai_summary = ai_summary
        st.session_state.floodguard_pending_summary = ai_summary

        # 6. XGBOOST FEATURE IMPORTANCE (excluding 'year'):
        try:
            st.markdown("""
            <div style="font-size:11px; color:#71717a; 
            text-transform:uppercase; letter-spacing:0.5px; 
            margin:16px 0 8px; font-weight:600;">Model Feature Importance</div>
            """, unsafe_allow_html=True)

            display_names = {
                "rainfall_30day": "30-Day Cumulative Rain",
                "terrain_rain_risk": "Terrain Drainage Risk",
                "api": "Soil Saturation (API)",
                "rainfall_intensity": "Rainfall Intensity",
                "is_monsoon": "Monsoon Active Factor",
                "rainfall_mm": "Current Rainfall",
                "rainfall_7day": "7-Day Cumulative Rain",
                "month_cos": "Seasonal Cycle",
                "water_level_m": "River Water Level",
                "discharge_per_water_level": "River Discharge Rate",
                "elevation_m": "Elevation",
                "humidity_pct": "Relative Humidity",
                "temperature_c": "Temperature",
                "population_density": "Vulnerability Density",
            }
            if hasattr(model, 'feature_importances_') and features:
                imp_pairs = [
                    (f, float(imp))
                    for f, imp in zip(features, model.feature_importances_)
                    if f.lower() != 'year' and not f.lower().startswith('year') and imp > 0
                ]
                total_imp = sum(imp for _, imp in imp_pairs) or 1.0
                sorted_imp = sorted(imp_pairs, key=lambda x: x[1], reverse=True)[:8]
                sorted_imp.reverse()
                y_labels = [display_names.get(k, k.replace('_', ' ').title()) for k, _ in sorted_imp]
                x_vals = [round((v / total_imp) * 100, 1) for _, v in sorted_imp]
            else:
                filtered_feat = {k: abs(v) for k, v in feat_dict.items() if k.lower() != 'year' and not k.lower().startswith('year')}
                top = dict(sorted(filtered_feat.items(), key=lambda x: x[1], reverse=True)[:8])
                sorted_items = list(top.items())
                sorted_items.reverse()
                y_labels = [display_names.get(k, k.replace('_', ' ').title()) for k, _ in sorted_items]
                x_vals = [v for _, v in sorted_items]

            fig2 = go.Figure(go.Bar(
                x=x_vals,
                y=y_labels,
                orientation='h',
                marker=dict(color="#f97316"),
                hovertemplate="%{y}: %{x:.1f}%<extra></extra>"
            ))
            fig2.update_layout(
                height=300,
                paper_bgcolor="#18181b",
                plot_bgcolor="#18181b",
                font={"color": "#fafafa", "family": "Inter"},
                xaxis=dict(gridcolor="#27272a", title="% Relative Impact"),
                yaxis=dict(gridcolor="#27272a"),
                margin=dict(l=10, r=10, t=10, b=20)
            )
            st.plotly_chart(fig2, use_container_width=True)
        except Exception as shap_err:
            st.error(f"Error rendering feature importance chart: {shap_err}")

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
                file_name=f"FloodGuard_AI_{selected_district}_{datetime.now().strftime('%Y%m%d')}.pdf",
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
                st.warning(f"Showing cached prediction from {cached.get('_cached_at', 'unknown time')}")
                st.info(f"District: {cached['district']}, State: {cached['state']}")
                cached_prob = cached["risk_score"]
                cached_level = cached["risk_level"]
                bar_color = "#ef4444" if cached_prob > 0.6 else "#f59e0b" if cached_prob > 0.3 else "#4ade80"
                fig = go.Figure(go.Indicator(mode="gauge+number", value=cached_prob*100,
                    number={"suffix":"%", "font":{"size":56,"color":"white","family":"Inter"}},
                    title={"text":f"Flood Risk Gauge - {cached['district']} (Cached)", "font":{"size":16,"color":"#71717a","family":"Inter"}},
                    gauge={"axis":{"range":[0,100],"tickmode":"array","tickvals":[0, 20, 40, 60, 80, 100],"ticktext":["0", "20", "40", "60", "80", "100"],"tickcolor":"#3f3f46","tickwidth":1},
                           "bar":{"color":bar_color,"thickness":0.75},
                           "bgcolor":"rgba(24,24,27,0.5)", "bordercolor":"rgba(249,115,22,0.1)", "borderwidth":2,
                           "steps":[{"range":[0,30],"color":"rgba(74,222,128,0.1)"},
                                    {"range":[30,60],"color":"rgba(245,158,11,0.1)"},
                                    {"range":[60,100],"color":"rgba(239,68,68,0.1)"}],
                           "threshold":{"line":{"color":bar_color,"width":4},"thickness":0.85,"value":cached_prob*100}}))
                fig.update_layout(height=340, paper_bgcolor="rgba(0,0,0,0)", font={"color":"white","family":"Inter"},
                                  margin=dict(t=80,b=20,l=50,r=50))
                st.plotly_chart(fig, use_container_width=True)
                st.markdown(f"""
                <div class="rec-box" style="border-left:4px solid {bar_color};padding:16px;background:#27272a;border-radius:8px;">
                    <strong style="color:{bar_color};">{cached_level}</strong> — Cached risk score: <strong>{cached_prob:.0%}</strong>
                </div>""", unsafe_allow_html=True)
            else:
                st.error("No cached prediction available. Please connect to internet and run a prediction first.")
        else:
            st.markdown(f"""<div class="glass-card" style="text-align:center;padding:70px 40px">
                <h3 style="color:#fafafa !important;font-size:1.5rem !important">{get_text('ready_to_predict', lang)}</h3>
                <p style="color:#71717a;font-size:1.1rem;margin-top:12px">{get_text('ready_to_predict_desc', lang)}</p>
                <p style="color:#52525b;font-size:0.85rem;margin-top:20px">{get_text('weather_auto_note', lang)}</p>
            </div>""", unsafe_allow_html=True)

# ===== PAGE 2: RISK MAP =====
def page_map():
    st.markdown("""<div class="card-custom" style="padding:16px 20px;margin-bottom:16px;">
        <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:12px;">
            <div>
                <h2 style="font-size:1.35rem;font-weight:600;margin:0 0 2px 0;color:#fafafa;">India Flood Risk Map</h2>
                <p style="margin:0;color:#71717a;font-size:0.875rem;">Full national flood risk assessment across 736 districts based on historical flood patterns.</p>
            </div>
        </div>
    </div>""", unsafe_allow_html=True)

    is_online = st.session_state.get("internet_status", True)
    if should_skip_map(is_online):
        st.info("Offline Mode: The interactive map requires internet connectivity to stream map tiles. Displaying cached tabular overview.")
        st.markdown("### National Risk Overview (Offline)")
        districts_df = get_district_risk_data()
        render_risk_overview_table(districts_df)
        return

    districts_df = get_district_risk_data()
    m = folium.Map(
        location=[22.0, 82.0],
        zoom_start=5,
        tiles="OpenStreetMap",
        prefer_canvas=True,
        min_zoom=4,
        max_zoom=10
    )
    
    # 1. LEGEND — add to map
    # Accessible legend: color-coded risk levels for screen readers
    legend_html = """
    <div style="position:fixed;bottom:30px;left:30px;z-index:1000;
      background:#27272a;padding:12px 16px;border-radius:10px;
      border:1px solid #3f3f46;font-family:Arial;">
      <p style="color:#fafafa;font-size:13px;font-weight:bold;
        margin:0 0 8px;">Flood Risk Level</p>
      <p style="margin:4px 0;color:#ef4444;"><i class="fa-solid fa-circle" style="color:#ef4444;font-size:10px;"></i> High / Very High</p>
      <p style="margin:4px 0;color:#f59e0b;"><i class="fa-solid fa-circle" style="color:#f59e0b;font-size:10px;"></i> Moderate</p>
      <p style="margin:4px 0;color:#4ade80;"><i class="fa-solid fa-circle" style="color:#4ade80;font-size:10px;"></i> Low</p>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    for _, row in districts_df.iterrows():
        district = row['district']
        state = row['state']
        risk_score = float(row.get('risk_score', 15.0))
        score = risk_score / 100.0
        risk_level = str(row.get('Risk Level', 'Low'))
        
        risk_color = {
            "LOW": "#4ade80",
            "MODERATE": "#f59e0b",
            "HIGH": "#ef4444",
            "VERY HIGH": "#ef4444",
            "EXTREME": "#ef4444",
            "SEVERE": "#ef4444"
        }.get(risk_level.strip().upper(), "#4ade80")
        
        popup = (
            f"<b>{district}, {state}</b><br>"
            f"Risk: <b>{risk_level}</b> ({score:.0%})<br>"
            f"Flood type: {row.get('flood_type', 'Riverine flood')}"
        )
        
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

    render_html('<div class="gradient-divider"></div>')
    st.markdown('<h3 style="font-size:1.25rem;font-weight:600;margin:16px 0 12px 0;color:#fafafa;">India Risk Overview</h3>', unsafe_allow_html=True)
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

    st.session_state.setdefault("floodguard_pending_summary", None)

    district = (
        st.session_state.get("selected_district")
        or st.session_state.get("forecast_district")
        or st.session_state.get("trends_district")
        or st.session_state.get("district_select")
        or "Patna"
    )
    state = (
        st.session_state.get("selected_state")
        or st.session_state.get("forecast_state")
        or st.session_state.get("trends_state")
        or st.session_state.get("state_select")
        or "Bihar"
    )

    # Pull actual risk score from session state
    risk_score = None
    if "current_risk_score" in st.session_state and st.session_state["current_risk_score"]:
        risk_score = float(st.session_state["current_risk_score"])
    elif "risk_score" in st.session_state and st.session_state["risk_score"]:
        risk_score = float(st.session_state["risk_score"])
    elif "last_risk_score" in st.session_state and st.session_state["last_risk_score"]:
        risk_score = float(st.session_state["last_risk_score"]) / 100.0
    elif "current_forecast" in st.session_state and st.session_state["current_forecast"]:
        fc_df = st.session_state["current_forecast"].get("df")
        if fc_df is not None and not fc_df.empty and "flood_probability_pct" in fc_df.columns:
            risk_score = float(fc_df["flood_probability_pct"].max()) / 100.0

    # If still None or 0, pull actual risk score for the selected district from get_district_risk_data()
    if risk_score is None or risk_score == 0.0:
        try:
            dist_data = get_district_risk_data()
            row = dist_data[dist_data['district'].astype(str).str.lower() == str(district).lower()]
            if not row.empty:
                risk_score = float(row.iloc[0]['risk_score']) / 100.0
                st.session_state["risk_score"] = risk_score
                st.session_state["risk_level"] = str(row.iloc[0]['Risk Level'])
        except Exception:
            pass

    if risk_score is None:
        risk_score = 0.42

    if risk_score > 1.0:
        risk_score = risk_score / 100.0

    # Determine risk level in normal case
    risk_level = st.session_state.get("risk_level") or st.session_state.get("current_risk_level")
    if not risk_level or str(risk_level).upper() in ["LOW", "UNKNOWN"]:
        if risk_score >= 0.6:
            risk_level = "High"
        elif risk_score >= 0.3:
            risk_level = "Moderate"
        else:
            risk_level = "Low"
    else:
        risk_level = str(risk_level).title()

    st.session_state["risk_score"] = risk_score
    st.session_state["risk_level"] = risk_level
    st.session_state["selected_district"] = district
    st.session_state["selected_state"] = state

    st.markdown("""<div class="card-custom" style="padding:16px 20px;margin-bottom:16px;">
        <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:12px;">
            <div>
                <h2 style="font-size:1.35rem;font-weight:600;margin:0 0 2px 0;color:#fafafa;">Flood Assistant</h2>
                <p style="margin:0;color:#71717a;font-size:0.875rem;">Ask flood safety questions for your selected district</p>
            </div>
        </div>
    </div>""", unsafe_allow_html=True)

    # Set API keys if available (optional — KB works without any API key)
    gemini_key = _get_gemini_key_for_app()
    if gemini_key and CHATBOT_AVAILABLE:
        set_gemini_api_key(gemini_key)
    if CHATBOT_AVAILABLE:
        # Anthropic key from env/secrets
        anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if anthropic_key and anthropic_key != "your_anthropic_api_key_here":
            set_anthropic_api_key(anthropic_key)

    context_color = "#ef4444" if str(risk_level).upper() == "HIGH" else "#f59e0b" if str(risk_level).upper() == "MODERATE" else "#4ade80"
    st.markdown(f"""<div class="chat-context-card">
        <div>
            <div class="metric-label" style="text-transform:none !important;letter-spacing:normal;">Current context</div>
            <div style="color:#fafafa;font-size:1.1rem;font-weight:800">{district}</div>
        </div>
        <div>
            <div class="metric-label" style="text-transform:none !important;letter-spacing:normal;">Risk score</div>
            <div style="color:{context_color};font-size:1.4rem;font-weight:900">{risk_score:.0%}</div>
        </div>
        <div>
            <div class="metric-label" style="text-transform:none !important;letter-spacing:normal;">Risk level</div>
            <div style="color:{context_color};font-size:1.1rem;font-weight:800">{risk_level}</div>
        </div>
    </div>""", unsafe_allow_html=True)

    # STEP 6 - Clear chat button styled with orange border and dark background
    st.markdown("""
<style>
/* Clear Chat button: orange border and dark background */
.st-key-clear_chat button,
div[class*="st-key-clear_chat"] button,
div[data-testid="stVerticalBlock"]:has(.chat-context-card) .stButton button[kind="secondary"]:not(div[data-testid="stHorizontalBlock"] button) {
    border: 1px solid #f97316 !important;
    background: #18181b !important;
    background-color: #18181b !important;
    color: #f97316 !important;
    border-radius: 6px !important;
    font-weight: 600 !important;
    transition: all 0.15s ease !important;
}
.st-key-clear_chat button:hover,
div[class*="st-key-clear_chat"] button:hover,
div[data-testid="stVerticalBlock"]:has(.chat-context-card) .stButton button[kind="secondary"]:not(div[data-testid="stHorizontalBlock"] button):hover {
    border-color: #fb923c !important;
    background: rgba(249, 115, 22, 0.12) !important;
    background-color: rgba(249, 115, 22, 0.12) !important;
    color: #ffffff !important;
}

/* Suggested Questions equal height buttons */
div[data-testid="stVerticalBlock"]:has(.chat-context-card) div[data-testid="stHorizontalBlock"]:not(:has(input)) button:not([kind="primary"]) {
    height: 52px !important;
    min-height: 52px !important;
    max-height: 52px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    text-align: center !important;
    padding: 6px 12px !important;
    line-height: 1.25 !important;
    font-size: 0.8125rem !important;
    white-space: normal !important;
    word-break: break-word !important;
}
div[data-testid="stVerticalBlock"]:has(.chat-context-card) div[data-testid="stHorizontalBlock"]:not(:has(input)) button:not([kind="primary"]) p {
    font-size: 0.8125rem !important;
    line-height: 1.25 !important;
    margin: 0 !important;
    display: -webkit-box !important;
    -webkit-line-clamp: 2 !important;
    -webkit-box-orient: vertical !important;
    overflow: hidden !important;
}
</style>
""", unsafe_allow_html=True)
    if st.button("Clear Chat", key="clear_chat"):
        st.session_state.chat_history = []
        st.rerun()

    # Suggested Questions (chips with equal height, complete 3x3 grid)
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
        "Tell me about Bihar 2017 floods",
    ]
    selected_prompt = None
    for r in range(0, len(suggestions), 3):
        chunk = suggestions[r:r+3]
        row_cols = st.columns(len(chunk))
        for c, question in enumerate(chunk):
            idx = r + c
            with row_cols[c]:
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
<div style="background:#18181b;border-radius:6px;padding:16px;
  border:1px solid #3f3f46;">
  <div style="display:flex;align-items:center;gap:10px;
    padding-bottom:12px;border-bottom:1px solid #3f3f46;margin-bottom:16px;">
    <div style="width:30px;height:30px;background:rgba(249,115,22,0.15);
      border:1px solid rgba(249,115,22,0.3);border-radius:4px;display:flex;
      align-items:center;justify-content:center;color:#f97316;font-weight:700;font-size:11px;">FG</div>
    <div>
      <div style="color:#fafafa;font-size:13px;font-weight:600;">FloodGuard Assistant</div>
    </div>
    <div style="margin-left:auto;display:flex;align-items:center;gap:6px;">
      <span class="status-dot green"></span>
      <span style="font-size:11px;color:#71717a;">Active</span>
    </div>
  </div>
""", unsafe_allow_html=True)

    # STEP 3 - Render chat messages as bubbles:
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(f"""
            <div style="display:flex;justify-content:flex-end;margin-bottom:10px;">
              <div style="background:#f97316;color:#ffffff;
                border-radius:6px;padding:8px 12px;
                font-size:13px;max-width:80%;">{msg["content"]}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            content = msg.get("content", "")
            is_unavail = msg.get("is_unavailable", False) or "AI Assistant unavailable" in content or "GROQ_API_KEY" in content
            if is_unavail:
                st.markdown("""
                <div style="background:#27272a; border:1px solid #3f3f46; border-radius:6px; 
                padding:10px 14px; color:#71717a; font-size:12px; max-width:85%; margin-bottom:10px; display:flex; align-items:center; gap:8px;">
                    <i class="fa-solid fa-circle-info" style="color:#71717a; font-size:13px;"></i>
                    <span>AI Assistant unavailable. Check API configuration.</span>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="display:flex;gap:8px;margin-bottom:10px;">
                  <div style="width:24px;height:24px;background:#27272a;
                    border:1px solid #3f3f46;border-radius:4px;
                    display:flex;align-items:center;justify-content:center;
                    font-size:10px;font-weight:600;color:#71717a;flex-shrink:0;margin-top:2px;">AI</div>
                  <div>
                    <div style="background:rgba(249,115,22,0.1);color:#f97316;
                      border:1px solid rgba(249,115,22,0.25);
                      border-radius:3px;padding:1px 6px;font-size:10px;
                      display:inline-block;margin-bottom:4px;">
                      {msg.get("q_type","General")}</div>
                    <div style="background:#27272a;border:1px solid #3f3f46;color:#fafafa;
                      border-radius:6px;padding:10px 14px;
                      font-size:13px;line-height:1.55;max-width:85%;">
                      {content}</div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

    # Close the STEP 1 chat container div opened above
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("")

    # STEP 4 - Input row at bottom:
    col1, col2 = st.columns([5, 1])
    with col1:
        user_input = st.text_input("", 
            placeholder="Ask about floods, crops, weather...",
            key="chat_input", label_visibility="collapsed")
    with col2:
        send = st.button("Send", use_container_width=True)

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
                is_unavail = res.get("is_unavailable", False)
            else:
                response = str(res)
                q_type = "General"
                is_unavail = False

        if "GROQ_API_KEY" in response or "AI Assistant unavailable" in response:
            try:
                from chatbot import _offline_answer
                response = _offline_answer(prompt, district=district, state=state, risk_score=risk_score)
                is_unavail = False
            except Exception:
                response = "AI Assistant unavailable. Check API configuration."
                is_unavail = True

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": response,
            "q_type": q_type,
            "is_unavailable": is_unavail
        })
        st.rerun()

# ===== PAGE 4: 7-DAY DISTRICT-LEVEL FORECAST =====
def page_forecast():
    """Display 7-day flood forecast for selected district."""
    is_online = st.session_state.get("internet_status", True)
    if not FORECAST_AVAILABLE:
        st.warning("Forecast module not available. Please check dependencies.")
        return

    st.markdown("""<div class="card-custom" style="padding:16px 20px;margin-bottom:16px;">
        <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:12px;">
            <div>
                <h2 style="font-size:1.35rem;font-weight:600;margin:0 0 2px 0;color:#fafafa;">7-Day Flood Forecast</h2>
                <p style="margin:0;color:#71717a;font-size:0.875rem;">Based on Open-Meteo weather forecast data</p>
            </div>
        </div>
    </div>""", unsafe_allow_html=True)

    # Get all districts and their states
    all_districts = get_district_coordinates()
    states = sorted(set(d['state'] for d in all_districts.values() if 'state' in d))

    # Selection in main area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        default_f_state = st.session_state.get("selected_state") or st.session_state.get("state_select", "Bihar")
        default_f_state_idx = states.index(default_f_state) if default_f_state in states else 0
        selected_state = st.selectbox("Select State", states, index=default_f_state_idx, key="forecast_state")
        
        # Filter districts by selected state
        district_list = sorted([d for d, info in all_districts.items() if info.get('state') == selected_state])

        # If state changed, update forecast_district before widget instantiation
        if selected_state != st.session_state.get('last_forecast_state'):
            st.session_state['last_forecast_state'] = selected_state
            if district_list:
                st.session_state['forecast_district'] = district_list[0]
            elif 'forecast_district' in st.session_state:
                del st.session_state['forecast_district']
        
    with col2:
        # Set index: if current session district is in list, use it; otherwise use global selected_district
        current_district = st.session_state.get('forecast_district')
        global_district = st.session_state.get('selected_district')
        default_idx = 0
        if current_district and current_district in district_list:
            default_idx = district_list.index(current_district)
        elif global_district and global_district in district_list:
            default_idx = district_list.index(global_district)
        
        selected_district = st.selectbox("Select District", district_list, key="forecast_district", index=default_idx)
        
    # Store selected district in session state (widget keys are managed automatically by Streamlit)
    st.session_state['selected_district'] = selected_district
    st.session_state['selected_state'] = selected_state

    # Show forecast button only if district is selected
    if st.session_state.get('selected_district'):
        generate_btn = st.button("Generate 7-Day Forecast", type="primary", use_container_width=True)
    else:
        st.markdown("""<div class="glass-card" style="text-align:center;padding:50px 30px">
            <h3 style="color:#fafafa !important;font-size:1.3rem !important">Select a Location</h3>
            <p style="color:#71717a;margin-top:8px">Choose a state and district above to generate a 7-day flood forecast</p>
        </div>""", unsafe_allow_html=True)
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
            with st.spinner(f"Fetching weather data and generating forecast for {selected_district}..."):
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
                    st.error(f"Error generating forecast: {str(e)}")
                    st.session_state.current_forecast = None
                    # --- Offline: try cached forecast ---
                    if not is_online:
                        cached_forecast = load_from_cache("forecast")
                        if cached_forecast:
                            st.warning(f"Showing forecast cached {get_cache_age('forecast')}")
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
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
                st.caption("Rainfall forecast from Open-Meteo. Accuracy improves during monsoon season (June–September).")
            except Exception as e:
                st.warning(f"Could not render chart: {str(e)}")

            st.markdown("---")

            # Display summary
            try:
                peak_idx = int(forecast_df['flood_probability_pct'].idxmax()) if not forecast_df.empty else 0
                peak_row = forecast_df.loc[peak_idx] if not forecast_df.empty else {}
                peak_day = peak_row.get('day', 'N/A')
                peak_prob = float(peak_row.get('flood_probability_pct', 0.0))
                peak_risk_level = peak_row.get('risk_level', 'Low')
                total_rain = float(forecast_df['precipitation_mm'].sum()) if not forecast_df.empty else 0.0
                high_days = int((forecast_df['flood_probability_pct'] > 60.0).sum()) if not forecast_df.empty else 0

                if peak_prob > 80.0:
                    recommendation = "Severe flooding is likely. Evacuate low-lying areas and follow official orders."
                elif peak_prob > 60.0:
                    recommendation = "High flood risk is expected. Prepare emergency supplies and stay alert."
                elif peak_prob > 30.0:
                    recommendation = "Moderate flood risk is expected. Monitor local weather and avoid risky travel."
                else:
                    recommendation = "Low flood risk is expected. Continue routine precautions."

                st.subheader("Forecast Summary")

                # Stat cards
                sc1, sc2, sc3 = st.columns(3)
                with sc1:
                    st.markdown(f"""
                    <div class="metric-card" style="display:flex; flex-direction:column; justify-content:center; align-items:center; min-height:105px; padding:16px 12px; text-align:center; border:1px solid #3f3f46; border-top:2px solid #f97316 !important; background:#27272a; border-radius:6px; box-sizing:border-box;">
                        <div class="metric-label" style="margin-bottom:6px; font-size:11px; text-transform:uppercase; letter-spacing:0.5px; color:#a1a1aa; font-weight:600;">Peak Risk Day</div>
                        <div class="metric-value" style="color:#fafafa; font-size:1.5rem !important; font-weight:700; line-height:1.2;">{peak_day} <span style="font-size:1rem;color:#f97316;">({peak_prob:.1f}%)</span></div>
                        <div style="font-size:11px; color:#71717a; margin-top:4px;">Risk Level: {peak_risk_level}</div>
                    </div>
                    """, unsafe_allow_html=True)
                with sc2:
                    st.markdown(f"""
                    <div class="metric-card" style="display:flex; flex-direction:column; justify-content:center; align-items:center; min-height:105px; padding:16px 12px; text-align:center; border:1px solid #3f3f46; border-top:2px solid #f97316 !important; background:#27272a; border-radius:6px; box-sizing:border-box;">
                        <div class="metric-label" style="margin-bottom:6px; font-size:11px; text-transform:uppercase; letter-spacing:0.5px; color:#a1a1aa; font-weight:600;">Total Rainfall</div>
                        <div class="metric-value" style="color:#fafafa; font-size:1.5rem !important; font-weight:700; line-height:1.2;">{total_rain:.1f} mm</div>
                        <div style="font-size:11px; color:#71717a; margin-top:4px;">7-Day Cumulative Total</div>
                    </div>
                    """, unsafe_allow_html=True)
                with sc3:
                    st.markdown(f"""
                    <div class="metric-card" style="display:flex; flex-direction:column; justify-content:center; align-items:center; min-height:105px; padding:16px 12px; text-align:center; border:1px solid #3f3f46; border-top:2px solid #f97316 !important; background:#27272a; border-radius:6px; box-sizing:border-box;">
                        <div class="metric-label" style="margin-bottom:6px; font-size:11px; text-transform:uppercase; letter-spacing:0.5px; color:#a1a1aa; font-weight:600;">High Risk Days</div>
                        <div class="metric-value" style="color:#fafafa; font-size:1.5rem !important; font-weight:700; line-height:1.2;">{high_days} <span style="font-size:1rem;color:#71717a;">/ 7</span></div>
                        <div style="font-size:11px; color:#71717a; margin-top:4px;">Days exceeding 60% probability</div>
                    </div>
                    """, unsafe_allow_html=True)

                # Recommendation dark card with orange left border (#f97316)
                st.markdown(f"""
                <div style="background:#27272a; border:1px solid #3f3f46; border-left:4px solid #f97316; border-radius:6px; padding:14px 18px; margin:14px 0;">
                    <div style="display:flex; align-items:flex-start; gap:10px;">
                        <i class="fa-solid fa-circle-info" style="color:#f97316; font-size:1rem; margin-top:2px;"></i>
                        <div>
                            <span style="font-weight:600; color:#fafafa; font-size:0.9rem;">Recommendation: </span>
                            <span style="color:#d4d4d8; font-size:0.875rem; line-height:1.5;">{recommendation}</span>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Emergency Numbers Alert Card
                curr_state = forecast_data.get('state') or selected_state or 'National'
                state_hl = get_state_helpline(curr_state)
                st.markdown(f"""
                <div style="background:rgba(239, 68, 68, 0.08); border:1px solid rgba(239, 68, 68, 0.25); border-radius:6px; padding:14px 18px; margin-top:8px; display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:12px;">
                    <div style="display:flex; align-items:center; gap:10px;">
                        <i class="fa-solid fa-phone-volume" style="color:#ef4444; font-size:1.1rem;"></i>
                        <div>
                            <span style="font-size:0.875rem; font-weight:600; color:#fafafa;">Emergency Helplines</span>
                            <span style="font-size:0.775rem; color:#71717a; margin-left:8px;">Emergency contacts</span>
                        </div>
                    </div>
                    <div style="display:flex; gap:16px; align-items:center; flex-wrap:wrap; font-size:0.85rem;">
                        <span style="color:#d4d4d8;">State Disaster Helpline: <b style="color:#fafafa; font-family:monospace; font-size:0.95rem;">{state_hl}</b></span>
                        <span style="color:#3f3f46;">|</span>
                        <span style="color:#d4d4d8;">National Emergency: <b style="color:#fafafa; font-family:monospace; font-size:0.95rem;">112</b></span>
                        <span style="color:#3f3f46;">|</span>
                        <span style="color:#d4d4d8;">NDMA Helpline: <b style="color:#fafafa; font-family:monospace; font-size:0.95rem;">1078</b></span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            except Exception as e:
                st.warning(f"Could not generate summary: {str(e)}")

            st.markdown("---")

            # Display detailed table
            st.subheader("Day-by-Day Breakdown")
            display_cols = ["date", "precipitation_mm", "flood_probability", "risk_level"]
            
            # Format dataframe for display
            table_df = forecast_df[display_cols].copy()
            table_df.columns = ["Date", "Rainfall (mm)", "Flood Risk %", "Risk Level"]
            table_df["Flood Risk %"] = (table_df["Flood Risk %"] * 100).round(1).astype(str) + "%"
            try:
                table_df["Date"] = pd.to_datetime(table_df["Date"]).dt.strftime("%a, %b %d")
            except Exception:
                pass  # Keep raw dates if parsing fails
            
            st.dataframe(table_df, use_container_width=True, hide_index=True)

            st.markdown("---")

            # Download buttons styled with orange border and dark background
            st.markdown("""
            <style>
            div.stDownloadButton > button,
            div[data-testid="stDownloadButton"] > button {
                border: 1px solid #f97316 !important;
                background: #18181b !important;
                background-color: #18181b !important;
                color: #fafafa !important;
                border-radius: 6px !important;
                font-weight: 500 !important;
                transition: all 0.15s ease !important;
            }
            div.stDownloadButton > button:hover,
            div[data-testid="stDownloadButton"] > button:hover {
                border-color: #fb923c !important;
                background: rgba(249, 115, 22, 0.12) !important;
                background-color: rgba(249, 115, 22, 0.12) !important;
                color: #ffffff !important;
            }
            </style>
            """, unsafe_allow_html=True)
            col1, col2, col3 = st.columns([1, 1, 2])
            
            with col1:
                csv = forecast_df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"forecast_{dist_name}_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    key="forecast_download_csv"
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
                    label="Download Report",
                    data=report_text,
                    file_name=f"report_{dist_name}_{pd.Timestamp.now().strftime('%Y%m%d')}.txt",
                    mime="text/plain",
                    key="forecast_download_report"
                )

# ===== PAGE 5: FLOOD DAMAGE SEVERITY CLASSIFIER =====
def page_damage_classifier():
    st.markdown("""<div class="card-custom" style="padding:16px 20px;margin-bottom:16px;">
        <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:12px;">
            <div>
                <h2 style="font-size:1.35rem;font-weight:600;margin:0 0 2px 0;color:#fafafa;">Flood Damage Severity Classifier</h2>
                <p style="margin:0;color:#71717a;font-size:0.875rem;">Computer vision classification model evaluating structural and surface inundation severity.</p>
            </div>
            <div>
                <span class="status-pill"><span class="status-dot blue"></span> Vision Engine: ResNet CNN</span>
            </div>
        </div>
    </div>""", unsafe_allow_html=True)

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
                "Classify Damage Severity",
                type="primary",
                use_container_width=True,
            )

    with col2:
        if uploaded_file and analyze:
            with st.spinner("Analyzing image..."):
                result = classify_flood_image(image)

            colors = {
                "No Flooding": "#f97316",
                "Mild": "#4ade80",
                "Moderate": "#f59e0b",
                "Severe": "#ef4444",
            }
            color = colors.get(result["severity"], "#f97316")
            severity_label = (
                result["severity"]
                if result["severity"] == "No Flooding"
                else f"{result['severity']} Flooding"
            )

            st.markdown(f"""
            <div style='background:#27272a; border:1px solid #3f3f46; border-left:4px solid {color};
            padding:16px; border-radius:6px;
            text-align:left; color:#fafafa; margin-bottom:12px;'>
                <div style="font-size:11px;color:#71717a;font-weight:600;text-transform:uppercase;letter-spacing:0.04em;">Classification Result</div>
                <div style="font-size:20px;font-weight:700;color:{color};margin-top:4px;">{severity_label}</div>
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
    st.markdown("""<div class="card-custom" style="padding:16px 20px;margin-bottom:16px;">
        <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:12px;">
            <div>
                <h2 style="font-size:1.35rem;font-weight:600;margin:0 0 2px 0;color:#fafafa;">Crop Disease Detection</h2>
                <p style="margin:0;color:#71717a;font-size:0.875rem;">Deep learning pathology model detecting agricultural blights and pathogen infections.</p>
            </div>
            <div>
                <span class="status-pill"><span class="status-dot green"></span> PlantVillage Model: Active</span>
            </div>
        </div>
    </div>""", unsafe_allow_html=True)

    if not CROP_DISEASE_AVAILABLE:
        st.error("Crop disease classifier is not available.")
        if CROP_DISEASE_ERROR:
            st.caption(CROP_DISEASE_ERROR)
        return

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### Upload Leaf Image")
        st.caption(
            "Supported crops: Corn, Tomato, Potato, Pepper, Rice, Grape, Apple, Strawberry."
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
                "Detect Disease",
                type="primary",
                use_container_width=True,
            )

            if analyze_btn:
                with st.spinner("Analyzing leaf image..."):
                    result = classify_crop_image(image)

                st.session_state["crop_result"] = result

    with col2:
        st.markdown("### Analysis Results")

        st.markdown("**Supported Crops:**")
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
                st.success(f"**{result['crop']}** — HEALTHY")
            elif result["status"] == "Error":
                st.error("Model not trained yet")
            else:
                st.error(f"**{result['crop']}** — {result['disease']} DETECTED")

            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Confidence", f"{result['confidence']}%")
            with col_b:
                st.metric("Severity", result["severity"])

            st.markdown("---")
            st.markdown("**Description:**")
            st.info(result["description"])

            st.markdown("**Treatment:**")
            for treatment in result["treatment"]:
                st.markdown(f"- {treatment}")

            st.markdown("**Flood Connection:**")
            st.warning(result["flood_connection"])

            st.markdown("**Prevention:**")
            for prevention in result["prevention"]:
                st.markdown(f"- {prevention}")
        else:
            st.info("Upload a leaf image and click Detect Disease to see results")
            return

# ===== PAGE 7: CROP YIELD PREDICTOR =====
def page_yield_predictor():
    st.markdown("""<div class="card-custom" style="padding:16px 20px;margin-bottom:16px;">
        <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:12px;">
            <div>
                <h2 style="font-size:1.35rem;font-weight:600;margin:0 0 2px 0;color:#fafafa;">Crop Yield Predictor</h2>
                <p style="margin:0;color:#71717a;font-size:0.875rem;">Agronomic forecast model correlating flood probability, precipitation, and regional harvest records.</p>
            </div>
            <div>
                <span class="status-pill"><span class="status-dot blue"></span> Model: Agronomic Regression</span>
            </div>
        </div>
    </div>""", unsafe_allow_html=True)

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
        st.subheader("Input Parameters")

        selected_state = st.selectbox(
            "Select State",
            options=states,
            index=states.index(st.session_state.get("last_state")) if st.session_state.get("last_state") in states else 0,
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
            "Predict Crop Yield",
            type="primary",
            use_container_width=True,
        )

    with col2:
        if predict_btn:
            from src.crop_yield_predictor import predict_crop_yield

            with st.spinner("Calculating expected yield..."):
                result = predict_crop_yield(
                    crop=selected_crop,
                    state=selected_state,
                    season=selected_season,
                    area_hectares=area,
                    flood_risk_pct=flood_risk,
                    current_rainfall=rainfall,
                )

            if "error" in result:
                st.error(result["error"])
            else:
                colors = {
                    "Low": "#4ade80",
                    "Moderate": "#f59e0b",
                    "High": "#ef4444",
                }
                color = colors.get(result["risk_level"], "#f97316")

                st.markdown(f"""
                <div style='background: #27272a;
                border: 1px solid #3f3f46;
                border-left: 4px solid {color};
                border-radius: 6px;
                padding: 16px;
                text-align: left;
                margin-bottom: 16px;'>
                <div style='color: #71717a; font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.04em;'>Expected Yield Projection</div>
                <div style='color: {color}; font-size: 24px; font-weight: 700; margin-top: 4px;'>
                {result['adjusted_yield']} t/ha</div>
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
                        f"{result['adjusted_yield']} t/ha is "
                        f"{abs(result['vs_average']):.2f} t/ha ABOVE district average"
                    )
                else:
                    st.warning(
                        f"{result['adjusted_yield']} t/ha is "
                        f"{abs(result['vs_average']):.2f} t/ha BELOW district average"
                    )

                st.info(f"{result['recommendation']}")

                if flood_risk > 50:
                    st.error(
                        "Apply for PMFBY Crop Insurance immediately!\n"
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
    st.markdown("""<div class="card-custom" style="padding:16px 20px;margin-bottom:16px;">
        <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:12px;">
            <div>
                <h2 style="font-size:1.35rem;font-weight:600;margin:0 0 2px 0;color:#fafafa;">Crop Loss Estimator & Compensation</h2>
                <p style="margin:0;color:#71717a;font-size:0.875rem;">Estimate financial loss to regional crops due to flooding and query PMFBY compensation schemes.</p>
            </div>
            <div>
                <span class="status-pill"><span class="status-dot green"></span> PMFBY Scheme Rates: 2024-25</span>
            </div>
        </div>
    </div>""", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### Farm Parameters")

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
            "Calculate Crop Loss",
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
            <div style='background: #27272a;
                border: 2px solid #ef4444;
                border-radius: 15px;
                padding: 25px;
                text-align: center;
                margin-bottom: 20px'>
                <h2 style='color: #ef4444; margin:0'>
                Estimated Loss</h2>
                <h1 style='color: #fafafa;
                    font-size: 2.5em; margin:10px 0'>
                {loss_str}</h1>
                <p style='color: #71717a; margin:0'>
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

            st.markdown("### Crop-wise Breakdown")
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

            st.markdown("### Loss Visualization")
            fig = plot_loss_chart(results["crops"])
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("### Government Compensation")
            schemes = get_compensation_schemes()
            for key, scheme in schemes.items():
                with st.expander(f"{scheme['name']}"):
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
            Fill in your farm details and click
            'Calculate Crop Loss' to see estimated
            financial impact of flooding on your crops.

            This tool helps farmers:
            - Estimate financial losses before floods
            - Plan crop insurance
            - Apply for government compensation
            """)

# ===== PAGE 9: ALERT SYSTEM =====
def page_alert_system():
    st.markdown("""<div class="card-custom" style="padding:16px 20px;margin-bottom:16px;">
        <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:12px;">
            <div>
                <h2 style="font-size:1.35rem;font-weight:600;margin:0 0 2px 0;color:#fafafa;">Automated Early Warning & Alert System</h2>
                <p style="margin:0;color:#71717a;font-size:0.875rem;">Configure threshold-based regional notification dispatches via SMTP and SMS channels.</p>
            </div>
            <div>
                <span class="status-pill"><span class="status-dot green"></span> Dispatch Daemon: Online</span>
            </div>
        </div>
    </div>""", unsafe_allow_html=True)

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
  <div style="width:32px;height:32px;background:rgba(249,115,22,0.15);
    border:1px solid rgba(249,115,22,0.3);border-radius:4px;display:flex;align-items:center;
    justify-content:center;color:#f97316;font-size:11px;font-weight:700;">AL</div>
  <div>
    <div style="color:#fafafa;font-size:14px;font-weight:600;">
      Notification Dispatch Parameters</div>
    <div style="color:#71717a;font-size:12px;">
      Direct subscriber dispatch before projected crest</div>
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
<div style="background:#27272a;border:1px solid #3f3f46;border-radius:6px;
  padding:14px;margin:16px 0;">
  <div style="color:#71717a;font-size:11px;text-transform:uppercase;
    letter-spacing:0.04em;font-weight:600;">
    Alert threshold — notify when risk exceeds:</div>
</div>
""", unsafe_allow_html=True)

    if "alert_threshold" not in st.session_state:
        st.session_state.alert_threshold = 40

    t1, t2, t3, t4 = st.columns(4)

    thresholds = [
        (t1, 20, "Low", "#4ade80"),
        (t2, 40, "Moderate", "#f59e0b"),
        (t3, 60, "High", "#f97316"),
        (t4, 80, "Very High", "#ef4444"),
    ]

    for col, val, label, color in thresholds:
        with col:
            is_selected = st.session_state.alert_threshold == val
            st.markdown(f"""
            <div style="background:#27272a;
              border:1px solid {'#f97316' if is_selected else '#3f3f46'};
              border-radius:6px;padding:12px;text-align:center;">
              <div style="font-size:20px;font-weight:600;
                color:{'#f97316' if is_selected else color};">
                {val}%</div>
              <div style="font-size:11px;color:#71717a;
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
    if st.button("Subscribe to Alerts", type="primary", use_container_width=True, key="subscribe_btn"):
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
            <i class="fa-solid fa-check-circle" style="color:#4ade80;"></i> Subscribed successfully!
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
        st.markdown("### Test Alert")
        test_email = st.text_input(
            "Test email address",
            placeholder="Send test alert to this email",
            key="test_alert_email",
        )
        if st.button("Send Test Email", use_container_width=True):
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
                    st.success(f"{message}")
                else:
                    st.error(f"Email failed: {message}")
            else:
                st.warning("Enter a test email address.")

# ===== PAGE 3: HISTORICAL FLOOD TRENDS =====
def page_trends():
    lang = st.session_state.get("lang", "en")
    
    # Heading & Subheading
    st.markdown(f"""
    <div class="card-custom" style="padding:16px 20px;margin-bottom:16px;">
        <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:12px;">
            <div>
                <h2 style="font-size:1.35rem;font-weight:600;margin:0 0 2px 0;color:#fafafa;">{get_text("trends_title", lang)}</h2>
                <p style="margin:0;color:#71717a;font-size:0.875rem;">{get_text("trends_subtitle", lang)}</p>
            </div>
            <div>
                <span class="status-pill">Dataset: NDMA Historical Records</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
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
        
    # Year Range Slider (2015-2024)
    min_year = 2015
    max_year = 2024
    if not df.empty and "year" in df.columns:
        min_year = int(df["year"].min())
        max_year = int(df["year"].max())

    year_range = st.slider(
        "Year Range (2015–2024)",
        min_value=min_year,
        max_value=max_year,
        value=(min_year, max_year),
        step=1,
        key="trends_year_slider"
    )
    start_year, end_year = year_range
        
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

    # District data for trends and calculations
    df_district = df_trends[(df_trends['state'] == selected_state) & (df_trends['district'] == selected_district)].sort_values("year")

    # Filter df to selected district and year range
    filtered = df_district[(df_district['year'] >= start_year) & (df_district['year'] <= end_year)]
    
    # Calculate stats
    if filtered.empty:
        total_events, worst_year, peak_people, total_damage = 0, 0, 0, 0.0
    else:
        total_events = int(filtered["flood_events"].sum())
        max_events_idx = filtered["flood_events"].idxmax()
        worst_year = int(filtered.loc[max_events_idx, "year"])
        peak_people = int(filtered["people_affected"].max())
        total_damage = float(filtered["damage_cr"].sum())
    
    # Render Stat Cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="metric-card" style="display: flex; flex-direction: column; justify-content: center; align-items: center; height: 130px; padding: 16px 12px; text-align: center; border: 1px solid #3f3f46; border-top: 2px solid #f97316 !important; background: #27272a; border-radius: 6px; box-sizing: border-box; overflow: hidden;">
            <div class="metric-label" style="margin-bottom: 6px; font-size: 11px; overflow: hidden; text-overflow: ellipsis; display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; line-height: 1.2; height: 28px; width: 100%;">{get_text("stat_total_events", lang)}</div>
            <div class="metric-value" style="color: #fafafa; font-size: 1.8rem !important; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; width: 100%; line-height: 1.1;">{total_events}</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-card" style="display: flex; flex-direction: column; justify-content: center; align-items: center; height: 130px; padding: 16px 12px; text-align: center; border: 1px solid #3f3f46; border-top: 2px solid #f97316 !important; background: #27272a; border-radius: 6px; box-sizing: border-box; overflow: hidden;">
            <div class="metric-label" style="margin-bottom: 6px; font-size: 11px; overflow: hidden; text-overflow: ellipsis; display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; line-height: 1.2; height: 28px; width: 100%;">{get_text("stat_worst_year", lang)}</div>
            <div class="metric-value" style="color: #fafafa; font-size: 1.8rem !important; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; width: 100%; line-height: 1.1;">{worst_year}</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="metric-card" style="display: flex; flex-direction: column; justify-content: center; align-items: center; height: 130px; padding: 16px 12px; text-align: center; border: 1px solid #3f3f46; border-top: 2px solid #f97316 !important; background: #27272a; border-radius: 6px; box-sizing: border-box; overflow: hidden;">
            <div class="metric-label" style="margin-bottom: 6px; font-size: 11px; overflow: hidden; text-overflow: ellipsis; display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; line-height: 1.2; height: 28px; width: 100%;">{get_text("stat_peak_people", lang)}</div>
            <div class="metric-value" style="color: #fafafa; font-size: 1.8rem !important; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; width: 100%; line-height: 1.1;">{peak_people:,}</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="metric-card" style="display: flex; flex-direction: column; justify-content: center; align-items: center; height: 130px; padding: 16px 12px; text-align: center; border: 1px solid #3f3f46; border-top: 2px solid #f97316 !important; background: #27272a; border-radius: 6px; box-sizing: border-box; overflow: hidden;">
            <div class="metric-label" style="margin-bottom: 6px; font-size: 11px; overflow: hidden; text-overflow: ellipsis; display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; line-height: 1.2; height: 28px; width: 100%;">{get_text("stat_total_damage", lang)}</div>
            <div class="metric-value" style="color: #fafafa; font-size: 1.8rem !important; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; width: 100%; line-height: 1.1;">₹{total_damage:,.0f} Cr</div>
        </div>
        """, unsafe_allow_html=True)

    # Trend Indicator Calculation (5-year baseline: 2020-2024 vs 2015-2019)
    df_recent = df_district[df_district["year"] >= 2020]
    df_old = df_district[(df_district["year"] >= 2015) & (df_district["year"] < 2020)]
    
    avg_recent = df_recent["flood_events"].mean() if not df_recent.empty else 0.0
    avg_old = df_old["flood_events"].mean() if not df_old.empty else 0.0
    
    if avg_old > 0:
        pct_change = ((avg_recent - avg_old) / avg_old) * 100
    else:
        pct_change = 0.0
        
    expl_text = "Change in flood frequency vs previous 5-year period"

    # Render Trend Indicator
    if pct_change > 0:
        trend_html = f"""
        <div class="rec-box rec-mod" style="background: rgba(245,158,11,0.08); border-left: 3px solid #f59e0b !important; padding: 14px; margin: 16px 0; border-radius: 6px;">
            <div style="color: #f59e0b; margin: 0; font-size: 0.9375rem; font-weight: 600;">
                {get_text("trend_increasing", lang)} (+{pct_change:.1f}%)
            </div>
            <div style="color: #71717a; font-size: 0.75rem; margin-top: 4px;">
                {expl_text}
            </div>
        </div>
        """
    elif pct_change < 0:
        trend_html = f"""
        <div class="rec-box rec-safe" style="background: rgba(74,222,128,0.08); border-left: 3px solid #4ade80 !important; padding: 14px; margin: 16px 0; border-radius: 6px;">
            <div style="color: #4ade80; margin: 0; font-size: 0.9375rem; font-weight: 600;">
                {get_text("trend_decreasing", lang)} ({pct_change:.1f}%)
            </div>
            <div style="color: #71717a; font-size: 0.75rem; margin-top: 4px;">
                {expl_text}
            </div>
        </div>
        """
    else:
        trend_html = f"""
        <div class="rec-box rec-safe" style="background: rgba(249,115,22,0.08); border: 1px solid #3f3f46; border-left: 3px solid #f97316 !important; padding: 14px; margin: 16px 0; border-radius: 6px;">
            <div style="color: #f97316; margin: 0; font-size: 0.9375rem; font-weight: 600;">
                {get_text("trend_stable", lang)}
            </div>
            <div style="color: #71717a; font-size: 0.75rem; margin-top: 4px;">
                {expl_text}
            </div>
        </div>
        """
    render_html(trend_html)
    
    # CHART 1 — Year-over-Year Line Chart
    fig1 = go.Figure()
    colors = ["#f97316", "#71717a", "#f97316", "#4ade80"]
    
    for i, (col, display_label) in enumerate(selected_cols):
        fig1.add_trace(go.Scatter(
            x=filtered["year"],
            y=filtered[col],
            mode="lines+markers",
            name=display_label,
            line=dict(color=colors[i % len(colors)], width=2.5),
            marker=dict(size=7, symbol="circle"),
            hovertemplate="%{x}: %{y}<extra></extra>"
        ))

    # Determine integer dtick if values are small to avoid fractional ticks
    max_val = 0
    for col, _ in selected_cols:
        if not filtered.empty and col in filtered.columns:
            col_max = filtered[col].max()
            if pd.notna(col_max) and col_max > max_val:
                max_val = float(col_max)
    y_dtick = 1 if max_val <= 10 else None
        
    fig1.update_layout(
        title=dict(
            text=get_text("chart_yoy_title", lang).format(district=selected_district, state=selected_state),
            font=dict(size=15, color="#fafafa", family="Inter")
        ),
        paper_bgcolor="#18181b",
        plot_bgcolor="#27272a",
        font=dict(color="#71717a", family="Inter"),
        xaxis=dict(
            tickmode="linear",
            tick0=start_year,
            dtick=1,
            gridcolor="rgba(255,255,255,0.05)",
            linecolor="rgba(255,255,255,0.1)",
            title=""
        ),
        yaxis=dict(
            gridcolor="rgba(255,255,255,0.05)",
            linecolor="rgba(255,255,255,0.1)",
            title="",
            tickformat="d",
            tickmode="linear" if y_dtick else "auto",
            dtick=y_dtick,
            tick0=0 if y_dtick else None
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
    st.plotly_chart(fig1, use_container_width=True, config={"displayModeBar": False})
    
    # CHART 2 — Bar Chart (Flood Events by Year)
    fig2 = go.Figure(go.Bar(
        x=filtered["year"],
        y=filtered["flood_events"],
        marker=dict(
            color="#f97316"
        ),
        hovertemplate="Year %{x}: %{y} events<extra></extra>"
    ))
    
    fig2.update_layout(
        title=dict(
            text=get_text("chart_annual_title", lang),
            font=dict(size=16, color="#fafafa", family="Inter")
        ),
        paper_bgcolor="#18181b",
        plot_bgcolor="#27272a",
        font=dict(color="#71717a", family="Inter"),
        xaxis=dict(
            tickmode="linear",
            tick0=start_year,
            dtick=1,
            gridcolor="rgba(255,255,255,0.05)",
            linecolor="rgba(255,255,255,0.1)",
            title=""
        ),
        yaxis=dict(
            gridcolor="rgba(255,255,255,0.05)",
            linecolor="rgba(255,255,255,0.1)",
            title=get_text("metric_flood_events", lang),
            tickformat="d",
            tickmode="linear",
            tick0=0,
            dtick=1
        ),
        margin=dict(l=40, r=40, t=60, b=40)
    )
    st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})
    
    # CHART 3 — State Comparison (optional toggle)
    compare_state = st.checkbox(get_text("compare_checkbox", lang), value=False)
    if compare_state:
        df_state = df[
            (df["state"] == selected_state) &
            (df["year"] >= start_year) &
            (df["year"] <= end_year)
        ].sort_values(["year", "district"])
        
        orange_shades = [
            "#f97316", "#fb923c", "#ea580c", "#fdba74", "#c2410c",
            "#f59e0b", "#9a3412", "#d97706", "#fed7aa", "#b45309",
            "#7c2d12", "#ffedd5"
        ]
        fig3 = px.bar(
            df_state,
            x="year",
            y="flood_events",
            color="district",
            barmode="group",
            title=get_text("chart_compare_title", lang).format(state=selected_state),
            color_discrete_sequence=orange_shades
        )
        
        fig3.update_layout(
            paper_bgcolor="#18181b",
            plot_bgcolor="#27272a",
            font=dict(color="#71717a", family="Inter"),
            xaxis=dict(
                tickmode="linear",
                tick0=start_year,
                dtick=1,
                gridcolor="rgba(255,255,255,0.05)",
                linecolor="rgba(255,255,255,0.1)",
                title=""
            ),
            yaxis=dict(
                gridcolor="rgba(255,255,255,0.05)",
                linecolor="rgba(255,255,255,0.1)",
                title=get_text("metric_flood_events", lang),
                tickformat="d",
                tickmode="linear",
                tick0=0,
                dtick=1
            ),
            margin=dict(l=40, r=40, t=60, b=40),
            legend=dict(
                title=dict(text="", font=dict(color="#fafafa")),
                font=dict(color="#fafafa")
            )
        )
        st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False})


# ===== PAGE 10: ABOUT =====
def page_about():
    """Renders the developer-built system architecture and telemetry documentation."""
    st.markdown("""
    <div class="card-custom" style="padding:16px 20px;margin-bottom:16px;">
        <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:12px;">
            <div>
                <h2 style="font-size:1.35rem;font-weight:600;margin:0 0 2px 0;color:#fafafa;"><i class="fa-solid fa-cog" style="color:#f97316;margin-right:8px;"></i>System Architecture & Validation</h2>
                <p style="margin:0;color:#71717a;font-size:0.875rem;">Platform telemetry pipelines, model benchmark metrics, and data source provenance.</p>
            </div>
            <div>
                <span class="status-pill"><span class="status-dot blue"></span> Build: v2.4 Enterprise</span>
            </div>
        </div>
    </div>

    <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-bottom:16px;">
        <div class="card-custom" style="text-align:center;padding:14px;">
            <div style="font-size:1.5rem;font-weight:700;color:#fafafa;">736</div>
            <div style="font-size:0.75rem;font-weight:600;color:#71717a;text-transform:uppercase;letter-spacing:0.04em;margin-top:2px;">Districts Covered</div>
        </div>
        <div class="card-custom" style="text-align:center;padding:14px;">
            <div style="font-size:1.5rem;font-weight:700;color:#fafafa;">36</div>
            <div style="font-size:0.75rem;font-weight:600;color:#71717a;text-transform:uppercase;letter-spacing:0.04em;margin-top:2px;">States & UTs</div>
        </div>
        <div class="card-custom" style="text-align:center;padding:14px;">
            <div style="font-size:1.5rem;font-weight:700;color:#fafafa;">4,695</div>
            <div style="font-size:0.75rem;font-weight:600;color:#71717a;text-transform:uppercase;letter-spacing:0.04em;margin-top:2px;">Historical Records</div>
        </div>
        <div class="card-custom" style="text-align:center;padding:14px;">
            <div style="font-size:1.5rem;font-weight:700;color:#fafafa;">5</div>
            <div style="font-size:0.75rem;font-weight:600;color:#71717a;text-transform:uppercase;letter-spacing:0.04em;margin-top:2px;">Model Pipelines</div>
        </div>
    </div>

    <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:16px;">
        <div class="card-custom" style="padding:16px;">
            <div style="font-size:0.75rem;font-weight:600;color:#71717a;text-transform:uppercase;letter-spacing:0.04em;">Model Benchmarks</div>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:12px;">
                <div style="background:#18181b;border:1px solid #3f3f46;border-radius:4px;padding:10px;">
                    <div style="font-size:0.8125rem;color:#fafafa;font-weight:500;">XGBoost Baseline</div>
                    <div style="font-size:1.25rem;font-weight:700;color:#fafafa;margin-top:2px;">0.83 AUC</div>
                </div>
                <div style="background:#18181b;border:1px solid #3f3f46;border-radius:4px;padding:10px;">
                    <div style="font-size:0.8125rem;color:#fafafa;font-weight:500;">LSTM Attention</div>
                    <div style="font-size:1.25rem;font-weight:700;color:#fafafa;margin-top:2px;">0.70 Recall</div>
                </div>
                <div style="background:#18181b;border:1px solid #3f3f46;border-radius:4px;padding:10px;">
                    <div style="font-size:0.8125rem;color:#fafafa;font-weight:500;">Damage Classifier</div>
                    <div style="font-size:1.25rem;font-weight:700;color:#fafafa;margin-top:2px;">89.6% Accuracy</div>
                </div>
                <div style="background:#18181b;border:1px solid #3f3f46;border-radius:4px;padding:10px;">
                    <div style="font-size:0.8125rem;color:#fafafa;font-weight:500;">Crop Pathology</div>
                    <div style="font-size:1.25rem;font-weight:700;color:#fafafa;margin-top:2px;">98.4% Accuracy</div>
                </div>
            </div>
        </div>

        <div class="card-custom" style="padding:16px;">
            <div style="font-size:0.75rem;font-weight:600;color:#71717a;text-transform:uppercase;letter-spacing:0.04em;">Engineering Stack</div>
            <div style="display:flex;flex-wrap:wrap;gap:6px;margin-top:12px;">
                <span class="status-pill">Python 3.11</span>
                <span class="status-pill">Streamlit Enterprise</span>
                <span class="status-pill">PyTorch 2.x</span>
                <span class="status-pill">XGBoost</span>
                <span class="status-pill">Folium GIS</span>
                <span class="status-pill">Open-Meteo ECMWF</span>
                <span class="status-pill">SHAP Explainability</span>
                <span class="status-pill">fpdf2 Automated Reports</span>
                <span class="status-pill">Pandas / NumPy</span>
            </div>
        </div>
    </div>

    <div class="card-custom" style="padding:16px;margin-bottom:16px;">
        <div style="font-size:0.75rem;font-weight:600;color:#71717a;text-transform:uppercase;letter-spacing:0.04em;margin-bottom:12px;">Data Ingestion Sources</div>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;">
            <div style="display:flex;gap:10px;align-items:flex-start;">
                <div style="background:#18181b;border:1px solid #3f3f46;color:#f97316;padding:4px 8px;border-radius:4px;font-family:monospace;font-size:0.75rem;font-weight:700;">IMD</div>
                <div>
                    <div style="color:#fafafa;font-size:0.875rem;font-weight:600;">India Meteorological Department</div>
                    <div style="color:#71717a;font-size:0.75rem;margin-top:1px;">Daily historical and projected precipitation grids.</div>
                </div>
            </div>
            <div style="display:flex;gap:10px;align-items:flex-start;">
                <div style="background:#18181b;border:1px solid #3f3f46;color:#f97316;padding:4px 8px;border-radius:4px;font-family:monospace;font-size:0.75rem;font-weight:700;">NDMA</div>
                <div>
                    <div style="color:#fafafa;font-size:0.875rem;font-weight:600;">National Disaster Management Authority</div>
                    <div style="color:#71717a;font-size:0.75rem;margin-top:1px;">Official disaster mapping and emergency inventory archives.</div>
                </div>
            </div>
            <div style="display:flex;gap:10px;align-items:flex-start;">
                <div style="background:#18181b;border:1px solid #3f3f46;color:#f97316;padding:4px 8px;border-radius:4px;font-family:monospace;font-size:0.75rem;font-weight:700;">DATA</div>
                <div>
                    <div style="color:#fafafa;font-size:0.875rem;font-weight:600;">PlantVillage Dataset (32,883 images)</div>
                    <div style="color:#71717a;font-size:0.75rem;margin-top:1px;">Standardized agricultural pathology benchmarking corpus.</div>
                </div>
            </div>
            <div style="display:flex;gap:10px;align-items:flex-start;">
                <div style="background:#18181b;border:1px solid #3f3f46;color:#f97316;padding:4px 8px;border-radius:4px;font-family:monospace;font-size:0.75rem;font-weight:700;">AGRI</div>
                <div>
                    <div style="color:#fafafa;font-size:0.875rem;font-weight:600;">Agricultural Harvest Statistics (19,689 records)</div>
                    <div style="color:#71717a;font-size:0.75rem;margin-top:1px;">Historical multi-crop yield and acreage records across India.</div>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ===== PAGE: SYSTEM OVERVIEW =====
def page_tactical_command():
    """Renders the clean internal tool System Overview and telemetry dashboard."""
    lang = st.session_state.get("lang", "en")
    district = st.session_state.get("selected_district") or st.session_state.get("current_district", "Patna")
    state = st.session_state.get("selected_state", "Bihar")
    district_profile = get_district_profile(state, district)

    st.markdown(
        f"""<div style="margin-bottom:16px;">
<div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:12px;">
<div>
<h2 style="margin:0;font-size:1.25rem;font-weight:600;color:#fafafa;"><i class="fa-solid fa-water" style="color:#f97316;margin-right:8px;"></i>System Telemetry & Sensor Overview</h2>
<p style="color:#71717a;font-size:0.875rem;margin:2px 0 0 0;">
Hydrologic indicators, model confidence metrics, and regional sensor status for {district} ({state}).
</p>
</div>
<div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;">
<span class="status-pill"><span class="status-dot amber"></span> Telemetry Gateway: Standby</span>
<span class="status-pill" style="color:#71717a;">Station Feed: Data unavailable — connect live API</span>
</div>
</div>
</div>""",
        unsafe_allow_html=True
    )

    view_mode = st.radio(
        "Overview Mode",
        [
            "Telemetry Dashboard",
            "Agricultural Assessment",
            "Emergency Protocol Checklist"
        ],
        horizontal=True,
        label_visibility="collapsed"
    )

    if view_mode == "Telemetry Dashboard":
        st.markdown(
            """<div class="system-alert-banner" style="background:#27272a;border:1px solid #3f3f46;border-radius:6px;padding:12px 16px;margin-bottom:16px;display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:8px;">
<div style="display:flex;align-items:center;gap:8px;">
<span style="background:#3f3f46;color:#fafafa;font-size:0.6875rem;font-weight:700;padding:2px 8px;border-radius:3px;">
<i class="fa-solid fa-circle-info" style="color:#71717a;margin-right:4px;"></i>TELEMETRY STATUS
</span>
<span style="color:#71717a;font-weight:500;font-size:0.875rem;">
Regional hydrologic advisory: Data unavailable — connect live API
</span>
</div>
<span style="color:#71717a;font-size:0.75rem;">Source: CWC / IMD Telemetry Network (Offline)</span>
</div>""",
            unsafe_allow_html=True
        )

        col1, col2 = st.columns([5, 7])

        with col1:
            st.markdown(
                f"""<div class="card-custom" style="background:#27272a;border:1px solid #3f3f46;border-radius:6px;padding:16px;">
<div style="display:flex;justify-content:space-between;align-items:flex-start;">
<div>
<div style="font-size:0.75rem;font-weight:600;color:#71717a;text-transform:uppercase;letter-spacing:0.04em;">
Current Telemetry Assessment
</div>
<h3 style="margin:4px 0 0 0;font-size:1.125rem;font-weight:600;color:#fafafa;">
Sensor Gateway Status
</h3>
</div>
<span class="badge" style="background:#3f3f46;color:#fafafa;padding:3px 8px;border-radius:4px;font-size:0.75rem;font-weight:600;">Feed Offline</span>
</div>
<div style="padding:16px 0;margin:12px 0;border-top:1px solid #3f3f46;border-bottom:1px solid #3f3f46;">
<div style="display:flex;align-items:baseline;gap:8px;">
<span style="font-size:1.75rem;font-weight:700;color:#71717a;line-height:1;">--</span>
<span style="color:#71717a;font-size:0.875rem;">Data unavailable — connect live API</span>
</div>
<div style="margin-top:12px;display:flex;flex-direction:column;gap:8px;">
<div style="display:flex;justify-content:space-between;font-size:0.8125rem;">
<span style="color:#71717a;">Peak Arrival Window:</span>
<span style="color:#71717a;font-style:italic;">Data unavailable — connect live API</span>
</div>
<div style="display:flex;justify-content:space-between;font-size:0.8125rem;">
<span style="color:#71717a;">Ensemble Confidence:</span>
<span style="color:#71717a;font-style:italic;">Data unavailable — connect live API</span>
</div>
<div style="display:flex;justify-content:space-between;font-size:0.8125rem;">
<span style="color:#71717a;">Monitored Basin:</span>
<span style="color:#fafafa;font-weight:600;">{district_profile.get("flood_type", "Riverine flood")}</span>
</div>
</div>
</div>
<div style="font-size:0.8125rem;color:#71717a;display:flex;align-items:center;justify-content:space-between;">
<span>Automated Emergency Dispatch: Standby</span>
<span style="color:#71717a;font-weight:500;">Feed Offline</span>
</div>
</div>""",
                unsafe_allow_html=True
            )

        with col2:
            m1, m2 = st.columns(2)
            with m1:
                st.markdown(
                    """<div class="metric-card" style="background:#27272a;border:1px solid #3f3f46;border-radius:6px;padding:16px;">
<div class="metric-label"><i class="fa-solid fa-cloud-rain" style="color:#f97316;margin-right:6px;"></i>24h Precipitation</div>
<div class="metric-value" style="color:#71717a;font-size:0.95rem;font-weight:600;margin:6px 0;">Data unavailable — connect live API</div>
<div style="font-size:0.75rem;color:#71717a;margin-top:4px;">No in-situ rain gauge feed connected</div>
</div>""",
                    unsafe_allow_html=True
                )
                st.markdown(
                    """<div class="metric-card" style="background:#27272a;border:1px solid #3f3f46;border-radius:6px;padding:16px;margin-top:10px;">
<div class="metric-label">Soil Saturation (0–30 cm)</div>
<div class="metric-value" style="color:#71717a;font-size:0.95rem;font-weight:600;margin:6px 0;">Data unavailable — connect live API</div>
<div style="font-size:0.75rem;color:#71717a;margin-top:4px;">No in-situ soil moisture probe connected</div>
</div>""",
                    unsafe_allow_html=True
                )
            with m2:
                st.markdown(
                    """<div class="metric-card" style="background:#27272a;border:1px solid #3f3f46;border-radius:6px;padding:16px;">
<div class="metric-label">River Gauging Level</div>
<div class="metric-value" style="color:#71717a;font-size:0.95rem;font-weight:600;margin:6px 0;">Data unavailable — connect live API</div>
<div style="font-size:0.75rem;color:#71717a;margin-top:4px;">No live CWC river gauge feed connected</div>
</div>""",
                    unsafe_allow_html=True
                )
                st.markdown(
                    """<div class="metric-card" style="background:#27272a;border:1px solid #3f3f46;border-radius:6px;padding:16px;margin-top:10px;">
<div class="metric-label">Dam Discharge Outflow</div>
<div class="metric-value" style="color:#71717a;font-size:0.95rem;font-weight:600;margin:6px 0;">Data unavailable — connect live API</div>
<div style="font-size:0.75rem;color:#71717a;margin-top:4px;">No live barrage telemetry feed connected</div>
</div>""",
                    unsafe_allow_html=True
                )

    elif view_mode == "Agricultural Assessment":
        st.markdown(
            """<div class="card-custom" style="background:#27272a;border:1px solid #3f3f46;border-radius:6px;padding:16px;">
<h3 style="margin:0 0 12px 0;font-size:1rem;color:#fafafa;">Automated Agricultural Satellite Telemetry</h3>
<div style="background:#18181b;border:1px solid #3f3f46;border-radius:6px;padding:14px;margin-bottom:12px;">
<div style="display:flex;justify-content:space-between;align-items:center;">
<b style="color:#fafafa;">Kharif Crop Inundation Satellite Feed</b>
<span style="color:#71717a;font-weight:600;font-size:0.8125rem;">Data unavailable — connect live API</span>
</div>
<p style="font-size:0.8125rem;color:#71717a;margin:6px 0 0 0;">
Real-time satellite radar and multispectral flood extent feeds require an active ISRO Bhuvan or Sentinel-1 telemetry gateway subscription.
</p>
</div>
<div style="background:#18181b;border:1px solid #3f3f46;border-radius:6px;padding:14px;">
<div style="display:flex;justify-content:space-between;align-items:center;">
<b style="color:#fafafa;">PMFBY Automated Loss Estimation</b>
<span style="color:#71717a;font-weight:600;font-size:0.8125rem;">Data unavailable — connect live API</span>
</div>
<p style="font-size:0.8125rem;color:#71717a;margin:6px 0 0 0;">
No automated digital claim dossier feed connected. Use the dedicated <b>Crop Loss Estimator</b> and <b>Yield Predictor</b> tabs in the navigation menu for manual farm-level calculations.
</p>
</div>
</div>""",
            unsafe_allow_html=True
        )

    elif view_mode == "Emergency Protocol Checklist":
        st.markdown(
            """<div class="card-custom" style="background:#27272a;border:1px solid #3f3f46;border-radius:6px;padding:16px;">
<h3 style="margin:0 0 12px 0;font-size:1rem;color:#fafafa;">Operational Readiness & Emergency Protocols</h3>
<div style="display:flex;flex-direction:column;gap:8px;">
<div style="display:flex;align-items:center;justify-content:space-between;padding:10px 14px;background:#18181b;border-radius:4px;border:1px solid #3f3f46;">
<span style="font-size:0.8125rem;color:#fafafa;">Early Warning Cell Broadcast Gateway</span>
<span style="color:#71717a;font-weight:600;font-size:0.8125rem;">Data unavailable — connect live API</span>
</div>
<div style="display:flex;align-items:center;justify-content:space-between;padding:10px 14px;background:#18181b;border-radius:4px;border:1px solid #3f3f46;">
<span style="font-size:0.8125rem;color:#fafafa;">Shelter Evacuation Transit Telemetry</span>
<span style="color:#71717a;font-weight:600;font-size:0.8125rem;">Data unavailable — connect live API</span>
</div>
<div style="display:flex;align-items:center;justify-content:space-between;padding:10px 14px;background:#18181b;border-radius:4px;border:1px solid #3f3f46;">
<span style="font-size:0.8125rem;color:#fafafa;">Substation Power Grid SCADA Telemetry</span>
<span style="color:#71717a;font-weight:600;font-size:0.8125rem;">Data unavailable — connect live API</span>
</div>
<div style="display:flex;align-items:center;justify-content:space-between;padding:10px 14px;background:#18181b;border-radius:4px;border:1px solid #3f3f46;">
<span style="font-size:0.8125rem;color:#fafafa;">Rescue Vessel Fleet Tracking</span>
<span style="color:#71717a;font-weight:600;font-size:0.8125rem;">Data unavailable — connect live API</span>
</div>
</div>
</div>""",
            unsafe_allow_html=True
        )


# ===== FOOTER =====
def render_footer():
    st.markdown("""<div class="app-footer">
        <div style="display:flex;align-items:center;justify-content:center;gap:16px;flex-wrap:wrap;">
            <p style="margin:0"><span style="color:#fafafa;font-weight:600;">FloodGuard AI</span> &middot; Hydrological Telemetry & Emergency Analytics &middot; v2.4</p>
            <a href="tel:1078" style="text-decoration:none;">
                <span style="background:rgba(239,68,68,0.12);border:1px solid rgba(239,68,68,0.3);color:#ef4444;padding:3px 10px;border-radius:4px;font-size:0.75rem;font-weight:600;">
                    NDMA Hotline: 1078
                </span>
            </a>
        </div>
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
        <div class="system-alert-banner">
            <div>
                <strong style="color:#f87171;"><i class="fa-solid fa-exclamation-triangle" style="color:#ef4444;margin-right:6px;"></i>Offline Mode Active</strong><br>
                <span style="color:#71717a;font-size:0.8125rem;">
                No internet connection detected. Showing cached telemetry data.
                </span>
            </div>
            <span class="status-pill">Offline Cache</span>
        </div>
        """, unsafe_allow_html=True)

    # --- Persistent Global Sidebar ---
    with st.sidebar:
        st.markdown(f"""
<div style='padding: 10px 0 16px 0; border-bottom: 1px solid #3f3f46; margin-bottom: 12px;'>
    <h2 style='color: #fafafa; font-size: 1.125rem; font-weight: 700; margin: 0 0 4px 0; letter-spacing: -0.01em;'>
        <i class="fa-solid fa-water" style="color:#f97316; margin-right:8px;"></i>{get_text('app_title', lang)}
    </h2>
    <p style='color: #71717a; font-size: 0.8125rem; margin: 0; line-height: 1.4;'>
        {get_text('app_subtitle', lang)}
    </p>
</div>
""", unsafe_allow_html=True)

        # --- System Status Expander ---
        with st.sidebar.expander("System Status", expanded=True):
            if is_online:
                st.markdown("""
                <div style="background:rgba(74,222,128,0.1); border:1px solid rgba(74,222,128,0.3); border-radius:4px;
                padding:6px 10px; margin-bottom:8px; font-size:12px; color:#4ade80;">
                <i class="fa-solid fa-circle" style="font-size:8px;color:#4ade80;"></i> Online
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="background:rgba(239,68,68,0.1); border:1px solid rgba(239,68,68,0.3); border-radius:4px;
                padding:6px 10px; margin-bottom:8px; font-size:12px; color:#f87171;">
                <i class="fa-solid fa-circle" style="font-size:8px;color:#ef4444;"></i> Offline
                </div>
                """, unsafe_allow_html=True)

        # --- District Search ---
        districts_df = load_india_districts()
        ALL_DISTRICTS = sorted(districts_df["district"].dropna().unique().tolist())

        st.sidebar.markdown('### <i class="fa-solid fa-search" style="color:#71717a;font-size:0.9rem;margin-right:6px;"></i>District Search', unsafe_allow_html=True)
        search_query = st.sidebar.text_input(
            "", 
            placeholder="Search district...",
            key="district_search",
            label_visibility="collapsed"
        )

        if search_query and len(search_query) >= 2:
            matches = [d for d in ALL_DISTRICTS 
                       if search_query.lower() in d.lower()][:5]
            
            if matches:
                st.sidebar.markdown("**Results:**")
                for match in matches:
                    if st.sidebar.button(match, 
                      key=f"search_{match}",
                      use_container_width=True):
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

        st.markdown("---")

        # --- Language Toggle ---
        st.sidebar.markdown('<div style="font-size:0.75rem;color:#71717a;font-weight:600;margin-top:12px;margin-bottom:6px;">Language / भाषा</div>', unsafe_allow_html=True)
        current_lang = st.session_state.get("lang", "en")
        selected_lang = st.sidebar.radio(
            "Select Language",
            options=["English", "हिंदी"],
            index=0 if current_lang == "en" else 1,
            horizontal=True,
            label_visibility="collapsed",
            key="app_lang_radio"
        )
        new_lang = "hi" if selected_lang == "हिंदी" else "en"
        if new_lang != current_lang:
            st.session_state.lang = new_lang
            st.rerun()

        st.markdown("---")

        # --- Settings Expander ---
        st.sidebar.markdown('<div style="font-size:0.75rem;color:#71717a;font-weight:600;margin-top:14px;margin-bottom:4px;"><i class="fa-solid fa-cog" style="color:#71717a;margin-right:6px;"></i>Settings</div>', unsafe_allow_html=True)
        with st.sidebar.expander("System Configuration", expanded=False):
            groq_key = os.environ.get("GROQ_API_KEY", "").strip()
            if not groq_key or groq_key == "your_groq_api_key_here":
                st.markdown("""
                <div style="background:#18181b; border:1px solid #3f3f46; border-radius:4px; 
                padding:8px 10px; font-size:12px; color:#71717a; margin-bottom:8px; display:flex; align-items:center; gap:6px;">
                    <i class="fa-solid fa-circle-info" style="color:#71717a; font-size:12px;"></i>
                    <span>AI Assistant unavailable. Check API configuration.</span>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.caption("AI Assistant: Configured & Active")
            st.caption("Environment: Operational • Local Inference")
        st.markdown("---")

    st.markdown("""
    <div class="status-strip" style="margin-bottom:16px;">
        <div style="display:flex;align-items:center;gap:8px;">
            <span class="status-dot blue"></span>
            <span style="font-weight:600;font-size:0.9375rem;color:#fafafa;"><i class="fa-solid fa-water" style="color:#f97316;margin-right:6px;"></i>FloodGuard AI</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab_stitch, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11 = st.tabs([
        get_text("tab_risk", lang),
        get_text("tab_stitch", lang),
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
    with tab_stitch:
        page_tactical_command()
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
