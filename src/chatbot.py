"""
FloodGuard AI chatbot — powered by Groq API.
Every question goes directly to Groq. No pre-filtering.
"""
import logging
import os
import re
import sys

logger = logging.getLogger(__name__)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(PROJECT_ROOT, ".env"))
except ImportError:
    pass

from groq import Groq

try:
    from config import GROQ_API_KEY as _CFG_KEY
except ImportError:
    _CFG_KEY = ""

RUNTIME_GROQ_API_KEY = None

# ── API key management ──────────────────────────────────────────────
def set_gemini_api_key(api_key):
    global RUNTIME_GROQ_API_KEY
    RUNTIME_GROQ_API_KEY = api_key.strip() if api_key else None
    if api_key:
        os.environ["GROQ_API_KEY"] = api_key.strip()

def _get_api_key():
    return (RUNTIME_GROQ_API_KEY
            or os.environ.get("GROQ_API_KEY", "")
            or _CFG_KEY or "")

def has_gemini_api_key():
    k = _get_api_key()
    return bool(k and k != "your_groq_api_key_here")

# Backward compat stubs
def set_anthropic_api_key(api_key):
    pass

def has_anthropic_api_key():
    return False

# ── Question type detection ─────────────────────────────────────────
def detect_question_type(msg):
    m = msg.lower()
    if any(w in m for w in ["travel", "safe", "go", "visit", "drive", "road"]): return "travel_safety"
    if any(w in m for w in ["prepare", "kit", "emergency", "ready", "stock"]): return "emergency_prep"
    if any(w in m for w in ["risk", "score", "percent", "level", "danger"]): return "risk_explanation"
    if any(w in m for w in ["why", "cause", "how", "what is", "explain", "happen"]): return "education"
    if any(w in m for w in ["evacuate", "escape", "run", "leave", "move"]): return "evacuation"
    return "general"

# ── Risk helpers ────────────────────────────────────────────────────
def _risk_percent(score):
    s = float(score or 0)
    return round(s * 100) if s <= 1 else round(s)

def _risk_level_from_score(score):
    s = float(score or 0)
    if s > 1: s /= 100
    if s >= 0.6: return "HIGH"
    if s >= 0.3: return "MODERATE"
    return "LOW"

# ── Location detection (for UI label) ───────────────────────────────
def detect_location(question):
    q = question.lower()
    for state, kws in {
        "kerala": ["kerala","kochi","trivandrum","idukki","thrissur","alappuzha","wayanad"],
        "assam": ["assam","guwahati","kamrup","brahmaputra","dibrugarh","barpeta","dhubri"],
        "bihar": ["bihar","patna","kosi","darbhanga","muzaffarpur","sitamarhi","gandak"],
        "mumbai": ["mumbai","bombay","mithi","dharavi","thane"],
        "chennai": ["chennai","madras","tamil nadu","adyar","velachery"],
        "uttarakhand": ["uttarakhand","kedarnath","chamoli","mandakini","alaknanda"],
        "odisha": ["odisha","mahanadi","puri","cuttack","kendrapara"],
        "west bengal": ["west bengal","kolkata","damodar","hooghly","howrah"],
        "delhi": ["delhi","new delhi","yamuna"],
        "karnataka": ["karnataka","bengaluru","bangalore"],
    }.items():
        for kw in kws:
            if kw in q:
                return state
    return None


# ══════════════════════════════════════════════════════════════════════
#  MAIN FUNCTION — EVERY question goes to Groq API directly
# ══════════════════════════════════════════════════════════════════════
def _offline_answer(question, district=None, state=None, risk_score=None):
    q = question.lower()
    loc = district or state or "your region"
    if "kerala" in q or "2018" in q:
        return ("In the Kerala 2018 floods, extraordinary monsoon downpours (42% above normal) led to 35 overflowing rivers "
                "and controlled discharge from over 80 major dams, affecting all 14 districts. "
                "Key lesson: heed early dam discharge alerts and evacuate low-lying floodplains promptly.")
    elif "mumbai" in q or "2005" in q:
        return ("On 26 July 2005, Mumbai experienced a historic cloudburst of 944 mm rain in 24 hours coinciding with high tide, "
                "causing the Mithi River to overflow and submerging municipal transit networks. "
                "Key safety rule: avoid low-lying underpasses and flooded roads during heavy rainfall.")
    elif "chennai" in q or "2015" in q:
        return ("The November–December 2015 Chennai floods were caused by anomalous Northeast Monsoon rainfall combined with "
                "rapid discharge from the Chembarambakkam reservoir into the Adyar River basin.")
    elif "kedarnath" in q or "2013" in q or "uttarakhand" in q:
        return ("In June 2013, extreme multi-day cloudbursts and the breach of Chorabari Lake triggered devastating flash floods "
                "and debris flows down the Mandakini valley in Uttarakhand. In hilly terrain, always move to high ground away from river channels.")
    elif "assam" in q or "brahmaputra" in q:
        return ("The Brahmaputra river system in Assam regularly breaches danger levels during the South-West monsoon due to heavy catchment rainfall "
                "in Arunachal Pradesh and Bhutan, sediment siltation, and seasonal bank erosion across 25+ districts.")
    elif "bihar" in q or "kosi" in q:
        return ("Northern Bihar faces recurrent inundation from transboundary rivers like the Kosi, Gandak, and Kamla originating in Nepal. "
                "State flood management protocols emphasize embankment patrols and designated elevated shelter staging.")
    elif any(w in q for w in ["what should i do", "water enter", "house", "home", "protect", "kit", "safety"]):
        return (f"Immediate Flood Safety Protocols for {loc}:\n\n"
                "1. Turn off main circuit breakers and LPG gas cylinders before floodwaters enter living spaces.\n"
                "2. Relocate medicines, potable water, battery torches, and vital documents to upper floors.\n"
                "3. Never attempt to drive or wade through flood currents; 15 cm of moving water can knock an adult down.\n"
                "4. Keep emergency hotlines at hand: National Helpline: 112 | NDMA: 1078 | Flood Control: 1070.")
    elif any(w in q for w in ["why", "cause", "how do flood", "happen"]):
        return ("Floods in India result from intense concentrated monsoon spells, inadequate urban stormwater drainage capacity, "
                "riverbed siltation, and saturated upstream catchment basins releasing excessive runoff.")
    else:
        return (f"Hydrological Intelligence Advisory for {loc}:\n\n"
                "Antecedent precipitation, river levels, and regional saturation are monitored continuously. "
                "For localized emergencies, contact National Emergency: 112, NDMA Hotline: 1078, or State Flood Control: 1070.")


def get_chatbot_response(question, district=None, state=None, risk_score=None):
    """Send question to Groq API if configured, otherwise provide offline hydrological knowledge."""
    from dotenv import load_dotenv
    load_dotenv()
    api_key = _get_api_key()
    if not api_key or api_key == "your_groq_api_key_here":
        return {
            "answer": "AI Assistant unavailable. Check API configuration.",
            "question_type": "Info",
            "is_unavailable": True,
            "location_detected": None,
        }

    client = Groq(api_key=api_key)

    # Build context
    context = ""
    if district and state:
        context = f"User location: {district}, {state}. "
    if risk_score is not None:
        level = ("Low" if risk_score < 0.3 else
                 "Moderate" if risk_score < 0.6 else
                 "High" if risk_score < 0.8 else "Severe")
        context += f"Flood risk: {level} ({risk_score*100:.1f}%). "

    system_prompt = """You are FloodGuard AI, India's flood expert assistant.
Answer ALL questions about floods in India accurately.

Key facts you must know:
- Kerala 2018: 35 rivers flooded, 80+ dam releases, 483 deaths, 1.5M displaced, 42% above normal rain
- Assam 2020: Brahmaputra overflow, 25 districts, 4M affected
- Mumbai 2005: 944mm in 24hrs, 1094 deaths
- Chennai 2015: NE monsoon, Adyar overflow, 500 deaths
- Bihar 2017: Kosi river, 514 deaths, 8.5M affected
- Uttarakhand 2013: Kedarnath glacier burst, 5700 deaths
- Delhi 2023: Yamuna highest level in 45 years
- Emergency numbers: 112, NDMA: 1078, Flood Control: 1070

Rules:
- Answer EVERY question fully and helpfully
- Never say "please ask a specific question"
- Give detailed, informative answers always
- Kerala question = Kerala answer only
- Mumbai question = Mumbai answer only
- Never mix up states or events"""

    user_message = f"{context}User question: {question}\n\nGive a clear, detailed, helpful answer:"

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            max_tokens=1000,
            temperature=0.7
        )
        answer = response.choices[0].message.content

        # Detect question type for UI label
        q_lower = question.lower()
        if any(w in q_lower for w in ["2018", "2020", "2005", "2015", "2013", "2017", "2023"]):
            q_type = "Historical Event"
        elif any(w in q_lower for w in ["safe", "do", "help", "emergency", "evacuate"]):
            q_type = "Safety Advice"
        elif any(w in q_lower for w in ["why", "how", "cause", "happen", "what", "tell", "explain"]):
            q_type = "Education"
        else:
            q_type = "General"

        return {
            "answer": answer,
            "question_type": q_type,
            "location_detected": detect_location(question),
        }

    except Exception as e:
        logger.warning(f"Groq API error: {e}")
        return {
            "answer": "AI Assistant unavailable. Check API configuration.",
            "question_type": "Info",
            "is_unavailable": True,
            "location_detected": detect_location(question),
        }


# ── Legacy API (backward compatible with streamlit_app.py) ──────────
def get_chat_response(user_message, chat_history=None, district=None,
                      risk_score=None, risk_level=None):
    """Legacy string-returning API used by streamlit_app.py."""
    result = get_chatbot_response(question=user_message, district=district, risk_score=risk_score)
    if isinstance(result, dict):
        return result.get("answer", "")
    return result


def generate_risk_explanation(district, risk_score, shap_values, weather_data):
    pct = _risk_percent(risk_score)
    level = _risk_level_from_score(risk_score)
    return get_chat_response(
        f"Create a 3-sentence flood risk summary for {district}. "
        f"Risk: {pct}% ({level}). Drivers: {shap_values}. Weather: {weather_data}.",
        district=district, risk_score=risk_score,
    )


def get_safety_tips(risk_level, district):
    score = 0.8 if str(risk_level).upper() == "HIGH" else 0.45 if str(risk_level).upper() == "MODERATE" else 0.15
    return get_chat_response(
        f"Give 5 short flood safety tips for {district}. Risk level is {risk_level}.",
        district=district, risk_score=score,
    )
