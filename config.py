"""Configuration for the Flood Risk Prediction System."""

import os

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

# OpenWeatherMap API key. You can also set OPENWEATHERMAP_API_KEY in a .env file.
OPENWEATHERMAP_API_KEY = os.getenv("OPENWEATHERMAP_API_KEY", "your_api_key_here")

# Google Gemini API key. You can also set GEMINI_API_KEY in a .env file.
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "your_gemini_api_key_here")

# District coordinates (lat, lon) for API lookups.
DISTRICT_COORDS = {
    "Guwahati": (26.1445, 91.7362),
    "Jorhat": (26.7509, 94.2037),
    "Dibrugarh": (27.4728, 94.9120),
    "Silchar": (24.8333, 92.7789),
    "Tezpur": (26.6338, 92.8000),
    "Nagaon": (26.3500, 92.6833),
    "Dhubri": (26.0200, 89.9800),
    "Barpeta": (26.3200, 91.0000),
    "Sivasagar": (26.9800, 94.6400),
    "Lakhimpur": (27.2350, 94.1010),
    "Jamshedpur": (22.8046, 86.2029),
    "Chandigarh": (30.7333, 76.7794),
    "Dhanbad": (23.7957, 86.4304),
    "Ranchi": (23.3441, 85.3096),
    "Patna": (25.5941, 85.1376),
    "Kolkata": (22.5726, 88.3639),
    "Bhopal": (23.2599, 77.4126),
    "Indore": (22.7196, 75.8577),
    "Nagpur": (21.1458, 79.0882),
    "Pune": (18.5204, 73.8567),
    "Hyderabad": (17.3850, 78.4867),
    "Chennai": (13.0827, 80.2707),
    "Kochi": (9.9312, 76.2673),
    "Bengaluru": (12.9716, 77.5946),
    "Ahmedabad": (23.0225, 72.5714),
    "Jaipur": (26.9124, 75.7873),
    "Delhi": (28.6139, 77.2090),
    "Amritsar": (31.6340, 74.8723),
    "Dehradun": (30.3165, 78.0322),
    "Lucknow": (26.8467, 80.9462),
    "Varanasi": (25.3176, 82.9739),
    "Visakhapatnam": (17.6868, 83.2185),
    "Thiruvananthapuram": (8.5241, 76.9366),
    "Shillong": (25.5788, 91.8933),
    "Imphal": (24.8170, 93.9368),
    "Agartala": (23.8315, 91.2868),
    "Siliguri": (26.7271, 88.3953),
    "Bhubaneswar": (20.2961, 85.8245),
}
