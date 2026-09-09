"""
OpenWeatherMap integration for India district weather.

The free OpenWeatherMap endpoints provide current weather and a 5-day
forecast in 3-hour intervals. Forecast data is aggregated into daily rainfall
totals and returned as up to 7 calendar days when the API supplies them.
"""

import logging
import math
import os
import sys
from datetime import datetime

import pandas as pd
import requests

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

try:
    from config import DISTRICT_COORDS, OPENWEATHERMAP_API_KEY
except ImportError:
    DISTRICT_COORDS = {}
    OPENWEATHERMAP_API_KEY = "your_api_key_here"

logger = logging.getLogger(__name__)

BASE_URL = "https://api.openweathermap.org/data/2.5"
RUNTIME_OPENWEATHERMAP_API_KEY = None

BASE_RIVER_LEVELS = {
    "Guwahati": 4.5,
    "Jorhat": 4.0,
    "Dibrugarh": 3.8,
    "Silchar": 5.2,
    "Tezpur": 4.2,
    "Nagaon": 4.8,
    "Dhubri": 5.5,
    "Barpeta": 5.0,
    "Sivasagar": 3.5,
    "Lakhimpur": 3.8,
}


def _finite_float(value, default: float = 0.0) -> float:
    """Coerce an externally supplied value to a finite float."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _nonnegative_float(value, default: float = 0.0) -> float:
    """Coerce externally supplied measurements to a usable non-negative float."""
    return max(_finite_float(value, default), 0.0)


def calculate_daily_flood_risk(rainfall_mm, state):
    """Rule-based daily forecast risk using IMD daily rainfall thresholds."""
    rainfall_mm = _nonnegative_float(rainfall_mm)

    if rainfall_mm < 7.5:
        base_risk = 5
    elif rainfall_mm < 35.5:
        base_risk = 20
    elif rainfall_mm < 64.5:
        base_risk = 45
    elif rainfall_mm < 115.5:
        base_risk = 65
    else:
        base_risk = 85

    flood_prone = {
        "Assam": 1.4,
        "Bihar": 1.3,
        "West Bengal": 1.3,
        "Uttar Pradesh": 1.2,
        "Odisha": 1.2,
        "Kerala": 1.2,
        "Andhra Pradesh": 1.1,
        "Maharashtra": 1.1,
        "Jharkhand": 1.1,
        "Tripura": 1.1,
        "Manipur": 1.1,
        "Meghalaya": 1.2,
        "Arunachal Pradesh": 1.2,
        "Nagaland": 1.1,
        "Gujarat": 1.1,
        "Rajasthan": 0.9,
        "Himachal Pradesh": 1.1,
        "Uttarakhand": 1.1,
    }
    state_key = str(state or "").strip().title()
    multiplier = flood_prone.get(state_key, 1.0)
    return min(round(base_risk * multiplier, 1), 95)


def get_daily_risk_level(risk_pct):
    risk_pct = _nonnegative_float(risk_pct)
    if risk_pct < 20:
        return "Low"
    if risk_pct < 40:
        return "Moderate"
    if risk_pct <= 60:
        return "High"
    return "Severe"

def _load_india_district_coords() -> dict:
    path = os.path.join(PROJECT_ROOT, "data", "india_districts.csv")
    if not os.path.exists(path):
        return {}
    try:
        df = pd.read_csv(path)
        coords = {}
        for _, row in df.dropna(subset=["district", "state", "lat", "lon"]).iterrows():
            lat_lon = (float(row["lat"]), float(row["lon"]))
            coords[str(row["district"])] = lat_lon
            coords[f"{row['district']}, {row['state']}"] = lat_lon
        return coords
    except Exception as exc:
        logger.warning("Could not load India district coordinates: %s", exc)
        return {}


DISTRICT_COORDS = {**DISTRICT_COORDS, **_load_india_district_coords()}


def _find_district_coords(district_name: str):
    """Resolve a district name or district,state pair to lat/lon coordinates."""
    if not district_name or not isinstance(district_name, str):
        return None

    normalized = district_name.strip()
    if not normalized:
        return None

    if normalized in DISTRICT_COORDS:
        return DISTRICT_COORDS[normalized]

    normalized_casefold = normalized.casefold()
    for known_name, coordinates in DISTRICT_COORDS.items():
        if known_name.casefold() == normalized_casefold:
            return coordinates

    parts = [part.strip() for part in normalized.split(",") if part.strip()]
    if len(parts) >= 2:
        exact = f"{parts[0]}, {parts[1]}"
        if exact in DISTRICT_COORDS:
            return DISTRICT_COORDS[exact]
        if parts[0] in DISTRICT_COORDS:
            return DISTRICT_COORDS[parts[0]]

    title_normalized = normalized.title()
    if title_normalized in DISTRICT_COORDS:
        return DISTRICT_COORDS[title_normalized]

    if len(parts) >= 2:
        exact_title = f"{parts[0].title()}, {parts[1].title()}"
        if exact_title in DISTRICT_COORDS:
            return DISTRICT_COORDS[exact_title]
        if parts[0].title() in DISTRICT_COORDS:
            return DISTRICT_COORDS[parts[0].title()]

    return None


def set_openweathermap_api_key(api_key: str | None):
    """Set an API key at runtime from the Streamlit UI."""
    global RUNTIME_OPENWEATHERMAP_API_KEY
    RUNTIME_OPENWEATHERMAP_API_KEY = api_key.strip() if api_key else None


def _get_api_key():
    """Read the newest available API key without requiring an app restart."""
    return (
        RUNTIME_OPENWEATHERMAP_API_KEY
        or os.getenv("OPENWEATHERMAP_API_KEY")
        or OPENWEATHERMAP_API_KEY
    )


def has_openweathermap_api_key():
    """Return True when a non-placeholder OpenWeatherMap key is configured."""
    api_key = _get_api_key()
    return bool(api_key and api_key != "your_api_key_here")


def _api_key_error():
    return {
        "error": (
            "OpenWeatherMap is not connected yet. Paste your API key in the sidebar "
            "or add OPENWEATHERMAP_API_KEY to .env."
        )
    }


def _request_openweather(endpoint: str, district_name: str) -> dict:
    coords = _find_district_coords(district_name)
    if not coords:
        raise ValueError(f"Unknown district: {district_name}")
    api_key = _get_api_key()
    if not api_key or api_key == "your_api_key_here":
        raise PermissionError(_api_key_error()["error"])

    lat, lon = coords
    response = requests.get(
        f"{BASE_URL}/{endpoint}",
        params={
            "lat": lat,
            "lon": lon,
            "appid": api_key,
            "units": "metric",
        },
        timeout=10,
    )
    response.raise_for_status()
    return response.json()


def get_current_weather(district_name: str) -> dict:
    """
    Fetch real-time weather for an Indian district.

    Returns rainfall_mm, rainfall, temperature_c, humidity_pct, wind_speed,
    weather_description, weather_icon, weather_main, and fetched_at.
    """
    try:
        data = _request_openweather("weather", district_name)
        rain = data.get("rain", {})
        rainfall_mm = rain.get("1h", rain.get("3h", 0.0))
        weather = data.get("weather", [{}])[0]

        return {
            "rainfall_mm": round(float(rainfall_mm), 1),
            "rainfall": round(float(rainfall_mm), 1),
            "temperature_c": round(float(data["main"]["temp"]), 1),
            "humidity_pct": int(data["main"]["humidity"]),
            "wind_speed": round(float(data.get("wind", {}).get("speed", 0.0)) * 3.6, 1),
            "weather_description": weather.get("description", "Unknown").title(),
            "weather_icon": weather.get("icon", "01d"),
            "weather_main": weather.get("main", "Unknown"),
            "feels_like": round(float(data["main"].get("feels_like", data["main"]["temp"])), 1),
            "pressure": data["main"].get("pressure"),
            "fetched_at": datetime.now().isoformat(timespec="seconds"),
            "error": None,
        }
    except PermissionError as exc:
        return {"error": str(exc)}
    except ValueError as exc:
        return {"error": str(exc)}
    except requests.exceptions.HTTPError as exc:
        status_code = exc.response.status_code if exc.response is not None else None
        if status_code == 401:
            return {
                "error": (
                    "OpenWeatherMap rejected this API key. Check that your email is verified, "
                    "the key was copied correctly, and wait 10-30 minutes if it is new."
                )
            }
        logger.error("OpenWeatherMap current weather failed for %s: %s", district_name, exc)
        return {"error": "Could not fetch live weather right now. Manual input is still available."}
    except requests.exceptions.RequestException as exc:
        logger.error("OpenWeatherMap current weather failed for %s: %s", district_name, exc)
        return {"error": "Could not fetch live weather right now. Manual input is still available."}
    except (KeyError, TypeError, ValueError) as exc:
        logger.error("OpenWeatherMap current weather parse failed: %s", exc)
        return {"error": "Live weather response was incomplete. Manual input is still available."}


def get_rainfall_forecast(district_name: str) -> list:
    """
    Get 7-day rainfall forecast from keyless Open-Meteo API.
    
    Returns a list of daily summaries:
    {date, date_display, rainfall_mm, rainfall, precipitation_mm, description, flood_probability, flood_probability_pct}
    """
    coords = _find_district_coords(district_name)
    if not coords:
        logger.warning(f"Could not find coordinates for {district_name}")
        return []

    lat, lon = coords
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": "precipitation_sum,precipitation_probability_max,weathercode,temperature_2m_max,temperature_2m_min",
        "timezone": "Asia/Kolkata",
        "forecast_days": 7,
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        forecast_data = response.json()
        
        daily = forecast_data.get("daily", {})
        times = daily.get("time", [])
        precipitation_sums = daily.get("precipitation_sum", [])
        precip_prob_max = daily.get("precipitation_probability_max", [])
        weathercodes = daily.get("weathercode", [])
        temp_maxs = daily.get("temperature_2m_max", [])
        temp_mins = daily.get("temperature_2m_min", [])
        
        forecast = []
        for i in range(min(7, len(times))):
            date_str = times[i]
            dt = datetime.strptime(date_str, "%Y-%m-%d")
            date_display = dt.strftime("%b %d")
            
            # Get precip probability
            precip_prob = float(precip_prob_max[i]) if i < len(precip_prob_max) and precip_prob_max[i] is not None else 0.0
            
            # Calculate flood probability
            flood_prob = min(precip_prob * 1.2, 100.0)
            
            # Decode weathercode
            wmo_code = int(weathercodes[i]) if i < len(weathercodes) and weathercodes[i] is not None else 0
            
            if wmo_code == 0:
                precip_mm, desc = 0.0, "Clear Sky ☀️"
            elif wmo_code in (1, 2, 3):
                precip_mm, desc = 0.0, "Partly Cloudy ⛅"
            elif wmo_code in (45, 48):
                precip_mm, desc = 0.0, "Foggy 🌫️"
            elif wmo_code in (51, 53, 55):
                precip_mm, desc = 6.0, "Drizzle 🌦️"
            elif wmo_code in (61, 63, 65):
                precip_mm, desc = 25.0, "Rainy 🌧️"
            elif wmo_code in (66, 67):
                precip_mm, desc = 25.0, "Freezing Rain 🌧️"
            elif wmo_code in (71, 73, 75, 77):
                precip_mm, desc = 10.0, "Snowy ❄️"
            elif wmo_code in (80, 81, 82):
                precip_mm, desc = 25.0, "Rain Showers 🌧️"
            elif wmo_code in (85, 86):
                precip_mm, desc = 10.0, "Snow Showers ❄️"
            elif wmo_code in (95, 96, 99):
                precip_mm, desc = 50.0, "Thunderstorm ⛈️"
            else:
                precip_mm, desc = 0.0, "Cloudy ☁️"

            if i < len(precipitation_sums):
                precip_mm = _nonnegative_float(precipitation_sums[i], precip_mm)
                
            t_max = _finite_float(temp_maxs[i], 28.0) if i < len(temp_maxs) else 28.0
            t_min = _finite_float(temp_mins[i], 20.0) if i < len(temp_mins) else 20.0
            temp_avg = round((t_max + t_min) / 2.0, 1)
            
            forecast.append({
                "date": date_str,
                "date_display": date_display,
                "rainfall_mm": precip_mm,
                "rainfall": precip_mm,
                "precipitation_mm": precip_mm,
                "description": desc,
                "temp_avg": temp_avg,
                "flood_probability": round(flood_prob / 100.0, 3),
                "flood_probability_pct": round(flood_prob, 1),
                "risk_probability": round(flood_prob, 1),
                "risk_level": get_daily_risk_level(flood_prob),
            })
            
        return forecast
    except Exception as exc:
        logger.error(f"Open-Meteo forecast failed: {exc}")
        return []


def get_river_level_estimate(rainfall_mm: float, district_name: str) -> float:
    """
    Estimate river level from rainfall on a 0-10 scale.

    Free weather APIs do not include river gauge observations, so this gives a
    simple district-specific estimate for app input only.
    """
    base = BASE_RIVER_LEVELS.get(district_name, 4.5)
    month = datetime.now().month
    seasonal_factor = 1.25 if month in [6, 7, 8, 9] else 0.9 if month in [11, 12, 1, 2] else 1.0

    rainfall_mm = _nonnegative_float(rainfall_mm)
    if rainfall_mm <= 10:
        rain_effect = rainfall_mm * 0.04
    elif rainfall_mm <= 50:
        rain_effect = 0.4 + (rainfall_mm - 10) * 0.06
    elif rainfall_mm <= 150:
        rain_effect = 2.8 + (rainfall_mm - 50) * 0.025
    else:
        rain_effect = 5.3 + (rainfall_mm - 150) * 0.01

    level = base * seasonal_factor + rain_effect
    return round(min(max(level, 0.0), 10.0), 1)


def get_weather_emoji(weather_main: str) -> str:
    """Map OpenWeatherMap weather groups to a compact display emoji."""
    mapping = {
        "Clear": "☀️",
        "Clouds": "☁️",
        "Rain": "🌧️",
        "Drizzle": "🌦️",
        "Thunderstorm": "⛈️",
        "Snow": "❄️",
        "Mist": "🌫️",
        "Fog": "🌫️",
        "Haze": "🌫️",
        "Smoke": "🌫️",
    }
    return mapping.get(weather_main, "🌤️")


if __name__ == "__main__":
    for district in ["Guwahati", "Dhubri"]:
        print(f"\n=== {district} ===")
        print(get_current_weather(district))
        print(get_rainfall_forecast(district))
        print(f"River estimate for 50mm rain: {get_river_level_estimate(50, district)}")
