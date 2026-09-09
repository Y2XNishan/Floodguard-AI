"""
Offline caching utilities for FloodGuard AI.

Enables the app to work without internet by caching the last successful
prediction, forecast, and weather data locally. Designed for rural India
where connectivity is unreliable.
"""

import json
import os
import socket
import tempfile
from datetime import datetime
from pathlib import Path

CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "offline_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

CACHE_FILES = {
    "prediction": CACHE_DIR / "last_prediction.json",
    "forecast": CACHE_DIR / "last_forecast.json",
    "weather": CACHE_DIR / "last_weather.json",
    "map_data": CACHE_DIR / "last_map_data.json",
    "crop_data": CACHE_DIR / "last_crop_data.json",
}


def _cache_file_for(key: str) -> Path:
    """Resolve a supported cache key with a useful error for callers."""
    try:
        return CACHE_FILES[key]
    except KeyError as exc:
        raise ValueError(f"Unsupported cache key: {key}") from exc


def save_to_cache(key: str, data: dict):
    """Save data to local JSON cache file with timestamp."""
    try:
        cache_data = dict(data)
        cache_data["_cached_at"] = datetime.now().strftime("%d %b %Y, %I:%M %p")
        cache_data["_cache_version"] = "1.0"
        cache_file = _cache_file_for(key)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=cache_file.parent,
            delete=False,
        ) as file:
            json.dump(cache_data, file, ensure_ascii=False, indent=2)
            temp_path = Path(file.name)
        temp_path.replace(cache_file)
        return True
    except Exception as e:
        print(f"Cache save failed for {key}: {e}")
        return False


def load_from_cache(key: str) -> dict | None:
    """Load cached data if it exists."""
    try:
        cache_file = _cache_file_for(key)
        if cache_file.exists():
            with open(cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data if isinstance(data, dict) else None
        return None
    except Exception as e:
        print(f"Cache load failed for {key}: {e}")
        return None


def is_cache_available(key: str) -> bool:
    """Check if cache file exists and is not empty."""
    try:
        cache_file = _cache_file_for(key)
        return cache_file.exists() and cache_file.stat().st_size > 0
    except Exception:
        return False


def get_cache_age(key: str) -> str:
    """Return human-readable age of cache file."""
    try:
        cache_file = _cache_file_for(key)
        if cache_file.exists():
            mtime = datetime.fromtimestamp(cache_file.stat().st_mtime)
            diff = datetime.now() - mtime
            total_seconds = max(int(diff.total_seconds()), 0)
            hours, remainder = divmod(total_seconds, 3600)
            minutes = remainder // 60
            if hours > 0:
                return f"{hours} hour{'s' if hours > 1 else ''} ago"
            return f"{minutes} minute{'s' if minutes > 1 else ''} ago"
        return "No cache"
    except Exception:
        return "Unknown"


def clear_all_cache():
    """Delete all cache files."""
    cleared = 0
    for path in CACHE_FILES.values():
        try:
            if path.exists():
                path.unlink()
                cleared += 1
        except Exception:
            pass
    return cleared


def is_internet_available() -> bool:
    """Check internet connectivity by pinging a reliable server."""
    try:
        with socket.create_connection(("8.8.8.8", 53), timeout=3):
            pass
        return True
    except Exception:
        return False


# ----- Rural India specific features -----

HIGH_RISK_DISTRICTS = [
    "Patna", "Muzaffarpur", "Darbhanga", "Sitamarhi",
    "Guwahati", "Dibrugarh", "Lakhimpur", "Dhemaji",
    "Puri", "Kendrapara", "Jagatsinghpur", "Bhadrak",
    "Varanasi", "Gorakhpur", "Bahraich", "Lakhimpur Kheri",
    "Murshidabad", "Malda", "North 24 Parganas", "Hooghly",
    "East Singhbum", "Sahibganj", "Godda", "Pakur",
]


def get_high_risk_districts():
    """Return list of high-risk districts for pre-caching."""
    return list(HIGH_RISK_DISTRICTS)


def should_skip_map(is_online: bool) -> bool:
    """Skip map rendering in offline mode to save memory."""
    return not is_online
