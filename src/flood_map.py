"""
Interactive Folium flood risk map for Assam districts.

Creates a choropleth-style map with circle markers colored from
green (low risk) to red (high risk), with tooltips and a legend.
"""

import os
import logging
import numpy as np
import pandas as pd
import folium
from folium import plugins
import joblib

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# District coordinates (lat, lon)
DISTRICT_COORDS = {
    "Guwahati":  (26.1445, 91.7362),
    "Jorhat":    (26.7509, 94.2037),
    "Dibrugarh": (27.4728, 94.9120),
    "Silchar":   (24.8333, 92.7789),
    "Tezpur":    (26.6338, 92.8000),
    "Nagaon":    (26.3500, 92.6800),
    "Dhubri":    (26.0200, 89.9800),
    "Barpeta":   (26.3200, 91.0000),
    "Sivasagar": (26.9800, 94.6300),
    "Lakhimpur": (27.2400, 94.1000),
}


def risk_color(score: float) -> str:
    """
    Map a risk score (0-1) to a color from green to red.

    Args:
        score: Risk score between 0 and 1.

    Returns:
        Hex color string.
    """
    if score < 0.3:
        return "#2ecc71"  # green
    elif score < 0.5:
        return "#f1c40f"  # yellow
    elif score < 0.7:
        return "#e67e22"  # orange
    else:
        return "#e74c3c"  # red


def risk_level(score: float) -> str:
    """Map risk score to human-readable level."""
    if score < 0.3:
        return "🟢 Low Risk"
    elif score < 0.5:
        return "🟡 Moderate Risk"
    elif score < 0.7:
        return "🟠 High Risk"
    else:
        return "🔴 Very High Risk"


def compute_district_risk() -> dict:
    """
    Compute flood risk scores per district from sample data.

    Uses historical flood frequency and recent conditions to
    estimate current risk. Falls back to random scores if data unavailable.

    Returns:
        Dict mapping district names to risk scores (0-1).
    """
    try:
        data_path = os.path.join(PROJECT_ROOT, "data", "sample_data.csv")
        df = pd.read_csv(data_path, parse_dates=["date"])

        # Use last year's monsoon data for risk calculation
        recent = df[df["date"] >= df["date"].max() - pd.Timedelta(days=365)]

        risk_scores = {}
        for district in DISTRICT_COORDS:
            d = recent[recent["district"] == district]
            if len(d) == 0:
                risk_scores[district] = 0.5
                continue

            # Risk based on: flood frequency, avg rainfall, avg river level
            flood_freq = d["flood_occurred"].mean()
            rain_factor = min(d["rainfall_mm"].mean() / 100, 1.0)
            river_factor = min(d["river_level_m"].mean() / 10, 1.0)

            score = 0.5 * flood_freq + 0.25 * rain_factor + 0.25 * river_factor
            risk_scores[district] = round(min(max(score, 0), 1), 3)

        return risk_scores

    except Exception as e:
        logger.warning(f"Could not compute risk from data: {e}. Using defaults.")
        np.random.seed(42)
        return {d: round(np.random.uniform(0.1, 0.9), 3) for d in DISTRICT_COORDS}


def create_flood_map(output_path: str = None) -> folium.Map:
    """
    Create an interactive Folium map of Assam flood risk.

    Args:
        output_path: Path to save the HTML map. Defaults to project root.

    Returns:
        Folium Map object.
    """
    logger.info("Creating flood risk map...")

    risk_scores = compute_district_risk()

    # Center map on Assam
    m = folium.Map(
        location=[26.2006, 92.9376],
        zoom_start=7,
        tiles="CartoDB positron",
    )

    # Title
    title_html = """
    <div style="position: fixed; top: 10px; left: 50%; transform: translateX(-50%);
                z-index: 1000; background: rgba(0,0,0,0.8); color: white;
                padding: 12px 24px; border-radius: 8px; font-size: 18px;
                font-family: Arial, sans-serif; font-weight: bold;">
        🌊 Assam Flood Risk Map — Brahmaputra Basin
    </div>
    """
    m.get_root().html.add_child(folium.Element(title_html))

    # Add district markers
    for district, (lat, lon) in DISTRICT_COORDS.items():
        score = risk_scores.get(district, 0.5)
        color = risk_color(score)
        level = risk_level(score)

        tooltip_text = (
            f"<b>{district}</b><br>"
            f"Risk Score: <b>{score:.1%}</b><br>"
            f"Level: {level}"
        )

        folium.CircleMarker(
            location=[lat, lon],
            radius=20 + score * 30,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.7,
            popup=folium.Popup(tooltip_text, max_width=200),
            tooltip=tooltip_text,
        ).add_to(m)

        # Add district label
        folium.Marker(
            location=[lat, lon],
            icon=folium.DivIcon(
                html=f'<div style="font-size:11px; font-weight:bold; '
                     f'color:#333; text-align:center;">{district}</div>',
                icon_size=(80, 20),
                icon_anchor=(40, -15),
            ),
        ).add_to(m)

    # Legend
    legend_html = """
    <div style="position: fixed; bottom: 30px; right: 30px; z-index: 1000;
                background: white; padding: 15px; border-radius: 8px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.3); font-family: Arial;">
        <h4 style="margin:0 0 8px 0;">Risk Level</h4>
        <p style="margin:3px 0;"><span style="color:#2ecc71;">●</span> Low (&lt;30%)</p>
        <p style="margin:3px 0;"><span style="color:#f1c40f;">●</span> Moderate (30-50%)</p>
        <p style="margin:3px 0;"><span style="color:#e67e22;">●</span> High (50-70%)</p>
        <p style="margin:3px 0;"><span style="color:#e74c3c;">●</span> Very High (&gt;70%)</p>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    # Save
    if output_path is None:
        output_path = os.path.join(PROJECT_ROOT, "flood_risk_map.html")
    m.save(output_path)
    logger.info(f"Flood risk map saved to {output_path}")

    return m


if __name__ == "__main__":
    create_flood_map()
