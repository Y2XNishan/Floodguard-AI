"""Crop loss estimation helpers for FloodGuard AI."""

from __future__ import annotations

from typing import Any

import plotly.graph_objects as go


def get_crop_data() -> dict[str, list[str]]:
    """Return major crops grown in selected Indian states."""
    return {
        "Assam": ["Rice", "Tea", "Jute", "Mustard", "Vegetables", "Sugarcane"],
        "Bihar": ["Rice", "Wheat", "Maize", "Sugarcane", "Vegetables", "Potato"],
        "Kerala": ["Rice", "Coconut", "Banana", "Rubber", "Spices", "Vegetables"],
        "Maharashtra": ["Rice", "Sugarcane", "Cotton", "Soybean", "Onion", "Wheat"],
        "West Bengal": ["Rice", "Jute", "Potato", "Vegetables", "Tea", "Maize"],
        "Uttar Pradesh": ["Wheat", "Rice", "Sugarcane", "Potato", "Vegetables", "Maize"],
        "Odisha": ["Rice", "Jute", "Groundnut", "Vegetables", "Pulses", "Maize"],
        "Andhra Pradesh": ["Rice", "Sugarcane", "Cotton", "Groundnut", "Chilli", "Maize"],
        "Tamil Nadu": ["Rice", "Sugarcane", "Banana", "Coconut", "Vegetables", "Maize"],
        "Karnataka": ["Rice", "Sugarcane", "Cotton", "Groundnut", "Ragi", "Maize"],
        "Punjab": ["Wheat", "Rice", "Maize", "Sugarcane", "Cotton", "Potato"],
        "Madhya Pradesh": ["Wheat", "Rice", "Soybean", "Cotton", "Pulses", "Maize"],
        "Gujarat": ["Cotton", "Groundnut", "Sugarcane", "Wheat", "Rice", "Vegetables"],
        "Rajasthan": ["Wheat", "Bajra", "Mustard", "Pulses", "Cotton", "Maize"],
        "Jharkhand": ["Rice", "Maize", "Pulses", "Vegetables", "Oilseeds", "Wheat"],
        "Chhattisgarh": ["Rice", "Maize", "Pulses", "Oilseeds", "Vegetables", "Wheat"],
        "Uttarakhand": ["Rice", "Wheat", "Maize", "Vegetables", "Pulses", "Soybean"],
        "Himachal Pradesh": ["Wheat", "Maize", "Rice", "Vegetables", "Apple", "Potato"],
        "Telangana": ["Rice", "Cotton", "Maize", "Sugarcane", "Chilli", "Pulses"],
        "Haryana": ["Wheat", "Rice", "Sugarcane", "Cotton", "Maize", "Potato"],
    }


def get_crop_prices() -> dict[str, int]:
    """Return MSP or market price per quintal in INR."""
    return {
        "Rice": 2300,
        "Wheat": 2275,
        "Maize": 2090,
        "Sugarcane": 315,
        "Cotton": 6620,
        "Jute": 5050,
        "Tea": 18000,
        "Rubber": 20000,
        "Coconut": 3200,
        "Banana": 2000,
        "Potato": 1200,
        "Onion": 2000,
        "Tomato": 2500,
        "Vegetables": 2000,
        "Mustard": 5650,
        "Groundnut": 6377,
        "Soybean": 4892,
        "Pulses": 7755,
        "Bajra": 2500,
        "Ragi": 3846,
        "Oilseeds": 5440,
        "Spices": 25000,
        "Chilli": 8000,
        "Apple": 12000,
    }


def get_yield_per_hectare() -> dict[str, int]:
    """Return average crop yield per hectare in quintals."""
    return {
        "Rice": 26,
        "Wheat": 32,
        "Maize": 28,
        "Sugarcane": 700,
        "Cotton": 13,
        "Jute": 22,
        "Tea": 18,
        "Rubber": 12,
        "Coconut": 55,
        "Banana": 250,
        "Potato": 130,
        "Onion": 110,
        "Tomato": 200,
        "Vegetables": 90,
        "Mustard": 13,
        "Groundnut": 16,
        "Soybean": 16,
        "Pulses": 9,
        "Bajra": 17,
        "Ragi": 19,
        "Oilseeds": 13,
        "Spices": 10,
        "Chilli": 15,
        "Apple": 100,
    }


def get_crop_vulnerability() -> dict[str, float]:
    """Return flood vulnerability factors where higher values are more vulnerable."""
    return {
        "Rice": 0.85,
        "Wheat": 0.75,
        "Maize": 0.80,
        "Sugarcane": 0.55,
        "Cotton": 0.80,
        "Jute": 0.30,
        "Tea": 0.45,
        "Rubber": 0.35,
        "Coconut": 0.25,
        "Banana": 0.75,
        "Potato": 0.95,
        "Onion": 0.90,
        "Tomato": 0.95,
        "Vegetables": 0.95,
        "Mustard": 0.70,
        "Groundnut": 0.70,
        "Soybean": 0.75,
        "Pulses": 0.80,
        "Bajra": 0.60,
        "Ragi": 0.65,
        "Oilseeds": 0.70,
        "Spices": 0.60,
        "Chilli": 0.85,
        "Apple": 0.50,
    }


def estimate_crop_loss(
    state: str,
    selected_crops: list[str],
    area_per_crop: dict[str, float],
    flood_probability: float,
    flood_duration_days: int,
    season: str,
) -> dict[str, Any]:
    """Estimate crop-wise and total flood losses."""
    crop_data = get_crop_data()
    prices = get_crop_prices()
    yields = get_yield_per_hectare()
    vulnerabilities = get_crop_vulnerability()

    if state not in crop_data:
        raise ValueError(f"Unsupported state: {state}")

    if flood_probability < 0.20:
        base_damage = 0.03
    elif flood_probability < 0.40:
        base_damage = 0.15
    elif flood_probability < 0.60:
        base_damage = 0.35
    elif flood_probability < 0.80:
        base_damage = 0.60
    else:
        base_damage = 0.85

    duration_mult = min(2.0, 1.0 + (flood_duration_days - 1) * 0.12)
    season_mult = {
        "Kharif": 1.2,
        "Rabi": 0.8,
        "Zaid": 1.0,
        "Whole Year": 1.1,
    }.get(season, 1.0)

    results = []
    total_loss_inr = 0
    total_area_ha = 0.0
    weighted_damage = 0.0

    for crop in selected_crops:
        if crop not in prices or crop not in yields or crop not in vulnerabilities:
            raise ValueError(f"Unsupported crop: {crop}")

        vulnerability = vulnerabilities[crop]
        damage_pct = min(0.95, base_damage * duration_mult * season_mult * vulnerability)
        yield_ha = yields[crop]
        price = prices[crop]
        area = float(area_per_crop[crop])

        normal_yield = yield_ha * area
        damaged_yield = normal_yield * damage_pct
        loss_inr = damaged_yield * price * 100

        total_loss_inr += round(loss_inr)
        total_area_ha += area
        weighted_damage += damage_pct * area

        results.append(
            {
                "crop": crop,
                "area_ha": area,
                "damage_pct": round(damage_pct * 100, 1),
                "normal_yield_q": round(normal_yield, 1),
                "damaged_yield_q": round(damaged_yield, 1),
                "loss_inr": round(loss_inr),
                "loss_lakhs": round(loss_inr / 100000, 2),
                "loss_crores": round(loss_inr / 10000000, 3),
            }
        )

    avg_damage_pct = (weighted_damage / total_area_ha * 100) if total_area_ha else 0.0
    worst_crop = max(results, key=lambda item: item["loss_inr"])["crop"] if results else None
    safest_crop = min(results, key=lambda item: item["damage_pct"])["crop"] if results else None

    return {
        "crops": results,
        "total_loss_inr": total_loss_inr,
        "total_loss_lakhs": total_loss_inr / 100000,
        "total_loss_crores": total_loss_inr / 10000000,
        "total_area_ha": total_area_ha,
        "avg_damage_pct": avg_damage_pct,
        "worst_crop": worst_crop,
        "safest_crop": safest_crop,
    }


def get_compensation_schemes() -> dict[str, dict[str, str]]:
    """Return government compensation and support scheme information."""
    return {
        "PMFBY": {
            "name": "PM Fasal Bima Yojana",
            "coverage": "Up to 100% of crop loss",
            "premium": "2% for Kharif, 1.5% for Rabi",
            "how_to_apply": "Contact nearest bank or CSC center",
            "helpline": "1800-180-1111",
            "website": "pmfby.gov.in",
        },
        "SDRF": {
            "name": "State Disaster Relief Fund",
            "coverage": "Rs 6,800 per hectare for crop loss",
            "how_to_apply": "Apply through District Collector office",
            "helpline": "1070",
            "website": "ndma.gov.in",
        },
        "NDRF": {
            "name": "National Disaster Response Fund",
            "coverage": "Additional support for major disasters",
            "how_to_apply": "Automatic for declared disasters",
            "helpline": "1078",
            "website": "ndma.gov.in",
        },
        "KCC": {
            "name": "Kisan Credit Card",
            "coverage": "Crop loan up to Rs 3 lakh at 4% interest",
            "how_to_apply": "Apply at any bank branch",
            "helpline": "1800-11-0001",
            "website": "nabard.org",
        },
    }


def plot_loss_chart(results: list[dict[str, Any]]) -> go.Figure:
    """Create a crop loss bar chart colored by damage severity."""
    crops = [item["crop"] for item in results]
    losses = [item["loss_lakhs"] for item in results]
    damages = [item["damage_pct"] for item in results]
    colors = [
        "#10b981" if damage < 20 else "#f59e0b" if damage <= 50 else "#ef4444"
        for damage in damages
    ]

    fig = go.Figure(
        go.Bar(
            x=crops,
            y=losses,
            marker_color=colors,
            text=[f"{loss:.2f}" for loss in losses],
            textposition="outside",
            hovertemplate=(
                "<b>%{x}</b><br>Loss: Rs %{y:.2f} lakhs"
                "<br>Damage: %{customdata:.1f}%<extra></extra>"
            ),
            customdata=damages,
        )
    )
    fig.update_layout(
        title="Estimated Crop Loss by Type",
        xaxis_title="Crop",
        yaxis_title="Loss in Lakhs (INR)",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#f1f5f9"),
        margin=dict(l=20, r=20, t=60, b=40),
    )
    fig.update_yaxes(rangemode="tozero")
    return fig
