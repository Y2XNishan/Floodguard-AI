"""
Generate synthetic but realistic flood data for Assam, India (Brahmaputra basin).

This script creates 5 years (2019-2024) of daily data for 10 Assam districts
with realistic monsoon patterns, river levels, and flood occurrences (~15%).
"""

import pandas as pd
import numpy as np
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# District metadata — elevation & baseline river levels vary by district
# -------------------------------------------------------------------
DISTRICTS = {
    "Guwahati":   {"lat": 26.1445, "lon": 91.7362, "elevation": 55,  "base_river": 5.0},
    "Jorhat":     {"lat": 26.7509, "lon": 94.2037, "elevation": 86,  "base_river": 4.5},
    "Dibrugarh":  {"lat": 27.4728, "lon": 94.9120, "elevation": 108, "base_river": 4.8},
    "Silchar":    {"lat": 24.8333, "lon": 92.7789, "elevation": 22,  "base_river": 5.5},
    "Tezpur":     {"lat": 26.6338, "lon": 92.8000, "elevation": 48,  "base_river": 5.2},
    "Nagaon":     {"lat": 26.3500, "lon": 92.6800, "elevation": 60,  "base_river": 4.9},
    "Dhubri":     {"lat": 26.0200, "lon": 89.9800, "elevation": 28,  "base_river": 6.0},
    "Barpeta":    {"lat": 26.3200, "lon": 91.0000, "elevation": 35,  "base_river": 5.8},
    "Sivasagar":  {"lat": 26.9800, "lon": 94.6300, "elevation": 95,  "base_river": 4.3},
    "Lakhimpur":  {"lat": 27.2400, "lon": 94.1000, "elevation": 102, "base_river": 4.6},
}


def generate_sample_data(seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic daily flood-related data for 10 Assam districts
    over 5 years (2019-01-01 to 2024-12-31).

    The data follows realistic patterns:
      - Monsoon (Jun–Sep) has significantly higher rainfall
      - River levels correlate with recent rainfall
      - Flood probability increases with high rainfall + river level
      - ~15% of records are flood events
      - Flood severity is distributed: mild > moderate > severe

    Args:
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with all required columns, sorted by date and district.
    """
    np.random.seed(seed)
    logger.info("Generating synthetic flood data for Assam districts...")

    date_range = pd.date_range(start="2019-01-01", end="2024-12-31", freq="D")
    records = []

    for district_name, meta in DISTRICTS.items():
        logger.info(f"  Generating data for {district_name}...")
        days_since_flood = 180  # start with a long gap

        for date in date_range:
            month = date.month
            is_monsoon = 1 if month in [6, 7, 8, 9] else 0

            # ---- Rainfall (mm) ----
            if is_monsoon:
                # Monsoon: heavy rainfall with occasional extreme events
                base_rain = np.random.gamma(shape=3.0, scale=12.0)
                # Add occasional extreme events
                if np.random.random() < 0.08:
                    base_rain += np.random.uniform(80, 200)
            else:
                # Dry season: mostly light rain
                base_rain = np.random.exponential(scale=3.0)
                if np.random.random() < 0.7:
                    base_rain = 0.0  # many dry days

            # District-specific modifier (lower elevation = more rain effect)
            elev_factor = 1.0 + (100 - meta["elevation"]) / 500.0
            rainfall_mm = round(max(0, base_rain * elev_factor), 1)

            # ---- River Level (m) ----
            # Base level + monsoon surge + noise
            monsoon_surge = 2.5 * is_monsoon + 0.8 * np.sin(2 * np.pi * (month - 6) / 12)
            rain_effect = rainfall_mm * 0.015
            river_level_m = round(
                meta["base_river"] + monsoon_surge + rain_effect + np.random.normal(0, 0.3), 2
            )
            river_level_m = max(1.0, river_level_m)

            # ---- Temperature (°C) ----
            if is_monsoon:
                temperature_c = round(np.random.normal(30, 2.5), 1)
            else:
                if month in [12, 1, 2]:
                    temperature_c = round(np.random.normal(18, 3), 1)
                else:
                    temperature_c = round(np.random.normal(27, 3), 1)

            # ---- Humidity (%) ----
            humidity_pct = round(
                np.clip(np.random.normal(80 if is_monsoon else 55, 10), 20, 100), 1
            )

            # ---- Elevation (m) — constant per district with tiny noise ----
            elevation_m = meta["elevation"] + round(np.random.normal(0, 0.5), 1)

            # ---- NDVI (vegetation index, 0-1) ----
            ndvi = round(
                np.clip(np.random.normal(0.65 if is_monsoon else 0.40, 0.1), 0.05, 0.95), 3
            )

            # ---- Soil Moisture (0-1) ----
            soil_moisture = round(
                np.clip(
                    0.3 + is_monsoon * 0.25 + rainfall_mm * 0.002 + np.random.normal(0, 0.08),
                    0.05, 0.95
                ), 3
            )

            # ---- Flood Occurrence ----
            # Probability increases with rainfall, river level, soil moisture
            flood_prob = (
                0.05
                + 0.003 * max(0, rainfall_mm - 10)
                + 0.05 * max(0, river_level_m - 5.5)
                + 0.18 * max(0, soil_moisture - 0.50)
                + 0.08 * is_monsoon
                - 0.0008 * meta["elevation"]
            )
            flood_prob = np.clip(flood_prob, 0.02, 0.85)
            flood_occurred = int(np.random.random() < flood_prob)

            # ---- Days since last flood ----
            if flood_occurred:
                days_since_flood = 0
            else:
                days_since_flood += 1

            # ---- Flood Severity ----
            if flood_occurred:
                sev_roll = np.random.random()
                if rainfall_mm > 120 and river_level_m > 9:
                    # Extreme conditions → likely severe
                    flood_severity = 3 if sev_roll < 0.5 else (2 if sev_roll < 0.85 else 1)
                elif rainfall_mm > 60 or river_level_m > 7.5:
                    flood_severity = 2 if sev_roll < 0.4 else (1 if sev_roll < 0.8 else 3)
                else:
                    flood_severity = 1 if sev_roll < 0.6 else (2 if sev_roll < 0.9 else 3)
            else:
                flood_severity = 0

            records.append({
                "date": date,
                "district": district_name,
                "rainfall_mm": rainfall_mm,
                "river_level_m": river_level_m,
                "temperature_c": temperature_c,
                "humidity_pct": humidity_pct,
                "elevation_m": elevation_m,
                "ndvi": ndvi,
                "soil_moisture": soil_moisture,
                "days_since_last_flood": days_since_flood,
                "flood_occurred": flood_occurred,
                "flood_severity": flood_severity,
            })

    df = pd.DataFrame(records)
    df = df.sort_values(["date", "district"]).reset_index(drop=True)

    # ---- Inject ~2% missing values to make data realistic ----
    n_missing = int(len(df) * 0.02)
    cols_to_miss = ["rainfall_mm", "river_level_m", "temperature_c", "humidity_pct",
                    "ndvi", "soil_moisture"]
    for col in cols_to_miss:
        miss_idx = np.random.choice(df.index, size=n_missing // len(cols_to_miss), replace=False)
        df.loc[miss_idx, col] = np.nan

    logger.info(f"Generated {len(df)} records across {df['district'].nunique()} districts")
    logger.info(f"Flood rate: {df['flood_occurred'].mean():.2%}")
    logger.info(f"Date range: {df['date'].min()} to {df['date'].max()}")
    logger.info(f"Missing values:\n{df.isnull().sum()}")

    return df


def main():
    """Generate and save the sample dataset."""
    try:
        df = generate_sample_data()

        # Save to CSV
        output_dir = os.path.dirname(os.path.abspath(__file__))
        output_path = os.path.join(output_dir, "sample_data.csv")
        df.to_csv(output_path, index=False)
        logger.info(f"Sample data saved to {output_path}")

        # Print summary statistics
        print("\n" + "=" * 60)
        print("DATASET SUMMARY")
        print("=" * 60)
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(f"\nFlood occurrence distribution:")
        print(df["flood_occurred"].value_counts(normalize=True))
        print(f"\nFlood severity distribution (among floods):")
        print(df[df["flood_occurred"] == 1]["flood_severity"].value_counts(normalize=True))
        print(f"\nDistrict counts:")
        print(df["district"].value_counts())

    except Exception as e:
        logger.error(f"Failed to generate sample data: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
