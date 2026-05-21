"""Merge downloaded India flood datasets into a single master table."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
MERGED_PATH = PROCESSED_DIR / "india_flood_master.csv"
FLAGGED_PATH = PROCESSED_DIR / "flagged_records.csv"

STATION_DISTRICT_MAPPING = {
    "Dibrugarh": "Dibrugarh",
    "Guwahati": "Guwahati",
    "Patna": "Patna",
    "Varanasi": "Varanasi",
    "Prayagraj": "Prayagraj",
    "Mangaluru": "Mangalore",
}


def _normalize_name(value: str) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().lower().replace(".", "").replace("-", " ")


def _fill_rainfall_gaps(df: pd.DataFrame) -> pd.DataFrame:
    output = []
    for district, group in df.groupby([
        "district_norm",
        "state_norm",
    ]):
        group = group.sort_values("date").copy()
        group["rainfall_mm"] = group["rainfall_mm"].interpolate(method="linear", limit=5, limit_direction="both")
        output.append(group)
    return pd.concat(output, ignore_index=True)


def _copy_if_exists(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        logger.warning("Missing optional dataset: %s", path)
        return None
    try:
        return pd.read_csv(path, parse_dates=["date"], low_memory=False)
    except Exception as exc:
        logger.error("Failed to load %s: %s", path, exc)
        return None


def merge_all_datasets() -> Optional[Path]:
    """Merge all available raw datasets into one master India flood dataset."""
    imd_path = RAW_DIR / "imd_rainfall" / "imd_rainfall.csv"
    ndma_path = RAW_DIR / "ndma_flood_records" / "ndma_flood_records.csv"
    cwc_path = RAW_DIR / "cwc_river_levels" / "cwc_river_levels.csv"
    elev_path = RAW_DIR / "elevation" / "elevation.csv"

    if not imd_path.exists() or not ndma_path.exists():
        logger.error("IMD rainfall and NDMA flood records are required for merge.")
        return None

    rainfall = pd.read_csv(imd_path, parse_dates=["date"], low_memory=False)
    rainfall["district_norm"] = rainfall["district"].apply(_normalize_name)
    rainfall["state_norm"] = rainfall["state"].apply(_normalize_name)
    if "year" not in rainfall.columns:
        rainfall["year"] = rainfall["date"].dt.year

    ndma = pd.read_csv(ndma_path, low_memory=False)
    if "year" not in ndma.columns:
        if "date" in ndma.columns:
            ndma["date"] = pd.to_datetime(ndma["date"], errors="coerce")
            ndma["year"] = ndma["date"].dt.year
        else:
            logger.error("NDMA flood records must contain a year or date column.")
            return None
    ndma["district_norm"] = ndma["district"].apply(_normalize_name)
    ndma["state_norm"] = ndma["state"].apply(_normalize_name)
    ndma["flood_occurred"] = pd.to_numeric(ndma["flood_occurred"], errors="coerce").fillna(0).astype(int)
    ndma_summary = (
        ndma.groupby(["state_norm", "district_norm", "year"], as_index=False)
        .agg(
            flood_occurred=("flood_occurred", "max"),
            flood_severity=("flood_severity", "max"),
            area_affected_ha=("area_affected_ha", "sum"),
            people_affected=("people_affected", "sum"),
            crops_damaged=("crops_damaged", "sum"),
        )
    )

    merged = rainfall.merge(
        ndma_summary,
        how="left",
        on=["state_norm", "district_norm", "year"],
        validate="m:1",
    )
    merged["flood_occurred"] = merged["flood_occurred"].fillna(0).astype(int)
    merged["flood_severity"] = merged["flood_severity"].fillna(0).astype(int)

    river = _copy_if_exists(cwc_path)
    if river is not None:
        river["date"] = pd.to_datetime(river["date"], errors="coerce")
        river["station_norm"] = river["station_name"].apply(_normalize_name)
        river["district_norm"] = river["station_norm"].map(
            {k.lower(): v.lower() for k, v in STATION_DISTRICT_MAPPING.items()}
        )
        river["district_norm"] = river["district_norm"].fillna(river["district"].apply(_normalize_name))
        river["state_norm"] = river["state"].apply(_normalize_name)
        river = (
            river.groupby(["date", "state_norm", "district_norm"], as_index=False)
            .agg(
                water_level_m=("water_level_m", "mean"),
                danger_level_m=("danger_level_m", "mean"),
                warning_level_m=("warning_level_m", "mean"),
                discharge_cumecs=("discharge_cumecs", "mean"),
            )
        )
        merged = merged.merge(
            river,
            how="left",
            on=["date", "state_norm", "district_norm"],
            validate="m:1",
        )
        merged = merged.sort_values(["district_norm", "date"]).reset_index(drop=True)
        merged["water_level_m"] = merged.groupby(["state_norm", "district_norm"])["water_level_m"].ffill(limit=7)
        merged["danger_level_m"] = merged.groupby(["state_norm", "district_norm"])["danger_level_m"].ffill(limit=7)
        merged["warning_level_m"] = merged.groupby(["state_norm", "district_norm"])["warning_level_m"].ffill(limit=7)
        merged["discharge_cumecs"] = merged.groupby(["state_norm", "district_norm"])["discharge_cumecs"].ffill(limit=7)

    elevation = _copy_if_exists(elev_path)
    if elevation is not None:
        elevation["district_norm"] = elevation["district"].apply(_normalize_name)
        elevation["state_norm"] = elevation["state"].apply(_normalize_name)
        merged = merged.merge(
            elevation[
                ["district_norm", "state_norm", "mean_elevation_m", "min_elevation_m", "slope_degrees"]
            ],
            how="left",
            on=["state_norm", "district_norm"],
            validate="m:1",
        )

    soil_path = RAW_DIR / "soil" / "soil_data.csv"
    soil = _copy_if_exists(soil_path)
    if soil is not None:
        soil["district_norm"] = soil["district"].apply(_normalize_name)
        soil["state_norm"] = soil["state"].apply(_normalize_name)
        merged = merged.merge(
            soil,
            how="left",
            on=["state_norm", "district_norm"],
            validate="m:1",
        )

    merged = _fill_rainfall_gaps(merged)
    merged["rainfall_unreliable"] = merged["rainfall_mm"].isna()
    merged["missing_pct"] = merged.isna().mean(axis=1)

    district_stats = (
        merged.groupby(["state_norm", "district_norm"])["rainfall_mm"]
        .apply(lambda s: s.isna().mean())
        .reset_index(name="missing_fraction")
    )
    excluded = district_stats[district_stats["missing_fraction"] > 0.40]
    excluded_districts = excluded[["district_norm", "state_norm"]].apply(lambda row: f"{row['district_norm']} ({row['state_norm']})", axis=1).tolist()
    if excluded_districts:
        logger.warning("Excluding districts with >40%% missing data: %s", excluded_districts)
        merged = merged[~merged.set_index(["state_norm", "district_norm"]).index.isin(excluded.set_index(["state_norm","district_norm"]).index)]

    merged["date_range_start"] = merged["date"].min()
    merged["date_range_end"] = merged["date"].max()

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    merged.to_csv(MERGED_PATH, index=False)

    total_records = len(merged)
    district_count = merged["district_norm"].nunique()
    flood_rate = merged["flood_occurred"].mean() * 100
    missing_by_col = merged.isna().mean() * 100
    logger.info("Master merge complete: %s records, %s districts", total_records, district_count)
    logger.info("Flood event rate: %.2f%%", flood_rate)
    logger.info("Missing values by column:\n%s", missing_by_col.to_dict())

    print("\nDATA QUALITY REPORT")
    print("Total records:", total_records)
    print("Date range:", merged["date"].min(), "to", merged["date"].max())
    print("Districts covered:", district_count)
    print(f"Flood event rate: {flood_rate:.2f}%")
    print("Missing % per column:")
    for column, missing in missing_by_col.items():
        print(f"  {column}: {missing:.2f}%")
    print("Excluded districts:", excluded_districts or ["None"])

    return MERGED_PATH


def validate_data_quality(df: pd.DataFrame) -> pd.DataFrame:
    """Validate merged data and write flagged records to a CSV file."""
    flagged = pd.DataFrame()
    if "rainfall_mm" in df.columns:
        flagged = df[df["rainfall_mm"] < 0]
    if "water_level_m" in df.columns:
        flagged = pd.concat([flagged, df[df["water_level_m"] > 30]], ignore_index=True)
    if "rainfall_mm" in df.columns:
        flagged = pd.concat([flagged, df[df["rainfall_mm"] > 400]], ignore_index=True)
    duplicates = df.duplicated(subset=["district_norm", "state_norm", "date"]).sum()
    if duplicates:
        logger.warning("Found %s duplicate district-date rows.", duplicates)
        flagged = pd.concat([flagged, df[df.duplicated(subset=["district_norm", "state_norm", "date"], keep=False)]], ignore_index=True)

    all_zero_rain = df.groupby(["district_norm", "state_norm"])["rainfall_mm"].sum() == 0 if "rainfall_mm" in df.columns else pd.Series(dtype=bool)
    all_zero_districts = all_zero_rain[all_zero_rain].index.tolist()
    if all_zero_districts:
        logger.warning("Districts with all-zero rainfall: %s", all_zero_districts)

    if not flagged.empty:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        flagged.drop_duplicates(inplace=True)
        flagged.to_csv(FLAGGED_PATH, index=False)
        logger.info("Saved flagged records to %s", FLAGGED_PATH)
    else:
        logger.info("No flagged records found.")

    flood_rate = df["flood_occurred"].mean() * 100 if "flood_occurred" in df.columns else 0.0
    if flood_rate < 5 or flood_rate > 20:
        logger.warning("Flood event rate is outside the expected 5-20%% range: %.2f%%", flood_rate)

    return flagged
