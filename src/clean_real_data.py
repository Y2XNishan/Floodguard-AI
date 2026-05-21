"""Clean the real India flood master dataset for model training.

The master CSV is a concatenation of several real-world sources with different
schemas. This script standardizes the useful columns, imputes missing values,
engineers flood-risk features, balances the target to a 20% flood rate, and
writes chronological train/validation/test CSV files.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MASTER_PATH = PROCESSED_DIR / "india_flood_master.csv"
CLEAN_PATH = PROCESSED_DIR / "india_flood_clean.csv"
TRAIN_PATH = PROCESSED_DIR / "train.csv"
VAL_PATH = PROCESSED_DIR / "val.csv"
TEST_PATH = PROCESSED_DIR / "test.csv"

RANDOM_STATE = 42
TARGET_FLOOD_RATE = 0.20
MONSOON_MONTHS = {6, 7, 8, 9}


def standardize_column_name(name: str) -> str:
    """Return a stable snake_case ASCII-ish column name."""
    cleaned = str(name).strip().lower()
    replacements = {
        "°": "",
        "³": "3",
        "ł": "3",
        "%": "pct",
        "/": "_per_",
        "(": "",
        ")": "",
    }
    for old, new in replacements.items():
        cleaned = cleaned.replace(old, new)
    cleaned = re.sub(r"[^a-z0-9]+", "_", cleaned)
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned


def coalesce_columns(df: pd.DataFrame, aliases: list[str], default: object = np.nan) -> pd.Series:
    """Coalesce the first non-null value from any available alias column."""
    result = pd.Series(default, index=df.index)
    for alias in aliases:
        if alias in df.columns:
            result = result.combine_first(df[alias])
    return result


def parse_dates(df: pd.DataFrame) -> pd.Series:
    """Build a usable date from date/start_date/year-style source columns."""
    parsed = pd.to_datetime(coalesce_columns(df, ["date", "start_date"]), errors="coerce")
    if "year" in df.columns:
        year = pd.to_numeric(df["year"], errors="coerce")
        year_dates = pd.to_datetime(
            year.dropna().astype(int).astype(str) + "-01-01",
            errors="coerce",
        ).reindex(df.index)
        parsed = parsed.combine_first(year_dates)
    return parsed


def normalize_text(series: pd.Series, fallback: str) -> pd.Series:
    """Normalize state/district names without losing unknown rows."""
    normalized = (
        series.astype("string")
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
        .str.replace(r"^nan$", "", regex=True)
    )
    normalized = normalized.mask(normalized.isna() | normalized.eq(""), fallback)
    return normalized.str.title()


def first_district_name(series: pd.Series) -> pd.Series:
    """Use the first named district when inventory rows list many districts."""
    return (
        series.astype("string")
        .str.replace(r"\bparts of\b", "", case=False, regex=True)
        .str.split(",")
        .str[0]
        .str.strip()
    )


def load_and_standardize() -> tuple[pd.DataFrame, dict[str, int]]:
    """Load the master CSV and standardize the training columns."""
    print(f"Loading master data: {MASTER_PATH}")
    raw = pd.read_csv(MASTER_PATH, low_memory=False)
    original_rows = len(raw)
    raw.columns = [standardize_column_name(col) for col in raw.columns]

    df = pd.DataFrame(index=raw.index)
    df["date"] = parse_dates(raw)
    df["state"] = normalize_text(
        coalesce_columns(raw, ["state_name", "state", "state_ut_name", "subdivision"]),
        "Unknown",
    )
    df["district"] = normalize_text(
        first_district_name(coalesce_columns(raw, ["district", "districts", "location"])),
        "Statewide",
    )

    df["rainfall_mm"] = pd.to_numeric(
        coalesce_columns(raw, ["actual", "rainfall_mm", "annual", "normal"]),
        errors="coerce",
    )
    df["temperature_c"] = pd.to_numeric(coalesce_columns(raw, ["temperature_c"]), errors="coerce")
    df["humidity_pct"] = pd.to_numeric(coalesce_columns(raw, ["humidity_pct"]), errors="coerce")
    df["water_level_m"] = pd.to_numeric(coalesce_columns(raw, ["water_level_m"]), errors="coerce")
    df["river_discharge_m3_s"] = pd.to_numeric(
        coalesce_columns(raw, ["river_discharge_m3_per_s", "river_discharge_m3_s"]),
        errors="coerce",
    )
    df["elevation_m"] = pd.to_numeric(coalesce_columns(raw, ["elevation_m"]), errors="coerce")
    df["population_density"] = pd.to_numeric(coalesce_columns(raw, ["population_density"]), errors="coerce")

    severity = pd.to_numeric(coalesce_columns(raw, ["severity"]), errors="coerce")
    explicit_target = pd.to_numeric(coalesce_columns(raw, ["flood_occurred"]), errors="coerce")
    event_source = coalesce_columns(raw, ["main_cause", "event_source", "uei", "area_affected"])
    dated_event = raw.get("start_date", pd.Series(np.nan, index=raw.index)).notna() & event_source.notna()
    inferred_flood = severity.notna() | dated_event
    df["flood_occurred"] = explicit_target.where(explicit_target.notna(), inferred_flood.astype(int))
    df["flood_occurred"] = pd.to_numeric(df["flood_occurred"], errors="coerce").fillna(0).clip(0, 1).astype(int)

    before_date_filter = len(df)
    df = df[df["date"].notna()].copy()
    dropped_no_date = before_date_filter - len(df)

    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day_of_year"] = df["date"].dt.dayofyear
    df["is_monsoon"] = df["month"].isin(MONSOON_MONTHS).astype(int)

    df = df.drop_duplicates(subset=["date", "state", "district", "rainfall_mm", "flood_occurred"])
    df = df.sort_values(["state", "district", "date"]).reset_index(drop=True)

    stats = {
        "original_rows": original_rows,
        "dropped_no_date": dropped_no_date,
        "deduplicated_rows": before_date_filter - dropped_no_date - len(df),
    }
    return df, stats


def fill_numeric_by_context(df: pd.DataFrame, column: str) -> None:
    """Fill a numeric column using district/state/month context before globals."""
    if column not in df.columns:
        return
    group_levels = [
        ["state", "district", "month"],
        ["state", "month"],
        ["month"],
        ["state", "district"],
        ["state"],
    ]
    for levels in group_levels:
        fill_values = df.groupby(levels, dropna=False)[column].transform("median")
        df[column] = df[column].fillna(fill_values)

    median = df[column].median()
    if pd.isna(median):
        median = 0
    df[column] = df[column].fillna(median)


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values with time interpolation and contextual medians."""
    print("Handling missing values...")
    numeric_cols = [
        "rainfall_mm",
        "temperature_c",
        "humidity_pct",
        "water_level_m",
        "river_discharge_m3_s",
        "elevation_m",
        "population_density",
    ]

    df = df.sort_values(["state", "district", "date"]).reset_index(drop=True)
    for column in numeric_cols:
        df[column] = df.groupby(["state", "district"], dropna=False)[column].transform(
            lambda s: s.interpolate(limit_direction="both")
        )
        fill_numeric_by_context(df, column)

    df["rainfall_mm"] = df["rainfall_mm"].clip(lower=0)
    df["humidity_pct"] = df["humidity_pct"].clip(lower=0, upper=100)
    df["water_level_m"] = df["water_level_m"].clip(lower=0)
    df["river_discharge_m3_s"] = df["river_discharge_m3_s"].clip(lower=0)
    df["population_density"] = df["population_density"].clip(lower=0)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create requested time-series and seasonal features."""
    print("Engineering features...")
    df = df.sort_values(["state", "district", "date"]).reset_index(drop=True)
    grouped = df.groupby(["state", "district"], group_keys=False, dropna=False)

    df["rainfall_7day"] = grouped["rainfall_mm"].rolling(7, min_periods=1).sum().reset_index(level=[0, 1], drop=True)
    df["rainfall_30day"] = grouped["rainfall_mm"].rolling(30, min_periods=1).sum().reset_index(level=[0, 1], drop=True)
    df["api"] = grouped["rainfall_mm"].transform(lambda s: s.ewm(alpha=0.30, adjust=False).mean())
    df["river_rise_rate"] = grouped["water_level_m"].diff().fillna(0)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    df["rainfall_intensity"] = df["rainfall_mm"] / (df["temperature_c"].abs() + 1)
    df["rainfall_river_interaction"] = df["rainfall_7day"] * df["water_level_m"]
    df["discharge_per_water_level"] = df["river_discharge_m3_s"] / (df["water_level_m"] + 1)
    df["terrain_rain_risk"] = df["rainfall_30day"] / (df["elevation_m"].abs() + 1)
    return df


def balance_to_flood_rate(df: pd.DataFrame, target_rate: float = TARGET_FLOOD_RATE) -> pd.DataFrame:
    """Downsample non-flood rows so floods are target_rate of the final data."""
    print(f"Balancing classes to {target_rate:.0%} flood rate...")
    floods = df[df["flood_occurred"].eq(1)]
    non_floods = df[df["flood_occurred"].eq(0)]
    if floods.empty:
        print("WARNING: No flood rows found; skipping class balancing.")
        return df

    desired_non_floods = int(round(len(floods) * (1 - target_rate) / target_rate))
    desired_non_floods = min(desired_non_floods, len(non_floods))
    sampled_non_floods = non_floods.sample(n=desired_non_floods, random_state=RANDOM_STATE)
    balanced = pd.concat([floods, sampled_non_floods], ignore_index=True)
    return balanced.sort_values(["date", "state", "district"]).reset_index(drop=True)


def split_stratified(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create stratified 70/15/15 splits that preserve the flood rate."""
    print("Splitting with stratified random 70/15/15 split...")
    train, temp = train_test_split(
        df,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=df["flood_occurred"],
    )
    val, test = train_test_split(
        temp,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=temp["flood_occurred"],
    )
    train = train.sort_values(["date", "state", "district"]).reset_index(drop=True)
    val = val.sort_values(["date", "state", "district"]).reset_index(drop=True)
    test = test.sort_values(["date", "state", "district"]).reset_index(drop=True)
    return train, val, test


def print_dataset_summary(name: str, df: pd.DataFrame) -> None:
    """Print a compact but complete summary for a dataframe."""
    flood_rate = df["flood_occurred"].mean() if len(df) else 0
    print(f"\n{name}")
    print("-" * len(name))
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns):,}")
    print(f"Date range: {df['date'].min() if len(df) else 'N/A'} to {df['date'].max() if len(df) else 'N/A'}")
    print(f"States: {df['state'].nunique() if len(df) else 0:,}")
    print(f"Districts: {df['district'].nunique() if len(df) else 0:,}")
    print(f"Flood rows: {int(df['flood_occurred'].sum()) if len(df) else 0:,}")
    print(f"Flood rate: {flood_rate:.2%}")
    missing = (df.isna().mean() * 100).sort_values(ascending=False)
    missing = missing[missing.gt(0)]
    if missing.empty:
        print("Missing values: none")
    else:
        print("Missing values:")
        for column, pct in missing.items():
            print(f"  {column}: {pct:.2f}%")


def save_outputs(df: pd.DataFrame, train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> None:
    """Save cleaned full data and chronological splits."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLEAN_PATH, index=False)
    train.to_csv(TRAIN_PATH, index=False)
    val.to_csv(VAL_PATH, index=False)
    test.to_csv(TEST_PATH, index=False)
    print("\nSaved files:")
    print(f"  {CLEAN_PATH}")
    print(f"  {TRAIN_PATH}")
    print(f"  {VAL_PATH}")
    print(f"  {TEST_PATH}")


def main() -> None:
    df, stats = load_and_standardize()
    print(f"Loaded rows: {stats['original_rows']:,}")
    print(f"Dropped rows without usable date: {stats['dropped_no_date']:,}")
    print(f"Dropped duplicate rows: {stats['deduplicated_rows']:,}")

    df = handle_missing_values(df)
    df = engineer_features(df)
    df = balance_to_flood_rate(df)
    train, val, test = split_stratified(df)
    save_outputs(df, train, val, test)

    print("\n" + "=" * 72)
    print("CLEAN REAL DATA SUMMARY")
    print("=" * 72)
    print_dataset_summary("Full Balanced Clean Data", df)
    print_dataset_summary("Train Split (Stratified 70%)", train)
    print_dataset_summary("Validation Split (Stratified 15%)", val)
    print_dataset_summary("Test Split (Stratified 15%)", test)
    print("\nPre-event feature columns:")
    feature_cols = [c for c in df.columns if c not in {"date", "state", "district", "flood_occurred"}]
    for column in feature_cols:
        print(f"  - {column}")
    print("\nDone.")


if __name__ == "__main__":
    main()
