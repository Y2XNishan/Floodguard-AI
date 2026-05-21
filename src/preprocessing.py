"""
Preprocessing pipeline for Flood Risk Prediction System.

Handles data cleaning, feature engineering, normalization,
class balancing (SMOTE), and train/val/test splitting.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import joblib

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_data() -> pd.DataFrame:
    """Load raw sample data from CSV."""
    path = os.path.join(PROJECT_ROOT, "data", "sample_data.csv")
    logger.info(f"Loading data from {path}")
    df = pd.read_csv(path, parse_dates=["date"])
    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values using time-series interpolation per district.

    Args:
        df: Raw dataframe with potential NaN values.

    Returns:
        DataFrame with missing values filled.
    """
    logger.info(f"Missing values before interpolation:\n{df.isnull().sum()}")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Remove target columns from interpolation
    interp_cols = [c for c in numeric_cols if c not in ["flood_occurred", "flood_severity"]]

    # Interpolate within each district's time series
    df = df.sort_values(["district", "date"]).reset_index(drop=True)
    for district in df["district"].unique():
        mask = df["district"] == district
        subset = df.loc[mask, ["date"] + interp_cols].copy()
        subset = subset.set_index("date")
        subset = subset.interpolate(method="time").ffill().bfill()
        subset = subset.reset_index(drop=True)
        df.loc[mask, interp_cols] = subset[interp_cols].values

    # Fill any remaining NaN with column median
    for col in interp_cols:
        if df[col].isnull().any():
            df[col].fillna(df[col].median(), inplace=True)

    logger.info(f"Missing values after interpolation:\n{df.isnull().sum()}")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create engineered features for flood prediction.

    Features created:
        - rainfall_7day_cumsum: Rolling 7-day cumulative rainfall
        - rainfall_30day_cumsum: Rolling 30-day cumulative rainfall
        - api: Antecedent Precipitation Index (EWM span=7)
        - river_rise_rate: Daily change in river level
        - rainfall_river_interaction: rainfall_7day * river_level
        - is_monsoon: Binary flag for June-September
        - month_sin, month_cos: Cyclical month encoding

    Args:
        df: DataFrame with raw features.

    Returns:
        DataFrame with additional engineered features.
    """
    logger.info("Engineering features...")
    df = df.sort_values(["district", "date"]).reset_index(drop=True)

    engineered = []
    for district in df["district"].unique():
        d = df[df["district"] == district].copy()
        d = d.sort_values("date")

        # Rolling cumulative rainfall
        d["rainfall_7day_cumsum"] = d["rainfall_mm"].rolling(window=7, min_periods=1).sum()
        d["rainfall_30day_cumsum"] = d["rainfall_mm"].rolling(window=30, min_periods=1).sum()

        # Antecedent Precipitation Index
        d["api"] = d["rainfall_mm"].ewm(span=7, adjust=False).mean()

        # River rise rate (daily change)
        d["river_rise_rate"] = d["river_level_m"].diff().fillna(0)

        # Interaction feature
        d["rainfall_river_interaction"] = d["rainfall_7day_cumsum"] * d["river_level_m"]

        engineered.append(d)

    df = pd.concat(engineered, ignore_index=True)

    # Monsoon flag
    df["is_monsoon"] = df["date"].dt.month.isin([6, 7, 8, 9]).astype(int)

    # Cyclical month encoding
    df["month_sin"] = np.sin(2 * np.pi * df["date"].dt.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["date"].dt.month / 12)

    df["rainfall_intensity"] = df["rainfall_mm"] / (df["temperature_c"] + 1)
    df["flood_risk_index"] = df["rainfall_7day_cumsum"] * df["river_level_m"]
    df["seasonal_risk"] = df["is_monsoon"] * df["rainfall_30day_cumsum"]

    logger.info(f"Feature engineering complete. Shape: {df.shape}")
    return df


def engineer_real_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create extended real-world features for India flood risk modeling."""
    logger.info("Engineering real-world features...")
    df = df.sort_values(["district", "date"]).reset_index(drop=True)
    output: list[pd.DataFrame] = []
    flood_plain_districts = {
        "patna", "varanasi", "nagaon", "morigaon", "barpeta",
        "dibrugarh", "assam", "bhagalpur", "muzaffarpur", "darbhanga",
    }

    for district, group in df.groupby("district"):
        group = group.sort_values("date").copy()
        group["rainfall_3day"] = group["rainfall_mm"].rolling(window=3, min_periods=1).sum()
        group["rainfall_7day"] = group["rainfall_mm"].rolling(window=7, min_periods=1).sum()
        group["rainfall_15day"] = group["rainfall_mm"].rolling(window=15, min_periods=1).sum()
        group["rainfall_30day"] = group["rainfall_mm"].rolling(window=30, min_periods=1).sum()
        group["api"] = (0.9 * group["rainfall_mm"].shift(1).fillna(0)) + group["rainfall_mm"]

        mean_30yr = group["rainfall_mm"].rolling(window=365, min_periods=30).mean()
        std_30yr = group["rainfall_mm"].rolling(window=365, min_periods=30).std().replace(0, np.nan)
        group["rainfall_anomaly"] = (group["rainfall_mm"] - mean_30yr) / std_30yr
        group["rainfall_anomaly"] = group["rainfall_anomaly"].fillna(0)
        group["extreme_rain_days"] = group["rainfall_mm"].gt(115).rolling(window=7, min_periods=1).sum()

        if "water_level_m" in group.columns and "danger_level_m" in group.columns:
            group["river_level_pct"] = (group["water_level_m"] / group["danger_level_m"].replace(0, np.nan)) * 100
            group["river_rise_rate"] = group["water_level_m"].diff().fillna(0)
            group["days_above_warning"] = group["water_level_m"].gt(group["warning_level_m"]).rolling(window=7, min_periods=1).sum()
            seasonal_mean = group["water_level_m"].groupby(group["date"].dt.month).transform("mean")
            group["river_level_anomaly"] = (group["water_level_m"] - seasonal_mean) / seasonal_mean.replace(0, np.nan)
        else:
            group["river_level_pct"] = 0.0
            group["river_rise_rate"] = 0.0
            group["days_above_warning"] = 0.0
            group["river_level_anomaly"] = 0.0

        group["day_of_year"] = group["date"].dt.dayofyear
        group["week_of_year"] = group["date"].dt.isocalendar().week
        group["month_sin"] = np.sin(2 * np.pi * group["date"].dt.month / 12)
        group["month_cos"] = np.cos(2 * np.pi * group["date"].dt.month / 12)
        group["is_monsoon"] = group["date"].dt.month.isin([6, 7, 8, 9]).astype(int)
        group["monsoon_day"] = np.where(
            group["is_monsoon"] == 1,
            (group["date"] - pd.to_datetime(group["date"].dt.year.astype(str) + "-06-01")).dt.days.clip(lower=0),
            0,
        )
        group["is_peak_monsoon"] = group["date"].dt.month.isin([7, 8]).astype(int)
        group["flood_plain_flag"] = group["district"].str.lower().isin(flood_plain_districts).astype(int)

        group["rain_river_interaction"] = group["rainfall_7day"] * group["river_level_pct"] / 100
        group["terrain_rain_risk"] = group["rainfall_30day"] / (group.get("elevation_m", 0) + 1)

        group["flood_freq_5yr"] = group["flood_occurred"].rolling(window=365 * 5, min_periods=1).sum().shift(1).fillna(0)
        group["last_flood_days"] = (group["date"] - group.loc[group["flood_occurred"] == 1, "date"].ffill()).dt.days.fillna(999)
        month_rate = (
            group.groupby(group["date"].dt.month)["flood_occurred"].transform("mean")
        )
        group["same_month_flood_rate"] = month_rate.fillna(0)

        output.append(group)

    result = pd.concat(output, ignore_index=True)
    logger.info("Real feature engineering complete. Shape: %s", result.shape)
    return result


def prepare_splits(df: pd.DataFrame, target_col: str = "flood_occurred"):
    """
    Normalize features, apply SMOTE, and split into train/val/test sets.

    The split is chronological (70/15/15) to prevent data leakage.
    SMOTE is applied only to the training set.

    Args:
        df: DataFrame with all features engineered.
        target_col: Name of the target column.

    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test,
                  feature_names, scaler)
    """
    logger.info("Preparing train/val/test splits...")

    # Sort by date for chronological split
    df = df.sort_values("date").reset_index(drop=True)

    # Define feature columns (exclude non-feature cols and leaky features)
    # NOTE: days_since_last_flood is excluded because it's a target leak —
    # it equals 0 whenever flood_occurred=1, giving the model a trivial shortcut.
    exclude_cols = ["date", "district", "flood_occurred", "flood_severity", "days_since_last_flood"]
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    numeric_features = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()

    X = df[numeric_features].values
    y = df[target_col].values

    # Chronological split: 70% train, 15% val, 15% test
    n = len(df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]

    logger.info(f"Split sizes — Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    logger.info(f"Train flood rate: {y_train.mean():.2%}")

    # Normalize features using StandardScaler (fit on train only)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    os.makedirs(os.path.join(PROJECT_ROOT, "models"), exist_ok=True)
    # Save scaler for inference
    scaler_path = os.path.join(PROJECT_ROOT, "models", "scaler.pkl")
    joblib.dump(scaler, scaler_path)
    logger.info(f"Scaler saved to {scaler_path}")

    # Apply SMOTE to training set only
    logger.info("Applying SMOTE to training set...")
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)
    logger.info(f"After SMOTE — Train size: {len(X_train)}, Flood rate: {y_train.mean():.2%}")

    # Save feature names
    feature_names = numeric_features
    joblib.dump(feature_names, os.path.join(PROJECT_ROOT, "models", "feature_names.pkl"))

    return X_train, X_val, X_test, y_train, y_val, y_test, feature_names, scaler


def run_full_pipeline():
    """Run the complete preprocessing pipeline and save results."""
    try:
        # Load data
        df = load_data()

        # Handle missing values
        df = handle_missing_values(df)

        # Engineer features
        df = engineer_features(df)

        # Save processed data
        processed_path = os.path.join(PROJECT_ROOT, "data", "processed", "processed_data.csv")
        df.to_csv(processed_path, index=False)
        logger.info(f"Processed data saved to {processed_path}")

        # Prepare splits
        X_train, X_val, X_test, y_train, y_val, y_test, features, scaler = prepare_splits(df)

        # Save splits for model training
        splits_path = os.path.join(PROJECT_ROOT, "data", "processed")
        np.save(os.path.join(splits_path, "X_train.npy"), X_train)
        np.save(os.path.join(splits_path, "X_val.npy"), X_val)
        np.save(os.path.join(splits_path, "X_test.npy"), X_test)
        np.save(os.path.join(splits_path, "y_train.npy"), y_train)
        np.save(os.path.join(splits_path, "y_val.npy"), y_val)
        np.save(os.path.join(splits_path, "y_test.npy"), y_test)
        logger.info("All splits saved to data/processed/")

        print("\n" + "=" * 60)
        print("PREPROCESSING COMPLETE")
        print("=" * 60)
        print(f"Features ({len(features)}): {features}")
        print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    except Exception as e:
        logger.error(f"Preprocessing pipeline failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    run_full_pipeline()
