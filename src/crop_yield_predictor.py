import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


CROP_YIELD_DATA = "data/raw/crop_yield/crop_yield.csv"
RAINFALL_DATA = "data/raw/crop_yield/rainfall in india 1901-2015.csv"
MODEL_PATH = "models/crop_yield_model.pkl"
ENCODERS_PATH = "models/crop_yield_encoders.pkl"
METADATA_PATH = "models/crop_yield_metadata.json"


def load_and_preprocess():
    print("Loading crop yield data...")
    df = pd.read_csv(CROP_YIELD_DATA)

    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(df.head())

    df.columns = df.columns.str.strip()

    column_mapping = {}
    for col in df.columns:
        col_lower = col.lower().strip()
        if "state" in col_lower:
            column_mapping[col] = "State"
        elif "crop" in col_lower and "year" not in col_lower:
            column_mapping[col] = "Crop"
        elif "year" in col_lower:
            column_mapping[col] = "Year"
        elif "season" in col_lower:
            column_mapping[col] = "Season"
        elif "area" in col_lower:
            column_mapping[col] = "Area"
        elif "production" in col_lower:
            column_mapping[col] = "Production"
        elif "yield" in col_lower:
            column_mapping[col] = "Yield"
        elif "rainfall" in col_lower or "rain" in col_lower:
            column_mapping[col] = "Annual_Rainfall"
        elif "fertilizer" in col_lower:
            column_mapping[col] = "Fertilizer"
        elif "pesticide" in col_lower:
            column_mapping[col] = "Pesticide"

    df = df.rename(columns=column_mapping)
    print(f"Renamed columns: {df.columns.tolist()}")

    required_cols = ["State", "Crop", "Year"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns after preprocessing: {missing}")

    if "Yield" not in df.columns:
        if "Production" in df.columns and "Area" in df.columns:
            df["Yield"] = pd.to_numeric(df["Production"], errors="coerce") / pd.to_numeric(
                df["Area"], errors="coerce"
            )
            print("Calculated Yield from Production/Area")
        else:
            raise ValueError("Dataset must contain Yield or both Production and Area columns.")

    df["State"] = df["State"].astype(str).str.strip()
    df["Crop"] = df["Crop"].astype(str).str.strip()
    if "Season" in df.columns:
        df["Season"] = df["Season"].astype(str).str.strip()

    df["Yield"] = pd.to_numeric(df["Yield"], errors="coerce")
    df = df.dropna(subset=["Yield", "State", "Crop", "Year"])
    df = df[df["Yield"] > 0]
    df = df[df["Yield"] < df["Yield"].quantile(0.99)]

    print(f"After cleaning: {df.shape}")
    print(f"Unique crops: {df['Crop'].nunique()}")
    print(f"Unique states: {df['State'].nunique()}")
    print(f"Year range: {df['Year'].min()} - {df['Year'].max()}")

    return df


def engineer_features(df):
    if "Season" in df.columns:
        season_map = {
            "Kharif": 0,
            "Rabi": 1,
            "Zaid": 2,
            "Whole Year": 3,
            "Summer": 4,
            "Winter": 5,
            "Autumn": 6,
        }
        df["Season_encoded"] = df["Season"].map(season_map).fillna(3)
    else:
        df["Season_encoded"] = 0

    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df["Decade"] = (df["Year"] // 10) * 10
    df["Years_since_2000"] = df["Year"] - 2000

    flood_prone_states = {
        "Assam": 0.9,
        "Bihar": 0.85,
        "West Bengal": 0.8,
        "Uttar Pradesh": 0.7,
        "Odisha": 0.75,
        "Andhra Pradesh": 0.6,
        "Kerala": 0.65,
        "Punjab": 0.5,
        "Haryana": 0.45,
        "Manipur": 0.7,
        "Tripura": 0.65,
        "Meghalaya": 0.6,
        "Nagaland": 0.55,
        "Arunachal Pradesh": 0.6,
    }
    df["Flood_Risk_Historical"] = df["State"].map(flood_prone_states).fillna(0.3)

    flood_sensitive_crops = {
        "Rice": 0.3,
        "Wheat": 0.8,
        "Cotton": 0.75,
        "Sugarcane": 0.4,
        "Maize": 0.7,
        "Potato": 0.8,
        "Tomato": 0.85,
        "Jute": 0.2,
        "Groundnut": 0.75,
        "Soyabean": 0.7,
    }
    df["Flood_Sensitivity"] = df["Crop"].map(flood_sensitive_crops).fillna(0.5)

    if "Area" in df.columns:
        df["Area"] = pd.to_numeric(df["Area"], errors="coerce").fillna(0)
        df["Log_Area"] = np.log1p(df["Area"])
    else:
        df["Area"] = 0
        df["Log_Area"] = 0

    if "Annual_Rainfall" in df.columns:
        df["Annual_Rainfall"] = pd.to_numeric(df["Annual_Rainfall"], errors="coerce").fillna(1000)
        df["Rainfall_Category"] = (
            pd.cut(
                df["Annual_Rainfall"],
                bins=[0, 500, 1000, 1500, 2000, 5000],
                labels=[0, 1, 2, 3, 4],
            )
            .astype(float)
            .fillna(1)
        )
    else:
        df["Annual_Rainfall"] = 1000
        df["Rainfall_Category"] = 1

    if "Fertilizer" in df.columns:
        df["Fertilizer"] = pd.to_numeric(df["Fertilizer"], errors="coerce").fillna(0)
        df["Log_Fertilizer"] = np.log1p(df["Fertilizer"])
    else:
        df["Log_Fertilizer"] = 0

    if "Pesticide" in df.columns:
        df["Pesticide"] = pd.to_numeric(df["Pesticide"], errors="coerce").fillna(0)
        df["Log_Pesticide"] = np.log1p(df["Pesticide"])
    else:
        df["Log_Pesticide"] = 0

    return df


def train_model():
    df = load_and_preprocess()
    df = engineer_features(df)

    le_state = LabelEncoder()
    le_crop = LabelEncoder()

    df["State_encoded"] = le_state.fit_transform(df["State"].fillna("Unknown"))
    df["Crop_encoded"] = le_crop.fit_transform(df["Crop"].fillna("Unknown"))

    feature_cols = [
        "State_encoded",
        "Crop_encoded",
        "Year",
        "Season_encoded",
        "Area",
        "Log_Area",
        "Annual_Rainfall",
        "Rainfall_Category",
        "Flood_Risk_Historical",
        "Flood_Sensitivity",
        "Log_Fertilizer",
        "Log_Pesticide",
        "Decade",
        "Years_since_2000",
    ]

    available_features = [feature for feature in feature_cols if feature in df.columns]
    print(f"Using features: {available_features}")

    X = df[available_features].fillna(0)
    y = df["Yield"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    print(f"\nTraining set: {len(X_train)}")
    print(f"Test set: {len(X_test)}")

    print("\nTraining Random Forest...")
    rf_model = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    rf_model.fit(X_train, y_train)

    print("Training Gradient Boosting...")
    gb_model = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42,
    )
    gb_model.fit(X_train, y_train)

    rf_pred = rf_model.predict(X_test)
    gb_pred = gb_model.predict(X_test)
    ensemble_pred = (rf_pred + gb_pred) / 2

    rf_r2 = r2_score(y_test, rf_pred)
    gb_r2 = r2_score(y_test, gb_pred)
    ens_r2 = r2_score(y_test, ensemble_pred)

    rf_mae = mean_absolute_error(y_test, rf_pred)
    gb_mae = mean_absolute_error(y_test, gb_pred)
    ens_mae = mean_absolute_error(y_test, ensemble_pred)

    print("\n" + "=" * 50)
    print("CROP YIELD PREDICTION RESULTS")
    print("=" * 50)
    print(f"Random Forest  - R2: {rf_r2:.4f}, MAE: {rf_mae:.2f}")
    print(f"Gradient Boost - R2: {gb_r2:.4f}, MAE: {gb_mae:.2f}")
    print(f"Ensemble       - R2: {ens_r2:.4f}, MAE: {ens_mae:.2f}")

    best_name = "Random Forest" if rf_r2 > gb_r2 else "Gradient Boosting"
    print(f"\nBest model: {best_name}")

    os.makedirs("models", exist_ok=True)

    joblib.dump(
        {
            "rf_model": rf_model,
            "gb_model": gb_model,
            "feature_cols": available_features,
        },
        MODEL_PATH,
    )

    joblib.dump(
        {
            "state": le_state,
            "crop": le_crop,
        },
        ENCODERS_PATH,
    )

    avg_yield_by_crop = {
        str(crop): float(value)
        for crop, value in df.groupby("Crop")["Yield"].mean().to_dict().items()
    }
    metadata = {
        "crops": sorted([str(crop) for crop in df["Crop"].dropna().unique().tolist()]),
        "states": sorted([str(state) for state in df["State"].dropna().unique().tolist()]),
        "seasons": ["Kharif", "Rabi", "Zaid", "Whole Year", "Summer", "Winter", "Autumn"],
        "feature_cols": available_features,
        "rf_r2": float(rf_r2),
        "gb_r2": float(gb_r2),
        "ensemble_r2": float(ens_r2),
        "rf_mae": float(rf_mae),
        "gb_mae": float(gb_mae),
        "ensemble_mae": float(ens_mae),
        "year_range": [int(df["Year"].min()), int(df["Year"].max())],
        "avg_yield_by_crop": avg_yield_by_crop,
    }

    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nModels saved to {MODEL_PATH}")
    print(f"Metadata saved to {METADATA_PATH}")

    importances = rf_model.feature_importances_
    feat_imp = sorted(zip(available_features, importances), key=lambda x: x[1], reverse=True)
    print("\nTop Feature Importances:")
    for feat, imp in feat_imp[:8]:
        print(f"  {feat}: {imp:.4f}")

    return rf_r2, gb_r2, ens_r2


def predict_crop_yield(state, crop, season, area_hectares, flood_risk_pct, current_rainfall=None):
    if not os.path.exists(MODEL_PATH):
        return {"error": "Model not trained yet"}

    models = joblib.load(MODEL_PATH)
    encoders = joblib.load(ENCODERS_PATH)

    with open(METADATA_PATH, encoding="utf-8") as f:
        metadata = json.load(f)

    rf_model = models["rf_model"]
    gb_model = models["gb_model"]
    feature_cols = models["feature_cols"]

    le_state = encoders["state"]
    le_crop = encoders["crop"]

    try:
        state_enc = le_state.transform([state])[0]
    except ValueError:
        state_enc = 0

    try:
        crop_enc = le_crop.transform([crop])[0]
    except ValueError:
        crop_enc = 0

    season_map = {
        "Kharif": 0,
        "Rabi": 1,
        "Zaid": 2,
        "Whole Year": 3,
        "Summer": 4,
        "Winter": 5,
        "Autumn": 6,
    }

    flood_sensitive_crops = {
        "Rice": 0.3,
        "Wheat": 0.8,
        "Cotton": 0.75,
        "Sugarcane": 0.4,
        "Maize": 0.7,
        "Potato": 0.8,
        "Tomato": 0.85,
        "Jute": 0.2,
        "Groundnut": 0.75,
        "Soyabean": 0.7,
    }

    rainfall = current_rainfall or 1000

    input_data = {
        "State_encoded": state_enc,
        "Crop_encoded": crop_enc,
        "Year": 2026,
        "Season_encoded": season_map.get(season, 3),
        "Area": area_hectares,
        "Log_Area": np.log1p(area_hectares),
        "Annual_Rainfall": rainfall,
        "Rainfall_Category": min(4, int(rainfall / 500)),
        "Flood_Risk_Historical": flood_risk_pct / 100,
        "Flood_Sensitivity": flood_sensitive_crops.get(crop, 0.5),
        "Log_Fertilizer": 5.0,
        "Log_Pesticide": 3.0,
        "Decade": 2020,
        "Years_since_2000": 26,
    }

    X = pd.DataFrame([input_data])
    X = X.reindex(columns=feature_cols, fill_value=0)

    rf_pred = rf_model.predict(X)[0]
    gb_pred = gb_model.predict(X)[0]
    base_yield = max(0, (rf_pred + gb_pred) / 2)

    flood_impact = (flood_risk_pct / 100) * flood_sensitive_crops.get(crop, 0.5)
    adjusted_yield = base_yield * (1 - flood_impact * 0.6)
    adjusted_yield = max(0, adjusted_yield)

    avg_yield = metadata.get("avg_yield_by_crop", {}).get(crop, base_yield)

    yield_loss_pct = (
        max(0, ((base_yield - adjusted_yield) / base_yield * 100))
        if base_yield > 0
        else 0
    )

    total_production = adjusted_yield * area_hectares

    crop_prices = {
        "Rice": 20000,
        "Wheat": 22000,
        "Cotton": 65000,
        "Sugarcane": 3500,
        "Maize": 18000,
        "Potato": 12000,
        "Tomato": 15000,
        "Jute": 35000,
        "Groundnut": 55000,
        "Soyabean": 40000,
        "Bajra": 22000,
        "Jowar": 20000,
        "Ragi": 35000,
        "Arhar": 65000,
    }

    price_per_tonne = crop_prices.get(crop, 25000)
    economic_loss = (base_yield - adjusted_yield) * area_hectares * price_per_tonne

    if flood_risk_pct < 30:
        risk_level = "Low"
        recommendation = f"Good conditions for {crop} cultivation. Monitor weather regularly."
    elif flood_risk_pct < 60:
        risk_level = "Moderate"
        recommendation = f"Consider crop insurance for your {crop} crop. Ensure proper drainage."
    else:
        risk_level = "High"
        recommendation = (
            f"High flood risk! Consider delaying {crop} planting or switch to flood-tolerant "
            "varieties. Apply for PMFBY insurance immediately."
        )

    return {
        "crop": crop,
        "state": state,
        "season": season,
        "area_hectares": area_hectares,
        "base_yield": round(base_yield, 2),
        "adjusted_yield": round(adjusted_yield, 2),
        "yield_loss_pct": round(yield_loss_pct, 1),
        "total_production": round(total_production, 2),
        "economic_loss": round(economic_loss, 0),
        "price_per_tonne": price_per_tonne,
        "flood_risk_pct": flood_risk_pct,
        "risk_level": risk_level,
        "recommendation": recommendation,
        "avg_district_yield": round(avg_yield, 2),
        "vs_average": round(adjusted_yield - avg_yield, 2),
    }


if __name__ == "__main__":
    train_model()
