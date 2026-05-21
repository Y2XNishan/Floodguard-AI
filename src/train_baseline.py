"""Train an XGBoost flood classifier on the cleaned real India flood data."""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

matplotlib.use("Agg")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data" / "processed"
CLEAN_DATA_PATH = DATA_DIR / "india_flood_clean.csv"

FEATURE_COLUMNS = [
    "rainfall_mm",
    "temperature_c",
    "humidity_pct",
    "water_level_m",
    "river_discharge_m3_s",
    "elevation_m",
    "population_density",
    "year",
    "month",
    "day_of_year",
    "is_monsoon",
    "rainfall_7day",
    "rainfall_30day",
    "api",
    "river_rise_rate",
    "month_sin",
    "month_cos",
    "rainfall_intensity",
    "rainfall_river_interaction",
    "discharge_per_water_level",
    "terrain_rain_risk",
]
TARGET_COLUMN = "flood_occurred"
RANDOM_STATE = 42


def load_real_data() -> tuple[pd.DataFrame, pd.Series]:
    """Load cleaned real data and return feature and target frames."""
    if not CLEAN_DATA_PATH.exists():
        raise FileNotFoundError(f"Clean real data not found: {CLEAN_DATA_PATH}")

    df = pd.read_csv(CLEAN_DATA_PATH)
    missing = [column for column in FEATURE_COLUMNS + [TARGET_COLUMN] if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {CLEAN_DATA_PATH}: {missing}")

    X = df[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    if X.isna().any().any():
        X = X.fillna(X.median(numeric_only=True)).fillna(0)
    y = df[TARGET_COLUMN].astype(int)

    print(f"Loaded real clean data: {len(df):,} rows")
    print(f"Overall flood rate: {y.mean():.2%}")
    return X, y


def stratified_split(X: pd.DataFrame, y: pd.Series):
    """Create stratified 70/15/15 train/validation/test splits."""
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=y_temp,
    )

    print("\nStratified splits:")
    for name, labels in [("Train", y_train), ("Val", y_val), ("Test", y_test)]:
        print(f"  {name:5}: {len(labels):,} rows | flood rate {labels.mean():.2%}")

    return X_train, X_val, X_test, y_train, y_val, y_test


def build_model(y_train: pd.Series) -> xgb.XGBClassifier:
    """Build the requested XGBoost model with class weighting."""
    flood_count = int(y_train.sum())
    no_flood_count = int((y_train == 0).sum())
    scale_pos_weight = no_flood_count / max(flood_count, 1)

    return xgb.XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.01,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        use_label_encoder=False,
        eval_metric="aucpr",
        objective="binary:logistic",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        early_stopping_rounds=50,
    )


def evaluate_model(model: xgb.XGBClassifier, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    """Evaluate with the required metrics."""
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.50).astype(int)
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred, zero_division=0),
        "AUC": roc_auc_score(y_test, y_prob),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
    }

    print("\n" + "=" * 72)
    print("XGBOOST REAL DATA RESULTS")
    print("=" * 72)
    for name, value in metrics.items():
        print(f"{name:>10}: {value:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["No Flood", "Flood"], zero_division=0))
    return metrics


def generate_shap_outputs(model: xgb.XGBClassifier, X_train: pd.DataFrame, X_test: pd.DataFrame) -> None:
    """Generate SHAP summary plot and print top feature importances."""
    print("\nGenerating SHAP explanations...")
    sample_train = X_train.sample(n=min(1000, len(X_train)), random_state=RANDOM_STATE)
    sample_test = X_test.sample(n=min(500, len(X_test)), random_state=RANDOM_STATE)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample_test)

    import matplotlib.pyplot as plt

    shap.summary_plot(shap_values, sample_test, show=False, max_display=20)
    plt.tight_layout()
    shap_path = MODELS_DIR / "shap_real_summary.png"
    plt.savefig(shap_path, dpi=180, bbox_inches="tight")
    plt.close()

    joblib.dump(explainer, MODELS_DIR / "shap_real_explainer.pkl")

    mean_abs = np.abs(shap_values).mean(axis=0)
    top = (
        pd.Series(mean_abs, index=sample_test.columns)
        .sort_values(ascending=False)
        .head(10)
    )
    print("\nTop 10 most important SHAP features:")
    for rank, (feature, importance) in enumerate(top.items(), start=1):
        print(f"  {rank:2}. {feature}: {importance:.6f}")
    print(f"Saved SHAP summary plot to {shap_path}")


def train_and_evaluate() -> tuple[xgb.XGBClassifier, dict[str, float]]:
    """Train, evaluate, save artifacts, and generate SHAP plots."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    X, y = load_real_data()
    X_train, X_val, X_test, y_train, y_val, y_test = stratified_split(X, y)

    model = build_model(y_train)
    logger.info("Training XGBoost real-data model...")
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    metrics = evaluate_model(model, X_test, y_test)

    model_path = MODELS_DIR / "xgb_real_model.pkl"
    feature_path = MODELS_DIR / "real_feature_names.pkl"
    metrics_path = MODELS_DIR / "xgb_real_metrics.pkl"
    joblib.dump(model, model_path)
    joblib.dump(FEATURE_COLUMNS, feature_path)
    joblib.dump(metrics, metrics_path)
    logger.info("Saved model to %s", model_path)
    logger.info("Saved real feature names to %s", feature_path)
    logger.info("Saved metrics to %s", metrics_path)

    generate_shap_outputs(model, X_train, X_test)
    return model, metrics


if __name__ == "__main__":
    train_and_evaluate()
