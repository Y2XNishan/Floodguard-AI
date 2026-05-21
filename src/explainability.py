"""
SHAP-based explainability for the XGBoost flood prediction model.

Generates:
    - shap_summary_plot.png — feature importance overview
    - shap_force_plot.html — explanation for highest-risk prediction
    - shap_dependence_rainfall.png — dependence plot for top feature
"""

import os
import logging
import numpy as np
import joblib
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")


def generate_explanations():
    """
    Load the XGBoost model and test data, compute SHAP values,
    and generate all explainability plots.
    """
    try:
        # ----- Load model and data -----
        logger.info("Loading XGBoost model and test data...")
        model = joblib.load(os.path.join(MODELS_DIR, "xgb_model.pkl"))
        X_test = np.load(os.path.join(DATA_DIR, "X_test.npy"))
        feature_names = joblib.load(os.path.join(MODELS_DIR, "feature_names.pkl"))

        logger.info(f"Test set shape: {X_test.shape}")
        logger.info(f"Features: {feature_names}")

        # ----- Compute SHAP values -----
        logger.info("Computing SHAP values (this may take a moment)...")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)

        # ----- 1. Summary Plot -----
        logger.info("Generating SHAP summary plot...")
        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            shap_values, X_test,
            feature_names=feature_names,
            show=False,
            max_display=15,
        )
        plt.tight_layout()
        plt.savefig(os.path.join(MODELS_DIR, "shap_summary_plot.png"), dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("Saved shap_summary_plot.png")

        # ----- 2. Force Plot (highest risk prediction) -----
        logger.info("Generating SHAP force plot for highest-risk prediction...")
        y_prob = model.predict_proba(X_test)[:, 1]
        highest_risk_idx = np.argmax(y_prob)

        force_plot = shap.force_plot(
            explainer.expected_value,
            shap_values[highest_risk_idx],
            X_test[highest_risk_idx],
            feature_names=feature_names,
            matplotlib=False,
        )
        shap.save_html(
            os.path.join(MODELS_DIR, "shap_force_plot.html"),
            force_plot,
        )
        logger.info("Saved shap_force_plot.html")

        # ----- 3. Dependence Plot for top feature -----
        logger.info("Generating SHAP dependence plot...")
        # Find the most important feature
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        top_feature_idx = np.argmax(mean_abs_shap)
        top_feature_name = feature_names[top_feature_idx]

        plt.figure(figsize=(10, 7))
        shap.dependence_plot(
            top_feature_idx, shap_values, X_test,
            feature_names=feature_names,
            show=False,
        )
        plt.tight_layout()
        plt.savefig(
            os.path.join(MODELS_DIR, "shap_dependence_rainfall.png"),
            dpi=150, bbox_inches="tight"
        )
        plt.close()
        logger.info(f"Saved shap_dependence_rainfall.png (feature: {top_feature_name})")

        # ----- Print Top 5 Features -----
        sorted_idx = np.argsort(mean_abs_shap)[::-1]
        print("\n" + "=" * 60)
        print("TOP 5 MOST IMPORTANT FEATURES (by mean |SHAP|)")
        print("=" * 60)
        for rank, idx in enumerate(sorted_idx[:5], 1):
            print(f"  {rank}. {feature_names[idx]:>30s}  — mean |SHAP| = {mean_abs_shap[idx]:.4f}")

        # Save SHAP values for the Streamlit app
        joblib.dump(shap_values, os.path.join(MODELS_DIR, "shap_values.pkl"))
        joblib.dump(explainer, os.path.join(MODELS_DIR, "shap_explainer.pkl"))
        logger.info("SHAP analysis complete!")

    except Exception as e:
        logger.error(f"Explainability analysis failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    generate_explanations()
