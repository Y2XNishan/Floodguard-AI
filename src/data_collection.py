"""
Data collection module for Flood Risk Prediction System.
Provides functions to fetch data from public APIs with fallback to sample data.
"""

import os
import logging
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLE_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "sample_data.csv")


def load_all_data(use_sample: bool = True) -> pd.DataFrame:
    """
    Load the complete dataset from sample_data.csv or APIs.

    Args:
        use_sample: If True, load from sample_data.csv directly.

    Returns:
        Complete DataFrame with all features.
    """
    if use_sample:
        logger.info(f"Loading sample data from {SAMPLE_DATA_PATH}")
        try:
            df = pd.read_csv(SAMPLE_DATA_PATH, parse_dates=["date"])
            logger.info(f"Loaded {len(df)} records")
            return df
        except FileNotFoundError:
            logger.error("Sample data not found! Run 'python data/generate_data.py' first.")
            raise
    else:
        logger.warning("API fetching not configured. Falling back to sample data.")
        return load_all_data(use_sample=True)


if __name__ == "__main__":
    df = load_all_data()
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
