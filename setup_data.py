"""Master data setup script for the India flood risk prediction project."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import data_pipeline
import data_merger

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

RAW_DIR = ROOT / "data" / "raw"
PROCESS_DIR = ROOT / "data" / "processed"


def create_directories() -> None:
    for path in [
        ROOT / "data",
        RAW_DIR,
        RAW_DIR / "imd_rainfall",
        RAW_DIR / "ndma_flood_records",
        RAW_DIR / "cwc_river_levels",
        RAW_DIR / "elevation",
        PROCESS_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def run_full_setup() -> None:
    create_directories()
    load_dotenv(dotenv_path=ROOT / ".env")

    logger.info("Checking Kaggle configuration...")
    data_pipeline.setup_kaggle()

    logger.info("Downloading IMD rainfall data...")
    data_pipeline.download_imd_rainfall()

    logger.info("Downloading NDMA flood records...")
    data_pipeline.download_ndma_flood_records()

    logger.info("Downloading CWC river data (manual fallback expected)...")
    data_pipeline.download_cwc_river_data()

    logger.info("Downloading elevation data...")
    data_pipeline.download_elevation_data()

    logger.info("Checking download status...")
    data_pipeline.check_all_downloads()

    merged_path = data_merger.merge_all_datasets()
    if merged_path:
        logger.info("Validating merged dataset...")
        merged = pd.read_csv(merged_path, parse_dates=["date"], low_memory=False)
        data_merger.validate_data_quality(merged)
    else:
        logger.error("Merge step did not complete. Please resolve missing datasets first.")


def run_kaggle_downloads_only() -> None:
    create_directories()
    load_dotenv(dotenv_path=ROOT / ".env")
    if data_pipeline.setup_kaggle():
        data_pipeline.download_imd_rainfall(kaggle_only=True)
        data_pipeline.download_ndma_flood_records(kaggle_only=True)
    else:
        logger.error("Kaggle is not configured. Run setup_kaggle() first.")


def run_merge_only() -> None:
    create_directories()
    merged_path = data_merger.merge_all_datasets()
    if merged_path:
        merged = pd.read_csv(merged_path, parse_dates=["date"], low_memory=False)
        data_merger.validate_data_quality(merged)


def main() -> None:
    parser = argparse.ArgumentParser(description="Setup raw and processed data for the flood risk project.")
    parser.add_argument("--check", action="store_true", help="Check current download status only.")
    parser.add_argument("--kaggle", action="store_true", help="Download Kaggle datasets only.")
    parser.add_argument("--merge", action="store_true", help="Run only the merge step.")
    args = parser.parse_args()

    if args.check:
        create_directories()
        data_pipeline.check_all_downloads()
        return
    if args.kaggle:
        run_kaggle_downloads_only()
        return
    if args.merge:
        run_merge_only()
        return

    run_full_setup()


if __name__ == "__main__":
    main()
