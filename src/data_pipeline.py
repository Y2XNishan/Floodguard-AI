"""Data collection utilities for the India flood risk pipeline.

This module contains download helpers that prefer automatic sources and
fall back to clear manual instructions when data cannot be fetched.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

import requests
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
IMD_DIR = RAW_DIR / "imd_rainfall"
NDMA_DIR = RAW_DIR / "ndma_flood_records"
CWC_DIR = RAW_DIR / "cwc_river_levels"
ELEVATION_DIR = RAW_DIR / "elevation"


def _ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _is_kaggle_installed() -> bool:
    return importlib.util.find_spec("kaggle") is not None


def _validate_csv(path: Path) -> bool:
    if not path.exists() or path.stat().st_size < 100:
        return False

    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            reader = csv.reader(handle)
            headers = next(reader, None)
            first_row = next(reader, None)
            return bool(headers) and bool(first_row)
    except Exception as exc:
        logger.warning("CSV validation failed for %s: %s", path, exc)
        return False


def _download_file(url: str, dest_path: Path, params: Optional[dict] = None) -> bool:
    _ensure_parent_dir(dest_path)
    try:
        with requests.get(url, params=params, stream=True, timeout=30) as response:
            response.raise_for_status()
            total = int(response.headers.get("content-length", 0))
            with dest_path.open("wb") as handle:
                with tqdm(total=total, unit="B", unit_scale=True, desc=dest_path.name) as progress:
                    for chunk in response.iter_content(chunk_size=32_768):
                        if chunk:
                            handle.write(chunk)
                            progress.update(len(chunk))
        return dest_path.exists() and dest_path.stat().st_size > 0
    except Exception as exc:
        logger.warning("Failed to download %s: %s", url, exc)
        return False


def _run_kaggle_download(dataset: str, destination: Path) -> Optional[Path]:
    _ensure_parent_dir(destination)
    if not _is_kaggle_installed():
        logger.warning("Kaggle package not installed.")
        return None

    temp_dir = Path(tempfile.mkdtemp())
    try:
        command = [
            sys.executable,
            "-m",
            "kaggle",
            "datasets",
            "download",
            "-d",
            dataset,
            "-p",
            str(temp_dir),
            "--force",
        ]
        result = subprocess.run(command, capture_output=True, text=True, timeout=180)
        if result.returncode != 0:
            logger.warning("Kaggle download failed for %s: %s", dataset, result.stderr.strip())
            return None

        archive_files = list(temp_dir.glob("*.zip"))
        if not archive_files:
            logger.warning("No Kaggle archive found for dataset %s", dataset)
            return None

        archive_path = archive_files[0]
        shutil.unpack_archive(str(archive_path), extract_dir=str(temp_dir))

        csv_candidates = list(temp_dir.rglob("*.csv"))
        if not csv_candidates:
            logger.warning("No CSV found inside Kaggle archive for %s", dataset)
            return None

        best_file = csv_candidates[0]
        output = destination
        shutil.move(str(best_file), str(output))
        return output
    except Exception as exc:
        logger.warning("Unable to download Kaggle dataset %s: %s", dataset, exc)
        return None
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def setup_kaggle() -> bool:
    """Check whether Kaggle is configured and print setup instructions if not."""
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    package_installed = _is_kaggle_installed()
    if kaggle_json.exists() and package_installed:
        logger.info("Kaggle is configured and ready.")
        return True

    logger.warning("Kaggle is not fully configured.")
    print("Kaggle setup is required to download datasets from Kaggle.")
    print("1. Go to https://kaggle.com/settings")
    print("2. Click API -> Create New Token")
    print("3. Place the downloaded kaggle.json in ~/.kaggle/")
    print("4. Run: pip install kaggle")
    if not package_installed:
        print("   (python package missing)")
    if not kaggle_json.exists():
        print("   (configuration file ~/.kaggle/kaggle.json not found)")
    return False


def _print_manual_instructions(title: str, instructions: list[str]) -> None:
    print(f"\n=== {title} ===")
    for step in instructions:
        print(step)
    print("=== End of instructions ===\n")


def download_imd_rainfall(kaggle_only: bool = False) -> Optional[Path]:
    """Download district-wise rainfall data for India."""
    output_path = IMD_DIR / "imd_rainfall.csv"
    _ensure_parent_dir(output_path)
    if output_path.exists() and _validate_csv(output_path):
        logger.info("IMD rainfall data already exists at %s", output_path)
        return output_path

    if not kaggle_only:
        api_key = os.environ.get("DATA_GOV_API_KEY")
        if api_key:
            logger.info("Attempting automatic download from data.gov.in for IMD rainfall.")
            resource_id = __import__("os").environ.get(
                "IMD_RAINFALL_RESOURCE_ID",
                "9ef84268-d588-465a-a308-a864a43d0070",
            )
            try:
                url = f"https://api.data.gov.in/resource/{resource_id}"
                params = {
                    "api-key": api_key,
                    "format": "csv",
                    "offset": 0,
                    "limit": 100000,
                }
                if _download_file(url, output_path, params=params) and _validate_csv(output_path):
                    logger.info("Downloaded IMD rainfall data from data.gov.in")
                    return output_path
            except Exception as exc:
                logger.warning("data.gov.in download attempt failed: %s", exc)

    if setup_kaggle():
        logger.info("Attempting Kaggle download for IMD rainfall.")
        kaggle_path = IMD_DIR / "rainfall_in_india.csv"
        result = _run_kaggle_download("rajanand/rainfall-in-india", kaggle_path)
        if result and _validate_csv(result):
            result.rename(output_path)
            return output_path

    _print_manual_instructions(
        "Manual IMD Rainfall Download",
        [
            "1. Visit https://data.gov.in and register for a free API key.",
            "2. Search for 'District Wise Seasonal and Annual Rainfall'.",
            "3. Download the dataset as CSV.",
            f"4. Save the file to {output_path}" ,
            "5. Ensure columns: date, district, state, rainfall_mm.",
            "6. If API access fails, try browsing https://imdpune.gov.in for district rainfall reports.",
        ],
    )
    return None


def download_cwc_river_data() -> Optional[Path]:
    """Provide instructions for Central Water Commission river level data."""
    output_path = CWC_DIR / "cwc_river_levels.csv"
    _ensure_parent_dir(output_path)
    if output_path.exists() and _validate_csv(output_path):
        logger.info("CWC river level data already exists at %s", output_path)
        return output_path

    _print_manual_instructions(
        "Manual CWC River Level Download",
        [
            "1. Go to https://cwc.gov.in/ and open the Flood Forecasting section.",
            "2. Click on Historical Data and locate water level or flood forecasting archives.",
            "3. Download station-level river data for Brahmaputra, Ganga, Godavari, Krishna, Mahanadi, Kosi, Gandak, Yamuna, Narmada, Tapti.",
            "4. If the CWC portal is unavailable, try https://india-wris.nrsc.gov.in/ for river gauge data.",
            f"5. Save the file to {output_path} with columns: date, station_name, district, state, river_name, water_level_m, danger_level_m, warning_level_m, discharge_cumecs.",
            "6. If you must download multiple station files, merge them into one CSV with the same column names.",
            "7. Take screenshots of the download page or names of the files you downloaded for traceability.",
        ],
    )
    return None


def download_ndma_flood_records(kaggle_only: bool = False) -> Optional[Path]:
    """Download historical flood inventory records from NDMA sources."""
    output_path = NDMA_DIR / "ndma_flood_records.csv"
    _ensure_parent_dir(output_path)
    if output_path.exists() and _validate_csv(output_path):
        logger.info("NDMA flood records already exist at %s", output_path)
        return output_path

    if setup_kaggle():
        logger.info("Attempting Kaggle download for NDMA flood records.")
        kaggle_path = NDMA_DIR / "india_flood_inventory.csv"
        result = _run_kaggle_download("saswat9uhan/india-flood-inventory", kaggle_path)
        if result and _validate_csv(result):
            result.rename(output_path)
            return output_path
    if not kaggle_only:
        api_key = __import__("os").environ.get("DATA_GOV_API_KEY")
        if api_key:
            logger.info("Attempting data.gov.in download for NDMA flood records.")
            resource_id = __import__("os").environ.get("NDMA_FLOOD_RESOURCE_ID", "")
            if resource_id:
                try:
                    url = f"https://api.data.gov.in/resource/{resource_id}"
                    params = {
                        "api-key": api_key,
                        "format": "csv",
                        "offset": 0,
                        "limit": 100000,
                    }
                    if _download_file(url, output_path, params=params) and _validate_csv(output_path):
                        logger.info("Downloaded NDMA flood records from data.gov.in")
                        return output_path
                except Exception as exc:
                    logger.warning("data.gov.in NDMA download failed: %s", exc)

    _print_manual_instructions(
        "Manual NDMA Flood Records Download",
        [
            "1. Go to https://ndma.gov.in/ and open the Disaster Data section.",
            "2. Search for historical flood records or flood inventory datasets.",
            "3. Download the data in CSV format.",
            f"4. Save the file to {output_path}",
            "5. Ensure columns: year, state, district, flood_occurred, flood_severity (0-3), area_affected_ha, people_affected, crops_damaged.",
        ],
    )
    return None


def download_elevation_data() -> Optional[Path]:
    """Download elevation stats or raw elevation data for Indian districts."""
    output_path = ELEVATION_DIR / "elevation.csv"
    _ensure_parent_dir(output_path)
    if output_path.exists() and _validate_csv(output_path):
        logger.info("Elevation stats already exist at %s", output_path)
        return output_path

    api_key = os.environ.get("OPENTOPO_API_KEY")
    if api_key:
        logger.info("Attempting OpenTopography download for SRTM elevation data.")
        url = "https://portal.opentopography.org/API/globaldem"
        params = {
            "demtype": "SRTMGL1",
            "south": 6.0,
            "north": 38.0,
            "west": 68.0,
            "east": 98.0,
            "outputFormat": "CSV",
            "API_Key": api_key,
        }
        if _download_file(url, output_path, params=params) and _validate_csv(output_path):
            logger.info("Elevation data downloaded to %s", output_path)
            return output_path
        logger.warning("OpenTopography API download failed or did not return a CSV.")

    _print_manual_instructions(
        "Manual Elevation Data Download",
        [
            "1. Go to https://opentopography.org/ and register for a free API key.",
            "2. Use the GlobalDEM API to request elevation data for India.",
            "3. If the API cannot return district summaries, use USGS EarthExplorer or Bhuvan ISRO.",
            "4. Download elevation data for each district or a polygon covering India.",
            f"5. Compute per-district statistics and save them to {output_path}",
            "6. Ensure columns: district, state, mean_elevation_m, min_elevation_m, slope_degrees.",
        ],
    )
    return None


def check_all_downloads() -> dict[str, dict[str, str]]:
    """Check the current download status for all required datasets."""
    datasets = {
        "IMD Rainfall": {
            "path": IMD_DIR / "imd_rainfall.csv",
            "loader": _validate_csv,
        },
        "CWC River": {
            "path": CWC_DIR / "cwc_river_levels.csv",
            "loader": _validate_csv,
        },
        "NDMA Records": {
            "path": NDMA_DIR / "ndma_flood_records.csv",
            "loader": _validate_csv,
        },
        "Elevation": {
            "path": ELEVATION_DIR / "elevation.csv",
            "loader": _validate_csv,
        },
    }

    summary: dict[str, dict[str, str]] = {}
    print("Dataset          | Status    | Records | Path")
    print("---------------------------------------------------------------")
    for label, info in datasets.items():
        path = info["path"]
        ready = info["loader"](path)
        status = "✅ Ready" if ready else "❌ Missing"
        records = "-"
        if ready:
            try:
                with path.open("r", encoding="utf-8", errors="replace") as handle:
                    records = str(sum(1 for _ in handle) - 1)
            except Exception:
                records = "?"
        print(f"{label:15} | {status:8} | {records:7} | {path}")
        summary[label] = {
            "Status": status,
            "Records": records,
            "Path": str(path) if ready else "-",
        }
    return summary
