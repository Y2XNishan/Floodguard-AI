# Data Download Guide

## Overview
This guide documents the data sources required for the India Flood Risk Prediction system.
The repository expects raw downloads under `data/raw/` and processed outputs under `data/processed/`.

## Required datasets
- **IMD rainfall** — district-level daily rainfall for India
- **CWC river levels** — river gauge measurements for major river basins
- **NDMA flood inventory** — historical flood event labels and severity
- **Elevation data** — district-level elevation statistics

## Folder structure after download
```
data/
  raw/
    imd_rainfall/imd_rainfall.csv
    cwc_river_levels/cwc_river_levels.csv
    ndma_flood_records/ndma_flood_records.csv
    elevation/elevation.csv
```

## Source-specific instructions

### IMD Rainfall
- Preferred source: **data.gov.in**
- Register at: https://data.gov.in/user/register
- Store API key in environment variable: `DATA_GOV_API_KEY`
- If API download fails, visit:
  - https://data.gov.in
  - https://imdpune.gov.in
- Required file:
  - `data/raw/imd_rainfall/imd_rainfall.csv`
- Required columns:
  - `date`, `district`, `state`, `rainfall_mm`

#### Manual download fallback
1. Sign in or register on data.gov.in.
2. Search for `District Wise Seasonal and Annual Rainfall`.
3. Download the dataset in CSV format.
4. Save the file as `data/raw/imd_rainfall/imd_rainfall.csv`.

### NDMA Flood Records
- Preferred source: Kaggle dataset `saswat9uhan/india-flood-inventory`
- Alternative: data.gov.in with a search for `flood affected areas india`
- Manual source: https://ndma.gov.in → Disaster Data
- Required file:
  - `data/raw/ndma_flood_records/ndma_flood_records.csv`
- Required columns:
  - `year`, `state`, `district`, `flood_occurred`, `flood_severity`, `area_affected_ha`, `people_affected`, `crops_damaged`

#### Manual download fallback
1. Visit https://ndma.gov.in.
2. Open the Disaster Data or Publications section.
3. Download historical flood records for India.
4. Save as `data/raw/ndma_flood_records/ndma_flood_records.csv`.

### CWC River Data
- There is no reliable bulk API for CWC river data.
- Use these portals when possible:
  - https://cwc.gov.in/ → Flood Forecasting → Historical Data
  - https://india-wris.nrsc.gov.in/
- Required file:
  - `data/raw/cwc_river_levels/cwc_river_levels.csv`
- Required columns:
  - `date`, `station_name`, `district`, `state`, `river_name`, `water_level_m`, `danger_level_m`, `warning_level_m`, `discharge_cumecs`

#### Manual download fallback
1. Open https://cwc.gov.in/ and navigate to Flood Forecasting.
2. Find historical station-level river gauge data.
3. Download or export CSV for the listed rivers.
4. Combine station files into a single CSV, preserving the required columns.

### Elevation Data
- Preferred source: OpenTopography GlobalDEM API
- API docs: https://portal.opentopography.org/
- Store key in `.env` or environment variable: `OPENTOPO_API_KEY`
- Alternative sources:
  - https://earthexplorer.usgs.gov/
  - https://bhuvan.nrsc.gov.in/
- Required file:
  - `data/raw/elevation/elevation.csv`
- Required columns:
  - `district`, `state`, `mean_elevation_m`, `min_elevation_m`, `slope_degrees`

#### Manual download fallback
1. Register at https://portal.opentopography.org/.
2. Request SRTM or other global DEM tiles covering India.
3. Use GIS tools to compute district-level elevation statistics.
4. Save the aggregated district summary as `data/raw/elevation/elevation.csv`.

## Kaggle setup guide
1. Visit https://kaggle.com/settings.
2. Scroll to the API section.
3. Click **Create New Token**.
4. Download the generated `kaggle.json` file.
5. Place `kaggle.json` into `~/.kaggle/`.
6. Install the Kaggle package:
   ```bash
   pip install kaggle
   ```

### Common Kaggle issues
- `kaggle: command not found` → Install the Kaggle package and ensure Python scripts are on your PATH.
- `403 forbidden` or `API key invalid` → regenerate your Kaggle token and replace `~/.kaggle/kaggle.json`.

## Common errors and fixes
- `API key invalid`
  - Verify the key in `.env` or environment variables.
  - Reload the shell or restart the Python process.
- `District names don't match`
  - Normalize names by lowercasing and stripping punctuation.
  - Align district names to official state/district spellings.
- `kaggle: command not found`
  - Install with `pip install kaggle`.
  - Ensure `~/.kaggle/kaggle.json` exists.

## Expected file sizes
- IMD rainfall: ~50 MB
- NDMA flood inventory: ~5 MB
- CWC river levels: varies by source, expect 10-100 MB
- Elevation summaries: 1-10 MB for district stats

## How to add new years later
1. Download the latest raw data for each source.
2. Place the files in the corresponding `data/raw/...` path.
3. Run `python setup_data.py --merge` to refresh the merged dataset.
4. Re-run preprocessing and model training after the merge step.
