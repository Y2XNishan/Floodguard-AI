import pandas as pd
import os
from pathlib import Path

files = [
    'data/raw/imd_rainfall/daily-rainfall-at-state-level.csv',
    'data/raw/imd_rainfall/district wise rainfall normal.csv',
    'data/raw/imd_rainfall/rainfall in india 1901-2015.csv',
    'data/raw/ndma_flood_records/India_Floods_Inventory.csv',
    'data/raw/flood_risk/flood_risk_dataset_india.csv',
    'data/raw/flood_inventory/India_Flood_Inventory_v3.csv'
]

dfs = []
for file in files:
    if os.path.exists(file):
        df = pd.read_csv(file)
        print(f"\nFile: {file}")
        print("Columns:", list(df.columns))
        print("First 5 rows:")
        print(df.head())
        dfs.append(df)
    else:
        print(f"File not found: {file}")

# Now, identify columns
print("\n\nColumn Identification:")
print("Date/Year columns:")
for i, df in enumerate(dfs):
    cols = [c for c in df.columns if 'date' in c.lower() or 'year' in c.lower()]
    if cols:
        print(f"File {i+1}: {cols}")
    else:
        print(f"File {i+1}: None")

print("\nDistrict/State columns:")
for i, df in enumerate(dfs):
    cols = [c for c in df.columns if 'district' in c.lower() or 'state' in c.lower() or 'subdivision' in c.lower()]
    if cols:
        print(f"File {i+1}: {cols}")
    else:
        print(f"File {i+1}: None")

print("\nRainfall columns:")
for i, df in enumerate(dfs):
    cols = [c for c in df.columns if 'rain' in c.lower() or 'actual' in c.lower() or any(mon in c.upper() for mon in ['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC'])]
    if cols:
        print(f"File {i+1}: {cols}")
    else:
        print(f"File {i+1}: None")

print("\nFlood occurrence columns:")
for i, df in enumerate(dfs):
    cols = [c for c in df.columns if 'flood' in c.lower() or 'occur' in c.lower()]
    if cols:
        print(f"File {i+1}: {cols}")
    else:
        print(f"File {i+1}: None")

# For merging, since different structures, we'll concatenate all dataframes
master_df = pd.concat(dfs, ignore_index=True, sort=False)

# Save to CSV
output_path = 'data/processed/india_flood_master.csv'
master_df.to_csv(output_path, index=False)

# Summary
total_rows = len(master_df)
columns = list(master_df.columns)

# Flood rate: count rows where Flood Occurred == 1
flood_count = 0
if 'Flood Occurred' in master_df.columns:
    flood_count = master_df['Flood Occurred'].sum()
    flood_rate = flood_count / total_rows * 100
else:
    flood_rate = 0  # or estimate

# Date range
date_cols = [c for c in columns if 'date' in c.lower() or 'year' in c.lower()]
if date_cols:
    all_dates = []
    for col in date_cols:
        try:
            dates = pd.to_datetime(master_df[col], errors='coerce')
            all_dates.extend(dates.dropna())
        except:
            pass
    if all_dates:
        min_date = min(all_dates)
        max_date = max(all_dates)
        date_range = f"{min_date.date()} to {max_date.date()}"
    else:
        date_range = "N/A"
else:
    date_range = "N/A"

# States/districts
state_cols = [c for c in columns if 'state' in c.lower() or 'subdivision' in c.lower()]
district_cols = [c for c in columns if 'district' in c.lower()]
unique_states = set()
unique_districts = set()
for col in state_cols:
    unique_states.update(master_df[col].dropna().unique())
for col in district_cols:
    unique_districts.update(master_df[col].dropna().unique())
states_covered = len(unique_states)
districts_covered = len(unique_districts)

print(f"\n\nFinal Summary:")
print(f"Total rows: {total_rows}")
print(f"Columns available: {columns}")
print(f"Flood rate percentage: {flood_rate:.2f}%")
print(f"Date range covered: {date_range}")
print(f"States/districts covered: {states_covered} states, {districts_covered} districts")