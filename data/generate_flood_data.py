import pandas as pd
import numpy as np
import random
import os

districts_df = pd.read_csv("data/india_districts.csv")

HIGH_RISK = ["Assam","Bihar","Odisha","West Bengal","Kerala",
             "Uttar Pradesh","Uttarakhand","Manipur","Tripura","Meghalaya"]
MEDIUM_RISK = ["Maharashtra","Andhra Pradesh","Telangana",
               "Tamil Nadu","Gujarat","Madhya Pradesh","Jharkhand","Chhattisgarh"]

rows = []
for _, row in districts_df.iterrows():
    district = row["district"]  # adjust column name if different
    state = row["state"]        # adjust column name if different
    
    seed = hash(district) % 9999
    random.seed(seed)
    
    if state in HIGH_RISK:
        base_events = (3, 9)
        base_area = (10000, 50000)
        base_people = (50000, 200000)
        base_damage = (150, 1000)
    elif state in MEDIUM_RISK:
        base_events = (1, 5)
        base_area = (3000, 20000)
        base_people = (10000, 80000)
        base_damage = (50, 400)
    else:
        base_events = (0, 2)
        base_area = (500, 5000)
        base_people = (1000, 15000)
        base_damage = (10, 100)
    
    for year in range(2015, 2025):
        events = random.randint(*base_events)
        area = random.randint(*base_area)
        people = random.randint(*base_people)
        damage = random.randint(*base_damage)
        
        # Kerala 2018 spike
        if state == "Kerala" and year == 2018:
            events = min(events * 3, 15)
            area *= 3
            people *= 3
            damage *= 3
        
        # Assam/Bihar 2022 spike
        if state in ["Assam", "Bihar"] and year == 2022:
            events = min(int(events * 1.5), 12)
            area = int(area * 1.5)
            people = int(people * 1.5)
            damage = int(damage * 1.5)
        
        rows.append({
            "district": district,
            "state": state,
            "year": year,
            "flood_events": events,
            "area_affected_ha": area,
            "people_affected": people,
            "damage_cr": damage
        })

df = pd.DataFrame(rows)
df.to_csv("data/ndma_flood_history.csv", index=False)
print(f"Generated {len(df)} rows for {df['district'].nunique()} districts")
