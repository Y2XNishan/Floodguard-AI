import cv2
import numpy as np
import shutil
from pathlib import Path


dataset_dir = Path("data/raw/flood_images/Cyclone_Wildfire_Flood_Earthquake_Database/Cyclone_Wildfire_Flood_Earthquake_Database")
flood_dir = dataset_dir / "Flood"
non_flood_dirs = [
    dataset_dir / "Cyclone",
    dataset_dir / "Wildfire",
    dataset_dir / "Earthquake",
]
output_dir = Path("data/processed/flood_classified")
CLASSES = ["mild", "moderate", "severe", "no_flood"]
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


# Create output folders
if output_dir.exists():
    shutil.rmtree(output_dir)

for split in ["train", "val"]:
    for cls in CLASSES:
        (output_dir / split / cls).mkdir(parents=True, exist_ok=True)


def classify_image(img_path):
    return improved_classify_image(img_path)


def improved_classify_image(img_path):
    img = cv2.imread(str(img_path))
    if img is None:
        return "moderate"

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    h, w = img.shape[:2]
    total = h * w

    # Blue water (clear)
    blue = cv2.inRange(hsv, (90, 30, 30), (140, 255, 255))
    # Brown/muddy water
    brown = cv2.inRange(hsv, (8, 40, 20), (25, 255, 200))
    # Dark water
    dark = cv2.inRange(hsv, (0, 0, 0), (180, 255, 60))
    # Grey water
    grey = cv2.inRange(hsv, (0, 0, 60), (180, 30, 180))

    water = (
        cv2.countNonZero(blue)
        + cv2.countNonZero(brown)
        + cv2.countNonZero(dark) * 0.5
        + cv2.countNonZero(grey) * 0.3
    )

    ratio = water / total

    # Also check image brightness; retained for future threshold tuning.
    brightness = img.mean()

    # Lower half of image (ground level)
    lower_half = img[h // 2:, :]
    lower_hsv = cv2.cvtColor(lower_half, cv2.COLOR_BGR2HSV)
    lower_blue = cv2.inRange(lower_hsv, (90, 30, 30), (140, 255, 255))
    lower_brown = cv2.inRange(lower_hsv, (8, 40, 20), (25, 255, 200))
    lower_ratio = (
        cv2.countNonZero(lower_blue) + cv2.countNonZero(lower_brown)
    ) / (h // 2 * w)

    # Combined score
    score = ratio * 0.6 + lower_ratio * 0.4

    if score > 0.45:
        return "severe"
    if score > 0.20:
        return "moderate"
    return "mild"


def get_images(directory):
    return [
        path for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    ]


# Process all images
images = get_images(flood_dir)
non_flood_images = []
for directory in non_flood_dirs:
    non_flood_images.extend(get_images(directory))

print(f"Found {len(images)} flood images")
print(f"Found {len(non_flood_images)} non-flood images")

counts = {cls: 0 for cls in CLASSES}
for i, img_path in enumerate(images):
    label = classify_image(img_path)

    # 80% train, 20% val split
    split = "train" if i % 5 != 0 else "val"

    dest = output_dir / split / label / img_path.name
    shutil.copy2(img_path, dest)

    counts[label] += 1

    if i % 100 == 0:
        print(f"Processed {i}/{len(images)}...")

for i, img_path in enumerate(non_flood_images):
    split = "train" if i % 5 != 0 else "val"
    dest = output_dir / split / "no_flood" / f"{img_path.parent.name}_{img_path.name}"
    shutil.copy2(img_path, dest)
    counts["no_flood"] += 1

    if i % 250 == 0:
        print(f"Processed non-flood {i}/{len(non_flood_images)}...")

print("Done!")
print(
    f"Mild: {counts['mild']}, "
    f"Moderate: {counts['moderate']}, "
    f"Severe: {counts['severe']}, "
    f"No Flood: {counts['no_flood']}"
)
print(f"Saved to: {output_dir}")
