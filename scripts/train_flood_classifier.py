import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.flood_classifier import train_classifier


if __name__ == "__main__":
    train_classifier()
