"""Shared paths and constants so no module hard-codes them."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
DB_PATH = PROCESSED_DIR / "instamart.db"
OUTPUT_DIR = ROOT / "outputs"
MODELS_DIR = ROOT / "models"

SEED = 42
POLICY_DIR = ROOT / "data" / "policies"
