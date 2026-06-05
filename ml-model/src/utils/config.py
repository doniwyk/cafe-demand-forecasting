import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SALES_FORECASTING_DIR = PROCESSED_DIR / "sales_forecasting"
PREDICTIONS_DIR = DATA_DIR / "predictions"
MODELS_DIR = PROJECT_ROOT / "models"
BOM_DIR = RAW_DIR / "bom"
SALES_DIR = RAW_DIR / "sales"

DISCONTINUED_ITEMS = [
    "Menawan",
]

FEATURE_COLUMNS = [
    "Diff_1",
    "Lag_1",
    "Accel_2",
    "Seasonal_Strength",
    "Roll_Mean_7",
    "Roll_Mean_28",
    "Roll_Std_7",
    "Roll_Q95_7",
    "EWMA_7",
    "EWMA_28",
]
