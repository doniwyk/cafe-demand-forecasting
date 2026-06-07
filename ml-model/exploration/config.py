"""Local config for exploration — no dependency on src/."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
SALES_FORECASTING_DIR = PROCESSED_DIR / "sales_forecasting"
MODELS_DIR = PROJECT_ROOT / "models"

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
    "Trend_7",
    "Momentum_3",
    "Price_Level",
    "Lag_7",
    "Lag_14",
    "Lag_28",
    "Lag_182",
    "Weekly_Ratio",
    "Monthly_Ratio",
    "Seasonal_Diff",
    "DOW",
    "Is_Weekend",
]
