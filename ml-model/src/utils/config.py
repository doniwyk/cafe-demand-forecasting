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

REBRANDING_DATE = "2025-05-01"
INDONESIAN_HOLIDAYS = [
    "2024-01-01",
    "2024-04-10",
    "2024-04-11",
    "2024-04-12",
    "2024-05-01",
    "2024-05-23",
    "2024-06-01",
    "2024-06-17",
    "2024-12-25",
    "2025-01-01",
    "2025-03-01",
    "2025-03-30",
    "2025-03-31",
    "2025-05-01",
    "2025-05-12",
    "2025-06-01",
    "2025-06-07",
    "2025-12-25",
    "2026-01-01",
    "2026-02-17",
    "2026-03-09",
    "2026-05-01",
    "2026-05-21",
    "2026-12-25",
]

RAMADAN_RANGES = [
    ("2024-02-28", "2024-04-10"),
    ("2025-02-28", "2025-04-10"),
    ("2026-02-28", "2026-04-10"),
]

FEATURE_COLUMNS = [
    "Item_Code",
    "Month",
    "Week",
    "Year",
    "DOY",
    "DOW",
    "Sin_Week",
    "Cos_Week",
    "Weeks_Since_Start",
    "Is_Payday_Week",
    "Is_Weekend_Before_Payday",
    "Is_Holiday",
    "Is_Ramadan",
    "Is_Post_Rebranding",
    "Weeks_Since_Rebrand",
    "Post_Rebrand_Surge_Ratio",
    "Recent_vs_Old_Trend",
    "Lag_1",
    "Lag_2",
    "Lag_4",
    "Roll_Mean_4",
    "Roll_Mean_12",
    "Roll_Std_4",
    "Roll_Q95_4",
    "EWMA_4",
    "EWMA_12",
    "Diff_1",
    "Accel_2",
]

FEATURE_COLUMNS_DAILY = [
    "Item_Code",
    "Month",
    "Week",
    "Year",
    "DOY",
    "DOW",
    "Sin_Week",
    "Cos_Week",
    "Weeks_Since_Start",
    "Is_Payday_Week",
    "Is_Weekend_Before_Payday",
    "Is_Holiday",
    "Is_Ramadan",
    "Is_Post_Rebranding",
    "Weeks_Since_Rebrand",
    "Post_Rebrand_Surge_Ratio",
    "Recent_vs_Old_Trend",
    "Lag_1",
    "Lag_7",
    "Lag_14",
    "Roll_Mean_7",
    "Roll_Mean_28",
    "Roll_Std_7",
    "Roll_Q95_7",
    "EWMA_7",
    "EWMA_28",
    "Diff_1",
    "Accel_2",
]


def get_feature_columns(frequency: str) -> list:
    return FEATURE_COLUMNS_DAILY if frequency == "daily" else FEATURE_COLUMNS
