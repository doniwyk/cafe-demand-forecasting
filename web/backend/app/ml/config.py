import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
ML_MODEL_DIR = PROJECT_ROOT / "ml-model"

DATA_DIR = ML_MODEL_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SALES_FORECASTING_DIR = PROCESSED_DIR / "sales_forecasting"
PREDICTIONS_DIR = DATA_DIR / "predictions"
MODELS_DIR = ML_MODEL_DIR / "models"
BOM_DIR = RAW_DIR / "bom"
SALES_DIR = RAW_DIR / "sales"

DISCONTINUED_ITEMS = [
    "Menawan",
]

FEATURE_COLUMNS = [
    "DOW", "Is_Weekend", "Month", "Year", "WeekOfYear", "DayOfMonth",
    "Quarter", "MonthStart", "MonthEnd", "Is_Holiday_Season",
    "WeekOfMonth", "DaysFromStart", "DOW_Sin", "DOW_Cos", "Month_Sin", "Month_Cos",
    "Days_Since_Last_Sale", "Sales_Last_7D",
    "Lag_1", "Lag_7", "Lag_14", "Lag_28",
    "Day_Total_Qty", "Day_Total_Items_Sold", "Day_Total_Beverage",
    "Day_Total_Food", "Day_Total_Qty_7D",
]

DAILY_SALES_CSV = PROCESSED_DIR / "sales_forecasting" / "daily_item_sales.csv"
MODEL_TYPE = "xgboost"
METADATA_FILE = "model_metadata.json"

N_BACKTEST_WINDOWS = 8
BACKTEST_WINDOW_DAYS = 7
