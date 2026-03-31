import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
ROOT_DIR = BASE_DIR.parent.parent
ML_MODEL_DIR = ROOT_DIR / "ml-model"

ML_DATA_DIR = ML_MODEL_DIR / "data"
ML_PROCESSED_DIR = ML_DATA_DIR / "processed"
ML_PREDICTIONS_DIR = ML_DATA_DIR / "predictions"
ML_RAW_DIR = ML_DATA_DIR / "raw"
ML_BOM_DIR = ML_RAW_DIR / "bom"
ML_MODELS_DIR = ML_MODEL_DIR / "models"

DAILY_ITEM_SALES_PATH = ML_PROCESSED_DIR / "daily_item_sales.csv"
DAILY_CATEGORY_SALES_PATH = ML_PROCESSED_DIR / "daily_category_sales.csv"
DAILY_TOTAL_SALES_PATH = ML_PROCESSED_DIR / "daily_total_sales.csv"
FORECAST_PATH = ML_PREDICTIONS_DIR / "3_month_forecasts.csv"
MENU_BOM_PATH = ML_BOM_DIR / "menu_bom.csv"
CONDIMENT_BOM_PATH = ML_BOM_DIR / "condiment_bom.csv"
CLEANED_SALES_PATH = ML_PROCESSED_DIR / "sales_data_cleaned.csv"
DAILY_RAW_MATERIAL_PATH = ML_PROCESSED_DIR / "daily_raw_material_requirements.csv"
ASSOCIATION_RULES_PATH = ML_PROCESSED_DIR / "association_rules_fpgrowth.csv"
FORECAST_SUMMARY_PATH = ML_PREDICTIONS_DIR / "forecast_summary.json"

STATIC_DIR = BASE_DIR.parent / "frontend" / "dist"

sys_path = str(ML_MODEL_DIR)
if sys_path not in os.sys.path:
    os.sys.path.insert(0, sys_path)
