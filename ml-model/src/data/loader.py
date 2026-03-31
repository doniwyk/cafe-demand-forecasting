from __future__ import annotations

import pandas as pd
from pathlib import Path
from typing import Optional

from src.utils.config import (
    PROCESSED_DIR,
    RAW_DIR,
    BOM_DIR,
    SALES_DIR,
    PREDICTIONS_DIR,
)


def load_csv(filepath: str | Path, **kwargs) -> pd.DataFrame:
    return pd.read_csv(filepath, **kwargs)


def load_merged_sales() -> pd.DataFrame:
    return load_csv(PROCESSED_DIR / "sales_data.csv")


def load_cleaned_sales() -> pd.DataFrame:
    return load_csv(PROCESSED_DIR / "sales_data_cleaned.csv")


def load_daily_item_sales() -> pd.DataFrame:
    return load_csv(PROCESSED_DIR / "daily_item_sales.csv")


def load_daily_category_sales() -> pd.DataFrame:
    return load_csv(PROCESSED_DIR / "daily_category_sales.csv")


def load_daily_total_sales() -> pd.DataFrame:
    return load_csv(PROCESSED_DIR / "daily_total_sales.csv")


def load_forecasts() -> pd.DataFrame:
    return load_csv(PREDICTIONS_DIR / "3_month_forecasts.csv")


def load_menu_bom() -> pd.DataFrame:
    return load_csv(BOM_DIR / "menu_bom.csv")


def load_condiment_bom() -> pd.DataFrame:
    return load_csv(BOM_DIR / "condiment_bom.csv")
