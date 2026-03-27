"""
Sales Forecasting with XGBoost

Trains per-item XGBoost models and generates forecasts with ABC analysis.

Usage:
    python scripts/002_data_forecast.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import numpy as np

from src.models.features import create_features
from src.models.forecaster import train_and_predict, train_models
from src.evaluation.metrics import print_abc_report, generate_abc_analysis
from src.utils.config import PROCESSED_DIR, PREDICTIONS_DIR


def load_and_prep_data(filepath):
    print(f"Loading data from: {filepath}")
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df[~df["Item"].str.strip().str.lower().str.startswith("add")]
    df_weekly = (
        df.set_index("Date")
        .groupby("Item")
        .resample("W-MON")["Quantity_Sold"]
        .sum()
        .reset_index()
    )
    return df_weekly


def main():
    df_raw = load_and_prep_data(PROCESSED_DIR / "daily_item_sales.csv")
    df_feat = create_features(df_raw)
    test_pred = train_and_predict(df_feat)
    analysis = generate_abc_analysis(test_pred)
    print_abc_report(analysis)


if __name__ == "__main__":
    main()
