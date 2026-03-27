"""
Train Forecasting Models

Trains per-item XGBoost models and saves them as .pkl files for use by the API.

Usage:
    python scripts/train_models.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd

from src.models.features import create_features
from src.models.forecaster import train_models, generate_future_features, predict
from src.evaluation.metrics import generate_abc_analysis, print_abc_report
from src.utils.config import PROCESSED_DIR, PREDICTIONS_DIR, MODELS_DIR


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
    print(f"Aggregated to weekly: {len(df_weekly)} observations")
    print(f"Date range: {df_weekly['Date'].min()} to {df_weekly['Date'].max()}")
    return df_weekly


def main():
    print("=" * 80)
    print("MODEL TRAINING")
    print("=" * 80)

    df_raw = load_and_prep_data(PROCESSED_DIR / "daily_item_sales.csv")

    print("\nCreating features...")
    df_feat = create_features(df_raw)
    print(f"Features created: {df_feat.shape[1]} columns")

    print(f"\nSaving models to: {MODELS_DIR}")
    item_models, global_model, dow_factors = train_models(df_feat, MODELS_DIR)

    print("\n" + "=" * 80)
    print("GENERATING 3-MONTH FORECAST")
    print("=" * 80)

    future_features = generate_future_features(df_feat, future_weeks=12)
    print(f"Future feature rows: {len(future_features)}")

    future_predictions = predict(
        future_features,
        item_models=item_models,
        global_model=global_model,
        dow_factor_dict=dow_factors,
    )
    print(f"Predictions generated: {len(future_predictions)} rows")

    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    future_predictions[["Date", "Item", "Predicted"]].rename(
        columns={"Predicted": "Quantity_Sold"}
    ).to_csv(PREDICTIONS_DIR / "3_month_forecasts.csv", index=False)
    print(f"Forecasts saved to: {PREDICTIONS_DIR / '3_month_forecasts.csv'}")

    print("\n" + "=" * 80)
    print("MODEL EVALUATION")
    print("=" * 80)

    from src.models.forecaster import train_and_predict

    test_pred = train_and_predict(df_feat)
    analysis = generate_abc_analysis(test_pred)
    print_abc_report(analysis)

    print("\nTraining complete!")


if __name__ == "__main__":
    main()
