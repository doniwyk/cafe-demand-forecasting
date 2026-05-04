"""
Model Comparison: XGBoost vs Random Forest vs SARIMAX vs Prophet

Runs all four models on the same train/test split and compares metrics.

Usage:
    python scripts/07_compare_models.py -f weekly
    python scripts/07_compare_models.py -f daily
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import time
import argparse
import pandas as pd

from src.models.features import create_features
from src.models.forecaster import load_and_prep_data, train_and_predict as train_and_predict_xgb
from src.models.forecaster_rf import train_and_predict_rf
from src.models.forecaster_sarimax import train_and_predict_sarimax
from src.models.forecaster_prophet import train_and_predict_prophet
from src.evaluation.metrics import compute_item_metrics
from src.utils.config import SALES_FORECASTING_DIR


def run_model(name, fn, df_feat, frequency):
    print(f"\n{'=' * 70}", flush=True)
    print(f"  {name}", flush=True)
    print(f"{'=' * 70}", flush=True)
    t0 = time.time()
    try:
        pred = fn(df_feat, frequency=frequency)
        elapsed = time.time() - t0
        metrics = compute_item_metrics(pred["Quantity_Sold"], pred["Predicted"], pred["Item"])
        return {
            "model": name,
            "r2": metrics["r2"],
            "wmape": metrics["wmape"],
            "mae": metrics["mae"],
            "median_accuracy": metrics["median_period_accuracy"],
            "within_20": metrics["periods_within_20pct"],
            "within_50": metrics["periods_within_50pct"],
            "time_sec": round(elapsed, 1),
            "status": "OK",
        }
    except Exception as e:
        elapsed = time.time() - t0
        return {
            "model": name,
            "r2": 0,
            "wmape": 0,
            "mae": 0,
            "median_accuracy": 0,
            "within_20": 0,
            "within_50": 0,
            "time_sec": round(elapsed, 1),
            "status": f"FAIL: {e}",
        }


def compare_models(frequency: str = "weekly"):
    print("=" * 70)
    print(f"  MODEL COMPARISON ({frequency.upper()})")
    print("=" * 70)

    df_raw = load_and_prep_data(
        SALES_FORECASTING_DIR / "daily_item_sales.csv", frequency=frequency
    )
    print("\nCreating features...")
    df_feat = create_features(df_raw, frequency=frequency)
    print(f"Features: {df_feat.shape[1]} columns, {len(df_feat)} rows")

    models = {
        "XGBoost": lambda df, **kw: train_and_predict_xgb(df, frequency=kw["frequency"]),
        "RandomForest": lambda df, **kw: train_and_predict_rf(df, frequency=kw["frequency"]),
        "SARIMAX": lambda df, **kw: train_and_predict_sarimax(df, frequency=kw["frequency"]),
        "Prophet": lambda df, **kw: train_and_predict_prophet(df, frequency=kw["frequency"]),
    }

    results = []
    for name, fn in models.items():
        res = run_model(name, fn, df_feat, frequency)
        results.append(res)

    df = pd.DataFrame(results)

    print("\n" + "=" * 90)
    print(f"  COMPARISON RESULTS ({frequency.upper()})")
    print("=" * 90)
    header = (
        f"{'Model':<16} {'R2':>8} {'wMAPE':>8} {'MAE':>8} "
        f"{'Med.Acc':>8} {'±20%':>8} {'±50%':>8} {'Time':>8} {'Status'}"
    )
    print(header)
    print("-" * 90)

    best_r2 = max(r["r2"] for r in results if r["status"] == "OK")
    best_wmape = min(r["wmape"] for r in results if r["status"] == "OK")

    for _, row in df.iterrows():
        r2_marker = " *" if row["r2"] == best_r2 else ""
        wm_marker = " *" if row["wmape"] == best_wmape else ""
        marker = r2_marker or wm_marker
        print(
            f"{row['model']:<16} "
            f"{row['r2']:>8.4f}{r2_marker:<2}"
            f"{row['wmape']:>8.1f}%{wm_marker:<2}"
            f"{row['mae']:>8.2f}"
            f"{row['median_accuracy']:>8.1f}%"
            f"{row['within_20']:>7.1f}%"
            f"{row['within_50']:>7.1f}%"
            f"{row['time_sec']:>7.1f}s"
            f"  {row['status']}"
        )

    print("-" * 90)
    print("  * = best in category")
    print()

    return df


def main():
    print('start')
    parser = argparse.ArgumentParser(description="Compare forecasting models")
    print('parser')
    parser.add_argument(
        "-f", "--frequency",
        choices=["daily", "weekly"],
        default="weekly",
    )
    print('argument')
    args = parser.parse_args()
    print('args')
    compare_models(args.frequency)


if __name__ == "__main__":
    main()
