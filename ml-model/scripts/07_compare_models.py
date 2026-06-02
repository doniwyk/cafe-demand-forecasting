"""
Train and evaluate forecasting models with clean metrics output.

Usage:
    python scripts/07_compare_models.py -f daily
    python scripts/07_compare_models.py --model xgboost -f daily
    python scripts/07_compare_models.py --model xgboost --end-date 2026-05-25 -f daily
    python scripts/07_compare_models.py --model xgboost --end-date 2026-05-25 --sync-hus -f daily
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


def _load_from_db(end_date: str | None = None) -> pd.DataFrame:
    import sys, os
    backend_dir = os.path.join(os.path.dirname(__file__), "..", "..", "web", "backend")
    if backend_dir not in sys.path:
        sys.path.insert(0, backend_dir)
    from app.db.engine import sync_session
    from sqlalchemy import text

    session = sync_session()
    try:
        sql = (
            "SELECT dis.date, i.name as item, dis.quantity_sold "
            "FROM daily_item_sales dis JOIN items i ON dis.item_id = i.id"
        )
        params = {}
        if end_date:
            sql += " WHERE dis.date <= :end_date"
            params["end_date"] = end_date
        result = session.execute(text(sql), params)
        rows = result.fetchall()
        if not rows:
            return pd.DataFrame(columns=["Date", "Item", "Quantity_Sold"])
        df = pd.DataFrame(
            [tuple(row) for row in rows], columns=["Date", "Item", "Quantity_Sold"]
        )
        df["Date"] = pd.to_datetime(df["Date"])
        return df
    finally:
        session.close()


def _sync_hus():
    backend_dir = os.path.join(os.path.dirname(__file__), "..", "..", "web", "backend")
    if backend_dir not in sys.path:
        sys.path.insert(0, backend_dir)
    from scripts.sync_hus_sales import sync_sales
    result = sync_sales()
    print(f"  HUS sync: {result['inserted']} rows inserted, "
          f"{result['skipped_units']} units skipped")
    return result


def _clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df[~df["Item"].str.strip().str.lower().str.startswith("add")]
    return (
        df.set_index("Date")
        .groupby("Item")
        .resample("D")["Quantity_Sold"]
        .sum()
        .fillna(0)
        .reset_index()
    )


def _resample_to_frequency(df: pd.DataFrame, frequency: str) -> pd.DataFrame:
    freq_map = {"daily": "D", "weekly": "W-MON"}
    label = freq_map.get(frequency, "W-MON")
    return (
        df.set_index("Date")
        .groupby("Item")
        .resample(label)["Quantity_Sold"]
        .sum()
        .fillna(0)
        .reset_index()
    )


_FNS = {
    "XGBoost": lambda df, **kw: train_and_predict_xgb(df, frequency=kw["frequency"]),
    "RandomForest": lambda df, **kw: train_and_predict_rf(df, frequency=kw["frequency"]),
    "SARIMAX": lambda df, **kw: train_and_predict_sarimax(df, frequency=kw["frequency"]),
    "Prophet": lambda df, **kw: train_and_predict_prophet(df, frequency=kw["frequency"]),
}


def _print_results(results: list[dict], frequency: str):
    df = pd.DataFrame(results)
    print("\n" + "=" * 100)
    print(f"  RESULTS ({frequency.upper()})")
    print("=" * 100)
    header = (
        f"{'Model':<16} {'R2':>8} {'wMAPE':>8} {'MAE':>6} {'RMSE':>7} "
        f"{'Med.Acc':>8} {'±20%':>7} {'±50%':>7} {'Time':>7}  {'Status'}"
    )
    print(header)
    print("-" * 100)

    best_r2 = max(r["r2"] for r in results if r["status"] == "OK")
    best_wmape = min(r["wmape"] for r in results if r["status"] == "OK")

    for _, row in df.iterrows():
        r2_marker = " *" if row["r2"] == best_r2 else ""
        wm_marker = " *" if row["wmape"] == best_wmape else ""
        print(
            f"{row['model']:<16} "
            f"{row['r2']:>8.4f}{r2_marker:<2}"
            f"{row['wmape']:>8.1f}%{wm_marker:<2}"
            f"{row['mae']:>6.2f} "
            f"{row['rmse']:>6.2f} "
            f"{row['median_accuracy']:>7.1f}%"
            f"{row['within_20']:>7.1f}%"
            f"{row['within_50']:>7.1f}%"
            f"{row['time_sec']:>7.1f}s"
            f"  {row['status']}"
        )
    print("-" * 100)
    print("  * = best in category")
    print()


def _run_models(df_feat, models_to_run: list[str] | None, frequency: str) -> list[dict]:
    models = _FNS if models_to_run is None else {k: v for k, v in _FNS.items() if k in models_to_run}
    results = []
    for name, fn in models.items():
        print(f"\n{'=' * 70}", flush=True)
        print(f"  {name}", flush=True)
        print(f"{'=' * 70}", flush=True)
        t0 = time.time()
        try:
            pred = fn(df_feat, frequency=frequency)
            elapsed = time.time() - t0
            metrics = compute_item_metrics(pred["Quantity_Sold"], pred["Predicted"], pred["Item"])
            results.append({
                "model": name, "r2": metrics["r2"], "wmape": metrics["wmape"],
                "mae": metrics["mae"], "rmse": metrics.get("rmse", 0),
                "median_accuracy": metrics["median_period_accuracy"],
                "within_20": metrics["periods_within_20pct"],
                "within_50": metrics["periods_within_50pct"],
                "time_sec": round(elapsed, 1), "status": "OK",
            })
        except Exception as e:
            elapsed = time.time() - t0
            results.append({
                "model": name, "r2": 0, "wmape": 0, "mae": 0, "rmse": 0,
                "median_accuracy": 0, "within_20": 0, "within_50": 0,
                "time_sec": round(elapsed, 1), "status": f"FAIL: {e}",
            })
    return results


def main():
    model_names = {"xgboost", "random_forest", "sarimax", "prophet"}
    name_map = {"xgboost": "XGBoost", "random_forest": "RandomForest",
                "sarimax": "SARIMAX", "prophet": "Prophet"}

    parser = argparse.ArgumentParser(description="Train and evaluate forecasting models")
    parser.add_argument("-f", "--frequency", choices=["daily", "weekly"], default="daily")
    parser.add_argument("--model", default="xgboost",
                        choices=sorted(model_names | {"all"}),
                        help="Model to train (default: xgboost)")
    parser.add_argument("--end-date", help="Training data cutoff (e.g. 2026-05-25)")
    parser.add_argument("--sync-hus", action="store_true", help="Sync from HUS DB first")
    args = parser.parse_args()

    if args.sync_hus:
        print("Syncing from HUS DB...")
        _sync_hus()
        print("Sync complete.\n")

    use_db = bool(args.end_date or args.sync_hus)
    models_to_run = None if args.model == "all" else [name_map[args.model]]

    if use_db:
        print("Loading data from DB...")
        df_raw = _load_from_db(args.end_date)
        if df_raw.empty:
            print("No data loaded. Exiting.")
            return
        print(f"  Rows      : {len(df_raw)}")
        print(f"  Date range: {df_raw['Date'].min().date()} to {df_raw['Date'].max().date()}")

        df_clean = _clean_data(df_raw)
        df_freq = _resample_to_frequency(df_clean, args.frequency)
        print(f"  Aggregated: {len(df_freq)} observations ({args.frequency})")

        print("\nCreating features...")
        df_feat = create_features(df_freq, frequency=args.frequency)
        print(f"  Features  : {df_feat.shape[1]} columns, {len(df_feat)} rows")
    else:
        print("Loading data from CSV...")
        df_raw = load_and_prep_data(
            SALES_FORECASTING_DIR / "daily_item_sales.csv", frequency=args.frequency
        )
        print(f"  Rows      : {len(df_raw)}")
        print(f"  Date range: {df_raw['Date'].min().date()} to {df_raw['Date'].max().date()}")

        print("\nCreating features...")
        df_feat = create_features(df_raw, frequency=args.frequency)
        print(f"  Features  : {df_feat.shape[1]} columns, {len(df_feat)} rows")

    results = _run_models(df_feat, models_to_run, args.frequency)
    _print_results(results, args.frequency)


if __name__ == "__main__":
    main()
