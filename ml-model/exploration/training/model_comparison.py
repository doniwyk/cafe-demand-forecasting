"""Model Comparison: XGBoost (quantile) vs Random Forest
========================================================
Trains both models on the same data with the same feature set
(current 15 features from the inference pipeline), evaluates with
3-fold expanding window cross-validation, and compares results.

Uses cafe_db data, post-rebrand only, same feature builder as inference.

Run: python exploration/training/model_comparison.py
"""
from __future__ import annotations

import json
import pickle
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

BASE_DIR = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(BASE_DIR))

from inference.forecast import (
    load_all_items,
    build_item_features,
    FEATURE_COLS,
    QUANTILE,
    FRI_SAT_UPWEIGHT,
)
from config import MODELS_DIR
from metrics import compute_item_metrics, generate_abc_analysis, print_abc_report

SEP = "=" * 70
MIN_NONZERO_DAYS = 60
N_FOLDS = 3
VAL_RATIO = 0.15
OUTPUT_DIR = MODELS_DIR / "exploration" / "training"


def _load_quantile_params() -> dict:
    tuning_file = MODELS_DIR / "exploration" / "tuning" / "quantile_best_params.json"
    if tuning_file.exists():
        with open(tuning_file) as f:
            tuned = json.load(f)
        params = tuned.get("params", {})
        print(f"Loaded tuned params from {tuning_file}")
        return params
    print("No tuning file found, using defaults")
    return {
        "n_estimators": 200,
        "max_depth": 3,
        "learning_rate": 0.04,
        "min_child_weight": 5,
        "subsample": 0.8,
        "colsample_bytree": 1.0,
        "reg_alpha": 1.0,
        "reg_lambda": 2.0,
    }


def _load_rf_params() -> dict:
    tuning_file = MODELS_DIR / "exploration" / "tuning" / "rf_best_params.json"
    if tuning_file.exists():
        with open(tuning_file) as f:
            data = json.load(f)
        params = data.get("params", data)
        print(f"Loaded RF params from {tuning_file}")
        params["random_state"] = 42
        params["n_jobs"] = -1
        return params
    print("No RF tuning file, using defaults")
    return {"n_estimators": 200, "max_depth": 7, "random_state": 42, "n_jobs": -1}


def load_and_build_features() -> pd.DataFrame:
    df_all = load_all_items()
    print(f"\nBuilding features for {df_all['Item'].nunique()} items...")

    all_feat = []
    items = sorted(df_all["Item"].unique())
    for idx, item in enumerate(items):
        item_df = df_all[df_all["Item"] == item].copy()
        if len(item_df) < MIN_NONZERO_DAYS:
            continue
        feat_df = build_item_features(item_df)
        feat_df["Item"] = item
        all_feat.append(feat_df)

    combined = pd.concat(all_feat, ignore_index=True)
    combined = combined[combined["Quantity_Sold"] > 0].copy()
    print(f"Feature matrix: {len(combined):,} rows, {combined['Item'].nunique()} items")
    return combined


def expanding_window_cv(df: pd.DataFrame, n_folds: int = N_FOLDS):
    dates = sorted(df["Date"].unique())
    n_dates = len(dates)
    fold_size = n_dates // (n_folds + 1)

    folds = []
    for i in range(n_folds):
        train_end = fold_size * (i + 1)
        val_end = min(fold_size * (i + 2), n_dates)
        train_dates = set(dates[:train_end])
        val_dates = set(dates[train_end:val_end])
        train_df = df[df["Date"].isin(train_dates)]
        val_df = df[df["Date"].isin(val_dates)]
        folds.append((train_df, val_df))
    return folds


def train_xgboost_fold(train: pd.DataFrame, val: pd.DataFrame, features: list) -> dict:
    sample_weight = np.ones(len(train))
    fri_sat = train["DOW"].isin([4, 5])
    sample_weight[fri_sat] = FRI_SAT_UPWEIGHT

    params = _load_quantile_params()
    model = XGBRegressor(
        objective="reg:quantileerror",
        quantile_alpha=QUANTILE,
        random_state=42,
        **params,
    )
    model.fit(train[features], train["Quantity_Sold"], sample_weight=sample_weight, verbose=False)

    pred = np.maximum(0, model.predict(val[features]))
    return {"predictions": pred, "model": model}


def train_rf_fold(train: pd.DataFrame, val: pd.DataFrame, features: list) -> dict:
    params = _load_rf_params()
    model = RandomForestRegressor(**params)
    model.fit(train[features], train["Quantity_Sold"])

    pred = np.maximum(0, model.predict(val[features]))
    return {"predictions": pred, "model": model}


def run_cv(df: pd.DataFrame, features: list):
    folds = expanding_window_cv(df, N_FOLDS)

    xgb_all_metrics = []
    rf_all_metrics = []

    for fold_idx, (train, val) in enumerate(folds):
        print(f"\n{'='*70}")
        print(f"FOLD {fold_idx + 1}/{N_FOLDS}")
        print(f"Train: {train['Date'].min().date()} to {train['Date'].max().date()} ({len(train):,} rows)")
        print(f"Val:   {val['Date'].min().date()} to {val['Date'].max().date()} ({len(val):,} rows)")
        print(f"{'='*70}")

        t0 = time.time()
        xgb_result = train_xgboost_fold(train, val, features)
        xgb_time = time.time() - t0

        t0 = time.time()
        rf_result = train_rf_fold(train, val, features)
        rf_time = time.time() - t0

        val_xgb = val.copy()
        val_xgb["Predicted"] = xgb_result["predictions"]
        val_rf = val.copy()
        val_rf["Predicted"] = rf_result["predictions"]

        xgb_metrics = compute_item_metrics(val_xgb["Quantity_Sold"], val_xgb["Predicted"], val_xgb["Item"])
        rf_metrics = compute_item_metrics(val_rf["Quantity_Sold"], val_rf["Predicted"], val_rf["Item"])
        xgb_metrics["time"] = round(xgb_time, 1)
        rf_metrics["time"] = round(rf_time, 1)

        xgb_all_metrics.append(xgb_metrics)
        rf_all_metrics.append(rf_metrics)

        print(f"\n  {'Metric':<25s} {'XGBoost':>10s} {'RF':>10s}")
        print(f"  {'-'*50}")
        for key in ["rmse", "mae", "r2", "wmape", "periods_within_20pct", "periods_within_50pct"]:
            print(f"  {key:<25s} {xgb_metrics[key]:>10.2f} {rf_metrics[key]:>10.2f}")
        print(f"  {'time (s)':<25s} {xgb_metrics['time']:>10.1f} {rf_metrics['time']:>10.1f}")

    print(f"\n{SEP}")
    print("CROSS-VALIDATION SUMMARY (avg across folds)")
    print(SEP)

    print(f"\n  {'Metric':<25s} {'XGBoost':>10s} {'RF':>10s} {'Winner':>10s}")
    print(f"  {'-'*60}")
    for key in ["rmse", "mae", "r2", "wmape", "periods_within_20pct", "periods_within_50pct", "time"]:
        xgb_avg = np.mean([m[key] for m in xgb_all_metrics])
        rf_avg = np.mean([m[key] for m in rf_all_metrics])
        if key in ["r2", "periods_within_20pct", "periods_within_50pct"]:
            winner = "XGBoost" if xgb_avg > rf_avg else "RF"
        elif key == "time":
            winner = "XGBoost" if xgb_avg < rf_avg else "RF"
        else:
            winner = "XGBoost" if xgb_avg < rf_avg else "RF"
        print(f"  {key:<25s} {xgb_avg:>10.2f} {rf_avg:>10.2f} {winner:>10s}")

    return xgb_all_metrics, rf_all_metrics


def train_final_models(df: pd.DataFrame, features: list):
    print(f"\n{SEP}")
    print("TRAINING FINAL MODELS ON FULL DATA")
    print(SEP)

    n_val = max(1, int(len(df) * VAL_RATIO))
    train = df.iloc[:-n_val]
    val = df.iloc[-n_val:]
    print(f"Train: {len(train):,} rows | Val: {len(val):,} rows")

    xgb_result = train_xgboost_fold(train, val, features)
    rf_result = train_rf_fold(train, val, features)

    val_xgb = val.copy()
    val_xgb["Predicted"] = xgb_result["predictions"]
    val_rf = val.copy()
    val_rf["Predicted"] = rf_result["predictions"]

    xgb_analysis = generate_abc_analysis(val_xgb)
    rf_analysis = generate_abc_analysis(val_rf)

    print_abc_report(xgb_analysis, "XGBOOST QUANTILE")
    print_abc_report(rf_analysis, "RANDOM FOREST")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_DIR / "xgboost_model.pkl", "wb") as f:
        pickle.dump(xgb_result["model"], f)
    with open(OUTPUT_DIR / "rf_model.pkl", "wb") as f:
        pickle.dump(rf_result["model"], f)

    metadata = {
        "trained_at": datetime.now().isoformat(),
        "features": features,
        "n_items": df["Item"].nunique(),
        "n_records": len(df),
        "xgboost_params": _load_quantile_params(),
        "rf_params": _load_rf_params(),
        "xgboost_metrics": xgb_analysis["global_metrics"],
        "rf_metrics": rf_analysis["global_metrics"],
    }
    with open(OUTPUT_DIR / "comparison_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    print(f"\nModels saved to: {OUTPUT_DIR}")
    return xgb_analysis, rf_analysis


def main():
    print(SEP)
    print("MODEL COMPARISON: XGBoost (quantile) vs Random Forest")
    print(f"Features: {len(FEATURE_COLS)} | Quantile: {QUANTILE} | Folds: {N_FOLDS}")
    print(SEP)

    df = load_and_build_features()
    features = [f for f in FEATURE_COLS if f in df.columns]
    print(f"Using {len(features)} features: {features}")

    xgb_cv, rf_cv = run_cv(df, features)
    train_final_models(df, features)

    print(f"\n{SEP}")
    print("COMPARISON COMPLETE")
    print(SEP)


if __name__ == "__main__":
    main()
