"""Random Forest training pipeline."""
from __future__ import annotations

import pandas as pd
import numpy as np
import pickle
import json
import time
from pathlib import Path
from datetime import datetime

from sklearn.ensemble import RandomForestRegressor

from data import (
    load_data, prepare_features, get_feature_columns, split_train_val,
    time_series_cv, compute_dow_factors, filter_recent_data,
    MIN_TRAIN_RECORDS, N_FOLDS, TRAIN_MONTHS,
)
from metrics import generate_abc_analysis, print_abc_report, compute_metrics
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import MODELS_DIR

_TUNING_DIR = MODELS_DIR / "exploration" / "tuning"


def _load_rf_params() -> dict:
    """Load tuned params from tuning step, fallback to defaults."""
    params_file = _TUNING_DIR / "rf_best_params.json"
    defaults = {
        "n_estimators": 500,
        "max_depth": 15,
        "random_state": 42,
        "n_jobs": -1,
    }
    if params_file.exists():
        import json as _json
        with open(params_file) as f:
            tuned = _json.load(f)
        defaults.update(tuned)
        print(f"Loaded tuned params from {params_file}")
    else:
        print(f"No tuning results found, using defaults")
    return defaults

BLEND_ALPHA = 0.5


def train_global_model(train: pd.DataFrame, features: list) -> RandomForestRegressor:
    print("Training global Random Forest model...")
    t0 = time.time()

    model = RandomForestRegressor(**_load_rf_params())
    model.fit(train[features], train["Quantity_Sold"])

    print(f"Global model trained in {time.time() - t0:.1f}s")
    return model


def train_per_item_models(
    train: pd.DataFrame, features: list
) -> dict[str, RandomForestRegressor]:
    item_models = {}
    items = list(train["Item"].unique())
    total_items = len(items)

    print(f"Training per-item Random Forest models... total items: {total_items}")
    t0 = time.time()

    for idx, item in enumerate(items):
        if (idx + 1) % 20 == 0 or idx == 0:
            print(f"  Progress: {idx + 1}/{total_items} ({(idx + 1) / total_items * 100:.1f}%)")

        train_item = train[train["Item"] == item]
        if len(train_item) < MIN_TRAIN_RECORDS:
            continue

        model = RandomForestRegressor(**_load_rf_params())
        model.fit(train_item[features], train_item["Quantity_Sold"])

        item_models[item] = model

    print(f"Per-item models trained in {time.time() - t0:.1f}s ({len(item_models)} items)")
    return item_models


def evaluate(global_model, item_models, val: pd.DataFrame, features: list):
    preds = []
    for item in val["Item"].unique():
        item_val = val[val["Item"] == item].copy()

        if item in item_models:
            pred_item = item_models[item].predict(item_val[features])
            pred_global = global_model.predict(item_val[features])
            pred = BLEND_ALPHA * pred_item + (1 - BLEND_ALPHA) * pred_global
        else:
            pred = global_model.predict(item_val[features])

        item_val["Predicted"] = np.maximum(0, pred)
        preds.append(item_val)

    return pd.concat(preds)


def cross_validate(df: pd.DataFrame, features: list):
    """3-fold time series cross-validation."""
    print("\n" + "=" * 70)
    print(f"TIME SERIES CROSS-VALIDATION ({N_FOLDS} folds)")
    print("=" * 70)

    folds = time_series_cv(df, n_folds=N_FOLDS)
    all_metrics = []

    for fold_idx, (train, val) in enumerate(folds):
        print(f"\n{'='*70}")
        print(f"FOLD {fold_idx + 1}/{N_FOLDS}")
        print(f"Train: {train['Date'].min().date()} to {train['Date'].max().date()} ({len(train):,} rows)")
        print(f"Val:   {val['Date'].min().date()} to {val['Date'].max().date()} ({len(val):,} rows)")
        print(f"{'='*70}")

        global_model = train_global_model(train, features)
        item_models = train_per_item_models(train, features)

        result = evaluate(global_model, item_models, val, features)
        analysis = generate_abc_analysis(result)
        metrics = analysis["global_metrics"]
        all_metrics.append(metrics)

        print(f"\nFold {fold_idx + 1} Results:")
        print(f"  RMSE:  {metrics['rmse']:.2f}")
        print(f"  MAE:   {metrics['mae']:.2f}")
        print(f"  R²:    {metrics['r2']:.4f}")
        print(f"  wMAPE: {metrics['wmape']:.2f}%")
        print(f"  ±20%:  {metrics['periods_within_20pct']:.1f}%")
        print(f"  ±50%:  {metrics['periods_within_50pct']:.1f}%")

    print("\n" + "=" * 70)
    print("CROSS-VALIDATION SUMMARY")
    print("=" * 70)

    avg_metrics = {}
    for key in all_metrics[0]:
        values = [m[key] for m in all_metrics]
        avg_metrics[key] = np.mean(values)
        std_metrics = np.std(values)
        print(f"  {key:30s}: {np.mean(values):.4f} ± {std_metrics:.4f}")

    return avg_metrics


def main():
    df = load_data()
    df_feat = prepare_features(df)
    df_feat = filter_recent_data(df_feat, months=TRAIN_MONTHS)
    features = get_feature_columns(df_feat)

    # Cross-validation
    cv_metrics = cross_validate(df_feat, features)

    # Final model on full data
    print("\n" + "=" * 70)
    print("TRAINING FINAL MODEL ON FULL DATA")
    print("=" * 70)

    train, val = split_train_val(df_feat)
    print(f"Train: {len(train):,} rows | Val: {len(val):,} rows")

    global_model = train_global_model(train, features)
    item_models = train_per_item_models(train, features)

    result = evaluate(global_model, item_models, val, features)
    analysis = generate_abc_analysis(result)
    print_abc_report(analysis, "RANDOM FOREST (FINAL)")

    output_dir = MODELS_DIR / "exploration" / "random_forest"
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "global_model.pkl", "wb") as f:
        pickle.dump(global_model, f)
    with open(output_dir / "item_models.pkl", "wb") as f:
        pickle.dump(item_models, f)

    metadata = {
        "trained_at": datetime.now().isoformat(),
        "n_item_models": len(item_models),
        "items_with_models": sorted(item_models.keys()),
        "features": features,
        "rf_params": _load_rf_params(),
        "cv_metrics": cv_metrics,
        "n_records": len(df_feat),
        "train_months": TRAIN_MONTHS,
    }
    with open(output_dir / "model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nRandom Forest models saved to: {output_dir}")

    return cv_metrics


if __name__ == "__main__":
    main()
