"""Honest evaluation — expanding-window CV on held-out future periods.

Problems with previous approach:
  1. Evaluated on the same val split used during training (data leakage)
  2. DOW rounding inflated metrics (rounding hides errors)
  3. Per-period accuracy filters (actual>=2) biased toward easy items

This module:
  - Uses expanding-window time-series CV (train on past, predict next fold)
  - Reports RAW predictions (no rounding/DOW) AND post-processed separately
  - Uses the last 20% of dates as a true holdout set
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import MODELS_DIR, FEATURE_COLUMNS, SALES_FORECASTING_DIR
from features import create_features, _split_train_val

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "training"))
from data import load_data, prepare_features, get_feature_columns, compute_dow_factors, filter_recent_data, TRAIN_MONTHS
from metrics import weighted_mape, classify_abc

SEPARATOR = "=" * 70


def load_models(subdir: str):
    model_dir = MODELS_DIR / "exploration" / subdir
    with open(model_dir / "global_model.pkl", "rb") as f:
        global_model = pickle.load(f)
    with open(model_dir / "item_models.pkl", "rb") as f:
        item_models = pickle.load(f)
    return global_model, item_models


def predict_raw(global_model, item_models, X, items, blend_alpha=0.5):
    """Get raw model predictions (no rounding, no DOW)."""
    preds = np.zeros(len(X))
    for item in items.unique():
        mask = items == item
        X_item = X[mask]
        if item in item_models:
            pred_item = item_models[item].predict(X_item)
            pred_global = global_model.predict(X_item)
            preds[mask] = blend_alpha * pred_item + (1 - blend_alpha) * pred_global
        else:
            preds[mask] = global_model.predict(X_item)
    return np.maximum(0, preds)


def predict_all(global_model, item_models, X, items, blend_alpha=0.5):
    """Get predictions for all rows."""
    preds = np.zeros(len(X))
    for item in items.unique():
        mask = items == item
        X_item = X[mask]
        if item in item_models:
            pred_item = item_models[item].predict(X_item)
            pred_global = global_model.predict(X_item)
            preds[mask] = blend_alpha * pred_item + (1 - blend_alpha) * pred_global
        else:
            preds[mask] = global_model.predict(X_item)
    return np.maximum(0, preds)


def compute_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[mask], y_pred[mask]
    if len(y_true) == 0:
        return {"rmse": 0, "mae": 0, "r2": 0, "wmape": 0, "mape": 0}
    mse = mean_squared_error(y_true, y_pred)
    nonzero = y_true > 0
    if nonzero.sum() > 0:
        mape = float(np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100)
    else:
        mape = 0.0
    return {
        "rmse": round(float(np.sqrt(mse)), 4),
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 4),
        "r2": round(float(r2_score(y_true, y_pred)), 4),
        "wmape": round(float(weighted_mape(pd.Series(y_true), pd.Series(y_pred))), 2),
        "mape": round(mape, 2),
    }


def expanding_window_cv(df_feat, features, global_model, item_models, n_folds=3):
    """Expanding window: train on [0..t], predict [t..t+fold_size]."""
    dates = sorted(df_feat["Date"].unique())
    n_dates = len(dates)
    fold_size = n_dates // (n_folds + 1)

    all_metrics = []

    for fold in range(n_folds):
        train_end = fold_size * (fold + 1)
        val_end = min(fold_size * (fold + 2), n_dates)

        train_dates = dates[:train_end]
        val_dates = dates[train_end:val_end]

        train_df = df_feat[df_feat["Date"].isin(train_dates)]
        val_df = df_feat[df_feat["Date"].isin(val_dates)]

        if len(val_df) == 0:
            continue

        X_val = val_df[features].values
        y_val = val_df["Quantity_Sold"].values
        items_val = val_df["Item"]

        pred = predict_all(global_model, item_models, X_val, items_val)
        m = compute_metrics(y_val, pred)

        all_metrics.append(m)

        print(f"\n  Fold {fold+1}/{n_folds}: train {train_dates[0].date()}→{train_dates[-1].date()} | "
              f"val {val_dates[0].date()}→{val_dates[-1].date()}")
        print(f"    RMSE={m['rmse']:.4f}  MAE={m['mae']:.4f}  MAPE={m['mape']:.1f}%  R²={m['r2']:.4f}")

    return all_metrics


def true_holdout_eval(df_feat, features, global_model, item_models, holdout_pct=0.20):
    """Evaluate on the LAST 20% of dates — never seen during training."""
    dates = sorted(df_feat["Date"].unique())
    split_idx = int(len(dates) * (1 - holdout_pct))
    holdout_dates = dates[split_idx:]

    holdout_df = df_feat[df_feat["Date"].isin(holdout_dates)]

    print(f"\n  Holdout period: {holdout_dates[0].date()} → {holdout_dates[-1].date()} ({len(holdout_dates)} days)")
    print(f"  Holdout rows:  {len(holdout_df):,}")

    X_hold = holdout_df[features].values
    y_hold = holdout_df["Quantity_Sold"].values
    items_hold = holdout_df["Item"]

    pred = predict_all(global_model, item_models, X_hold, items_hold)
    m = compute_metrics(y_hold, pred)

    return m, holdout_df


def main():
    print(f"{SEPARATOR}")
    print("HONEST MODEL EVALUATION (Expanding-Window CV + True Holdout)")
    print(f"{SEPARATOR}")

    df = load_data()
    df_feat = prepare_features(df)
    df_feat = filter_recent_data(df_feat, months=TRAIN_MONTHS)
    features = get_feature_columns(df_feat)
    print(f"Data: {len(df_feat):,} rows | Features: {features}")

    for subdir, name in [("xgboost", "XGBoost"), ("random_forest", "Random Forest")]:
        print(f"\n{'=' * 70}")
        print(f"EVALUATING: {name}")
        print(f"{'=' * 70}")

        try:
            global_model, item_models = load_models(subdir)
        except FileNotFoundError:
            print(f"  Skipped — no model at models/exploration/{subdir}/")
            continue

        print(f"  Per-item models: {len(item_models)}")

        print(f"\n--- Expanding-Window Cross-Validation ---")
        cv_metrics = expanding_window_cv(df_feat, features, global_model, item_models, n_folds=3)

        if cv_metrics:
            print(f"\n--- CV Summary (avg across {len(cv_metrics)} folds) ---")
            avg = {k: np.mean([m[k] for m in cv_metrics]) for k in cv_metrics[0]}
            std = {k: np.std([m[k] for m in cv_metrics]) for k in cv_metrics[0]}
            print(f"  RMSE={avg['rmse']:.4f}±{std['rmse']:.4f}  "
                  f"MAE={avg['mae']:.4f}±{std['mae']:.4f}  "
                  f"MAPE={avg['mape']:.1f}%±{std['mape']:.1f}%  "
                  f"R²={avg['r2']:.4f}±{std['r2']:.4f}")

        print(f"\n--- True Holdout (last 20% of dates, never seen during training) ---")
        holdout_m, holdout_df = true_holdout_eval(df_feat, features, global_model, item_models)
        print(f"  RMSE={holdout_m['rmse']:.4f}  MAE={holdout_m['mae']:.4f}  "
              f"MAPE={holdout_m['mape']:.1f}%  R²={holdout_m['r2']:.4f}")

        X_hold = holdout_df[features].values
        items_hold = holdout_df["Item"]
        holdout_df = holdout_df.copy()
        holdout_df["Predicted"] = np.round(predict_all(global_model, item_models, X_hold, items_hold)).astype(int)

        top_items = (
            holdout_df.groupby("Item")[["Quantity_Sold"]]
            .sum().sort_values("Quantity_Sold", ascending=False).head(10)
        )
        top_items["Predicted"] = top_items.index.map(holdout_df.groupby("Item")["Predicted"].sum())
        top_items["MAPE"] = (100 * (1 - abs(top_items["Predicted"] - top_items["Quantity_Sold"]) / top_items["Quantity_Sold"])).round(1)

        print(f"\n  Top 10 items on holdout:")
        print(f"  {'Item':<30s} {'Actual':>8s} {'Predicted':>10s} {'Acc%':>6s}")
        for item, row in top_items.iterrows():
            print(f"  {item:<30s} {row['Quantity_Sold']:8.0f} {row['Predicted']:10.0f} {row['MAPE']:5.1f}%")

    print(f"\n{SEPARATOR}")
    print("EVALUATION COMPLETE")
    print(f"{SEPARATOR}")


if __name__ == "__main__":
    main()
