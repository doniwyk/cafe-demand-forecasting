"""Hyperparameter tuning for the Random Forest inference model.

Tunes on the target item (Kopi Susu Husgendam Ice) with:
  - Same feature set as XGBoost (no Lag_1/Diff_1)
  - MSE objective (regressor)
  - Fri/Sat sample upweighting
  - Sequential grid search, evaluated on held-out validation set

Run: python exploration/tuning/tune_rf.py
"""
from __future__ import annotations

import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

BASE_DIR = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(BASE_DIR))

from inference.forecast import (
    load_item_data,
    build_item_features,
    FEATURE_COLS,
    MIN_NONZERO_DAYS,
)
from config import MODELS_DIR

TARGET_ITEM = "Kopi Susu Husgendam Ice"
TUNING_DIR = MODELS_DIR / "exploration" / "tuning"


def make_train_val(df_feat: pd.DataFrame):
    """Time-based split: last 15% as validation."""
    nonzero = df_feat[df_feat["Quantity_Sold"] > 0].copy()
    cutoff = int(len(nonzero) * 0.85)
    train = nonzero.iloc[:cutoff]
    val = nonzero.iloc[cutoff:]
    return train, val


def eval_params(
    params: dict,
    train: pd.DataFrame,
    val_df: pd.DataFrame,
    features: list,
) -> float:
    """Train RF with given params, return MSE on validation."""
    sample_weight = np.ones(len(train))
    fri_sat = train["DOW"].isin([4, 5])
    sample_weight[fri_sat] = 3.0

    model = RandomForestRegressor(**params)
    model.fit(train[features], train["Quantity_Sold"], sample_weight=sample_weight)
    pred = model.predict(val_df[features])
    return float(mean_squared_error(val_df["Quantity_Sold"].values, pred))


def tune_sequential(train: pd.DataFrame, val_df: pd.DataFrame, features: list) -> dict:
    """Sequential grid search: tune one param at a time."""
    best = {
        "n_estimators": 200,
        "max_depth": 5,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "max_features": 1.0,
        "random_state": 42,
        "n_jobs": -1,
    }

    search_space = {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [3, 5, 7, 10, None],
        "min_samples_split": [2, 5, 10, 20],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2", 1.0],
    }

    for param, values in search_space.items():
        best_loss = float("inf")
        best_val = best[param]

        for v in values:
            params = best.copy()
            params[param] = v
            try:
                loss = eval_params(params, train, val_df, features)
                if loss < best_loss:
                    best_loss = loss
                    best_val = v
            except Exception:
                pass

        best[param] = best_val

    return best


def main():
    df = load_item_data(TARGET_ITEM)
    df_feat = build_item_features(df.copy())
    features = [f for f in FEATURE_COLS if f in df_feat.columns]

    train, val = make_train_val(df_feat)
    baseline = {
        "n_estimators": 200, "max_depth": 5, "min_samples_split": 5,
        "min_samples_leaf": 2, "max_features": 1.0,
        "random_state": 42, "n_jobs": -1,
    }
    baseline_loss = eval_params(baseline, train, val, features)

    best = tune_sequential(train, val, features)
    best_loss = eval_params(best, train, val, features)

    TUNING_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "item": TARGET_ITEM,
        "objective": "mse",
        "params": best,
        "baseline_mse": baseline_loss,
        "tuned_mse": best_loss,
        "features": features,
        "fri_sat_upweight": 3.0,
    }
    with open(TUNING_DIR / "rf_best_params.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"RF tune done: baseline MSE={baseline_loss:.4f} -> tuned MSE={best_loss:.4f} "
          f"({(baseline_loss - best_loss) / baseline_loss * 100:+.1f}%) -> saved {TUNING_DIR / 'rf_best_params.json'}")

    return best


if __name__ == "__main__":
    main()
