"""Hyperparameter tuning for the quantile XGBoost inference model.

Tunes on the target item (Kopi Susu Husgendam Ice) with:
  - Correct feature set (no Lag_1/Diff_1)
  - Quantile regression objective (reg:quantileerror)
  - Fri/Sat sample upweighting
  - Pinball loss evaluation metric

Uses sequential grid search: tune one param at a time while keeping
others at their current best. Evaluates with pinball loss at q=0.75
on a held-out validation set.

Run: python exploration/inference/tune_quantile.py
"""
from __future__ import annotations

import json
import numpy as np
import pandas as pd
from pathlib import Path
from xgboost import XGBRegressor

BASE_DIR = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(BASE_DIR))

from inference.forecast import (
    load_item_data,
    build_item_features,
    compute_dow_stats,
    FEATURE_COLS,
    QUANTILE,
    CAFE_DB_URL,
    MIN_NONZERO_DAYS,
)
from config import MODELS_DIR

TARGET_ITEM = "Kopi Susu Husgendam Ice"
TUNING_DIR = MODELS_DIR / "exploration" / "tuning"
SEP = "=" * 70


def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, quantile: float = QUANTILE) -> float:
    """Pinball/quantile loss — the proper metric for quantile regression."""
    diff = y_true - y_pred
    return float(np.mean(np.where(diff >= 0, quantile * diff, (quantile - 1) * diff)))


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
    """Train with given params, return pinball loss on validation."""
    sample_weight = np.ones(len(train))
    fri_sat = train["DOW"].isin([4, 5])
    sample_weight[fri_sat] = 3.0

    model = XGBRegressor(**params, random_state=42)
    model.fit(
        train[features], train["Quantity_Sold"],
        sample_weight=sample_weight,
        verbose=False,
    )
    pred = model.predict(val_df[features])
    return pinball_loss(val_df["Quantity_Sold"].values, pred)


def tune_sequential(train: pd.DataFrame, val_df: pd.DataFrame, features: list) -> dict:
    """Sequential grid search: tune one param at a time."""
    best = {
        "objective": "reg:quantileerror",
        "quantile_alpha": QUANTILE,
        "n_estimators": 600,
        "max_depth": 5,
        "learning_rate": 0.04,
        "min_child_weight": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 1.0,
        "reg_lambda": 2.0,
    }

    search_space = {
        "n_estimators": [200, 400, 600, 800, 1000],
        "max_depth": [3, 4, 5, 6, 7],
        "learning_rate": [0.01, 0.02, 0.03, 0.04, 0.05, 0.08],
        "min_child_weight": [1, 3, 5, 7, 10],
        "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        "reg_alpha": [0.0, 0.1, 0.5, 1.0, 2.0],
        "reg_lambda": [0.5, 1.0, 2.0, 5.0, 10.0],
    }

    print(f"\nTrain: {len(train)} rows | Val: {len(val_df)} rows")
    print(f"Features: {len(features)}")
    print(f"Quantile: {QUANTILE}")
    print()

    for param, values in search_space.items():
        print(f"Tuning {param}: testing {values}")
        best_loss = float("inf")
        best_val = best[param]
        results = []

        for v in values:
            params = best.copy()
            params[param] = v
            try:
                loss = eval_params(params, train, val_df, features)
                results.append((v, loss))
                marker = " <- best" if loss < best_loss else ""
                print(f"  {param}={v:>6}  pinball={loss:.4f}{marker}")
                if loss < best_loss:
                    best_loss = loss
                    best_val = v
            except Exception as e:
                print(f"  {param}={v:>6}  FAILED: {e}")

        best[param] = best_val
        print(f"  => Best {param}={best_val} (pinball={best_loss:.4f})\n")

    return best


def cross_validate_params(
    df_feat: pd.DataFrame,
    features: list,
    params: dict,
    n_folds: int = 3,
) -> dict:
    """Expanding window CV to validate tuned params."""
    nonzero = df_feat[df_feat["Quantity_Sold"] > 0].copy().reset_index(drop=True)
    fold_size = len(nonzero) // (n_folds + 1)

    print(f"\n{SEP}")
    print(f"CROSS-VALIDATION ({n_folds} folds)")
    print(SEP)

    metrics = []
    for i in range(n_folds):
        train_end = fold_size * (i + 1)
        val_end = min(fold_size * (i + 2), len(nonzero))

        train = nonzero.iloc[:train_end]
        val = nonzero.iloc[train_end:val_end]

        sw = np.ones(len(train))
        sw[train["DOW"].isin([4, 5])] = 3.0

        model = XGBRegressor(**params, random_state=42)
        model.fit(train[features], train["Quantity_Sold"], sample_weight=sw, verbose=False)
        pred = model.predict(val[features])

        pb = pinball_loss(val["Quantity_Sold"].values, pred)
        rmse = float(np.sqrt(((val["Quantity_Sold"] - pred) ** 2).mean()))
        mae = float(np.abs(val["Quantity_Sold"] - pred).mean())

        metrics.append({"pinball": pb, "rmse": rmse, "mae": mae})
        print(f"  Fold {i+1}: pinball={pb:.4f} | RMSE={rmse:.4f} | MAE={mae:.4f}")

    avg = {k: np.mean([m[k] for m in metrics]) for k in metrics[0]}
    print(f"\n  Avg: pinball={avg['pinball']:.4f} | RMSE={avg['rmse']:.4f} | MAE={avg['mae']:.4f}")
    return avg


def main():
    print(SEP)
    print(f"QUANTILE XGBOOST TUNING — {TARGET_ITEM}")
    print(f"Objective: reg:quantileerror (q={QUANTILE})")
    print(f"Evaluation: pinball loss")
    print(SEP)

    df = load_item_data(TARGET_ITEM)
    df_feat = build_item_features(df.copy())
    features = [f for f in FEATURE_COLS if f in df_feat.columns]

    train, val = make_train_val(df_feat)
    print(f"\nInitial params (before tuning):")
    baseline = {
        "objective": "reg:quantileerror",
        "quantile_alpha": QUANTILE,
        "n_estimators": 600, "max_depth": 5, "learning_rate": 0.04,
        "min_child_weight": 5, "subsample": 0.8, "colsample_bytree": 0.8,
        "reg_alpha": 1.0, "reg_lambda": 2.0,
    }
    for k, v in baseline.items():
        print(f"  {k}: {v}")
    baseline_loss = eval_params(baseline, train, val, features)
    print(f"\nBaseline pinball loss: {baseline_loss:.4f}")

    print(f"\n{SEP}")
    print("SEQUENTIAL GRID SEARCH")
    print(SEP)
    best = tune_sequential(train, val, features)

    best_loss = eval_params(best, train, val, features)
    print(f"\nTuned pinball loss: {best_loss:.4f}")
    print(f"Improvement: {(baseline_loss - best_loss) / baseline_loss * 100:+.1f}%")

    cv_metrics = cross_validate_params(df_feat, features, best)

    print(f"\n{SEP}")
    print("FINAL TUNED PARAMETERS")
    print(SEP)
    for k, v in best.items():
        print(f"  {k}: {v}")

    TUNING_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "item": TARGET_ITEM,
        "quantile": QUANTILE,
        "objective": "reg:quantileerror",
        "params": {k: v for k, v in best.items() if k != "objective" and k != "quantile_alpha"},
        "baseline_pinball": baseline_loss,
        "tuned_pinball": best_loss,
        "cv_metrics": cv_metrics,
        "features": features,
        "fri_sat_upweight": 3.0,
    }
    with open(TUNING_DIR / "quantile_best_params.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to: {TUNING_DIR / 'quantile_best_params.json'}")

    return best


if __name__ == "__main__":
    main()
