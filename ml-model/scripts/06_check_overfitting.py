"""
Overfitting Check for Trained Models

Compares train vs test metrics to diagnose overfitting.
Run after training models with 04_forecast.py.

Usage:
    python scripts/06_check_overfitting.py
    python scripts/06_check_overfitting.py -f daily
    python scripts/06_check_overfitting.py -f weekly
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import numpy as np
import pandas as pd

from src.models.features import create_features
from src.models.forecaster import (
    load_and_prep_data,
    load_models,
    predict,
    FREQ_MAP,
    get_min_train_records,
    _xgboost_params,
    _split_train_val,
    _EARLY_STOPPING_ROUNDS,
    _BASE_GLOBAL_PARAMS,
    _BASE_ITEM_PARAMS,
)
from xgboost import XGBRegressor
from src.evaluation.metrics import compute_metrics
from src.utils.config import PROCESSED_DIR, SALES_FORECASTING_DIR


def check_overfitting(frequency: str = "weekly"):
    print("=" * 70)
    print(f"OVERFITTING CHECK ({frequency.upper()})")
    print("=" * 70)

    df_raw = load_and_prep_data(SALES_FORECASTING_DIR / "daily_item_sales.csv", frequency=frequency)
    df_feat = create_features(df_raw, frequency=frequency)

    features = [
        c for c in df_feat.columns
        if c
        in [
            "Month", "Week", "Year", "DOY", "DOW", "Sin_Week", "Cos_Week",
            "Weeks_Since_Start", "Is_Payday_Week", "Is_Weekend_Before_Payday",
            "Is_Holiday", "Is_Ramadan", "Is_Post_Rebranding", "Weeks_Since_Rebrand",
            "Lag_1", "Lag_2", "Lag_4", "Roll_Mean_4", "Roll_Mean_12",
            "Roll_Std_4", "Roll_Q95_4", "EWMA_4", "EWMA_12",
            "Diff_1", "Accel_2", "Recent_vs_Old_Trend",
            "Post_Rebrand_Surge_Ratio", "Item_Code",
        ]
        and c in df_feat.columns
    ]

    if frequency == "daily":
        split_date = df_feat["Date"].max() - pd.Timedelta(days=12 * 7)
    else:
        split_date = df_feat["Date"].max() - pd.Timedelta(weeks=12)

    train = df_feat[df_feat["Date"] < split_date].copy()
    test = df_feat[df_feat["Date"] >= split_date].copy()

    train_core, train_val = _split_train_val(train)

    print(f"\n  Train: {train['Date'].min().date()} -> {train['Date'].max().date()} ({len(train)} rows)")
    print(f"    Core: {train_core['Date'].min().date()} -> {train_core['Date'].max().date()} ({len(train_core)} rows)")
    print(f"    Val:  {train_val['Date'].min().date()} -> {train_val['Date'].max().date()} ({len(train_val)} rows)")
    print(f"  Test:  {test['Date'].min().date()} -> {test['Date'].max().date()} ({len(test)} rows)")

    min_recs = get_min_train_records(frequency)

    print(f"\n  {'Metric':<20} {'Train':>10} {'Test':>10} {'Gap':>10}")
    print(f"  {'─' * 50}")

    global_model = XGBRegressor(
        **_xgboost_params(_BASE_GLOBAL_PARAMS),
        early_stopping_rounds=_EARLY_STOPPING_ROUNDS,
    )
    global_model.fit(
        train_core[features], train_core["Quantity_Sold"],
        eval_set=[(train_val[features], train_val["Quantity_Sold"])],
        verbose=False,
    )

    train["Predicted"] = np.maximum(0, global_model.predict(train[features]))
    test["Predicted"] = np.maximum(0, global_model.predict(test[features]))

    tm = compute_metrics(train["Quantity_Sold"], train["Predicted"])
    em = compute_metrics(test["Quantity_Sold"], test["Predicted"])

    print(f"\n  GLOBAL MODEL")
    print(f"  {'Metric':<20} {'Train':>10} {'Test':>10} {'Gap':>10}")
    print(f"  {'─' * 50}")
    for k in ["r2", "wmape", "mae"]:
        gap = em[k] - tm[k]
        print(f"  {k:<20} {tm[k]:>10.4f} {em[k]:>10.4f} {gap:>+10.4f}")

    item_models = {}
    n_items = 0
    for item in test["Item"].unique():
        tr = train_core[train_core["Item"] == item]
        if len(tr) < min_recs:
            continue
        val_item = train_val[train_val["Item"] == item]
        has_val = len(val_item) >= 1
        model_params = _xgboost_params(_BASE_ITEM_PARAMS)
        if has_val:
            model_params["early_stopping_rounds"] = _EARLY_STOPPING_ROUNDS
        m = XGBRegressor(**model_params)
        eval_set = (
            [(val_item[features], val_item["Quantity_Sold"])]
            if has_val
            else None
        )
        m.fit(tr[features], tr["Quantity_Sold"], eval_set=eval_set, verbose=False)
        item_models[item] = m
        n_items += 1

    from src.models.forecaster import _BLEND_ALPHA

    print(f"\n  PER-ITEM MODELS ({n_items} items, rest use global, blend α={_BLEND_ALPHA})")

    def apply_predictions(df, models, fallback):
        preds = []
        for item in df["Item"].unique():
            sub = df[df["Item"] == item].copy()
            m = models.get(item)
            if m is not None:
                pred_item = m.predict(sub[features])
                pred_global = fallback.predict(sub[features])
                sub["Predicted"] = np.maximum(
                    0, _BLEND_ALPHA * pred_item + (1 - _BLEND_ALPHA) * pred_global
                )
            else:
                sub["Predicted"] = np.maximum(0, fallback.predict(sub[features]))
            preds.append(sub)
        return pd.concat(preds)

    train_pred = apply_predictions(train, item_models, global_model)
    test_pred = apply_predictions(test, item_models, global_model)

    tm2 = compute_metrics(train_pred["Quantity_Sold"], train_pred["Predicted"])
    em2 = compute_metrics(test_pred["Quantity_Sold"], test_pred["Predicted"])

    print(f"  {'Metric':<20} {'Train':>10} {'Test':>10} {'Gap':>10}")
    print(f"  {'─' * 50}")
    for k in ["r2", "wmape", "mae"]:
        gap = em2[k] - tm2[k]
        print(f"  {k:<20} {tm2[k]:>10.4f} {em2[k]:>10.4f} {gap:>+10.4f}")

    if abs(tm2["r2"] - em2["r2"]) > 0.15:
        print(f"\n  ⚠ SIGNIFICANT OVERFITTING: Train R² ({tm2['r2']:.4f}) >> Test R² ({em2['r2']:.4f})")
    elif abs(tm2["r2"] - em2["r2"]) > 0.05:
        print(f"\n  ⚡ MODERATE OVERFITTING: Train R² ({tm2['r2']:.4f}) >> Test R² ({em2['r2']:.4f})")
    else:
        print(f"\n  ✓ Overfitting is within acceptable range")

    print()


def main():
    parser = argparse.ArgumentParser(description="Check model overfitting")
    parser.add_argument(
        "-f", "--frequency",
        choices=["daily", "weekly"],
        default="weekly",
    )
    args = parser.parse_args()
    check_overfitting(args.frequency)


if __name__ == "__main__":
    main()
