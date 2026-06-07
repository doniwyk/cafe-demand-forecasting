"""Blend weight tuning for the inference pipeline.
=============================================
Searches over baseline percentile, model weight, and RF weight
using backtested MAE and pinball loss. Every weight is tuned from data.

Search space:
  - weekend_baseline: which DOW percentile to use (P50/P75/P90/P95)
  - weekend_model_w:  how much weight the model gets on Fri/Sat (0.1-0.9)
  - weekday_model_w:  how much weight the model gets on weekdays (0.1-0.9)
  - rf_weight:        RF vs XGB ratio in model prediction (0.0-1.0)

Evaluation: backtest across 5 historical periods, all items.
Metric: pinball loss at q=0.75 (supply planning — overpredict OK, underpredict bad)
        + MAE for reference.

Run: python exploration/tuning/tune_blend.py
"""
from __future__ import annotations

import json
import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(BASE_DIR))

from inference.forecast import (
    load_all_items,
    build_item_features,
    compute_dow_stats,
    train_models,
    forecast_item,
    FEATURE_COLS,
    QUANTILE,
    MIN_NONZERO_DAYS,
    _should_skip,
)
from config import MODELS_DIR

TUNING_DIR = MODELS_DIR / "exploration" / "tuning"
SEP = "=" * 70

TEST_PERIODS = [
    ("2026-03-15", "2026-03-21"),
    ("2026-04-05", "2026-04-11"),
    ("2026-04-19", "2026-04-25"),
    ("2026-05-03", "2026-05-09"),
    ("2026-05-17", "2026-05-23"),
]

SEARCH_SPACE = {
    "weekend_baseline": ["P50", "P75", "P90", "P95"],
    "weekend_model_w": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
    "weekday_model_w": [0.2, 0.3, 0.4, 0.5, 0.6],
    "rf_weight": [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0],
}


def _default_config():
    return {
        "weekend_baseline": "P75",
        "weekend_model_w": 0.3,
        "weekday_model_w": 0.4,
        "rf_weight": 0.5,
    }


def pinball_loss(y_true, y_pred, q=QUANTILE):
    diff = y_true - y_pred
    return float(np.mean(np.where(diff >= 0, q * diff, (q - 1) * diff)))


def get_baseline_value(dow_stats_row, dow, stat_name):
    col = f"DOW_{stat_name}"
    if col in dow_stats_row.index:
        return dow_stats_row[col]
    return 3.0


def evaluate_config(
    all_preds: list,
    weekend_baseline: str,
    weekend_model_w: float,
    weekday_model_w: float,
    rf_weight: float,
) -> dict:
    all_actuals = []
    all_predicted = []
    fri_sat_actuals = []
    fri_sat_preds = []

    for rec in all_preds:
        dow = rec["dow"]
        model_pred = rf_weight * rec["rf"] + (1 - rf_weight) * rec["xgb"]

        if dow in (4, 5):
            baseline = rec["dow_stats"].get(weekend_baseline, 3.0)
            blend_w = weekend_model_w
        else:
            baseline = rec["dow_stats"].get("DOW_Median", 3.0)
            blend_w = weekday_model_w

        pred = blend_w * model_pred + (1 - blend_w) * baseline

        all_actuals.append(rec["actual"])
        all_predicted.append(pred)
        if dow in (4, 5):
            fri_sat_actuals.append(rec["actual"])
            fri_sat_preds.append(pred)

    if not all_actuals:
        return {"pinball": 999, "mae": 999, "fri_sat_mae": 999, "bias": 999}

    y_true = np.array(all_actuals)
    y_pred = np.array(all_predicted)

    return {
        "pinball": pinball_loss(y_true, y_pred),
        "mae": float(np.mean(np.abs(y_true - y_pred))),
        "fri_sat_mae": float(np.mean(np.abs(np.array(fri_sat_actuals) - np.array(fri_sat_preds)))) if fri_sat_actuals else 999,
        "bias": float(np.mean(y_pred - y_true)),
    }


def _precompute_predictions(df_all: pd.DataFrame) -> list:
    """Train models once, compute raw XGB+RF predictions for all test rows."""
    all_preds = []
    items = [i for i in df_all["Item"].unique() if not _should_skip(i)]

    for p_idx, (test_start, test_end) in enumerate(TEST_PERIODS):
        ts = pd.Timestamp(test_start)
        te = pd.Timestamp(test_end)

        for item in items:
            df_item = df_all[df_all["Item"] == item].copy()
            train_df = df_item[df_item["Date"] < ts]
            test_df = df_item[(df_item["Date"] >= ts) & (df_item["Date"] <= te)]

            nonzero_train = (train_df["Quantity_Sold"] > 0).sum()
            if nonzero_train < MIN_NONZERO_DAYS or len(test_df) == 0:
                continue

            df_feat = build_item_features(train_df.copy())
            features = [f for f in FEATURE_COLS if f in df_feat.columns]

            try:
                xgb, rf = train_models(df_feat, features)
            except Exception:
                continue

            dow_stats = compute_dow_stats(train_df)
            n_days = (te - train_df["Date"].max()).days
            if n_days <= 0 or n_days > 14:
                continue

            fc = forecast_item(xgb, rf, dow_stats, train_df, features, n_days=n_days)
            fc = fc[fc["Date"].isin(test_df["Date"])]
            test_matched = test_df[test_df["Date"].isin(fc["Date"])]

            if len(fc) == 0:
                continue

            ds_map = {}
            for dow in range(7):
                ds_row = dow_stats[dow_stats["DOW"] == dow]
                if ds_row.empty:
                    ds_map[dow] = {"DOW_Median": 3.0, "P50": 3.0, "P75": 3.0, "P90": 3.0, "P95": 3.0}
                else:
                    ds = ds_row.iloc[0]
                    ds_map[dow] = {
                        "DOW_Median": ds.get("DOW_Median", 3.0),
                        "P50": ds.get("DOW_Median", 3.0),
                        "P75": ds.get("DOW_P75", 3.0),
                        "P90": ds.get("DOW_P90", 3.0),
                        "P95": ds.get("DOW_P95", 3.0),
                    }

            for _, row in fc.iterrows():
                actual_row = test_matched[test_matched["Date"] == row["Date"]]
                if len(actual_row) == 0:
                    continue

                actual = actual_row.iloc[0]["Quantity_Sold"]
                dow = row["DOW"]

                all_preds.append({
                    "xgb": row["XGB"],
                    "rf": row["RF"],
                    "actual": actual,
                    "dow": dow,
                    "dow_stats": ds_map[dow],
                })

        print(f"  Period {p_idx+1}: {len(all_preds)} predictions so far")

    return all_preds


def main():
    print(SEP)
    print("BLEND WEIGHT TUNING (Sequential Search)")
    print(f"Quantile: {QUANTILE} | Metric: pinball loss + MAE")
    print(f"Test periods: {len(TEST_PERIODS)}")
    print(SEP)

    print("\nLoading data and pre-computing predictions...")
    df_all = load_all_items()
    all_preds = _precompute_predictions(df_all)
    print(f"\nTotal predictions: {len(all_preds)}")

    best = _default_config()

    print(f"\n{SEP}")
    print("SEQUENTIAL GRID SEARCH")
    print(SEP)

    baseline_cfg = best.copy()
    baseline_m = evaluate_config(all_preds,
        baseline_cfg["weekend_baseline"], baseline_cfg["weekend_model_w"],
        baseline_cfg["weekday_model_w"], baseline_cfg["rf_weight"])
    print(f"\nBaseline: {baseline_cfg}")
    print(f"  pinball={baseline_m['pinball']:.4f} MAE={baseline_m['mae']:.2f} "
          f"FriSatMAE={baseline_m['fri_sat_mae']:.2f} bias={baseline_m['bias']:+.2f}")

    for param, values in SEARCH_SPACE.items():
        print(f"\nTuning {param}: {values}")
        best_loss = float("inf")
        best_val = best[param]

        for val in values:
            cfg = best.copy()
            cfg[param] = val
            m = evaluate_config(all_preds,
                cfg["weekend_baseline"], cfg["weekend_model_w"],
                cfg["weekday_model_w"], cfg["rf_weight"],
            )
            marker = " <- best" if m["pinball"] < best_loss else ""
            print(f"  {param}={str(val):>5s}  pinball={m['pinball']:.4f} MAE={m['mae']:.2f} "
                  f"FriSatMAE={m['fri_sat_mae']:.2f} bias={m['bias']:+.2f}{marker}")
            if m["pinball"] < best_loss:
                best_loss = m["pinball"]
                best_val = val

        best[param] = best_val
        print(f"  => Best {param}={best_val} (pinball={best_loss:.4f})")

    final_m = evaluate_config(all_preds,
        best["weekend_baseline"], best["weekend_model_w"],
        best["weekday_model_w"], best["rf_weight"])

    print(f"\n{SEP}")
    print("FINAL TUNED BLEND CONFIG")
    print(SEP)
    for k, v in best.items():
        print(f"  {k}: {v}")
    print(f"\n  pinball={final_m['pinball']:.4f} MAE={final_m['mae']:.2f} "
          f"FriSatMAE={final_m['fri_sat_mae']:.2f} bias={final_m['bias']:+.2f}")
    print(f"\n  vs baseline: pinball {baseline_m['pinball']:.4f} -> {final_m['pinball']:.4f} "
          f"({(baseline_m['pinball'] - final_m['pinball']) / baseline_m['pinball'] * 100:+.1f}%)")

    output = {
        "best_config": best,
        "best_pinball": final_m["pinball"],
        "best_mae": final_m["mae"],
        "best_fri_sat_mae": final_m["fri_sat_mae"],
        "best_bias": final_m["bias"],
        "baseline_config": baseline_cfg,
        "baseline_pinball": baseline_m["pinball"],
    }
    TUNING_DIR.mkdir(parents=True, exist_ok=True)
    with open(TUNING_DIR / "blend_best_params.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to: {TUNING_DIR / 'blend_best_params.json'}")

    return best


if __name__ == "__main__":
    main()
