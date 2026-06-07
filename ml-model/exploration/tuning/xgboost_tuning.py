"""XGBoost hyperparameter tuning — grid search with RMSE evaluation."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import SALES_FORECASTING_DIR, MODELS_DIR
from features import _split_train_val

SEPARATOR = "=" * 70

TUNING_DIR = MODELS_DIR / "exploration" / "tuning"


def load_daily_data() -> pd.DataFrame:
    from pathlib import Path as _P
    csv_path = SALES_FORECASTING_DIR / "daily_item_sales.csv"
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    date_col = "Date_Only" if "Date_Only" in df.columns else "Date"
    qty_col = "Quantity" if "Quantity" in df.columns else "Quantity_Sold"
    df["Date"] = pd.to_datetime(df[date_col])
    df["Quantity_Sold"] = df[qty_col]
    df = df[~df["Item"].str.strip().str.lower().str.startswith("add")]
    df_freq = (
        df.set_index("Date")
        .groupby("Item")
        .resample("D")["Quantity_Sold"]
        .sum()
        .reset_index()
    )
    return df_freq


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    data = df[["Item", "Date", "Quantity_Sold"]].copy().sort_values(["Item", "Date"]).reset_index(drop=True)
    for item in data["Item"].unique():
        mask = data["Item"] == item
        g = data.loc[mask, "Quantity_Sold"]
        shifted = g.shift(1)
        data.loc[mask, "Lag_1"] = shifted.values
        data.loc[mask, "Diff_1"] = g.diff(1).values
        data.loc[mask, "Accel_2"] = g.diff(1).diff(1).values
        g_lag1, g_lag4 = g.shift(1), g.shift(4)
        data.loc[mask, "Seasonal_Strength"] = (g_lag1 / (g_lag4 + 1) - 1).values
        data.loc[mask, "Roll_Mean_7"] = shifted.rolling(7, min_periods=1).mean().values
        data.loc[mask, "Roll_Mean_28"] = shifted.rolling(28, min_periods=1).mean().values
        data.loc[mask, "EWMA_7"] = shifted.ewm(span=7, adjust=False).mean().values
        data.loc[mask, "EWMA_28"] = shifted.ewm(span=28, adjust=False).mean().values
        roll7 = shifted.rolling(7, min_periods=1).mean()
        roll28 = shifted.rolling(28, min_periods=1).mean()
        data.loc[mask, "Trend_7"] = ((roll7 - roll28) / (roll28 + 1)).values
        recent3 = shifted.rolling(3, min_periods=1).mean()
        data.loc[mask, "Momentum_3"] = ((recent3 - roll7) / (roll7 + 1)).values
        data.loc[mask, "Price_Level"] = (shifted / (roll28 + 1)).values
    return data.fillna(0).replace([np.inf, -np.inf], 0)


def _eval_rmse(model, X_val, y_val) -> float:
    pred = model.predict(X_val)
    return float(np.sqrt(((y_val - pred) ** 2).mean()))


def tune_xgboost(data: pd.DataFrame, features: list):
    """Sequential grid search over XGBoost hyperparameters."""
    target = "Quantity_Sold"
    train_data, val_data = _split_train_val(data)

    best = {
        "objective": "reg:pseudohubererror",
        "max_depth": 5,
        "learning_rate": 0.03,
        "n_estimators": 200,
        "min_child_weight": 3,
        "subsample": 0.8,
        "colsample_bytree": 0.7,
        "reg_alpha": 0.5,
        "reg_lambda": 1.0,
    }

    search_space = {
        "max_depth": [3, 4, 5, 6, 7],
        "learning_rate": [0.01, 0.03, 0.05, 0.1],
        "n_estimators": [100, 200, 300, 500],
        "min_child_weight": [1, 3, 5, 7],
        "subsample": [0.6, 0.7, 0.8, 0.9],
        "colsample_bytree": [0.5, 0.6, 0.7, 0.8, 0.9],
        "reg_alpha": [0.0, 0.1, 0.5, 1.0],
        "reg_lambda": [0.5, 1.0, 2.0, 5.0],
    }

    print(f"\n{SEPARATOR}")
    print("XGBOOST HYPERPARAMETER TUNING")
    print(f"{SEPARATOR}")
    print(f"Features: {features}")
    print(f"Train: {len(train_data):,} rows | Val: {len(val_data):,} rows\n")

    for param, values in search_space.items():
        print(f"Tuning {param}: {values}")
        best_rmse = float("inf")
        best_val = best[param]

        for val in values:
            params = best.copy()
            params[param] = val
            try:
                model = XGBRegressor(**params, random_state=42, early_stopping_rounds=20)
                model.fit(
                    train_data[features], train_data[target],
                    eval_set=[(val_data[features], val_data[target])],
                    verbose=False,
                )
                rmse = _eval_rmse(model, val_data[features], val_data[target])
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_val = val
            except Exception:
                continue

        best[param] = best_val
        print(f"  -> Best {param}={best_val} (RMSE={best_rmse:.4f})\n")

    print(f"\n{SEPARATOR}")
    print("BEST XGBOOST PARAMETERS")
    print(f"{SEPARATOR}")
    for k, v in best.items():
        print(f"  {k}: {v}")

    TUNING_DIR.mkdir(parents=True, exist_ok=True)
    with open(TUNING_DIR / "xgboost_best_params.json", "w") as f:
        json.dump(best, f, indent=2)
    print(f"\nSaved to: {TUNING_DIR / 'xgboost_best_params.json'}")

    return best


def main():
    print("Loading data...")
    df = load_daily_data()
    print(f"Loaded {len(df):,} observations")

    print("\nBuilding feature matrix...")
    data = build_feature_matrix(df)
    features = [
        "Lag_1", "Diff_1", "Accel_2",
        "Roll_Mean_7", "Roll_Mean_28", "EWMA_7", "EWMA_28",
        "Roll_Std_7", "Roll_Q95_7", "Seasonal_Strength",
        "Trend_7", "Momentum_3", "Price_Level",
    ]
    features = [f for f in features if f in data.columns]

    best_params = tune_xgboost(data, features)
    return best_params


if __name__ == "__main__":
    main()
