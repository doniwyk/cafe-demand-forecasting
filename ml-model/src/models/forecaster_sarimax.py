from __future__ import annotations

import pandas as pd
import numpy as np
import pickle
import json
import warnings
from pathlib import Path
from typing import Optional
from datetime import datetime

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller

from src.utils.config import MODELS_DIR

MIN_TRAIN_WEEKS = 52

DEFAULT_ORDER = (1, 1, 1)
DEFAULT_SEASONAL_ORDER = (1, 1, 1, 52)
STATIONARY_ORDER = (2, 0, 2)
STATIONARY_SEASONAL_ORDER = (0, 0, 0, 0)


def _check_stationarity(series: pd.Series, threshold: float = 0.05) -> bool:
    result = adfuller(series.dropna())
    return result[1] < threshold


def _fit_item(ts: pd.Series) -> "SARIMAXResultsWrapper | None":
    ts = ts.astype(float)
    if len(ts) < MIN_TRAIN_WEEKS:
        return None

    ts = ts.copy()
    ts.index = pd.DatetimeIndex(ts.index, freq="W-MON")

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

        if _check_stationarity(ts):
            order = STATIONARY_ORDER
            seasonal_order = STATIONARY_SEASONAL_ORDER
        else:
            order = DEFAULT_ORDER
            seasonal_order = DEFAULT_SEASONAL_ORDER

        model = SARIMAX(
            ts,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        return model.fit(disp=False, maxiter=100)
    except Exception:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
            model = SARIMAX(
                ts,
                order=(1, 1, 0),
                seasonal_order=(0, 0, 0, 0),
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            return model.fit(disp=False, maxiter=50)
        except Exception:
            return None


def train_and_predict_sarimax(
    df_weekly: pd.DataFrame,
    n_test_weeks: int = 12,
) -> pd.DataFrame:
    split_date = df_weekly["Date"].max() - pd.Timedelta(weeks=n_test_weeks)
    train = df_weekly[df_weekly["Date"] < split_date].copy()
    test = df_weekly[df_weekly["Date"] >= split_date].copy()

    print(f"Training: {train['Date'].min().date()} -> {train['Date'].max().date()}")
    print(f"Testing : {test['Date'].min().date()} -> {test['Date'].max().date()}")

    dow_pattern = (
        train.groupby(["Item", train["Date"].dt.weekday])["Quantity_Sold"]
        .mean()
        .reset_index()
    )
    item_avg = (
        train.groupby("Item")["Quantity_Sold"]
        .mean()
        .reset_index()
        .rename(columns={"Quantity_Sold": "item_avg"})
    )
    dow_pattern = dow_pattern.merge(item_avg, on="Item")
    dow_pattern["dow_factor"] = dow_pattern["Quantity_Sold"] / dow_pattern["item_avg"]
    dow_factor_dict = (
        dow_pattern.pivot(index="Item", columns="Date", values="dow_factor")
        .fillna(1.0)
        .to_dict("index")
    )

    global_mean = train["Quantity_Sold"].mean()

    print("Training per-item SARIMAX models (backtest)...")
    predictions = []
    items = list(test["Item"].unique())
    total_items = len(items)
    print(f"  Total items to train: {total_items}")
    for i, item in enumerate(items):
        print(f"  [{i + 1}/{total_items}] Training {item}...")

        train_item = train[train["Item"] == item].sort_values("Date")
        test_item = test[test["Item"] == item].copy()

        ts = train_item.set_index("Date")["Quantity_Sold"]
        result = _fit_item(ts)

        if result is not None:
            steps = len(test_item)
            forecast = result.get_forecast(steps=steps)
            pred = forecast.predicted_mean.values
        else:
            pred = np.full(len(test_item), global_mean)

        test_item["Raw_Pred"] = np.maximum(0, pred)
        test_item["DOW"] = test_item["Date"].dt.weekday

        factors = dow_factor_dict.get(item, {j: 1.0 for j in range(7)})
        test_item["dow_factor"] = test_item["DOW"].map(factors).fillna(1.0)
        test_item["Predicted"] = (
            test_item["Raw_Pred"] * test_item["dow_factor"]
        ).round(0)
        test_item["Predicted"] = np.maximum(0, test_item["Predicted"])

        predictions.append(test_item)

    print(f"  Completed all {total_items} items")
    print("Building combined SARIMAX backtest predictions...")

    return pd.concat(predictions).sort_values(["Item", "Date"])


def _compute_dow_factors(df: pd.DataFrame) -> dict:
    dow_pattern = (
        df.groupby(["Item", df["Date"].dt.weekday])["Quantity_Sold"]
        .mean()
        .reset_index()
    )
    item_avg = (
        df.groupby("Item")["Quantity_Sold"]
        .mean()
        .reset_index()
        .rename(columns={"Quantity_Sold": "item_avg"})
    )
    dow_pattern = dow_pattern.merge(item_avg, on="Item")
    dow_pattern["dow_factor"] = dow_pattern["Quantity_Sold"] / dow_pattern["item_avg"]
    return (
        dow_pattern.pivot(index="Item", columns="Date", values="dow_factor")
        .fillna(1.0)
        .to_dict("index")
    )


def train_models_sarimax(
    df_weekly: pd.DataFrame,
    output_dir: str | Path | None = None,
) -> tuple[dict, None, dict]:
    output_dir = Path(output_dir) if output_dir else MODELS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Training per-item SARIMAX models...")
    item_models = {}
    items = list(df_weekly["Item"].unique())
    total = len(items)
    print(f"  Total items: {total}")
    for i, item in enumerate(items):
        print(f"  [{i + 1}/{total}] Training {item}...")

        train_item = df_weekly[df_weekly["Item"] == item].sort_values("Date")
        ts = train_item.set_index("Date")["Quantity_Sold"]
        result = _fit_item(ts)
        if result is not None:
            item_models[item] = result

    print("Saving SARIMAX item models to disk...")
    with open(output_dir / "item_models_sarimax.pkl", "wb") as f:
        pickle.dump(item_models, f)
    print("Saved SARIMAX item models")

    print("Computing DOW factors...")
    dow_factor_dict = _compute_dow_factors(df_weekly)
    with open(output_dir / "dow_factors_sarimax.json", "w") as f:
        json.dump(dow_factor_dict, f, indent=2)
    print("Saved DOW factors")

    global_mean = float(df_weekly["Quantity_Sold"].mean())
    metadata = {
        "model_type": "sarimax",
        "trained_at": datetime.now().isoformat(),
        "n_item_models": len(item_models),
        "items_with_models": sorted(item_models.keys()),
        "n_records": len(df_weekly),
        "date_range": [str(df_weekly["Date"].min()), str(df_weekly["Date"].max())],
        "global_mean": global_mean,
    }
    with open(output_dir / "model_metadata_sarimax.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print("Saved SARIMAX metadata")

    print(f"SARIMAX Models saved to: {output_dir}")
    print(f"  - Per-item models: {len(item_models)} items in item_models_sarimax.pkl")
    print(f"  Completed all {total} items")
    return item_models, None, dow_factor_dict


def load_models_sarimax(
    model_dir: str | Path | None = None,
) -> tuple[dict, None, dict]:
    model_dir = Path(model_dir) if model_dir else MODELS_DIR

    with open(model_dir / "item_models_sarimax.pkl", "rb") as f:
        item_models = pickle.load(f)

    with open(model_dir / "dow_factors_sarimax.json", "r") as f:
        dow_factor_dict = json.load(f)

    return item_models, None, dow_factor_dict


def predict_sarimax(
    df_weekly: pd.DataFrame,
    item_models: dict | None = None,
    global_model: None = None,
    dow_factor_dict: dict | None = None,
    model_dir: str | Path | None = None,
) -> pd.DataFrame:
    if item_models is None or dow_factor_dict is None:
        item_models, _, dow_factor_dict = load_models_sarimax(model_dir)

    meta_path = (
        Path(model_dir) if model_dir else MODELS_DIR
    ) / "model_metadata_sarimax.json"
    global_mean = 0
    if meta_path.exists():
        with open(meta_path) as f:
            global_mean = json.load(f).get("global_mean", 0)

    print(f"Running SARIMAX inference for {df_weekly['Item'].nunique()} items...")
    predictions = []
    for item in df_weekly["Item"].unique():
        test_item = df_weekly[df_weekly["Item"] == item].copy()

        if item in item_models:
            steps = len(test_item)
            try:
                forecast = item_models[item].get_forecast(steps=steps)
                pred = forecast.predicted_mean.values
            except Exception:
                pred = np.full(steps, global_mean)
        else:
            pred = np.full(len(test_item), global_mean)

        test_item["Raw_Pred"] = np.maximum(0, pred)
        test_item["DOW"] = test_item["Date"].dt.weekday

        factors = dow_factor_dict.get(item, {str(i): 1.0 for i in range(7)})
        factors = {
            int(k)
            if isinstance(k, str) and k.isdigit()
            else int(k)
            if isinstance(k, (int, float))
            else k: v
            for k, v in factors.items()
        }
        test_item["dow_factor"] = test_item["DOW"].map(factors).fillna(1.0)
        test_item["Predicted"] = (
            test_item["Raw_Pred"] * test_item["dow_factor"]
        ).round(0)
        test_item["Predicted"] = np.maximum(0, test_item["Predicted"])

        predictions.append(test_item)

    return pd.concat(predictions).sort_values(["Item", "Date"])


def generate_future_weekly(
    df_weekly: pd.DataFrame,
    future_weeks: int = 12,
) -> pd.DataFrame:
    max_date = df_weekly["Date"].max()
    items = df_weekly["Item"].unique()

    future_dates = pd.date_range(
        start=max_date + pd.Timedelta(weeks=1),
        periods=future_weeks,
        freq="W-MON",
    )

    future_df = pd.DataFrame(
        {
            "Date": np.repeat(future_dates, len(items)),
            "Item": np.tile(items, len(future_dates)),
        }
    )
    future_df["Quantity_Sold"] = 0
    return future_df
