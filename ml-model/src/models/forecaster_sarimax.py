from __future__ import annotations

import warnings
import pandas as pd
import numpy as np
import pickle
import json
import time
from pathlib import Path
from typing import Optional
from datetime import datetime

warnings.filterwarnings("ignore")

from statsmodels.tsa.statespace.sarimax import SARIMAX


MIN_TRAIN_RECORDS = 40
_DEFAULT_ORDER = (1, 1, 1)
_DEFAULT_SEASONAL_ORDER = (1, 1, 1, 52)
_FIT_KWARGS = {"maxiter": 100, "disp": False}


def _fit_item_sarimax(
    series: pd.Series,
    order: tuple = _DEFAULT_ORDER,
    seasonal_order: tuple = _DEFAULT_SEASONAL_ORDER,
) -> "SARIMAXResultsWrapper":
    model = SARIMAX(
        series,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    return model.fit(**_FIT_KWARGS)


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


def _apply_dow_adjustment(df: pd.DataFrame, dow_factor_dict: dict) -> pd.DataFrame:
    df = df.copy()
    df["DOW"] = df["Date"].dt.weekday
    for item in df["Item"].unique():
        mask = df["Item"] == item
        factors = dow_factor_dict.get(item, {i: 1.0 for i in range(7)})
        df.loc[mask, "dow_factor"] = df.loc[mask, "DOW"].map(factors).fillna(1.0)
    df["Predicted"] = (df["Raw_Pred"] * df["dow_factor"]).round(0)
    df["Predicted"] = np.maximum(0, df["Predicted"])
    return df


def _get_global_avg(df: pd.DataFrame) -> float:
    return df["Quantity_Sold"].mean()


def train_and_predict_sarimax(
    df_weekly: pd.DataFrame,
    n_test_periods: int = 12,
) -> pd.DataFrame:
    split_date = df_weekly["Date"].max() - pd.Timedelta(weeks=n_test_periods)
    train = df_weekly[df_weekly["Date"] < split_date].copy()
    test = df_weekly[df_weekly["Date"] >= split_date].copy()

    dow_factor_dict = _compute_dow_factors(train)
    global_avg = _get_global_avg(train)

    print("[SARIMAX] Training per-item models...")
    predictions = []
    items = list(test["Item"].unique())
    for idx, item in enumerate(items):
        if (idx + 1) % 20 == 0 or idx == 0:
            print(f"  Progress: {idx + 1}/{len(items)} items")
        train_item = train[train["Item"] == item].set_index("Date").sort_index()
        test_item = test[test["Item"] == item].copy()

        if len(train_item) >= MIN_TRAIN_RECORDS:
            try:
                result = _fit_item_sarimax(train_item["Quantity_Sold"])
                pred = result.forecast(steps=len(test_item))
                pred = pred.values
            except Exception:
                pred = np.full(len(test_item), global_avg)
        else:
            pred = np.full(len(test_item), global_avg)

        test_item["Raw_Pred"] = np.maximum(0, pred)
        predictions.append(test_item)

    result = pd.concat(predictions)
    return _apply_dow_adjustment(result.sort_values(["Item", "Date"]), dow_factor_dict)


def train_models_sarimax(
    df_weekly: pd.DataFrame,
    output_dir: str | Path | None = None,
) -> tuple[dict, float, dict]:
    output_dir = Path(output_dir) if output_dir else Path("models")
    output_dir.mkdir(parents=True, exist_ok=True)

    dow_factor_dict = _compute_dow_factors(df_weekly)
    global_avg = _get_global_avg(df_weekly)

    item_models = {}
    items = list(df_weekly["Item"].unique())
    print(f"[SARIMAX] Training per-item models... total items: {len(items)}")
    for idx, item in enumerate(items):
        if (idx + 1) % 20 == 0 or idx == 0:
            print(f"  Progress: {idx + 1}/{len(items)} items")
        train_item = df_weekly[df_weekly["Item"] == item].set_index("Date").sort_index()
        if len(train_item) < MIN_TRAIN_RECORDS:
            continue
        try:
            result = _fit_item_sarimax(train_item["Quantity_Sold"])
            item_models[item] = result
        except Exception:
            print(f"  [WARN] SARIMAX failed for {item}, skipping")

    with open(output_dir / "item_models_sarimax.pkl", "wb") as f:
        pickle.dump(item_models, f)
    with open(output_dir / "dow_factors_sarimax.json", "w") as f:
        json.dump(dow_factor_dict, f, indent=2)

    metadata = {
        "model_type": "sarimax",
        "trained_at": datetime.now().isoformat(),
        "n_item_models": len(item_models),
        "items_with_models": sorted(item_models.keys()),
        "n_records": len(df_weekly),
        "date_range": [str(df_weekly["Date"].min()), str(df_weekly["Date"].max())],
        "global_avg": global_avg,
    }
    with open(output_dir / "model_metadata_sarimax.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"[SARIMAX] Models saved to: {output_dir}")
    return item_models, global_avg, dow_factor_dict


def load_models_sarimax(
    model_dir: str | Path | None = None,
) -> tuple[dict, float, dict]:
    model_dir = Path(model_dir) if model_dir else Path("models")
    with open(model_dir / "item_models_sarimax.pkl", "rb") as f:
        item_models = pickle.load(f)
    with open(model_dir / "dow_factors_sarimax.json", "r") as f:
        dow_factor_dict = json.load(f)
    meta_path = model_dir / "model_metadata_sarimax.json"
    global_avg = 0.0
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
            global_avg = meta.get("global_avg", 0.0)
    return item_models, global_avg, dow_factor_dict


def predict_sarimax(
    df_weekly: pd.DataFrame,
    item_models: dict | None = None,
    global_model: float | None = None,
    dow_factor_dict: dict | None = None,
    model_dir: str | Path | None = None,
) -> pd.DataFrame:
    if item_models is None or global_model is None or dow_factor_dict is None:
        item_models, global_model, dow_factor_dict = load_models_sarimax(model_dir)

    predictions = []
    for item in df_weekly["Item"].unique():
        test_item = df_weekly[df_weekly["Item"] == item].copy()
        n_steps = len(test_item)

        if item in item_models:
            try:
                pred = item_models[item].forecast(steps=n_steps).values
            except Exception:
                pred = np.full(n_steps, global_model)
        else:
            pred = np.full(n_steps, global_model)

        test_item["Raw_Pred"] = np.maximum(0, pred)
        predictions.append(test_item)

    result = pd.concat(predictions)
    return _apply_dow_adjustment(result.sort_values(["Item", "Date"]), dow_factor_dict)


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
    future_df = pd.DataFrame({
        "Date": np.repeat(future_dates, len(items)),
        "Item": np.tile(items, len(future_dates)),
        "Quantity_Sold": 0,
    })
    return future_df
