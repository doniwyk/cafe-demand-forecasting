from __future__ import annotations

import logging
import pandas as pd
import numpy as np
import pickle
import json
import time
from pathlib import Path

logging.getLogger("prophet").setLevel(logging.WARNING)
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)

from prophet import Prophet


MIN_TRAIN_RECORDS_DAILY = 180
MIN_TRAIN_RECORDS_WEEKLY = 100


def get_min_train_records(frequency: str) -> int:
    return MIN_TRAIN_RECORDS_DAILY if frequency == "daily" else MIN_TRAIN_RECORDS_WEEKLY


def _fit_item_prophet(series: pd.Series) -> Prophet:
    df = pd.DataFrame({"ds": series.index, "y": series.values})
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode="additive",
        changepoint_prior_scale=0.05,
    )
    model.fit(df)
    return model


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


def train_models_prophet(
    df_features: pd.DataFrame,
    output_dir: str | Path | None = None,
    frequency: str = "weekly",
) -> tuple[dict, float, dict]:
    output_dir = Path(output_dir) if output_dir else None
    output_dir.mkdir(parents=True, exist_ok=True) if output_dir else None

    dow_factor_dict = _compute_dow_factors(df_features)
    global_avg = _get_global_avg(df_features)
    min_recs = get_min_train_records(frequency)

    item_models = {}
    items = list(df_features["Item"].unique())
    total_items = len(items)
    print(f"[Prophet] Training per-item models ({frequency})...", flush=True)
    for idx, item in enumerate(items):
        if (idx + 1) % 20 == 0 or idx == 0:
            print(
                f"  Progress: {idx + 1}/{total_items} items ({((idx + 1) / total_items * 100):.1f}%)",
                flush=True,
            )
        train_item = df_features[df_features["Item"] == item].set_index("Date").sort_index()
        if len(train_item) >= min_recs:
            try:
                model = _fit_item_prophet(train_item["Quantity_Sold"])
                item_models[item] = model
            except Exception:
                pass

    metadata = {
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_item_models": len(item_models),
        "items_with_models": sorted(item_models.keys()),
        "global_avg": global_avg,
    }

    if output_dir:
        with open(output_dir / "item_models_prophet.pkl", "wb") as f:
            pickle.dump(item_models, f)
        with open(output_dir / "global_model_prophet.json", "w") as f:
            json.dump({"global_avg": global_avg}, f, indent=2)
        with open(output_dir / "dow_factors_prophet.json", "w") as f:
            json.dump(dow_factor_dict, f, indent=2)
        with open(output_dir / "model_metadata_prophet.json", "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"[Prophet] Models saved to: {output_dir}")
        print(f"  - Per-item models: {len(item_models)} items in item_models_prophet.pkl")
        print(f"  - DOW factors: dow_factors_prophet.json")

    return item_models, global_avg, dow_factor_dict


def load_models_prophet(
    model_dir: str | Path | None = None,
) -> tuple[dict, float, dict]:
    model_dir = Path(model_dir) if model_dir else None

    with open(model_dir / "item_models_prophet.pkl", "rb") as f:
        item_models = pickle.load(f)
    with open(model_dir / "global_model_prophet.json", "r") as f:
        data = json.load(f)
    global_avg = data.get("global_avg", 0.0)
    with open(model_dir / "dow_factors_prophet.json", "r") as f:
        dow_factor_dict = json.load(f)

    return item_models, global_avg, dow_factor_dict


def predict_prophet(
    df_features: pd.DataFrame,
    item_models: dict | None = None,
    global_model: float | None = None,
    dow_factor_dict: dict | None = None,
    model_dir: str | Path | None = None,
    frequency: str = "weekly",
) -> pd.DataFrame:
    if item_models is None or global_model is None or dow_factor_dict is None:
        item_models, global_model, dow_factor_dict = load_models_prophet(model_dir)

    predictions = []
    items = list(df_features["Item"].unique())
    for item in items:
        test_item = df_features[df_features["Item"] == item].copy()
        pred = np.full(len(test_item), global_model)

        if item in item_models:
            try:
                model = item_models[item]
                future = pd.DataFrame({"ds": test_item["Date"].values})
                forecast = model.predict(future)
                pred = np.maximum(0, forecast["yhat"].values)
            except Exception:
                pass

        test_item["Raw_Pred"] = np.maximum(0, pred)
        predictions.append(test_item)

    result = pd.concat(predictions)
    return _apply_dow_adjustment(result.sort_values(["Item", "Date"]), dow_factor_dict)


def generate_future_weekly(
    df_daily: pd.DataFrame,
    future_weeks: int = 12,
) -> pd.DataFrame:
    max_date = df_daily["Date"].max()
    items = df_daily["Item"].unique()

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


def train_and_predict_prophet(
    df_features: pd.DataFrame,
    n_test_periods: int = 12,
    frequency: str = "weekly",
) -> pd.DataFrame:
    if frequency == "daily":
        split_date = df_features["Date"].max() - pd.Timedelta(days=n_test_periods * 7)
    else:
        split_date = df_features["Date"].max() - pd.Timedelta(weeks=n_test_periods)
    train = df_features[df_features["Date"] < split_date].copy()
    test = df_features[df_features["Date"] >= split_date].copy()

    dow_factor_dict = _compute_dow_factors(train)
    global_avg = _get_global_avg(train)
    min_recs = get_min_train_records(frequency)

    print(f"[Prophet] Training per-item models ({frequency})...", flush=True)
    predictions = []
    items = list(test["Item"].unique())
    total_items = len(items)
    for idx, item in enumerate(items):
        if (idx + 1) % 20 == 0 or idx == 0:
            print(f"  Progress: {idx + 1}/{total_items} items ({((idx + 1) / total_items * 100):.1f}%)", flush=True)
        train_item = train[train["Item"] == item].set_index("Date").sort_index()
        test_item = test[test["Item"] == item].copy()

        if len(train_item) >= min_recs:
            try:
                model = _fit_item_prophet(train_item["Quantity_Sold"])
                future = pd.DataFrame({"ds": test_item["Date"].values})
                forecast = model.predict(future)
                pred = forecast["yhat"].values
            except Exception:
                pred = np.full(len(test_item), global_avg)
        else:
            pred = np.full(len(test_item), global_avg)

        test_item["Raw_Pred"] = np.maximum(0, pred)
        predictions.append(test_item)

    result = pd.concat(predictions)
    return _apply_dow_adjustment(result.sort_values(["Item", "Date"]), dow_factor_dict)
