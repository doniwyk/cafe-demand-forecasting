from __future__ import annotations

import logging
import pandas as pd
import numpy as np
import pickle
import json
import warnings
from pathlib import Path
from typing import Optional
from datetime import datetime

from prophet import Prophet

from src.utils.config import MODELS_DIR

logger = logging.getLogger("prophet")
logger.setLevel(logging.WARNING)
logger.propagate = False

logging.getLogger("cmdstanpy").setLevel(logging.WARNING)

MIN_TRAIN_WEEKS = 26


def train_and_predict_prophet(
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

    print("Training per-item Prophet models...")
    predictions = []
    items = list(test["Item"].unique())
    total_items = len(items)
    for i, item in enumerate(items):
        if (i + 1) % 10 == 0 or i == 0:
            print(
                f"  Progress: {i + 1}/{total_items} items ({((i + 1) / total_items * 100):.1f}%)"
            )

        train_item = train[train["Item"] == item].sort_values("Date")
        test_item = test[test["Item"] == item].copy()

        if len(train_item) >= MIN_TRAIN_WEEKS:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    warnings.filterwarnings("ignore", category=FutureWarning)

                m = Prophet(
                    yearly_seasonality=True,
                    weekly_seasonality=False,
                    daily_seasonality=False,
                    growth="linear",
                    changepoint_prior_scale=0.05,
                    seasonality_prior_scale=10.0,
                    uncertainty_samples=0,
                )
                m.add_seasonality(
                    name="weekly",
                    period=7,
                    fourier_order=3,
                )

                train_prophet = pd.DataFrame(
                    {
                        "ds": train_item["Date"].values,
                        "y": train_item["Quantity_Sold"].values.astype(float),
                    }
                )
                m.fit(train_prophet)

                future_dates = test_item["Date"].values
                future = pd.DataFrame({"ds": future_dates})
                forecast = m.predict(future)
                pred = forecast["yhat"].values
            except Exception:
                pred = np.full(len(test_item), global_mean)
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


def train_models_prophet(
    df_weekly: pd.DataFrame,
    output_dir: str | Path | None = None,
) -> tuple[dict, None, dict]:
    output_dir = Path(output_dir) if output_dir else MODELS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Training per-item Prophet models...")
    item_models = {}
    items = list(df_weekly["Item"].unique())
    total = len(items)
    print(f"  Total items: {total}")
    for i, item in enumerate(items):
        print(f"  [{i + 1}/{total}] Training {item}...")

        train_item = df_weekly[df_weekly["Item"] == item].sort_values("Date")
        if len(train_item) < MIN_TRAIN_WEEKS:
            continue

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                warnings.filterwarnings("ignore", category=FutureWarning)

            m = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=False,
                daily_seasonality=False,
                growth="linear",
                changepoint_prior_scale=0.05,
                seasonality_prior_scale=10.0,
                uncertainty_samples=0,
            )
            m.add_seasonality(name="weekly", period=7, fourier_order=3)

            train_prophet = pd.DataFrame(
                {
                    "ds": train_item["Date"].values,
                    "y": train_item["Quantity_Sold"].values.astype(float),
                }
            )
            m.fit(train_prophet)
            item_models[item] = m
        except Exception:
            continue

    with open(output_dir / "item_models_prophet.pkl", "wb") as f:
        pickle.dump(item_models, f)

    dow_factor_dict = _compute_dow_factors(df_weekly)
    with open(output_dir / "dow_factors_prophet.json", "w") as f:
        json.dump(dow_factor_dict, f, indent=2)

    global_mean = float(df_weekly["Quantity_Sold"].mean())
    metadata = {
        "model_type": "prophet",
        "trained_at": datetime.now().isoformat(),
        "n_item_models": len(item_models),
        "items_with_models": sorted(item_models.keys()),
        "n_records": len(df_weekly),
        "date_range": [str(df_weekly["Date"].min()), str(df_weekly["Date"].max())],
        "global_mean": global_mean,
    }
    with open(output_dir / "model_metadata_prophet.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Prophet Models saved to: {output_dir}")
    print(f"  - Per-item models: {len(item_models)} items in item_models_prophet.pkl")
    print(f"  Completed all {total} items")
    return item_models, None, dow_factor_dict


def load_models_prophet(
    model_dir: str | Path | None = None,
) -> tuple[dict, None, dict]:
    model_dir = Path(model_dir) if model_dir else MODELS_DIR

    with open(model_dir / "item_models_prophet.pkl", "rb") as f:
        item_models = pickle.load(f)

    with open(model_dir / "dow_factors_prophet.json", "r") as f:
        dow_factor_dict = json.load(f)

    return item_models, None, dow_factor_dict


def predict_prophet(
    df_weekly: pd.DataFrame,
    item_models: dict | None = None,
    global_model: None = None,
    dow_factor_dict: dict | None = None,
    model_dir: str | Path | None = None,
) -> pd.DataFrame:
    if item_models is None or dow_factor_dict is None:
        item_models, _, dow_factor_dict = load_models_prophet(model_dir)

    meta_path = (
        Path(model_dir) if model_dir else MODELS_DIR
    ) / "model_metadata_prophet.json"
    global_mean = 0
    if meta_path.exists():
        with open(meta_path) as f:
            global_mean = json.load(f).get("global_mean", 0)

    predictions = []
    for item in df_weekly["Item"].unique():
        test_item = df_weekly[df_weekly["Item"] == item].copy()

        if item in item_models:
            try:
                future = pd.DataFrame({"ds": test_item["Date"].values})
                forecast = item_models[item].predict(future)
                pred = forecast["yhat"].values
            except Exception:
                pred = np.full(len(test_item), global_mean)
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
