from __future__ import annotations

import logging
import pandas as pd
import numpy as np
import pickle
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
    items = test["Item"].unique()
    for i, item in enumerate(items):
        if (i + 1) % 10 == 0:
            print(f"  Fitting item {i + 1}/{len(items)}...")

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
