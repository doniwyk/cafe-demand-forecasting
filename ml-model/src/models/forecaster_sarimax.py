from __future__ import annotations

import pandas as pd
import numpy as np
import pickle
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

    print("Training per-item SARIMAX models...")
    predictions = []
    items = test["Item"].unique()
    for i, item in enumerate(items):
        if (i + 1) % 10 == 0:
            print(f"  Fitting item {i + 1}/{len(items)}...")

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

    return pd.concat(predictions).sort_values(["Item", "Date"])
