from __future__ import annotations

import logging
import pandas as pd
import numpy as np
import time

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
