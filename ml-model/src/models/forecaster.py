from __future__ import annotations

import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
from typing import Optional
from datetime import datetime

from xgboost import XGBRegressor

from src.utils.config import FEATURE_COLUMNS, MODELS_DIR
from src.models.features import create_features


MIN_TRAIN_RECORDS = 40


def train_and_predict(
    df_features: pd.DataFrame,
    n_test_weeks: int = 12,
) -> pd.DataFrame:
    split_date = df_features["Date"].max() - pd.Timedelta(weeks=n_test_weeks)
    train = df_features[df_features["Date"] < split_date].copy()
    test = df_features[df_features["Date"] >= split_date].copy()

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

    features = FEATURE_COLUMNS

    print("Training global fallback model...")
    global_model = XGBRegressor(
        objective="reg:tweedie",
        tweedie_variance_power=1.5,
        n_estimators=2000,
        learning_rate=0.02,
        max_depth=7,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=-1,
        random_state=42,
    )
    global_model.fit(train[features], train["Quantity_Sold"])

    print("Training per-item models...")
    predictions = []
    for item in test["Item"].unique():
        train_item = train[train["Item"] == item]
        test_item = test[test["Item"] == item].copy()

        if len(train_item) >= MIN_TRAIN_RECORDS:
            model = XGBRegressor(
                objective="reg:tweedie",
                tweedie_variance_power=1.5,
                n_estimators=1500,
                learning_rate=0.03,
                max_depth=6,
                subsample=0.85,
                colsample_bytree=0.85,
                n_jobs=-1,
                random_state=42,
            )
            model.fit(train_item[features], train_item["Quantity_Sold"], verbose=False)
            pred = model.predict(test_item[features])
        else:
            model = None
            pred = global_model.predict(test_item[features])

        test_item["Raw_Pred"] = np.maximum(0, pred)
        test_item["DOW"] = test_item["Date"].dt.weekday

        factors = dow_factor_dict.get(item, {i: 1.0 for i in range(7)})
        test_item["dow_factor"] = test_item["DOW"].map(factors).fillna(1.0)
        test_item["Predicted"] = (
            test_item["Raw_Pred"] * test_item["dow_factor"]
        ).round(0)
        test_item["Predicted"] = np.maximum(0, test_item["Predicted"])

        predictions.append(test_item)

    return pd.concat(predictions).sort_values(["Item", "Date"])


def train_models(
    df_features: pd.DataFrame,
    output_dir: str | Path | None = None,
) -> tuple[dict, XGBRegressor, dict]:
    output_dir = Path(output_dir) if output_dir else MODELS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    features = FEATURE_COLUMNS

    print("Training global fallback model...")
    global_model = XGBRegressor(
        objective="reg:tweedie",
        tweedie_variance_power=1.5,
        n_estimators=2000,
        learning_rate=0.02,
        max_depth=7,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=-1,
        random_state=42,
    )
    global_model.fit(df_features[features], df_features["Quantity_Sold"])

    item_models = {}
    for item in df_features["Item"].unique():
        train_item = df_features[df_features["Item"] == item]
        if len(train_item) < MIN_TRAIN_RECORDS:
            continue

        model = XGBRegressor(
            objective="reg:tweedie",
            tweedie_variance_power=1.5,
            n_estimators=1500,
            learning_rate=0.03,
            max_depth=6,
            subsample=0.85,
            colsample_bytree=0.85,
            n_jobs=-1,
            random_state=42,
        )
        model.fit(train_item[features], train_item["Quantity_Sold"], verbose=False)
        item_models[item] = model

    with open(output_dir / "global_model.pkl", "wb") as f:
        pickle.dump(global_model, f)

    with open(output_dir / "item_models.pkl", "wb") as f:
        pickle.dump(item_models, f)

    dow_pattern = (
        df_features.groupby(["Item", df_features["Date"].dt.weekday])["Quantity_Sold"]
        .mean()
        .reset_index()
    )
    item_avg = (
        df_features.groupby("Item")["Quantity_Sold"]
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

    with open(output_dir / "dow_factors.json", "w") as f:
        json.dump(dow_factor_dict, f, indent=2)

    metadata = {
        "trained_at": datetime.now().isoformat(),
        "n_item_models": len(item_models),
        "items_with_models": sorted(item_models.keys()),
        "features": features,
        "n_records": len(df_features),
        "date_range": [
            str(df_features["Date"].min()),
            str(df_features["Date"].max()),
        ],
    }
    with open(output_dir / "model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Models saved to: {output_dir}")
    print(f"  - Global model: global_model.pkl")
    print(f"  - Per-item models: {len(item_models)} items in item_models.pkl")
    print(f"  - DOW factors: dow_factors.json")

    return item_models, global_model, dow_factor_dict


def load_models(
    model_dir: str | Path | None = None,
) -> tuple[dict, XGBRegressor, dict]:
    model_dir = Path(model_dir) if model_dir else MODELS_DIR

    with open(model_dir / "global_model.pkl", "rb") as f:
        global_model = pickle.load(f)

    with open(model_dir / "item_models.pkl", "rb") as f:
        item_models = pickle.load(f)

    with open(model_dir / "dow_factors.json", "r") as f:
        dow_factor_dict = json.load(f)

    return item_models, global_model, dow_factor_dict


def predict(
    df_features: pd.DataFrame,
    item_models: dict | None = None,
    global_model: XGBRegressor | None = None,
    dow_factor_dict: dict | None = None,
    model_dir: str | Path | None = None,
) -> pd.DataFrame:
    if item_models is None or global_model is None or dow_factor_dict is None:
        item_models, global_model, dow_factor_dict = load_models(model_dir)

    features = FEATURE_COLUMNS
    predictions = []

    for item in df_features["Item"].unique():
        test_item = df_features[df_features["Item"] == item].copy()

        if item in item_models:
            model = item_models[item]
            pred = model.predict(test_item[features])
        else:
            pred = global_model.predict(test_item[features])

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


def generate_future_features(
    df_daily: pd.DataFrame,
    future_weeks: int = 12,
) -> pd.DataFrame:
    max_date = df_daily["Date"].max()
    items = df_daily["Item"].unique()

    future_dates = pd.date_range(
        start=max_date + pd.Timedelta(days=1),
        periods=future_weeks * 7,
        freq="D",
    )

    future_df = pd.DataFrame(
        {
            "Date": np.repeat(future_dates, len(items)),
            "Item": np.tile(items, len(future_dates)),
        }
    )
    future_df["Quantity_Sold"] = 0

    all_df = pd.concat([df_daily, future_df], ignore_index=True)
    all_df = create_features(all_df)
    future_features = all_df[all_df["Date"] > max_date]

    return future_features
