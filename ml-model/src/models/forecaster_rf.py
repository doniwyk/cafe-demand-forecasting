from __future__ import annotations

import pandas as pd
import numpy as np
import pickle
import json
import time
from pathlib import Path
from typing import Optional
from datetime import datetime

from sklearn.ensemble import RandomForestRegressor

from src.utils.config import get_feature_columns


MIN_TRAIN_RECORDS = 40

_RF_GLOBAL_PARAMS = {
    "n_estimators": 500,
    "max_depth": 12,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "random_state": 42,
    "n_jobs": -1,
}

_RF_ITEM_PARAMS = {
    "n_estimators": 300,
    "max_depth": 10,
    "min_samples_split": 4,
    "min_samples_leaf": 2,
    "random_state": 42,
    "n_jobs": -1,
}


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


def train_and_predict_rf(
    df_features: pd.DataFrame,
    n_test_periods: int = 12,
) -> pd.DataFrame:
    split_date = df_features["Date"].max() - pd.Timedelta(weeks=n_test_periods)
    train = df_features[df_features["Date"] < split_date].copy()
    test = df_features[df_features["Date"] >= split_date].copy()

    features = get_feature_columns("weekly")
    dow_factor_dict = _compute_dow_factors(train)

    print("[RF] Training global fallback model...")
    t0 = time.time()
    global_model = RandomForestRegressor(**_RF_GLOBAL_PARAMS)
    global_model.fit(train[features], train["Quantity_Sold"])
    print(f"[RF] Global model trained in {time.time() - t0:.1f}s")

    print("[RF] Training per-item models...")
    predictions = []
    items = list(test["Item"].unique())
    for idx, item in enumerate(items):
        if (idx + 1) % 20 == 0 or idx == 0:
            print(f"  Progress: {idx + 1}/{len(items)} items")
        train_item = train[train["Item"] == item]
        test_item = test[test["Item"] == item].copy()

        if len(train_item) >= MIN_TRAIN_RECORDS:
            model = RandomForestRegressor(**_RF_ITEM_PARAMS)
            model.fit(train_item[features], train_item["Quantity_Sold"])
            pred = model.predict(test_item[features])
        else:
            pred = global_model.predict(test_item[features])

        test_item["Raw_Pred"] = np.maximum(0, pred)
        predictions.append(test_item)

    result = pd.concat(predictions)
    return _apply_dow_adjustment(result.sort_values(["Item", "Date"]), dow_factor_dict)


def train_models_rf(
    df_features: pd.DataFrame,
    output_dir: str | Path | None = None,
) -> tuple[dict, object, dict]:
    output_dir = Path(output_dir) if output_dir else Path("models")
    output_dir.mkdir(parents=True, exist_ok=True)

    features = get_feature_columns("weekly")

    print("[RF] Training global fallback model...")
    t0 = time.time()
    global_model = RandomForestRegressor(**_RF_GLOBAL_PARAMS)
    global_model.fit(df_features[features], df_features["Quantity_Sold"])
    print(f"[RF] Global model trained in {time.time() - t0:.1f}s")

    item_models = {}
    items = list(df_features["Item"].unique())
    print(f"[RF] Training per-item models... total items: {len(items)}")
    for idx, item in enumerate(items):
        if (idx + 1) % 20 == 0 or idx == 0:
            print(f"  Progress: {idx + 1}/{len(items)} items")
        train_item = df_features[df_features["Item"] == item]
        if len(train_item) < MIN_TRAIN_RECORDS:
            continue
        model = RandomForestRegressor(**_RF_ITEM_PARAMS)
        model.fit(train_item[features], train_item["Quantity_Sold"])
        item_models[item] = model

    with open(output_dir / "global_model_rf.pkl", "wb") as f:
        pickle.dump(global_model, f)
    with open(output_dir / "item_models_rf.pkl", "wb") as f:
        pickle.dump(item_models, f)

    dow_factor_dict = _compute_dow_factors(df_features)
    with open(output_dir / "dow_factors_rf.json", "w") as f:
        json.dump(dow_factor_dict, f, indent=2)

    metadata = {
        "model_type": "random_forest",
        "trained_at": datetime.now().isoformat(),
        "n_item_models": len(item_models),
        "items_with_models": sorted(item_models.keys()),
        "features": features,
        "n_records": len(df_features),
        "date_range": [str(df_features["Date"].min()), str(df_features["Date"].max())],
    }
    with open(output_dir / "model_metadata_rf.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"[RF] Models saved to: {output_dir}")
    return item_models, global_model, dow_factor_dict


def load_models_rf(
    model_dir: str | Path | None = None,
) -> tuple[dict, object, dict]:
    model_dir = Path(model_dir) if model_dir else Path("models")
    with open(model_dir / "global_model_rf.pkl", "rb") as f:
        global_model = pickle.load(f)
    with open(model_dir / "item_models_rf.pkl", "rb") as f:
        item_models = pickle.load(f)
    with open(model_dir / "dow_factors_rf.json", "r") as f:
        dow_factor_dict = json.load(f)
    return item_models, global_model, dow_factor_dict


def predict_rf(
    df_features: pd.DataFrame,
    item_models: dict | None = None,
    global_model: object | None = None,
    dow_factor_dict: dict | None = None,
    model_dir: str | Path | None = None,
) -> pd.DataFrame:
    if item_models is None or global_model is None or dow_factor_dict is None:
        item_models, global_model, dow_factor_dict = load_models_rf(model_dir)

    features = get_feature_columns("weekly")
    predictions = []

    for item in df_features["Item"].unique():
        test_item = df_features[df_features["Item"] == item].copy()
        if item in item_models:
            pred = item_models[item].predict(test_item[features])
        else:
            pred = global_model.predict(test_item[features])
        test_item["Raw_Pred"] = np.maximum(0, pred)
        predictions.append(test_item)

    result = pd.concat(predictions)
    return _apply_dow_adjustment(result.sort_values(["Item", "Date"]), dow_factor_dict)
