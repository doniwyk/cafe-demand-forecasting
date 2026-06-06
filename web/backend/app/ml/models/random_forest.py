from __future__ import annotations

import pandas as pd
import numpy as np
import time
import json
import pickle
from pathlib import Path

from sklearn.ensemble import RandomForestRegressor

from app.ml.config import FEATURE_COLUMNS


MIN_TRAIN_RECORDS = 60
_BLEND_ALPHA = 0.5

_RF_GLOBAL_PARAMS = {
    "n_estimators": 500,
    "max_depth": 6,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "random_state": 42,
    "n_jobs": -1,
}

_RF_ITEM_PARAMS = {
    "n_estimators": 400,
    "max_depth": 5,
    "min_samples_split": 5,
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
    df["Predicted"] = np.maximum(0, df["Predicted"])
    return df


def train_models_rf(
    df_features: pd.DataFrame,
    output_dir: str | Path | None = None,
) -> tuple[dict, RandomForestRegressor, dict]:
    output_dir = Path(output_dir) if output_dir else None
    output_dir.mkdir(parents=True, exist_ok=True) if output_dir else None

    dow_factor_dict = _compute_dow_factors(df_features)

    print("[RF] Training global fallback model...", flush=True)
    t0 = time.time()
    global_model = RandomForestRegressor(**_RF_GLOBAL_PARAMS)
    global_model.fit(df_features[FEATURE_COLUMNS], df_features["Quantity_Sold"])
    print(f"[RF] Global model trained in {time.time() - t0:.1f}s", flush=True)

    item_models = {}
    items = list(df_features["Item"].unique())
    total_items = len(items)
    print(f"[RF] Training per-item models... total items: {total_items}", flush=True)
    for idx, item in enumerate(items):
        if (idx + 1) % 20 == 0 or idx == 0:
            print(
                f"  Progress: {idx + 1}/{total_items} items ({((idx + 1) / total_items * 100):.1f}%)",
                flush=True,
            )
        train_item = df_features[df_features["Item"] == item]
        if len(train_item) < MIN_TRAIN_RECORDS:
            continue

        model = RandomForestRegressor(**_RF_ITEM_PARAMS)
        model.fit(train_item[FEATURE_COLUMNS], train_item["Quantity_Sold"])
        item_models[item] = model

    if output_dir:
        with open(output_dir / "global_model_rf.pkl", "wb") as f:
            pickle.dump(global_model, f)
        with open(output_dir / "item_models_rf.pkl", "wb") as f:
            pickle.dump(item_models, f)
        with open(output_dir / "dow_factors_rf.json", "w") as f:
            json.dump(dow_factor_dict, f, indent=2)
        print(f"[RF] Models saved to: {output_dir}")
        print(f"  - Global model: global_model_rf.pkl")
        print(f"  - Per-item models: {len(item_models)} items in item_models_rf.pkl")
        print(f"  - DOW factors: dow_factors_rf.json")

    return item_models, global_model, dow_factor_dict


def load_models_rf(
    model_dir: str | Path | None = None,
) -> tuple[dict, RandomForestRegressor, dict]:
    model_dir = Path(model_dir) if model_dir else None

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
    global_model: RandomForestRegressor | None = None,
    dow_factor_dict: dict | None = None,
    model_dir: str | Path | None = None,
) -> pd.DataFrame:
    if item_models is None or global_model is None or dow_factor_dict is None:
        item_models, global_model, dow_factor_dict = load_models_rf(model_dir)

    predictions = []

    for item in df_features["Item"].unique():
        test_item = df_features[df_features["Item"] == item].copy()

        if item in item_models:
            model = item_models[item]
            pred_item = model.predict(test_item[FEATURE_COLUMNS])
            pred_global = global_model.predict(test_item[FEATURE_COLUMNS])
            pred = _BLEND_ALPHA * pred_item + (1 - _BLEND_ALPHA) * pred_global
        else:
            pred = global_model.predict(test_item[FEATURE_COLUMNS])

        test_item["Raw_Pred"] = np.maximum(0, pred)
        predictions.append(test_item)

    result = pd.concat(predictions)
    return _apply_dow_adjustment(result.sort_values(["Item", "Date"]), dow_factor_dict)


def train_and_predict_rf(
    df_features: pd.DataFrame,
    n_test_periods: int = 12,
) -> pd.DataFrame:
    split_date = df_features["Date"].max() - pd.Timedelta(days=n_test_periods * 7)
    train = df_features[df_features["Date"] < split_date].copy()
    test = df_features[df_features["Date"] >= split_date].copy()

    dow_factor_dict = _compute_dow_factors(train)

    print("[RF] Training global fallback model...", flush=True)
    t0 = time.time()
    global_model = RandomForestRegressor(**_RF_GLOBAL_PARAMS)
    global_model.fit(train[FEATURE_COLUMNS], train["Quantity_Sold"])
    print(f"[RF] Global model trained in {time.time() - t0:.1f}s", flush=True)

    print("[RF] Training per-item models...", flush=True)
    predictions = []
    items = list(test["Item"].unique())
    total_items = len(items)
    for idx, item in enumerate(items):
        if (idx + 1) % 20 == 0 or idx == 0:
            print(f"  Progress: {idx + 1}/{total_items} items ({((idx + 1) / total_items * 100):.1f}%)", flush=True)
        train_item = train[train["Item"] == item]
        test_item = test[test["Item"] == item].copy()

        if len(train_item) >= MIN_TRAIN_RECORDS:
            model = RandomForestRegressor(**_RF_ITEM_PARAMS)
            model.fit(train_item[FEATURE_COLUMNS], train_item["Quantity_Sold"])
            pred_item = model.predict(test_item[FEATURE_COLUMNS])
            pred_global = global_model.predict(test_item[FEATURE_COLUMNS])
            pred = _BLEND_ALPHA * pred_item + (1 - _BLEND_ALPHA) * pred_global
        else:
            pred = global_model.predict(test_item[FEATURE_COLUMNS])

        test_item["Raw_Pred"] = np.maximum(0, pred)
        predictions.append(test_item)

    result = pd.concat(predictions)
    return _apply_dow_adjustment(result.sort_values(["Item", "Date"]), dow_factor_dict)
