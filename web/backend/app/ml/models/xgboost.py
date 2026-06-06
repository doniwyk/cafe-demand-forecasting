from __future__ import annotations

import pandas as pd
import numpy as np
import pickle
import json
import time
from pathlib import Path
from typing import Optional
from datetime import datetime

from xgboost import XGBRegressor

from app.ml.config import FEATURE_COLUMNS, MODELS_DIR
from app.ml.utils.gpu import get_xgboost_params
from app.ml.features import create_features

MIN_TRAIN_RECORDS = 60

_EARLY_STOPPING_ROUNDS = 30
_BLEND_ALPHA = 0.5

_BASE_GLOBAL_PARAMS = {
    "objective": "count:poisson",
    "n_estimators": 600,
    "learning_rate": 0.03,
    "max_depth": 5,
    "min_child_weight": 3,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "reg_alpha": 0.5,
    "reg_lambda": 1.0,
    "random_state": 42,
}

_BASE_ITEM_PARAMS = {
    "objective": "count:poisson",
    "n_estimators": 400,
    "learning_rate": 0.03,
    "max_depth": 4,
    "min_child_weight": 3,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "reg_alpha": 0.5,
    "reg_lambda": 1.0,
    "random_state": 42,
}


def load_and_prep_data(filepath: str | Path) -> pd.DataFrame:
    print(f"Loading data from: {filepath}")
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()

    date_col = "Date_Only" if "Date_Only" in df.columns else "Date"
    qty_col = "Quantity" if "Quantity" in df.columns else "Quantity_Sold"
    df["Date"] = pd.to_datetime(df[date_col])
    df["Quantity_Sold"] = df[qty_col]

    df = df[~df["Item"].str.strip().str.lower().str.startswith("add")]

    df_freq = (
        df.set_index("Date")
        .groupby("Item")
        .resample("D")["Quantity_Sold"]
        .sum()
        .reset_index()
    )

    print(f"Aggregated to daily: {len(df_freq)} observations")
    print(f"Date range: {df_freq['Date'].min().date()} to {df_freq['Date'].max().date()}")
    return df_freq


def _xgboost_params(base: dict) -> dict:
    return {**base, **get_xgboost_params()}


def _split_train_val(df: pd.DataFrame, val_ratio: float = 0.15):
    train_parts: list[pd.DataFrame] = []
    val_parts: list[pd.DataFrame] = []
    for item in df["Item"].unique():
        item_df = df[df["Item"] == item].sort_values("Date")
        n_val = max(1, int(len(item_df) * val_ratio))
        train_parts.append(item_df.iloc[: len(item_df) - n_val])
        val_parts.append(item_df.iloc[len(item_df) - n_val :])
    return pd.concat(train_parts, ignore_index=True), pd.concat(val_parts, ignore_index=True)


def train_and_predict(
    df_features: pd.DataFrame,
    n_test_periods: int = 12,
) -> pd.DataFrame:
    split_date = df_features["Date"].max() - pd.Timedelta(days=n_test_periods * 7)
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

    train_core, train_val = _split_train_val(train)

    print("Training global fallback model (with early stopping)...")
    t0 = time.time()
    global_model = XGBRegressor(
        **_xgboost_params(_BASE_GLOBAL_PARAMS),
        early_stopping_rounds=_EARLY_STOPPING_ROUNDS,
    )
    global_model.fit(
        train_core[FEATURE_COLUMNS], train_core["Quantity_Sold"],
        eval_set=[(train_val[FEATURE_COLUMNS], train_val["Quantity_Sold"])],
        verbose=False,
    )
    print(f"Global fallback model trained in {time.time() - t0:.1f}s")

    print("Training per-item models (with early stopping)...")
    predictions = []
    items = list(test["Item"].unique())
    total_items = len(items)
    for idx, item in enumerate(items):
        if (idx + 1) % 20 == 0 or idx == 0:
            print(
                f"  Progress: {idx + 1}/{total_items} items ({((idx + 1) / total_items * 100):.1f}%)"
            )
        train_item = train_core[train_core["Item"] == item]
        test_item = test[test["Item"] == item].copy()
        val_item = train_val[train_val["Item"] == item]

        if len(train_item) >= MIN_TRAIN_RECORDS:
            has_val = len(val_item) >= 1
            model_params = _xgboost_params(_BASE_ITEM_PARAMS)
            if has_val:
                model_params["early_stopping_rounds"] = _EARLY_STOPPING_ROUNDS
            model = XGBRegressor(**model_params)
            eval_set = (
                [(val_item[FEATURE_COLUMNS], val_item["Quantity_Sold"])]
                if has_val
                else None
            )
            model.fit(
                train_item[FEATURE_COLUMNS], train_item["Quantity_Sold"],
                eval_set=eval_set, verbose=False,
            )
            pred_item = model.predict(test_item[FEATURE_COLUMNS])
            pred_global = global_model.predict(test_item[FEATURE_COLUMNS])
            pred = _BLEND_ALPHA * pred_item + (1 - _BLEND_ALPHA) * pred_global
        else:
            model = None
            pred = global_model.predict(test_item[FEATURE_COLUMNS])

        test_item["Raw_Pred"] = np.maximum(0, pred)
        test_item["DOW"] = test_item["Date"].dt.weekday

        factors = dow_factor_dict.get(item, {i: 1.0 for i in range(7)})
        test_item["dow_factor"] = test_item["DOW"].map(factors).fillna(1.0)
        raw_adj = test_item["Raw_Pred"] * test_item["dow_factor"]
        test_item["Predicted"] = raw_adj.round(0)
        test_item["Predicted"] = np.maximum(0, test_item["Predicted"])
        test_item["Predicted"] = np.maximum(0, test_item["Predicted"])

        predictions.append(test_item)

    return pd.concat(predictions).sort_values(["Item", "Date"])


def train_models(
    df_features: pd.DataFrame,
    output_dir: str | Path | None = None,
) -> tuple[dict, XGBRegressor, dict]:
    output_dir = Path(output_dir) if output_dir else MODELS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    train_data, val_data = _split_train_val(df_features)

    print("Training global fallback model (with early stopping)...")
    t0 = time.time()
    global_model = XGBRegressor(
        **_xgboost_params(_BASE_GLOBAL_PARAMS),
        early_stopping_rounds=_EARLY_STOPPING_ROUNDS,
    )
    global_model.fit(
        train_data[FEATURE_COLUMNS], train_data["Quantity_Sold"],
        eval_set=[(val_data[FEATURE_COLUMNS], val_data["Quantity_Sold"])],
        verbose=False,
    )
    print(f"Global fallback model trained in {time.time() - t0:.1f}s")

    item_models = {}
    items = list(df_features["Item"].unique())
    total_items = len(items)
    print(f"Training per-item models (with early stopping)... total items: {total_items}")
    for idx, item in enumerate(items):
        if (idx + 1) % 20 == 0 or idx == 0:
            print(
                f"  Progress: {idx + 1}/{total_items} items ({((idx + 1) / total_items * 100):.1f}%)"
            )
        train_item = train_data[train_data["Item"] == item]
        if len(train_item) < MIN_TRAIN_RECORDS:
            continue

        val_item = val_data[val_data["Item"] == item]
        has_val = len(val_item) >= 1
        model_params = _xgboost_params(_BASE_ITEM_PARAMS)
        if has_val:
            model_params["early_stopping_rounds"] = _EARLY_STOPPING_ROUNDS
        model = XGBRegressor(**model_params)
        eval_set = (
            [(val_item[FEATURE_COLUMNS], val_item["Quantity_Sold"])]
            if has_val
            else None
        )
        model.fit(
            train_item[FEATURE_COLUMNS], train_item["Quantity_Sold"],
            eval_set=eval_set, verbose=False,
        )
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
        "features": FEATURE_COLUMNS,
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
        raw_adj = test_item["Raw_Pred"] * test_item["dow_factor"]
        test_item["Predicted"] = raw_adj.round(0)
        test_item["Predicted"] = np.maximum(0, test_item["Predicted"])
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

    last_known = (
        df_daily.sort_values("Date")
        .groupby("Item")
        .last()["Quantity_Sold"]
        .reset_index()
    )
    last_map = dict(zip(last_known["Item"], last_known["Quantity_Sold"]))

    all_features = []
    current = df_daily.copy()

    for _, next_date in enumerate(future_dates):
        next_df = pd.DataFrame(
            {"Date": [next_date] * len(items), "Item": np.array(items)}
        )
        next_df["Quantity_Sold"] = next_df["Item"].map(last_map).fillna(1)

        temp = pd.concat([current, next_df], ignore_index=True)
        feat = create_features(temp)
        future_rows = feat[feat["Date"] == next_date]
        all_features.append(future_rows)

        current = pd.concat([current, next_df], ignore_index=True)

    return pd.concat(all_features, ignore_index=True)
