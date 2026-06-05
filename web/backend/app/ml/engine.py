from __future__ import annotations

import importlib
import json
from typing import Any

import numpy as np
import pandas as pd

from app.config import ML_MODELS_DIR
from src.models.features import create_features
from src.evaluation.metrics import generate_abc_analysis
from src.utils.config import DISCONTINUED_ITEMS


_METADATA_FILE: dict[str, str] = {
    "xgboost": "model_metadata.json",
    "random_forest": "model_metadata_rf.json",
    "sarimax": "model_metadata_sarimax.json",
    "prophet": "model_metadata_prophet.json",
}


def _import_model_fns(model_type: str) -> dict[str, Any]:
    """Lazily import the right module and return its key functions."""
    if model_type == "xgboost":
        from src.models.forecaster import train_models, load_models, predict, train_and_predict
        return {
            "train": train_models,
            "load": load_models,
            "predict": predict,
            "train_and_predict": train_and_predict,
            "needs_features": True,
        }

    suffix = {"random_forest": "rf", "sarimax": "sarimax", "prophet": "prophet"}[model_type]
    module_name = f"src.models.forecaster_{suffix}"
    try:
        mod = importlib.import_module(module_name)
    except ImportError:
        raise ImportError(f"Required module {module_name} not available")

    prefix = "" if model_type == "xgboost" else f"_{suffix}"
    train_fn = getattr(mod, f"train_models{prefix}", None)
    load_fn = getattr(mod, f"load_models{prefix}", None)
    predict_fn = getattr(mod, f"predict{prefix}", None)
    tap_fn = getattr(mod, f"train_and_predict{prefix}", None)

    if any(fn is None for fn in [train_fn, load_fn, predict_fn, tap_fn]):
        raise ImportError(f"{module_name} missing required functions")

    return {
        "train": train_fn,
        "load": load_fn,
        "predict": predict_fn,
        "train_and_predict": tap_fn,
        "needs_features": model_type in ("xgboost", "random_forest"),
    }


_model_fns_cache: dict[str, dict[str, Any]] = {}
_models_cache: dict[str, dict[str, Any]] = {
    mt: {"item_models": None, "global_model": None, "dow_factors": None, "loaded": False}
    for mt in _METADATA_FILE
}


def _get_fns(model_type: str) -> dict[str, Any]:
    if model_type not in _model_fns_cache:
        _model_fns_cache[model_type] = _import_model_fns(model_type)
    return _model_fns_cache[model_type]


def _load_for_model(model_type: str):
    fns = _get_fns(model_type)
    im, gm, dow = fns["load"](ML_MODELS_DIR)
    return im, gm, dow


def _ensure_models_loaded(model_type: str = "xgboost"):
    cache = _models_cache[model_type]
    if cache["loaded"]:
        return
    cache["item_models"], cache["global_model"], cache["dow_factors"] = _load_for_model(model_type)
    cache["loaded"] = True


def _predict_dispatch(model_type: str, df, item_models, global_model, dow_factors):
    fns = _get_fns(model_type)
    return fns["predict"](df, item_models=item_models, global_model=global_model, dow_factor_dict=dow_factors)


def run_predict(df: pd.DataFrame, model_type: str = "xgboost") -> pd.DataFrame:
    _ensure_models_loaded(model_type)
    cache = _models_cache[model_type]
    return _predict_dispatch(model_type, df, cache["item_models"], cache["global_model"], cache["dow_factors"])


def _clean_and_prepare(df_daily: pd.DataFrame, model_type: str) -> pd.DataFrame:
    df = _clean_data(df_daily)
    fns = _get_fns(model_type)
    if fns["needs_features"]:
        return create_features(df)
    return df


def run_train_and_evaluate(df_daily: pd.DataFrame, model_type: str = "xgboost"):
    processed = _clean_and_prepare(df_daily, model_type)
    fns = _get_fns(model_type)
    fns["train"](processed, ML_MODELS_DIR)
    test_pred = fns["train_and_predict"](processed)
    analysis = generate_abc_analysis(test_pred)
    _models_cache[model_type]["loaded"] = False
    return analysis


def _clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df[~df["Item"].str.strip().str.lower().str.startswith("add")]
    if DISCONTINUED_ITEMS:
        df = df[~df["Item"].isin(DISCONTINUED_ITEMS)]
    df = (
        df.set_index("Date")
        .groupby("Item")
        .resample("D")["Quantity_Sold"]
        .sum()
        .fillna(0)
        .reset_index()
    )
    return df


def get_model_metadata(model_type: str = "xgboost") -> dict | None:
    meta_path = ML_MODELS_DIR / _METADATA_FILE.get(model_type, "model_metadata.json")
    if not meta_path.exists():
        return None
    with open(meta_path) as f:
        return json.load(f)


def generate_forecast(df_daily: pd.DataFrame, weeks: int = 12, model_type: str = "xgboost") -> pd.DataFrame:
    data = _clean_data(df_daily)
    max_date = data["Date"].max()
    items = data["Item"].unique()
    future_dates = pd.date_range(start=max_date + pd.Timedelta(days=1), periods=weeks * 7, freq="D")

    if len(future_dates) == 0:
        return pd.DataFrame()

    _ensure_models_loaded(model_type)
    cache = _models_cache[model_type]

    last_known = data.sort_values("Date").groupby("Item").last()["Quantity_Sold"].reset_index()
    last_map = dict(zip(last_known["Item"], last_known["Quantity_Sold"]))

    all_predictions: list[pd.DataFrame] = []
    current = data.copy()
    days = list(future_dates)
    is_iterative = model_type in ("sarimax", "prophet")

    if is_iterative:
        for next_date in days:
            next_df = pd.DataFrame({"Date": [next_date] * len(items), "Item": np.array(items)})
            next_df["Quantity_Sold"] = next_df["Item"].map(last_map).fillna(1)
            temp = pd.concat([current, next_df], ignore_index=True)
            future_row = temp[temp["Date"] == next_date].copy()
            pred_result = _predict_dispatch(model_type, future_row, cache["item_models"], cache["global_model"], cache["dow_factors"])
            pred_result["Date"] = next_date
            pred_result["Item"] = future_row["Item"].values
            all_predictions.append(pred_result[["Date", "Item", "Predicted"]])
            next_df["Quantity_Sold"] = pd.Series(pred_result["Predicted"].values, index=next_df.index)
            current = pd.concat([current, next_df], ignore_index=True)
    else:
        for batch_start in range(0, len(days), 7):
            batch_dates = days[batch_start:batch_start + 7]
            batch_rows = []
            for d in batch_dates:
                df = pd.DataFrame({"Date": [d] * len(items), "Item": np.array(items)})
                df["Quantity_Sold"] = df["Item"].map(last_map).fillna(1)
                batch_rows.append(df)
            batch_df = pd.concat(batch_rows, ignore_index=True)
            temp = pd.concat([current, batch_df], ignore_index=True)
            feat = create_features(temp)
            batch_features = feat[feat["Date"].isin(batch_dates)]
            pred_result = _predict_dispatch(model_type, batch_features, cache["item_models"], cache["global_model"], cache["dow_factors"])
            pred_df = batch_features.copy()
            pred_df["Predicted"] = pred_result["Predicted"].values
            all_predictions.append(pred_df)
            for _, row in pred_df.iterrows():
                batch_df.loc[(batch_df["Date"] == row["Date"]) & (batch_df["Item"] == row["Item"]), "Quantity_Sold"] = row["Predicted"]
            current = pd.concat([current, batch_df], ignore_index=True)

    result = pd.concat(all_predictions, ignore_index=True)
    print(f"[{model_type}] Daily recursive forecast for {weeks * 7} days")
    return result
