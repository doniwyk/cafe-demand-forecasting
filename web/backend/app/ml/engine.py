from __future__ import annotations

import json

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from app.config import ML_MODELS_DIR
from app.ml.config import FEATURE_COLUMNS, METADATA_FILE
from app.ml.features import create_features
from app.ml.forecaster import Forecaster
from app.ml.metrics import generate_abc_analysis
from app.ml.models.xgboost import (
    load_models as _load_xgb,
    predict as _predict_xgb,
    train_models as _train_xgb,
    train_and_predict as _tap_xgb,
)

_model: XGBRegressor | None = None
_item_errors: dict[str, float] = {}


def _clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip()
    df["Date"] = pd.to_datetime(df["Date"])

    items = sorted(df["Item"].unique())
    dates = pd.date_range(df["Date"].min(), df["Date"].max())
    grid = pd.DataFrame(
        [(d, i) for d in dates for i in items], columns=["Date", "Item"]
    )
    full = grid.merge(df, on=["Date", "Item"], how="left")
    full["Quantity_Sold"] = full["Quantity_Sold"].fillna(0).astype(float)

    if "Category" in full.columns:
        full["Category"] = full.groupby("Item")["Category"].transform(
            lambda x: x.mode().iloc[0] if not x.mode().empty else (
                x.dropna().iloc[0] if x.dropna().shape[0] > 0 else "unknown"))
    return full


def _ensure_loaded():
    global _model, _item_errors
    if _model is None:
        _model, _item_errors = _load_xgb(ML_MODELS_DIR)


def run_predict(df: pd.DataFrame) -> pd.DataFrame:
    _ensure_loaded()
    return _predict_xgb(df, _model)


def run_train_and_evaluate(df_daily: pd.DataFrame):
    global _model, _item_errors
    processed = create_features(_clean_data(df_daily))
    _model = _train_xgb(processed, ML_MODELS_DIR)
    test_pred = _tap_xgb(processed)
    analysis = generate_abc_analysis(test_pred)
    _model = None
    _item_errors = {}
    return analysis


def get_model_metadata() -> dict | None:
    meta_path = ML_MODELS_DIR / METADATA_FILE
    if not meta_path.exists():
        return None
    with open(meta_path) as f:
        return json.load(f)


_SERVICE_Z = {"A": 1.645, "B": 1.282, "C": 1.036}


def _classify_abc(data: pd.DataFrame) -> dict[str, str]:
    item_vol = data.groupby("Item")["Quantity_Sold"].sum().sort_values(ascending=False)
    total = item_vol.sum()
    cum_pct = item_vol.cumsum() / total
    return {item: "A" if p <= 0.70 else "B" if p <= 0.90 else "C" for item, p in cum_pct.items()}


def _compute_buffer(item: str, abc: dict[str, str]) -> tuple[float, float, float]:
    z = _SERVICE_Z.get(abc.get(item, "C"), 1.036)
    error_std = _item_errors.get(item, 0.0)
    if error_std == 0.0 and _item_errors:
        error_std = sum(_item_errors.values()) / len(_item_errors)
    buffer = round(z * error_std, 2)
    return error_std, buffer, z


def generate_forecast(df_daily: pd.DataFrame, weeks: int = 12) -> pd.DataFrame:
    data = _clean_data(df_daily)
    max_date = data["Date"].max()
    future_dates = pd.date_range(
        start=max_date + pd.Timedelta(days=1), periods=weeks * 7, freq="D"
    )
    if len(future_dates) == 0:
        return pd.DataFrame()

    _ensure_loaded()
    forecaster = Forecaster(data)
    abc_class = _classify_abc(data)

    last_day = data[data["Date"] == max_date]
    cross = {
        "total_qty": float(last_day["Quantity_Sold"].sum()),
        "total_items": float((last_day["Quantity_Sold"] > 0).sum()),
        "total_qty_7d": float(last_day["Quantity_Sold"].sum()),
    }
    totals: list[float] = [cross["total_qty"]]

    predictions: list[dict] = []
    for next_date in future_dates:
        ts = pd.Timestamp(next_date)

        Xf = forecaster.build_features(ts, cross)
        feats = [c for c in FEATURE_COLUMNS if c in Xf.columns]
        preds = np.maximum(_model.predict(Xf[feats].fillna(0)), 0)

        for i, item in enumerate(forecaster.items):
            error_std, buffer, z = _compute_buffer(item, abc_class)
            predictions.append({
                "Date": next_date,
                "Item": item,
                "Predicted": round(float(preds[i]), 2),
                "Error_Std": error_std,
                "Buffer": buffer,
                "Supply": round(float(preds[i]) + buffer, 1),
            })

        cross = forecaster.update(preds)
        forecaster.update_cross_7d(cross, totals)

    result = pd.DataFrame(predictions)
    print(f"[xgboost] Recursive forecast for {weeks * 7} days")
    return result
