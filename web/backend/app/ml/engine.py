from __future__ import annotations

import json
from typing import Optional

import pandas as pd

from app.config import ML_MODELS_DIR
from src.models.forecaster import (
    train_models,
    load_models,
    predict,
    generate_future_features,
)
from src.models.features import create_features
from src.evaluation.metrics import generate_abc_analysis


_models_cache = {
    "item_models": None,
    "global_model": None,
    "dow_factors": None,
    "loaded": False,
}


def _ensure_models_loaded():
    if _models_cache["loaded"]:
        return
    item_models, global_model, dow_factors = load_models(ML_MODELS_DIR)
    _models_cache["item_models"] = item_models
    _models_cache["global_model"] = global_model
    _models_cache["dow_factors"] = dow_factors
    _models_cache["loaded"] = True


def run_predict(df_features: pd.DataFrame) -> pd.DataFrame:
    _ensure_models_loaded()
    return predict(
        df_features,
        item_models=_models_cache["item_models"],
        global_model=_models_cache["global_model"],
        dow_factor_dict=_models_cache["dow_factors"],
    )


def run_train_and_evaluate(df_daily: pd.DataFrame):
    df_weekly = _to_weekly(df_daily)
    df_feat = create_features(df_weekly)
    train_models(df_feat, ML_MODELS_DIR)
    test_pred = _train_and_predict_eval(df_feat)
    analysis = generate_abc_analysis(test_pred)
    _models_cache["loaded"] = False
    return analysis


def run_evaluate(df_daily: pd.DataFrame):
    df_weekly = _to_weekly(df_daily)
    df_feat = create_features(df_weekly)
    test_pred = _train_and_predict_eval(df_feat)
    return generate_abc_analysis(test_pred)


def _to_weekly(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df[~df["Item"].str.strip().str.lower().str.startswith("add")]
    return (
        df.set_index("Date")
        .groupby("Item")
        .resample("W-MON")["Quantity_Sold"]
        .sum()
        .reset_index()
    )


def _train_and_predict_eval(df_feat: pd.DataFrame) -> pd.DataFrame:
    from src.models.forecaster import train_and_predict

    return train_and_predict(df_feat)


def get_model_metadata() -> dict | None:
    meta_path = ML_MODELS_DIR / "model_metadata.json"
    if not meta_path.exists():
        return None
    with open(meta_path) as f:
        return json.load(f)


def generate_forecast(df_daily: pd.DataFrame, weeks: int = 12) -> pd.DataFrame:
    df_weekly = _to_weekly(df_daily)
    df_feat = create_features(df_weekly)
    future_features = generate_future_features(df_feat, future_weeks=weeks)
    return run_predict(future_features)
