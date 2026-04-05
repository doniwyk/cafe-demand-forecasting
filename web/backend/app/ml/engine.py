from __future__ import annotations

import json
from datetime import datetime
from typing import Optional

import pandas as pd

from app.config import ML_MODELS_DIR
from src.models.forecaster import (
    train_models,
    load_models,
    predict,
    generate_future_features,
    train_and_predict,
)
from src.models.forecaster_rf import (
    train_and_predict_rf,
    train_models_rf,
    load_models_rf,
    predict_rf,
)
from src.models.forecaster_sarimax import (
    train_and_predict_sarimax,
    train_models_sarimax,
    load_models_sarimax,
    predict_sarimax,
    generate_future_weekly as generate_future_weekly_sarimax,
)
from src.models.forecaster_prophet import (
    train_and_predict_prophet,
    train_models_prophet,
    load_models_prophet,
    predict_prophet,
)
from src.models.features import create_features
from src.evaluation.metrics import generate_abc_analysis


VALID_MODEL_TYPES = {"xgboost", "random_forest", "sarimax", "prophet"}

_METADATA_FILE = {
    "xgboost": "model_metadata.json",
    "random_forest": "model_metadata_rf.json",
    "sarimax": "model_metadata_sarimax.json",
    "prophet": "model_metadata_prophet.json",
}

_models_cache: dict[str, dict] = {
    mt: {
        "item_models": None,
        "global_model": None,
        "dow_factors": None,
        "loaded": False,
    }
    for mt in VALID_MODEL_TYPES
}


def _load_for_model(model_type: str):
    if model_type == "xgboost":
        im, gm, dow = load_models(ML_MODELS_DIR)
    elif model_type == "random_forest":
        im, gm, dow = load_models_rf(ML_MODELS_DIR)
    elif model_type == "sarimax":
        im, gm, dow = load_models_sarimax(ML_MODELS_DIR)
    elif model_type == "prophet":
        im, gm, dow = load_models_prophet(ML_MODELS_DIR)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    return im, gm, dow


def _ensure_models_loaded(model_type: str = "xgboost"):
    cache = _models_cache[model_type]
    if cache["loaded"]:
        return
    im, gm, dow = _load_for_model(model_type)
    cache["item_models"] = im
    cache["global_model"] = gm
    cache["dow_factors"] = dow
    cache["loaded"] = True


def _predict_dispatch(model_type: str, df, item_models, global_model, dow_factors):
    if model_type == "xgboost":
        return predict(
            df,
            item_models=item_models,
            global_model=global_model,
            dow_factor_dict=dow_factors,
        )
    elif model_type == "random_forest":
        return predict_rf(
            df,
            item_models=item_models,
            global_model=global_model,
            dow_factor_dict=dow_factors,
        )
    elif model_type == "sarimax":
        return predict_sarimax(
            df,
            item_models=item_models,
            global_model=global_model,
            dow_factor_dict=dow_factors,
        )
    elif model_type == "prophet":
        return predict_prophet(
            df,
            item_models=item_models,
            global_model=global_model,
            dow_factor_dict=dow_factors,
        )


def run_predict(df: pd.DataFrame, model_type: str = "xgboost") -> pd.DataFrame:
    _ensure_models_loaded(model_type)
    cache = _models_cache[model_type]
    return _predict_dispatch(
        model_type,
        df,
        cache["item_models"],
        cache["global_model"],
        cache["dow_factors"],
    )


def run_train_and_evaluate(df_daily: pd.DataFrame, model_type: str = "xgboost"):
    df_weekly = _to_weekly(df_daily)

    if model_type in ("xgboost", "random_forest"):
        df_feat = create_features(df_weekly)
        if model_type == "xgboost":
            train_models(df_feat, ML_MODELS_DIR)
            test_pred = train_and_predict(df_feat)
        else:
            train_models_rf(df_feat, ML_MODELS_DIR)
            test_pred = train_and_predict_rf(df_feat)
    elif model_type == "sarimax":
        print("[SARIMAX] Training and saving per-item models...")
        train_models_sarimax(df_weekly, ML_MODELS_DIR)
        print("[SARIMAX] Running backtest evaluation...")
        test_pred = train_and_predict_sarimax(df_weekly)
        print("[SARIMAX] Backtest evaluation complete")
    elif model_type == "prophet":
        train_models_prophet(df_weekly, ML_MODELS_DIR)
        test_pred = train_and_predict_prophet(df_weekly)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    analysis = generate_abc_analysis(test_pred)

    _models_cache[model_type]["loaded"] = False
    return analysis


def run_evaluate(df_daily: pd.DataFrame, model_type: str = "xgboost"):
    df_weekly = _to_weekly(df_daily)

    if model_type in ("xgboost", "random_forest"):
        df_feat = create_features(df_weekly)
        if model_type == "xgboost":
            test_pred = train_and_predict(df_feat)
        else:
            test_pred = train_and_predict_rf(df_feat)
    elif model_type == "sarimax":
        test_pred = train_and_predict_sarimax(df_weekly)
    elif model_type == "prophet":
        test_pred = train_and_predict_prophet(df_weekly)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

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


def get_model_metadata(model_type: str = "xgboost") -> dict | None:
    meta_path = ML_MODELS_DIR / _METADATA_FILE.get(model_type, "model_metadata.json")
    if not meta_path.exists():
        return None
    with open(meta_path) as f:
        return json.load(f)


def generate_forecast(
    df_daily: pd.DataFrame, weeks: int = 12, model_type: str = "xgboost"
) -> pd.DataFrame:
    df_weekly = _to_weekly(df_daily)

    if model_type in ("xgboost", "random_forest"):
        df_feat = create_features(df_weekly)
        future_features = generate_future_features(df_feat, future_weeks=weeks)
    else:
        from src.models.forecaster_sarimax import generate_future_weekly as _gen_fw

        future_features = _gen_fw(df_weekly, future_weeks=weeks)

    print(f"[{model_type}] Forecast inference started for {weeks} weeks")
    return run_predict(future_features, model_type)
