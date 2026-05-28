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
from src.models.features import create_features
from src.evaluation.metrics import generate_abc_analysis


VALID_MODEL_TYPES = {"xgboost", "random_forest", "sarimax", "prophet"}

_METADATA_FILE = {
    "xgboost": "model_metadata.json",
    "random_forest": "model_metadata_rf.json",
    "sarimax": "model_metadata_sarimax.json",
    "prophet": "model_metadata_prophet.json",
}


def _try_import_rf():
    try:
        from src.models.forecaster_rf import (
            train_and_predict_rf,
            train_models_rf,
            load_models_rf,
            predict_rf,
        )
        return (train_and_predict_rf, train_models_rf, load_models_rf, predict_rf)
    except ImportError:
        return (None, None, None, None)


def _try_import_sarimax():
    try:
        from src.models.forecaster_sarimax import (
            train_and_predict_sarimax,
            train_models_sarimax,
            load_models_sarimax,
            predict_sarimax,
            generate_future_weekly as _gfw,
        )
        return (train_and_predict_sarimax, train_models_sarimax, load_models_sarimax, predict_sarimax, _gfw)
    except ImportError:
        return (None, None, None, None, None)


def _try_import_prophet():
    try:
        from src.models.forecaster_prophet import (
            train_and_predict_prophet,
            train_models_prophet,
            load_models_prophet,
            predict_prophet,
            generate_future_weekly as _gfw,
        )
        return (train_and_predict_prophet, train_models_prophet, load_models_prophet, predict_prophet, _gfw)
    except ImportError:
        return (None, None, None, None, None)

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
        _, train_models_rf, load_models_rf, _ = _try_import_rf()
        if load_models_rf is None:
            raise ImportError("forecaster_rf module missing required functions")
        im, gm, dow = load_models_rf(ML_MODELS_DIR)
    elif model_type == "sarimax":
        _, train_models_sarimax, load_models_sarimax, _, _ = _try_import_sarimax()
        if load_models_sarimax is None:
            raise ImportError("forecaster_sarimax module missing required functions")
        im, gm, dow = load_models_sarimax(ML_MODELS_DIR)
    elif model_type == "prophet":
        _, train_models_prophet, load_models_prophet, _, _ = _try_import_prophet()
        if load_models_prophet is None:
            raise ImportError("forecaster_prophet module missing required functions")
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
        return predict(df, item_models=item_models, global_model=global_model, dow_factor_dict=dow_factors)
    elif model_type == "random_forest":
        _, _, _, predict_rf = _try_import_rf()
        if predict_rf is None:
            raise ImportError("forecaster_rf module missing required functions")
        return predict_rf(df, item_models=item_models, global_model=global_model, dow_factor_dict=dow_factors)
    elif model_type == "sarimax":
        _, _, _, predict_sarimax, _ = _try_import_sarimax()
        if predict_sarimax is None:
            raise ImportError("forecaster_sarimax module missing required functions")
        return predict_sarimax(df, item_models=item_models, global_model=global_model, dow_factor_dict=dow_factors)
    elif model_type == "prophet":
        _, _, _, predict_prophet, _ = _try_import_prophet()
        if predict_prophet is None:
            raise ImportError("forecaster_prophet module missing required functions")
        return predict_prophet(df, item_models=item_models, global_model=global_model, dow_factor_dict=dow_factors)


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

    if model_type == "xgboost":
        df_feat = create_features(df_weekly)
        train_models(df_feat, ML_MODELS_DIR)
        test_pred = train_and_predict(df_feat)
    elif model_type == "random_forest":
        train_and_predict_rf, train_models_rf, _, _ = _try_import_rf()
        if train_models_rf is None:
            raise ImportError("forecaster_rf module missing required functions")
        df_feat = create_features(df_weekly)
        train_models_rf(df_feat, ML_MODELS_DIR)
        test_pred = train_and_predict_rf(df_feat)
    elif model_type == "sarimax":
        train_and_predict_sarimax, train_models_sarimax, _, _, _ = _try_import_sarimax()
        if train_models_sarimax is None:
            raise ImportError("forecaster_sarimax module missing required functions")
        train_models_sarimax(df_weekly, ML_MODELS_DIR)
        test_pred = train_and_predict_sarimax(df_weekly)
    elif model_type == "prophet":
        train_and_predict_prophet, train_models_prophet, _, _, _ = _try_import_prophet()
        if train_models_prophet is None:
            raise ImportError("forecaster_prophet module missing required functions")
        train_models_prophet(df_weekly, ML_MODELS_DIR)
        test_pred = train_and_predict_prophet(df_weekly)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    analysis = generate_abc_analysis(test_pred)

    _models_cache[model_type]["loaded"] = False
    return analysis


def run_evaluate(df_daily: pd.DataFrame, model_type: str = "xgboost"):
    df_weekly = _to_weekly(df_daily)

    if model_type == "xgboost":
        df_feat = create_features(df_weekly)
        test_pred = train_and_predict(df_feat)
    elif model_type == "random_forest":
        train_and_predict_rf, _, _, _ = _try_import_rf()
        if train_and_predict_rf is None:
            raise ImportError("forecaster_rf module missing required functions")
        df_feat = create_features(df_weekly)
        test_pred = train_and_predict_rf(df_feat)
    elif model_type == "sarimax":
        train_and_predict_sarimax, _, _, _, _ = _try_import_sarimax()
        if train_and_predict_sarimax is None:
            raise ImportError("forecaster_sarimax module missing required functions")
        test_pred = train_and_predict_sarimax(df_weekly)
    elif model_type == "prophet":
        train_and_predict_prophet, _, _, _, _ = _try_import_prophet()
        if train_and_predict_prophet is None:
            raise ImportError("forecaster_prophet module missing required functions")
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
    import numpy as np
    df_weekly = _to_weekly(df_daily)

    if model_type in ("xgboost", "random_forest"):
        max_date = df_weekly["Date"].max()
        items = df_weekly["Item"].unique()
        future_dates = pd.date_range(
            start=max_date + pd.Timedelta(weeks=1), periods=weeks, freq="W-MON"
        )

        if len(future_dates) == 0:
            return pd.DataFrame()

        _ensure_models_loaded(model_type)
        cache = _models_cache[model_type]

        last_known = (
            df_weekly.sort_values("Date")
            .groupby("Item")
            .last()["Quantity_Sold"]
            .reset_index()
        )
        last_map = dict(zip(last_known["Item"], last_known["Quantity_Sold"]))

        all_predictions = []
        current = df_weekly.copy()

        for next_date in future_dates:
            next_df = pd.DataFrame(
                {"Date": [next_date] * len(items), "Item": np.array(items)}
            )
            next_df["Quantity_Sold"] = next_df["Item"].map(last_map).fillna(1)

            temp = pd.concat([current, next_df], ignore_index=True)
            feat = create_features(temp)
            future_rows = feat[feat["Date"] == next_date]

            pred_result = _predict_dispatch(
                model_type,
                future_rows,
                cache["item_models"],
                cache["global_model"],
                cache["dow_factors"],
            )

            pred_df = future_rows.copy()
            pred_df["Predicted"] = pred_result["Predicted"].values
            all_predictions.append(pred_df)

            next_df["Quantity_Sold"] = pd.Series(
                pred_result["Predicted"].values, index=next_df.index
            )
            current = pd.concat([current, next_df], ignore_index=True)

        print(f"[{model_type}] Recursive forecast for {weeks} weeks")
        return pd.concat(all_predictions, ignore_index=True)

    elif model_type == "prophet":
        df_feat = create_features(df_weekly)
        _, _, _, _, generate_future_weekly_prophet = _try_import_prophet()
        if generate_future_weekly_prophet is None:
            raise ImportError("forecaster_prophet module missing required functions")
        future_features = generate_future_weekly_prophet(df_feat, future_weeks=weeks)
        print(f"[{model_type}] Forecast inference started for {weeks} weeks")
        return run_predict(future_features, model_type)

    elif model_type == "sarimax":
        df_feat = create_features(df_weekly)
        _, _, _, _, generate_future_weekly_sarimax = _try_import_sarimax()
        if generate_future_weekly_sarimax is None:
            raise ImportError("forecaster_sarimax module missing required functions")
        future_features = generate_future_weekly_sarimax(df_feat, future_weeks=weeks)
        print(f"[{model_type}] Forecast inference started for {weeks} weeks")
        return run_predict(future_features, model_type)

    return pd.DataFrame()
