from __future__ import annotations

import json
import pickle
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from app.ml.config import FEATURE_COLUMNS, MODELS_DIR, N_BACKTEST_WINDOWS, BACKTEST_WINDOW_DAYS

XGBOOST_PARAMS = {
    "objective": "count:poisson",
    "n_estimators": 200,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.5,
    "reg_lambda": 0.5,
    "random_state": 42,
    "verbosity": 0,
}


def _get_available_features(df: pd.DataFrame) -> list[str]:
    return [c for c in FEATURE_COLUMNS if c in df.columns]


def train_and_predict(
    df_features: pd.DataFrame,
) -> pd.DataFrame:
    features = _get_available_features(df_features)
    all_dates = sorted(df_features["Date"].unique())
    test_start = all_dates[-1] - pd.Timedelta(days=BACKTEST_WINDOW_DAYS * N_BACKTEST_WINDOWS)

    print(f"Running {N_BACKTEST_WINDOWS}-window expanding backtest...")
    predictions = []

    for w in range(N_BACKTEST_WINDOWS):
        ws = test_start + pd.Timedelta(days=BACKTEST_WINDOW_DAYS * w)
        we = ws + pd.Timedelta(days=BACKTEST_WINDOW_DAYS - 1)
        train = df_features[df_features["Date"] < ws].copy()
        test = df_features[(df_features["Date"] >= ws) & (df_features["Date"] <= we)].copy()

        if len(test) < 10:
            continue

        t0 = time.time()
        model = XGBRegressor(**XGBOOST_PARAMS)
        model.fit(train[features].fillna(0), train["Quantity_Sold"])
        if w == 0:
            print(f"  Window 1: {ws.date()} -> {we.date()} | train={len(train)}, test={len(test)}, {time.time()-t0:.1f}s")

        test["Raw_Pred"] = np.maximum(model.predict(test[features].fillna(0)), 0)
        test["Predicted"] = test["Raw_Pred"]
        predictions.append(test)

    return pd.concat(predictions).sort_values(["Item", "Date"])


def _compute_item_errors(test_pred: pd.DataFrame) -> dict[str, float]:
    errors = {}
    for item, grp in test_pred.groupby("Item"):
        y_true = grp["Quantity_Sold"].values
        y_pred = grp["Predicted"].values
        if len(y_true) >= 2:
            errors[item] = round(float(np.std(y_pred - y_true)), 3)
    return errors


def train_models(
    df_features: pd.DataFrame,
    output_dir: str | Path | None = None,
) -> XGBRegressor:
    output_dir = Path(output_dir) if output_dir else MODELS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    features = _get_available_features(df_features)

    print("Training XGBoost Poisson model...")
    t0 = time.time()
    model = XGBRegressor(**XGBOOST_PARAMS)
    model.fit(df_features[features].fillna(0), df_features["Quantity_Sold"])
    print(f"Model trained in {time.time() - t0:.1f}s")

    test_pred = train_and_predict(df_features)
    item_errors = _compute_item_errors(test_pred)

    with open(output_dir / "forecast_model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open(output_dir / "item_errors.json", "w") as f:
        json.dump(item_errors, f, indent=2)
    with open(output_dir / "model_metadata.json", "w") as f:
        json.dump({
            "trained_at": datetime.now().isoformat(),
            "features": features,
            "n_records": len(df_features),
            "n_items": len(item_errors),
            "date_range": [
                str(df_features["Date"].min()),
                str(df_features["Date"].max()),
            ],
        }, f, indent=2)

    print(f"Models saved to: {output_dir}")
    return model


def load_models(
    model_dir: str | Path | None = None,
) -> tuple[XGBRegressor, dict[str, float]]:
    model_dir = Path(model_dir) if model_dir else MODELS_DIR
    with open(model_dir / "forecast_model.pkl", "rb") as f:
        model = pickle.load(f)
    errors_path = model_dir / "item_errors.json"
    item_errors: dict[str, float] = {}
    if errors_path.exists():
        with open(errors_path) as f:
            item_errors = json.load(f)
    return model, item_errors


def predict(
    df_features: pd.DataFrame,
    model: XGBRegressor | None = None,
    model_dir: str | Path | None = None,
) -> pd.DataFrame:
    if model is None:
        model, _ = load_models(model_dir)

    features = _get_available_features(df_features)

    df = df_features.copy()
    df["Raw_Pred"] = np.maximum(model.predict(df[features].fillna(0)), 0)
    df["Predicted"] = df["Raw_Pred"]

    return df.sort_values(["Item", "Date"])
