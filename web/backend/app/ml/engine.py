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
    test_pred = train_and_predict(df_feat)
    analysis = generate_abc_analysis(test_pred)

    _save_model_run_to_db(analysis)

    _models_cache["loaded"] = False
    return analysis


def _save_model_run_to_db(analysis: dict):
    try:
        from src.db import SessionLocal
        from src.db.models import (
            ModelRun,
            ModelRunClassMetric,
            ModelRunTopItem,
            Forecast,
            Item,
            DailyItemSale,
        )
        from sqlalchemy import text, update

        session = SessionLocal()
        try:
            session.execute(
                update(ModelRun)
                .where(ModelRun.model_type == "xgboost")
                .values(is_active=False)
            )

            meta_path = ML_MODELS_DIR / "model_metadata.json"
            meta = {}
            if meta_path.exists():
                with open(meta_path) as f:
                    meta = json.load(f)

            gm = analysis["global_metrics"]
            run = ModelRun(
                model_type="xgboost",
                trained_at=datetime.now(),
                n_item_models=meta.get("n_item_models"),
                n_records=meta.get("n_records"),
                date_range_start=pd.to_datetime(meta["date_range"][0]).date()
                if meta.get("date_range")
                else None,
                date_range_end=pd.to_datetime(meta["date_range"][1]).date()
                if meta.get("date_range")
                else None,
                r2=gm.get("r2"),
                wmape=gm.get("wmape"),
                mae=gm.get("mae"),
                volume_accuracy=gm.get("volume_accuracy"),
                features=json.dumps(meta.get("features", [])),
                items_with_models=json.dumps(meta.get("items_with_models", [])),
                is_active=True,
            )
            session.add(run)
            session.flush()

            for cls, cm in analysis.get("class_metrics", {}).items():
                session.add(
                    ModelRunClassMetric(
                        model_run_id=run.id,
                        abc_class=cls,
                        n_items=cm["n_items"],
                        wmape=cm["wmape"],
                        volume_accuracy=cm["volume_accuracy"],
                    )
                )

            for t in analysis.get("top_items", []):
                session.add(
                    ModelRunTopItem(
                        model_run_id=run.id,
                        item_name=t["Item"],
                        quantity_sold=t["Quantity_Sold"],
                        predicted=t["Predicted"],
                        accuracy_pct=t["accuracy_pct"],
                    )
                )

            df_feat = create_features(
                _to_weekly(
                    pd.read_csv(
                        ML_MODELS_DIR.parent.parent
                        / "data"
                        / "processed"
                        / "daily_item_sales.csv"
                    )
                )
            )
            future_features = generate_future_features(df_feat, future_weeks=12)
            forecast_result = predict(
                future_features,
                item_models=_models_cache.get("item_models", {}),
                global_model=_models_cache.get("global_model"),
                dow_factor_dict=_models_cache.get("dow_factors", {}),
            )

            item_name_to_id = {}
            item_rows = session.execute(text("SELECT id, name FROM items")).fetchall()
            for row in item_rows:
                item_name_to_id[row[1]] = row[0]

            for _, row in forecast_result.iterrows():
                item_name = str(row["Item"])
                item_id = item_name_to_id.get(item_name)
                if item_id is None:
                    continue
                session.add(
                    Forecast(
                        model_run_id=run.id,
                        item_id=item_id,
                        date=pd.to_datetime(row["Date"]).date(),
                        quantity_predicted=float(row["Predicted"]),
                    )
                )

            session.commit()
            print(f"Model run saved to DB (id={run.id})")
        except Exception as e:
            session.rollback()
            print(f"Failed to save model run to DB: {e}")
        finally:
            session.close()
    except ImportError:
        pass


def run_evaluate(df_daily: pd.DataFrame):
    df_weekly = _to_weekly(df_daily)
    df_feat = create_features(df_weekly)
    test_pred = train_and_predict(df_feat)
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
