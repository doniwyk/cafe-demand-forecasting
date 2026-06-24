from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any

import pandas as pd
from sqlalchemy import text, update as sa_update

from app.db.models import ModelRun, ModelRunClassMetric, ModelRunTopItem
from app.ml.config import METADATA_FILE, MODEL_TYPE, DAILY_SALES_CSV
from app.ml.engine import run_train_and_evaluate, ML_MODELS_DIR
from app.services.forecast_service import invalidate_forecast_cache

logger = logging.getLogger(__name__)

_MAX_LOG_LINES = 5000
_TRAINING_DATA_CUTOFF = "2026-03-31"
_retrain_status: dict = {"status": "idle", "message": ""}
_retrain_logs: list[str] = []
_cancelled: bool = False
_executor = ThreadPoolExecutor(max_workers=4)


class RetrainState:
    @staticmethod
    def get_status() -> dict:
        return _retrain_status

    @staticmethod
    def get_logs(tail: int = 200) -> list[str]:
        return _retrain_logs[-tail:] if tail > 0 else []

    @staticmethod
    def get_log_count() -> int:
        return len(_retrain_logs)

    @staticmethod
    def is_cancelled() -> bool:
        return _cancelled

    @staticmethod
    def is_training() -> bool:
        return _retrain_status.get("status") == "training"


def _append_log(message: str) -> None:
    if not message:
        return
    _retrain_logs.append(message)
    overflow = len(_retrain_logs) - _MAX_LOG_LINES
    if overflow > 0:
        del _retrain_logs[:overflow]


class LogCapture(io.IOBase):
    def __init__(self):
        self.buffer = io.StringIO()

    def write(self, text):
        for line in text.splitlines():
            clean = line.strip()
            if clean:
                _append_log(clean)
        self.buffer.write(text)

    def flush(self):
        self.buffer.flush()

    @property
    def getvalue(self):
        return self.buffer.getvalue()


def _safe_cast(value: Any, target_type: type) -> Any | None:
    if value is None:
        return None
    if hasattr(value, "item"):
        value = value.item()
    try:
        return target_type(value)
    except (TypeError, ValueError):
        return None


def _load_training_data(max_items: int | None) -> pd.DataFrame:
    csv_path = str(DAILY_SALES_CSV)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Training CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    df = df.rename(columns={"Date_Only": "Date", "Quantity": "Quantity_Sold"})
    df["Date"] = pd.to_datetime(df["Date"])

    df = df[df["Date"] <= _TRAINING_DATA_CUTOFF].copy()
    _append_log(f"Loaded {len(df)} rows from CSV (cutoff {_TRAINING_DATA_CUTOFF})")

    if max_items is not None and max_items > 0:
        unique_items = sorted(df["Item"].dropna().unique().tolist())
        selected_items = unique_items[:max_items]
        df = df[df["Item"].isin(selected_items)].copy()
        _append_log(
            f"Test mode: training limited to {len(selected_items)} items (max_items={max_items})",
        )

    return df


def _save_run_to_db(session, analysis: dict, max_items: int | None = None) -> int:
    meta_path = ML_MODELS_DIR / METADATA_FILE
    meta = {}
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)

    session.execute(
        sa_update(ModelRun)
        .where(ModelRun.model_type == MODEL_TYPE)
        .values(is_active=False)
    )

    gm = analysis.get("global_metrics", {})
    run = ModelRun(
        model_type=MODEL_TYPE,
        trained_at=datetime.now(),
        n_item_models=None,
        n_records=_safe_cast(meta.get("n_records"), int),
        date_range_start=pd.to_datetime(meta["date_range"][0]).date()
        if meta.get("date_range") else None,
        date_range_end=pd.to_datetime(meta["date_range"][1]).date()
        if meta.get("date_range") else None,
        r2=_safe_cast(gm.get("r2"), float),
        wmape=_safe_cast(gm.get("wmape"), float),
        mae=_safe_cast(gm.get("mae"), float),
        rmse=_safe_cast(gm.get("rmse"), float),
        volume_accuracy=_safe_cast(gm.get("median_period_accuracy"), float),
        median_period_accuracy=_safe_cast(gm.get("median_period_accuracy"), float),
        periods_within_20pct=_safe_cast(gm.get("periods_within_20pct"), float),
        periods_within_50pct=_safe_cast(gm.get("periods_within_50pct"), float),
        features=json.dumps(meta.get("features", [])),
        items_with_models=json.dumps(meta.get("items_with_models", [])),
        params=json.dumps({"max_items": max_items}) if max_items else None,
        is_active=True,
    )
    session.add(run)
    session.flush()

    session.add_all([
        ModelRunClassMetric(
            model_run_id=run.id,
            abc_class=cls_name,
            n_items=_safe_cast(cm.get("n_items"), int) or 0,
            wmape=_safe_cast(cm.get("wmape"), float) or 0.0,
            r2=_safe_cast(cm.get("r2"), float) or 0.0,
            mae=_safe_cast(cm.get("mae"), float) or 0.0,
            rmse=_safe_cast(cm.get("rmse"), float) or 0.0,
            volume_accuracy=_safe_cast(cm.get("median_period_accuracy"), float) or 0.0,
            median_period_accuracy=_safe_cast(cm.get("median_period_accuracy"), float) or 0.0,
            periods_within_20pct=_safe_cast(cm.get("periods_within_20pct"), float) or 0.0,
            periods_within_50pct=_safe_cast(cm.get("periods_within_50pct"), float) or 0.0,
        )
        for cls_name, cm in analysis.get("class_metrics", {}).items()
    ])

    session.add_all([
        ModelRunTopItem(
            model_run_id=run.id,
            item_name=t["Item"],
            quantity_sold=_safe_cast(t["Quantity_Sold"], float) or 0.0,
            predicted=_safe_cast(t["Predicted"], float) or 0.0,
            accuracy_pct=_safe_cast(t["accuracy_pct"], float) or 0.0,
        )
        for t in analysis.get("top_items", [])
    ])

    session.commit()
    return run.id


def _run_training_sync(max_items: int | None, end_date: str | None = None) -> dict[str, Any]:
    from app.db.engine import sync_session

    log_capture = LogCapture()
    original_stdout = sys.stdout
    sys.stdout = log_capture

    session = sync_session()
    try:
        if RetrainState.is_cancelled():
            return {"status": "cancelled"}

        _append_log("Loading training data from CSV...")
        df = _load_training_data(max_items)

        if RetrainState.is_cancelled():
            return {"status": "cancelled"}

        _append_log("Running XGBoost training...")
        analysis = run_train_and_evaluate(df)
        _append_log(f"Training done. Metrics: {analysis.get('global_metrics', {})}")

        if RetrainState.is_cancelled():
            return {"status": "cancelled"}

        _append_log("Saving model run to DB...")
        run_id = _save_run_to_db(session, analysis, max_items)
        invalidate_forecast_cache()
        _append_log(f"Model run saved to DB (id={run_id})")

        return {
            "status": "success",
            "model_run_id": run_id,
            "global_metrics": analysis.get("global_metrics", {}),
            "class_metrics": analysis.get("class_metrics", {}),
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}
    finally:
        sys.stdout = original_stdout
        session.close()


async def start_retrain(
    max_items: int | None = None,
    include_new_products: bool = False,
    end_date: str | None = None,
) -> dict:
    global _retrain_status, _cancelled, _retrain_logs

    if _retrain_status["status"] == "training":
        return {"status": "already_training", "message": "Training already in progress"}

    _retrain_logs = []

    if include_new_products:
        try:
            from scripts.sync_hus_sales import sync_sales
            result = sync_sales(include_new=True)
            _append_log(f"[sync] {result['inserted']} rows synced from hus_db")
            if result.get("new_products"):
                _append_log(f"[sync] Added {result['new_products']} new products")
        except Exception as e:
            _append_log(f"[sync] Error: {e}")

    _retrain_status = {"status": "training", "message": "Retraining XGBoost..."}
    _append_log("Starting XGBoost training...")
    _cancelled = False

    def _on_complete(result: dict):
        global _retrain_status
        if result.get("status") == "cancelled":
            _append_log("Training was cancelled")
            _retrain_status = {"status": "idle", "message": "Cancelled by user"}
        elif result.get("status") == "success":
            _retrain_status = {"status": "success", "message": "XGBoost retraining completed"}
        else:
            _append_log(f"Error: {result.get('error', 'Unknown error')}")
            _retrain_status = {"status": "error", "message": result.get("error", "Unknown error")}

    loop = asyncio.get_event_loop()
    await loop.run_in_executor(
        _executor,
        lambda: _on_complete(_run_training_sync(max_items, end_date)),
    )

    return {"status": "started", "message": "XGBoost retraining has been started in the background"}


def cancel_retrain() -> dict:
    global _cancelled, _retrain_status
    _cancelled = True
    _retrain_status = {"status": "idle", "message": "Cancelled by user"}
    _append_log("Cancelling training...")
    return {"status": "cancelled"}
