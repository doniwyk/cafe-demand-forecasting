from __future__ import annotations

import asyncio
import io
import json
import logging
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any

import pandas as pd
from sqlalchemy import text

from app.core.constants import MODEL_TYPES
from app.ml.engine import (
    run_train_and_evaluate,
    _METADATA_FILE,
    ML_MODELS_DIR,
)
from app.services.forecast_service import invalidate_forecast_cache

logger = logging.getLogger(__name__)

_MAX_LOG_LINES = 5000

_retrain_status: dict[str, dict] = {mt: {"status": "idle", "message": ""} for mt in MODEL_TYPES}
_retrain_logs: dict[str, list[str]] = {mt: [] for mt in MODEL_TYPES}
_cancelled: dict[str, bool] = {mt: False for mt in MODEL_TYPES}
_executor = ThreadPoolExecutor(max_workers=4)


class RetrainState:
    @staticmethod
    def get_status(model_type: str) -> dict:
        return _retrain_status.get(model_type, {"status": "unknown", "message": ""})

    @staticmethod
    def get_logs(model_type: str, tail: int = 200) -> list[str]:
        logs = _retrain_logs.get(model_type, [])
        return logs[-tail:] if tail > 0 else []

    @staticmethod
    def get_log_count(model_type: str) -> int:
        return len(_retrain_logs.get(model_type, []))

    @staticmethod
    def is_cancelled(model_type: str) -> bool:
        return _cancelled.get(model_type, False)

    @staticmethod
    def is_training(model_type: str) -> bool:
        return _retrain_status.get(model_type, {}).get("status") == "training"


def _append_log(model_type: str, message: str) -> None:
    if not message:
        return
    logs = _retrain_logs.setdefault(model_type, [])
    logs.append(message)
    overflow = len(logs) - _MAX_LOG_LINES
    if overflow > 0:
        del logs[:overflow]


class LogCapture(io.IOBase):
    def __init__(self, model_type: str):
        self.model_type = model_type
        self.buffer = io.StringIO()

    def write(self, text):
        for line in text.splitlines():
            clean = line.strip()
            if clean:
                _append_log(self.model_type, clean)
        self.buffer.write(text)

    def flush(self):
        self.buffer.flush()

    @property
    def getvalue(self):
        return self.buffer.getvalue()


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    if hasattr(value, "item"):
        try:
            value = value.item()
        except Exception:
            pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_int(value: Any) -> int | None:
    if value is None:
        return None
    if hasattr(value, "item"):
        try:
            value = value.item()
        except Exception:
            pass
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _sync_hus_data(model_type: str, include_new: bool = False) -> dict:
    from scripts.sync_hus_sales import sync_sales

    result = sync_sales(include_new=include_new)
    _append_log(model_type, f"[sync] {result['inserted']} rows synced from hus_db")
    if result.get("new_products"):
        _append_log(model_type, f"[sync] Added {result['new_products']} new products")
    if result.get("skipped_units"):
        _append_log(
            model_type,
            f"[sync] Skipped {result['skipped_units']} units ({result['skipped_products']} products)",
        )
    return result


def _suppress_noisy_logs():
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning, module="tsa_model")
    warnings.filterwarnings("ignore", category=UserWarning, module="sarimax")
    warnings.filterwarnings("ignore", message=".*frequency.*")
    warnings.filterwarnings("ignore", message=".*Too few observations.*")
    warnings.filterwarnings("ignore", message=".*ConvergenceWarning.*")
    logging.getLogger("statsmodels").setLevel(logging.ERROR)
    logging.getLogger("prophet").setLevel(logging.ERROR)
    logging.getLogger("cmdstanpy").setLevel(logging.ERROR)


def _run_training_sync(model_type: str, max_items: int | None, end_date: str | None = None) -> dict[str, Any]:
    from sqlalchemy import update as sa_update
    from app.db.engine import sync_session as get_sync_session
    from app.db.models import (
        ModelRun,
        ModelRunClassMetric,
        ModelRunTopItem,
    )

    _suppress_noisy_logs()
    log_capture = LogCapture(model_type)
    original_stdout = sys.stdout
    sys.stdout = log_capture

    session = get_sync_session()
    try:
        if RetrainState.is_cancelled(model_type):
            return {"status": "cancelled"}

        _append_log(model_type, "Loading data from DB...")
        base_sql = (
            "SELECT dis.date, i.name as item, dis.quantity_sold "
            "FROM daily_item_sales dis JOIN items i ON dis.item_id = i.id"
        )
        if end_date:
            base_sql += " WHERE dis.date <= :end_date"
            result = session.execute(text(base_sql), {"end_date": end_date})
            _append_log(model_type, f"Training data limited to dates <= {end_date}")
        else:
            result = session.execute(text(base_sql))
        rows = result.fetchall()
        df = pd.DataFrame(
            [tuple(row) for row in rows],
            columns=["Date", "Item", "Quantity_Sold"],
        )
        df["Date"] = pd.to_datetime(df["Date"])

        if max_items is not None and max_items > 0:
            unique_items = sorted(df["Item"].dropna().unique().tolist())
            selected_items = unique_items[:max_items]
            df = df[df["Item"].isin(selected_items)].copy()
            _append_log(
                model_type,
                f"Test mode: training limited to {len(selected_items)} items (max_items={max_items})",
            )

        if RetrainState.is_cancelled(model_type):
            return {"status": "cancelled"}

        _append_log(model_type, f"Running {model_type} training...")
        analysis = run_train_and_evaluate(df, model_type=model_type)
        _append_log(
            model_type,
            f"Training done. Metrics: {analysis.get('global_metrics', {})}",
        )

        if RetrainState.is_cancelled(model_type):
            return {"status": "cancelled"}

        _append_log(model_type, "Saving model run to DB...")

        session.execute(
            sa_update(ModelRun)
            .where(ModelRun.model_type == model_type)
            .values(is_active=False)
        )

        meta_path = ML_MODELS_DIR / _METADATA_FILE.get(model_type, "model_metadata.json")
        meta = {}
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)

        gm = analysis.get("global_metrics", {})
        run = ModelRun(
            model_type=model_type,
            trained_at=datetime.now(),
            n_item_models=_as_int(meta.get("n_item_models")),
            n_records=_as_int(meta.get("n_records")),
            date_range_start=pd.to_datetime(meta["date_range"][0]).date()
            if meta.get("date_range") else None,
            date_range_end=pd.to_datetime(meta["date_range"][1]).date()
            if meta.get("date_range") else None,
            r2=_as_float(gm.get("r2")),
            wmape=_as_float(gm.get("wmape")),
            mae=_as_float(gm.get("mae")),
            rmse=_as_float(gm.get("rmse")),
            volume_accuracy=_as_float(gm.get("median_period_accuracy")),
            median_period_accuracy=_as_float(gm.get("median_period_accuracy")),
            periods_within_20pct=_as_float(gm.get("periods_within_20pct")),
            periods_within_50pct=_as_float(gm.get("periods_within_50pct")),
            features=json.dumps(meta.get("features", [])),
            items_with_models=json.dumps(meta.get("items_with_models", [])),
            params=json.dumps({"max_items": max_items}) if max_items else None,
            is_active=True,
        )
        session.add(run)
        session.flush()

        for cls_name, cm in analysis.get("class_metrics", {}).items():
            session.add(
                ModelRunClassMetric(
                    model_run_id=run.id,
                    abc_class=cls_name,
                    n_items=_as_int(cm.get("n_items")) or 0,
                    wmape=_as_float(cm.get("wmape")) or 0.0,
                    volume_accuracy=_as_float(cm.get("median_period_acc")) or 0.0,
                    median_period_accuracy=_as_float(cm.get("median_period_acc")) or 0.0,
                )
            )

        for t in analysis.get("top_items", []):
            session.add(
                ModelRunTopItem(
                    model_run_id=run.id,
                    item_name=t["Item"],
                    quantity_sold=_as_float(t["Quantity_Sold"]) or 0.0,
                    predicted=_as_float(t["Predicted"]) or 0.0,
                    accuracy_pct=_as_float(t["accuracy_pct"]) or 0.0,
                )
            )

        session.commit()
        invalidate_forecast_cache(model_type)
        _append_log(model_type, f"Model run saved to DB (id={run.id})")

        return {
            "status": "success",
            "model_run_id": run.id,
            "global_metrics": analysis.get("global_metrics", {}),
            "class_metrics": analysis.get("class_metrics", {}),
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}
    finally:
        sys.stdout = original_stdout
        session.close()


async def start_retrain(
    model_type: str,
    max_items: int | None = None,
    sync_hus: bool = False,
    include_new_products: bool = False,
    end_date: str | None = None,
) -> dict:
    if model_type not in _retrain_status:
        return {"status": "error", "message": f"Unknown model type: {model_type}"}
    if _retrain_status[model_type]["status"] == "training":
        return {"status": "already_training", "message": f"{model_type} is already training"}

    _retrain_logs[model_type] = []

    if sync_hus:
        try:
            _sync_hus_data(model_type, include_new_products)
        except Exception as e:
            _append_log(model_type, f"[sync] Error: {e}")

    _retrain_status[model_type] = {"status": "training", "message": f"Retraining {model_type}..."}
    _append_log(model_type, f"Starting {model_type} training...")
    _cancelled[model_type] = False

    def _on_complete(result: dict):
        if result.get("status") == "cancelled":
            _append_log(model_type, "Training was cancelled")
            _retrain_status[model_type] = {"status": "idle", "message": "Cancelled by user"}
        elif result.get("status") == "success":
            _retrain_status[model_type] = {"status": "success", "message": f"{model_type} retraining completed"}
        else:
            _append_log(model_type, f"Error: {result.get('error', 'Unknown error')}")
            _retrain_status[model_type] = {"status": "error", "message": result.get("error", "Unknown error")}

    loop = asyncio.get_event_loop()
    await loop.run_in_executor(
        _executor,
        lambda: _on_complete(_run_training_sync(model_type, max_items, end_date)),
    )

    return {"status": "started", "message": f"{model_type} retraining has been started in the background"}


def cancel_retrain(model_type: str) -> dict:
    if model_type not in _retrain_status:
        return {"status": "error", "message": f"Unknown model type: {model_type}"}
    _cancelled[model_type] = True
    _retrain_status[model_type] = {"status": "idle", "message": "Cancelled by user"}
    _append_log(model_type, f"Cancelling {model_type} training...")
    return {"status": "cancelled", "model_type": model_type}
