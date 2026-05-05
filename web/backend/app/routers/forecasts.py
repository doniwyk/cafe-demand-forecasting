from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from fastapi import APIRouter, Depends, Query, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.engine import async_session
from app.models.forecast import (
    ForecastPage,
    ForecastSummary,
    PredictRequest,
    PredictResponse,
    RetrainRequest,
    RetrainResponse,
)
from app.services import forecast_service

router = APIRouter(prefix="/api/forecasts", tags=["forecasts"])


async def get_session():
    async with async_session() as session:
        yield session


@router.get("", response_model=ForecastPage)
async def get_forecasts(
    session: AsyncSession = Depends(get_session),
    item: str | None = Query(None),
    start_date: str | None = Query(None),
    end_date: str | None = Query(None),
    model_type: str | None = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(100, ge=1, le=1000),
):
    return await forecast_service.get_forecasts(
        session, item, start_date, end_date, page, page_size, model_type
    )


@router.get("/summary", response_model=ForecastSummary)
async def get_forecast_summary(
    session: AsyncSession = Depends(get_session),
    model_type: str | None = Query(None),
):
    return await forecast_service.get_forecast_summary(session, model_type)


@router.post("/predict", response_model=PredictResponse)
async def predict_items(request: PredictRequest):
    return await forecast_service.predict_items(request)


_retrain_status: dict[str, dict] = {
    "xgboost": {"status": "idle", "message": ""},
    "random_forest": {"status": "idle", "message": ""},
    "sarimax": {"status": "idle", "message": ""},
    "prophet": {"status": "idle", "message": ""},
}

_retrain_logs: dict[str, list[str]] = {
    "xgboost": [],
    "random_forest": [],
    "sarimax": [],
    "prophet": [],
}

_MAX_LOG_LINES = 5000
_STATUS_TAIL_DEFAULT = 200

_cancelled: dict[str, bool] = {
    "xgboost": False,
    "random_forest": False,
    "sarimax": False,
    "prophet": False,
}

_executor = ThreadPoolExecutor(max_workers=4)


def _is_cancelled(model_type: str) -> bool:
    return _cancelled.get(model_type, False)


def _append_log(model_type: str, message: str) -> None:
    if not message:
        return
    logs = _retrain_logs.setdefault(model_type, [])
    logs.append(message)
    overflow = len(logs) - _MAX_LOG_LINES
    if overflow > 0:
        del logs[:overflow]


@router.post("/retrain", response_model=RetrainResponse)
async def retrain_models(
    background_tasks: BackgroundTasks, body: RetrainRequest = RetrainRequest()
):
    model_type = body.model_type
    max_items = body.max_items
    if model_type not in _retrain_status:
        return RetrainResponse(
            status="error",
            message=f"Unknown model type: {model_type}",
        )
    if _retrain_status[model_type]["status"] == "training":
        return RetrainResponse(
            status="already_training",
            message=f"{model_type} is already training",
        )

    _retrain_logs[model_type] = []

    async def _run_retrain():
        model = model_type
        _retrain_status[model] = {
            "status": "training",
            "message": f"Retraining {model}...",
        }
        _append_log(model, f"Starting {model} training...")
        _cancelled[model] = False

        def run_training():
            import pandas as pd
            import warnings

            warnings.filterwarnings("ignore", category=FutureWarning)
            warnings.filterwarnings("ignore", category=UserWarning, module="tsa_model")
            warnings.filterwarnings("ignore", category=UserWarning, module="sarimax")
            warnings.filterwarnings("ignore", message=".*frequency.*")
            warnings.filterwarnings("ignore", message=".*Too few observations.*")
            warnings.filterwarnings("ignore", message=".*ConvergenceWarning.*")
            import logging

            logging.getLogger("statsmodels").setLevel(logging.ERROR)
            logging.getLogger("prophet").setLevel(logging.ERROR)
            logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
            import asyncio
            import io
            import sys

            class LogCapture(io.IOBase):
                def __init__(self, logs_list):
                    self.logs = logs_list
                    self.buffer = io.StringIO()

                def write(self, text):
                    for line in text.splitlines():
                        clean = line.strip()
                        if clean:
                            _append_log(model, clean)
                    self.buffer.write(text)

                def flush(self):
                    self.buffer.flush()

                @property
                def getvalue(self):
                    return self.buffer.getvalue()

            log_capture = LogCapture(_retrain_logs[model])
            original_stdout = sys.stdout
            sys.stdout = log_capture

            try:
                from sqlalchemy import text
                from app.db.engine import sync_session
                from app.ml.engine import (
                    run_train_and_evaluate,
                    generate_forecast,
                    _METADATA_FILE,
                    ML_MODELS_DIR,
                )

                def _as_float(value):
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

                def _as_int(value):
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

                if _is_cancelled(model):
                    return {"status": "cancelled"}

                session = sync_session()

                query = text(
                    "SELECT dis.date, i.name as item, dis.quantity_sold FROM daily_item_sales dis JOIN items i ON dis.item_id = i.id"
                )
                result = session.execute(query)
                rows = result.fetchall()
                df = pd.DataFrame(
                    [tuple(row) for row in rows],
                    columns=["Date", "Item", "Quantity_Sold"],
                )
                df["Date"] = pd.to_datetime(df["Date"])

                _append_log(model, "Loading data from DB...")

                if max_items is not None and max_items > 0:
                    unique_items = sorted(df["Item"].dropna().unique().tolist())
                    selected_items = unique_items[:max_items]
                    df = df[df["Item"].isin(selected_items)].copy()
                    _append_log(
                        model,
                        f"Test mode: training limited to {len(selected_items)} items (max_items={max_items})",
                    )

                if _is_cancelled(model):
                    return {"status": "cancelled"}

                _append_log(model, f"Running {model} training...")
                analysis = run_train_and_evaluate(df, model_type=model)
                _append_log(
                    model,
                    f"Training done. Metrics: {analysis.get('global_metrics', {})}"
                )

                if _is_cancelled(model):
                    return {"status": "cancelled"}

                _append_log(model, "Generating forecasts...")
                forecast_result = generate_forecast(df, weeks=12, model_type=model)
                _append_log(
                    model,
                    f"Generated {len(forecast_result)} forecast records"
                )

                if _is_cancelled(model):
                    return {"status": "cancelled"}

                from datetime import datetime as _dt
                import json as _json
                from sqlalchemy import update as sa_update
                from app.db.models import (
                    ModelRun,
                    ModelRunClassMetric,
                    ModelRunTopItem,
                    Forecast,
                )

                _append_log(model, "Saving model run and forecasts to DB...")

                session.execute(
                    sa_update(ModelRun)
                    .where(ModelRun.model_type == model)
                    .values(is_active=False)
                )

                meta_path = ML_MODELS_DIR / _METADATA_FILE.get(model, "model_metadata.json")
                meta = {}
                if meta_path.exists():
                    with open(meta_path) as f:
                        meta = _json.load(f)

                gm = analysis.get("global_metrics", {})
                run = ModelRun(
                    model_type=model,
                    trained_at=_dt.now(),
                    n_item_models=_as_int(meta.get("n_item_models")),
                    n_records=_as_int(meta.get("n_records")),
                    date_range_start=pd.to_datetime(meta["date_range"][0]).date()
                    if meta.get("date_range")
                    else None,
                    date_range_end=pd.to_datetime(meta["date_range"][1]).date()
                    if meta.get("date_range")
                    else None,
                    r2=_as_float(gm.get("r2")),
                    wmape=_as_float(gm.get("wmape")),
                    mae=_as_float(gm.get("mae")),
                    volume_accuracy=_as_float(gm.get("median_period_accuracy")),
                    median_period_accuracy=_as_float(gm.get("median_period_accuracy")),
                    periods_within_20pct=_as_float(gm.get("periods_within_20pct")),
                    periods_within_50pct=_as_float(gm.get("periods_within_50pct")),
                    features=_json.dumps(meta.get("features", [])),
                    items_with_models=_json.dumps(meta.get("items_with_models", [])),
                    params=_json.dumps({"max_items": max_items}) if max_items else None,
                    is_active=True,
                )
                session.add(run)
                session.flush()

                for cls, cm in analysis.get("class_metrics", {}).items():
                    session.add(
                        ModelRunClassMetric(
                            model_run_id=run.id,
                            abc_class=cls,
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

                item_rows = session.execute(text("SELECT id, name FROM items"))
                item_name_to_id = {row[1]: row[0] for row in item_rows.fetchall()}

                saved_forecasts = 0
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
                    saved_forecasts += 1

                session.commit()
                _append_log(
                    model,
                    f"Model run saved to DB (id={run.id}, forecasts={saved_forecasts})",
                )

                return {
                    "status": "success",
                    "model_run_id": run.id,
                    "saved_forecasts": saved_forecasts,
                    "global_metrics": analysis.get("global_metrics", {}),
                    "class_metrics": analysis.get("class_metrics", {}),
                }
            except Exception as e:
                return {"status": "error", "error": str(e)}
            finally:
                sys.stdout = original_stdout
                try:
                    session.close()
                except:
                    pass

        def log_output(result):
            if result.get("status") == "cancelled":
                _append_log(model, "Training was cancelled")
                _retrain_status[model] = {
                    "status": "idle",
                    "message": "Cancelled by user",
                }
            elif result.get("status") == "success":
                _retrain_status[model] = {
                    "status": "success",
                    "message": f"{model} retraining completed",
                }
            else:
                _append_log(
                    model,
                    f"Error: {result.get('error', 'Unknown error')}"
                )
                _retrain_status[model] = {
                    "status": "error",
                    "message": result.get("error", "Unknown error"),
                }

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(_executor, lambda: log_output(run_training()))

    background_tasks.add_task(_run_retrain)
    _retrain_status[model_type] = {
        "status": "training",
        "message": f"{model_type} retraining started",
    }
    return RetrainResponse(
        status="started",
        message=f"{model_type} retraining has been started in the background",
    )


@router.get("/retrain/status")
def get_retrain_status(
    model_type: str | None = Query(None),
    tail: int = Query(_STATUS_TAIL_DEFAULT, ge=0, le=_MAX_LOG_LINES),
):
    model_types = [model_type] if model_type else list(_retrain_status.keys())
    payload: dict[str, dict] = {}
    for mt in model_types:
        status = _retrain_status.get(mt)
        if status is None:
            continue
        logs = _retrain_logs.get(mt, [])
        tail_logs = logs[-tail:] if tail > 0 else []
        payload[mt] = {
            **status,
            "logs": tail_logs,
            "log_count": len(logs),
        }
    return payload


@router.post("/retrain/cancel")
def cancel_retrain(body: RetrainRequest = RetrainRequest()):
    model_type = body.model_type
    if model_type in _retrain_status:
        _cancelled[model_type] = True
        _retrain_status[model_type] = {"status": "idle", "message": "Cancelled by user"}
        _append_log(model_type, f"Cancelling {model_type} training...")
        return {"status": "cancelled", "model_type": model_type}
    return {"status": "error", "message": f"Unknown model type: {model_type}"}


@router.post("/cleanup")
async def cleanup_stale_data(session: AsyncSession = Depends(get_session)):
    from app.db.models import (
        ModelRun,
        ModelRunClassMetric,
        ModelRunTopItem,
        Forecast,
    )
    from sqlalchemy import delete, select, func

    inactive_runs = (
        (await session.execute(select(ModelRun.id).where(ModelRun.is_active == False)))
        .scalars()
        .all()
    )

    if not inactive_runs:
        return {"deleted_runs": 0, "deleted_forecasts": 0}

    run_ids = inactive_runs
    del_forecasts = (
        await session.execute(
            delete(Forecast).where(Forecast.model_run_id.in_(run_ids))
        )
    ).rowcount
    del_class = (
        await session.execute(
            delete(ModelRunClassMetric).where(
                ModelRunClassMetric.model_run_id.in_(run_ids)
            )
        )
    ).rowcount
    del_top = (
        await session.execute(
            delete(ModelRunTopItem).where(ModelRunTopItem.model_run_id.in_(run_ids))
        )
    ).rowcount
    del_runs = (
        await session.execute(delete(ModelRun).where(ModelRun.id.in_(run_ids)))
    ).rowcount

    await session.commit()
    return {
        "deleted_runs": del_runs,
        "deleted_class_metrics": del_class,
        "deleted_top_items": del_top,
        "deleted_forecasts": del_forecasts,
    }
