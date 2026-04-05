from __future__ import annotations

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


@router.post("/retrain", response_model=RetrainResponse)
async def retrain_models(
    background_tasks: BackgroundTasks, body: RetrainRequest = RetrainRequest()
):
    model_type = body.model_type
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

    def log(msg: str):
        _retrain_logs[model_type].append(msg)

    async def _run_retrain():
        _retrain_status[model_type] = {
            "status": "training",
            "message": f"Retraining {model_type}...",
        }
        try:
            import io
            import sys
            from contextlib import redirect_stdout, redirect_stderr

            log_output = io.StringIO()
            async with async_session() as session:
                with redirect_stdout(log_output), redirect_stderr(log_output):
                    result = await forecast_service.retrain(
                        session, model_type=model_type
                    )
            log_output.seek(0)
            for line in log_output.readlines():
                log(line.strip())
            _retrain_status[model_type] = {
                "status": "success",
                "message": f"{model_type} retraining completed",
                "result": result,
            }
        except Exception as e:
            log(f"Error: {str(e)}")
            _retrain_status[model_type] = {"status": "error", "message": str(e)}

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
def get_retrain_status():
    return {
        mt: {**status, "logs": _retrain_logs.get(mt, [])}
        for mt, status in _retrain_status.items()
    }


@router.post("/retrain/cancel")
def cancel_retrain(body: RetrainRequest = RetrainRequest()):
    model_type = body.model_type
    if model_type in _retrain_status:
        _retrain_status[model_type] = {"status": "idle", "message": "Cancelled by user"}
        _retrain_logs[model_type] = []
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
