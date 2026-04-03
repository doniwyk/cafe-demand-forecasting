from __future__ import annotations

from fastapi import APIRouter, Depends, Query, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.engine import async_session
from app.models.forecast import (
    ForecastPage,
    ForecastSummary,
    PredictRequest,
    PredictResponse,
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
    page: int = Query(1, ge=1),
    page_size: int = Query(100, ge=1, le=1000),
):
    return await forecast_service.get_forecasts(
        session, item, start_date, end_date, page, page_size
    )


@router.get("/summary", response_model=ForecastSummary)
async def get_forecast_summary(session: AsyncSession = Depends(get_session)):
    return await forecast_service.get_forecast_summary(session)


@router.post("/predict", response_model=PredictResponse)
async def predict_items(request: PredictRequest):
    return await forecast_service.predict_items(request)


_retrain_status = {"status": "idle", "message": ""}


@router.post("/retrain", response_model=RetrainResponse)
async def retrain_models(background_tasks: BackgroundTasks):
    if _retrain_status["status"] == "training":
        return RetrainResponse(
            status="already_training",
            message="Model retraining is already in progress",
        )

    async def _run_retrain():
        global _retrain_status
        _retrain_status = {"status": "training", "message": "Retraining models..."}
        try:
            async with async_session() as session:
                result = await forecast_service.retrain(session)
            _retrain_status = {
                "status": "success",
                "message": "Model retraining completed",
                "result": result,
            }
        except Exception as e:
            _retrain_status = {"status": "error", "message": str(e)}

    background_tasks.add_task(_run_retrain)
    _retrain_status = {"status": "training", "message": "Model retraining started"}
    return RetrainResponse(
        status="started",
        message="Model retraining has been started in the background",
    )


@router.get("/retrain/status")
def get_retrain_status():
    return _retrain_status
