from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.deps import get_session
from app.models.forecast import (
    ForecastPage,
    ForecastSummary,
    PredictRequest,
    PredictResponse,
    RetrainRequest,
    RetrainResponse,
)
from app.services import forecast_service
from app.services.retrain_service import RetrainState, start_retrain, cancel_retrain

router = APIRouter(prefix="/api/forecasts", tags=["forecasts"])


@router.get("", response_model=ForecastPage)
async def get_forecasts(
    session: AsyncSession = Depends(get_session),
    item: str | None = Query(None),
    start_date: str | None = Query(None),
    end_date: str | None = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(10000, ge=1, le=100000),
):
    return await forecast_service.get_forecasts(session, item, start_date, end_date, page, page_size)


@router.get("/summary", response_model=ForecastSummary)
async def get_forecast_summary(
    session: AsyncSession = Depends(get_session),
):
    return await forecast_service.get_forecast_summary(session)


@router.post("/predict", response_model=PredictResponse)
async def predict_items(request: PredictRequest):
    return await forecast_service.predict_items(request)


@router.post("/retrain", response_model=RetrainResponse)
async def retrain_models(
    body: RetrainRequest = RetrainRequest(),
):
    result = await start_retrain(
        max_items=body.max_items,
        include_new_products=body.include_new_products,
        end_date=body.end_date,
    )
    return RetrainResponse(status=result["status"], message=result["message"])


@router.get("/retrain/status")
def get_retrain_status(
    tail: int = Query(200, ge=0, le=5000),
):
    status = RetrainState.get_status()
    return {
        "xgboost": {
            "status": status["status"],
            "message": status.get("message", ""),
            "logs": RetrainState.get_logs(tail),
            "log_count": RetrainState.get_log_count(),
        }
    }


@router.post("/retrain/cancel")
def cancel_retrain_endpoint():
    return cancel_retrain()
