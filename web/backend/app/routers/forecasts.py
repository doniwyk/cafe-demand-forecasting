from fastapi import APIRouter, Query, BackgroundTasks

from app.models.forecast import (
    ForecastPage,
    ForecastSummary,
    PredictRequest,
    PredictResponse,
    RetrainResponse,
)
from app.services import forecast_service

router = APIRouter(prefix="/api/forecasts", tags=["forecasts"])


@router.get("", response_model=ForecastPage)
def get_forecasts(
    item: str | None = Query(None),
    start_date: str | None = Query(None),
    end_date: str | None = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(100, ge=1, le=1000),
):
    return forecast_service.get_forecasts(item, start_date, end_date, page, page_size)


@router.get("/summary", response_model=ForecastSummary)
def get_forecast_summary():
    return forecast_service.get_forecast_summary()


@router.post("/predict", response_model=PredictResponse)
def predict_items(request: PredictRequest):
    return forecast_service.predict_items(request)


_retrain_status = {"status": "idle", "message": ""}


@router.post("/retrain", response_model=RetrainResponse)
def retrain_models(background_tasks: BackgroundTasks):
    if _retrain_status["status"] == "training":
        return RetrainResponse(
            status="already_training",
            message="Model retraining is already in progress",
        )

    def _run_retrain():
        global _retrain_status
        _retrain_status = {"status": "training", "message": "Retraining models..."}
        try:
            result = forecast_service.retrain()
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
