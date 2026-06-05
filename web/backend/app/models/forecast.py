from pydantic import BaseModel


class ForecastRecord(BaseModel):
    date: str
    item: str
    quantity_sold: float
    actual: float = 0.0


class ForecastPage(BaseModel):
    data: list[ForecastRecord]
    total: int
    page: int
    page_size: int


class ModelMetrics(BaseModel):
    r2: float
    wmape: float
    mae: float
    rmse: float = 0.0
    median_period_accuracy: float = 0.0
    periods_within_20pct: float = 0.0
    periods_within_50pct: float = 0.0


class ClassMetrics(BaseModel):
    n_items: int
    wmape: float
    r2: float = 0.0
    mae: float = 0.0
    rmse: float = 0.0
    median_period_accuracy: float = 0.0
    periods_within_20pct: float = 0.0
    periods_within_50pct: float = 0.0


class TopItem(BaseModel):
    item: str
    quantity_sold: float
    predicted: float
    accuracy_pct: float


class ForecastSummary(BaseModel):
    global_metrics: ModelMetrics
    class_metrics: dict[str, ClassMetrics]
    top_items: list[TopItem]
    latest_training_date: str | None = None


class PredictRequest(BaseModel):
    items: list[str]
    weeks: int = 12
    model_type: str = "xgboost"


class RetrainRequest(BaseModel):
    model_type: str = "xgboost"
    max_items: int | None = None
    include_new_products: bool = False
    end_date: str | None = None


class PredictResponse(BaseModel):
    data: list[ForecastRecord]
    total: int


class RetrainResponse(BaseModel):
    status: str
    message: str
