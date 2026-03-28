from pydantic import BaseModel


class ForecastRecord(BaseModel):
    date: str
    item: str
    quantity_sold: float


class ForecastPage(BaseModel):
    data: list[ForecastRecord]
    total: int
    page: int
    page_size: int


class ModelMetrics(BaseModel):
    r2: float
    wmape: float
    mae: float
    volume_accuracy: float


class ClassMetrics(BaseModel):
    n_items: int
    wmape: float
    volume_accuracy: float


class TopItem(BaseModel):
    item: str
    quantity_sold: float
    predicted: float
    accuracy_pct: float


class ForecastSummary(BaseModel):
    global_metrics: ModelMetrics
    class_metrics: dict[str, ClassMetrics]
    top_items: list[TopItem]


class PredictRequest(BaseModel):
    items: list[str]
    weeks: int = 12


class PredictResponse(BaseModel):
    data: list[ForecastRecord]
    total: int


class RetrainResponse(BaseModel):
    status: str
    message: str
