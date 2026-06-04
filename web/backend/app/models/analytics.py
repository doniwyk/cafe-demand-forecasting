from pydantic import BaseModel


class ABCItem(BaseModel):
    item: str
    vol: float
    cum: float
    pct: float
    class_label: str


class ABCAnalysisResponse(BaseModel):
    class_metrics: dict[str, dict]
    classifications: list[ABCItem]
