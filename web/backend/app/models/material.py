from pydantic import BaseModel


class DailyMaterialRequirement(BaseModel):
    date: str
    raw_material: str
    quantity_required: float


class MaterialRequirementPage(BaseModel):
    data: list[DailyMaterialRequirement]
    total: int
    page: int
    page_size: int
