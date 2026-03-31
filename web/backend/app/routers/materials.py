from __future__ import annotations

from fastapi import APIRouter, Query

from app.models.material import MaterialRequirementPage
from app.services import material_service

router = APIRouter(prefix="/api/materials", tags=["materials"])


@router.get("/daily", response_model=MaterialRequirementPage)
def get_daily_materials(
    material: str | None = Query(None),
    start_date: str | None = Query(None),
    end_date: str | None = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(100, ge=1, le=1000),
):
    return material_service.get_daily_materials(
        material, start_date, end_date, page, page_size
    )


@router.get("/forecast", response_model=MaterialRequirementPage)
def get_material_forecast(
    material: str | None = Query(None),
    start_date: str | None = Query(None),
    end_date: str | None = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(100, ge=1, le=1000),
):
    return material_service.get_material_forecast(
        material, start_date, end_date, page, page_size
    )
