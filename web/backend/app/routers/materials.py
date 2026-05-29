from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.deps import get_session
from app.models.material import MaterialRequirementPage
from app.services import material_service
from app.services import recipe_material_service

router = APIRouter(prefix="/api/materials", tags=["materials"])


@router.get("/daily", response_model=MaterialRequirementPage)
async def get_daily_materials(
    session: AsyncSession = Depends(get_session),
    material: str | None = Query(None),
    start_date: str | None = Query(None),
    end_date: str | None = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=100),
):
    return await material_service.get_daily_materials(session, material, start_date, end_date, page, page_size)


@router.get("/forecast", response_model=MaterialRequirementPage)
async def get_material_forecast(
    session: AsyncSession = Depends(get_session),
    material: str | None = Query(None),
    start_date: str | None = Query(None),
    end_date: str | None = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=100),
    model_type: str | None = Query(None),
):
    return await material_service.get_material_forecast(
        session, material, start_date, end_date, page, page_size, model_type
    )


@router.get("/daily-forecast", response_model=MaterialRequirementPage)
async def get_daily_material_forecast(
    start_date: str | None = Query(None),
    end_date: str | None = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=100),
    model_type: str | None = Query(None),
):
    return await recipe_material_service.get_daily_material_forecast(
        start_date, end_date, page, page_size, model_type
    )
