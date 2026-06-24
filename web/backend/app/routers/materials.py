from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.deps import get_session
from app.models.material import MaterialRequirementPage
from app.services import material_service

router = APIRouter(prefix="/api/materials", tags=["materials"])


@router.get("/forecast", response_model=MaterialRequirementPage)
async def get_material_forecast(
    session: AsyncSession = Depends(get_session),
    material: str | None = Query(None),
    start_date: str | None = Query(None),
    end_date: str | None = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=100),
):
    return await material_service.get_material_forecast(
        session, material, start_date, end_date, page, page_size
    )
