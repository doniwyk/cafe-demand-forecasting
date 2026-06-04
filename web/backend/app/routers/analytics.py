from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.deps import get_session
from app.models.analytics import ABCAnalysisResponse
from app.services import analytics_service

router = APIRouter(prefix="/api/analytics", tags=["analytics"])


@router.get("/abc", response_model=ABCAnalysisResponse)
async def get_abc_analysis(
    session: AsyncSession = Depends(get_session),
    model_type: str | None = Query(None),
):
    return await analytics_service.get_abc_analysis(session, model_type)


@router.get("/metrics")
async def get_metrics(
    session: AsyncSession = Depends(get_session),
    model_type: str | None = Query(None),
):
    return await analytics_service.get_metrics(session, model_type)


@router.get("/top-items")
async def get_top_items(
    session: AsyncSession = Depends(get_session),
    n: int = Query(20, ge=1, le=100),
):
    return await analytics_service.get_top_items(session, n)
