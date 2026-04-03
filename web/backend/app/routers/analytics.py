from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.engine import async_session
from app.models.analytics import ABCAnalysisResponse, AssociationRule
from app.services import analytics_service

router = APIRouter(prefix="/api/analytics", tags=["analytics"])


async def get_session():
    async with async_session() as session:
        yield session


@router.get("/abc", response_model=ABCAnalysisResponse)
async def get_abc_analysis(session: AsyncSession = Depends(get_session)):
    return await analytics_service.get_abc_analysis(session)


@router.get("/metrics")
async def get_metrics(session: AsyncSession = Depends(get_session)):
    return await analytics_service.get_metrics(session)


@router.get("/top-items")
async def get_top_items(
    session: AsyncSession = Depends(get_session),
    n: int = Query(20, ge=1, le=100),
):
    return await analytics_service.get_top_items(session, n)


@router.get("/association-rules", response_model=list[AssociationRule])
async def get_association_rules(
    session: AsyncSession = Depends(get_session),
    min_confidence: float = Query(0.3, ge=0, le=1),
    min_lift: float = Query(1.0, ge=0),
):
    return await analytics_service.get_association_rules(
        session, min_confidence, min_lift
    )
