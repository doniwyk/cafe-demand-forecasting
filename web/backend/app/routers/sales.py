from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.deps import get_session
from app.models.sales import DailySalePage, ItemInfo
from app.services import sales_service

router = APIRouter(prefix="/api/sales", tags=["sales"])


@router.get("/daily", response_model=DailySalePage)
async def get_daily_sales(
    session: AsyncSession = Depends(get_session),
    item: str | None = Query(None),
    start_date: str | None = Query(None),
    end_date: str | None = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(100, ge=1, le=1000),
):
    return await sales_service.get_daily_sales(session, item, start_date, end_date, page, page_size)


@router.get("/items", response_model=list[ItemInfo])
async def get_items(session: AsyncSession = Depends(get_session)):
    return await sales_service.get_items(session)


@router.get("/categories", response_model=list[str])
async def get_categories(session: AsyncSession = Depends(get_session)):
    return await sales_service.get_categories(session)


@router.get("/latest-date", response_model=str | None)
async def get_latest_date(session: AsyncSession = Depends(get_session)):
    return await sales_service.get_latest_date(session)
