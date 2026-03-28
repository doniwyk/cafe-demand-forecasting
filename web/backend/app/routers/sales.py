from fastapi import APIRouter, Query

from app.models.sales import DailySalePage, DailyTotalSale, DailyCategorySale, ItemInfo
from app.services import sales_service

router = APIRouter(prefix="/api/sales", tags=["sales"])


@router.get("/daily", response_model=DailySalePage)
def get_daily_sales(
    item: str | None = Query(None),
    start_date: str | None = Query(None),
    end_date: str | None = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(100, ge=1, le=1000),
):
    return sales_service.get_daily_sales(item, start_date, end_date, page, page_size)


@router.get("/daily/total", response_model=list[DailyTotalSale])
def get_daily_total_sales(
    start_date: str | None = Query(None),
    end_date: str | None = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(100, ge=1, le=1000),
):
    return sales_service.get_daily_total_sales(start_date, end_date, page, page_size)


@router.get("/daily/category", response_model=list[DailyCategorySale])
def get_daily_category_sales(
    category: str | None = Query(None),
    start_date: str | None = Query(None),
    end_date: str | None = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(100, ge=1, le=1000),
):
    return sales_service.get_daily_category_sales(
        category, start_date, end_date, page, page_size
    )


@router.get("/items", response_model=list[ItemInfo])
def get_items():
    return sales_service.get_items()


@router.get("/categories", response_model=list[str])
def get_categories():
    return sales_service.get_categories()
