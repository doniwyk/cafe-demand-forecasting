from __future__ import annotations

from app.models.sales import (
    DailySale,
    DailySalePage,
    DailyTotalSale as DailyTotalSaleSchema,
    DailyCategorySale as DailyCategorySaleSchema,
    ItemInfo,
)
from app.repositories.sales_repository import SalesRepository
from app.core.deps import get_session


async def get_daily_sales(
    session,
    item: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    page: int = 1,
    page_size: int = 100,
) -> DailySalePage:
    repo = SalesRepository(session)
    rows, total = await repo.get_daily_sales(item, start_date, end_date, page, page_size)

    return DailySalePage(
        data=[
            DailySale(
                date=str(row.DailyItemSale.date),
                item=row.name,
                quantity_sold=row.DailyItemSale.quantity_sold,
            )
            for row in rows
        ],
        total=total,
        page=page,
        page_size=page_size,
    )


async def get_daily_total_sales(
    session,
    start_date: str | None = None,
    end_date: str | None = None,
    page: int = 1,
    page_size: int = 100,
) -> list[DailyTotalSaleSchema]:
    repo = SalesRepository(session)
    rows = await repo.get_daily_total_sales(start_date, end_date, page, page_size)

    return [
        DailyTotalSaleSchema(
            date=str(row.date),
            quantity=row.quantity,
            net_sales=row.net_sales,
            gross_sales=row.gross_sales,
            unique_items=row.unique_items,
        )
        for row in rows
    ]


async def get_daily_category_sales(
    session,
    category: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    page: int = 1,
    page_size: int = 100,
) -> list[DailyCategorySaleSchema]:
    repo = SalesRepository(session)
    rows = await repo.get_daily_category_sales(category, start_date, end_date, page, page_size)

    return [
        DailyCategorySaleSchema(
            date=str(row.date),
            category=row.category,
            quantity=row.quantity,
            net_sales=row.net_sales,
            gross_sales=row.gross_sales,
            unique_items=row.unique_items,
        )
        for row in rows
    ]


async def get_items(session) -> list[ItemInfo]:
    repo = SalesRepository(session)
    rows = await repo.get_items()
    return [ItemInfo(name=row[0], category=row[1]) for row in rows]


async def get_categories(session) -> list[str]:
    repo = SalesRepository(session)
    return await repo.get_categories()
