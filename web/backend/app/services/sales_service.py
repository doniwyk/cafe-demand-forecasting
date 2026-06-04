from __future__ import annotations

from app.models.sales import (
    DailySale,
    DailySalePage,
    ItemInfo,
)
from app.repositories.sales_repository import SalesRepository


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


async def get_items(session) -> list[ItemInfo]:
    repo = SalesRepository(session)
    rows = await repo.get_items()
    return [ItemInfo(name=row[0], category=row[1]) for row in rows]


async def get_categories(session) -> list[str]:
    repo = SalesRepository(session)
    return await repo.get_categories()
