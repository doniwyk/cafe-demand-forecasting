from __future__ import annotations

from datetime import date

from sqlalchemy import select, func, distinct
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import (
    Category,
    Item,
    DailyItemSale,
    DailyCategorySale,
    DailyTotalSale,
)
from app.models.sales import (
    DailySale,
    DailySalePage,
    DailyTotalSale as DailyTotalSaleSchema,
    DailyCategorySale as DailyCategorySaleSchema,
    ItemInfo,
)


async def get_daily_sales(
    session: AsyncSession,
    item: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    page: int = 1,
    page_size: int = 100,
) -> DailySalePage:
    query = select(DailyItemSale, Item.name).join(Item)
    if item:
        query = query.where(Item.name == item)
    if start_date:
        query = query.where(DailyItemSale.date >= date.fromisoformat(start_date))
    if end_date:
        query = query.where(DailyItemSale.date <= date.fromisoformat(end_date))

    count_q = select(func.count()).select_from(query.subquery())
    total = (await session.execute(count_q)).scalar() or 0

    query = query.order_by(DailyItemSale.date, Item.name)
    query = query.offset((page - 1) * page_size).limit(page_size)

    result = await session.execute(query)
    rows = result.all()

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
    session: AsyncSession,
    start_date: str | None = None,
    end_date: str | None = None,
    page: int = 1,
    page_size: int = 100,
) -> list[DailyTotalSaleSchema]:
    query = select(DailyTotalSale)
    if start_date:
        query = query.where(DailyTotalSale.date >= date.fromisoformat(start_date))
    if end_date:
        query = query.where(DailyTotalSale.date <= date.fromisoformat(end_date))

    query = query.order_by(DailyTotalSale.date)
    query = query.offset((page - 1) * page_size).limit(page_size)

    result = await session.execute(query)
    rows = result.scalars().all()

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
    session: AsyncSession,
    category: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    page: int = 1,
    page_size: int = 100,
) -> list[DailyCategorySaleSchema]:
    query = select(DailyCategorySale)
    if category:
        query = query.where(DailyCategorySale.category == category)
    if start_date:
        query = query.where(DailyCategorySale.date >= date.fromisoformat(start_date))
    if end_date:
        query = query.where(DailyCategorySale.date <= date.fromisoformat(end_date))

    query = query.order_by(DailyCategorySale.date)
    query = query.offset((page - 1) * page_size).limit(page_size)

    result = await session.execute(query)
    rows = result.scalars().all()

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


async def get_items(session: AsyncSession) -> list[ItemInfo]:
    query = select(Item.name, Category.name).outerjoin(Category).order_by(Item.name)
    result = await session.execute(query)
    return [ItemInfo(name=row[0], category=row[1]) for row in result.all()]


async def get_categories(session: AsyncSession) -> list[str]:
    query = (
        select(DailyCategorySale.category)
        .distinct()
        .order_by(DailyCategorySale.category)
    )
    result = await session.execute(query)
    return [row[0] for row in result.all()]
