from __future__ import annotations

from datetime import date

from sqlalchemy import select, func, text
from sqlalchemy.orm import joinedload

from app.db.models import (
    Category,
    Item,
    DailyItemSale,
    DailyCategorySale,
    DailyTotalSale,
)
from app.repositories import BaseRepository


class SalesRepository(BaseRepository):
    async def get_daily_sales(
        self,
        item: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        page: int = 1,
        page_size: int = 100,
    ):
        query = select(DailyItemSale, Item.name).join(Item)
        if item:
            query = query.where(Item.name == item)
        if start_date:
            query = query.where(DailyItemSale.date >= date.fromisoformat(start_date))
        if end_date:
            query = query.where(DailyItemSale.date <= date.fromisoformat(end_date))
        query = query.order_by(DailyItemSale.date, Item.name)
        return await self._paginate(query, page, page_size)

    async def get_daily_total_sales(
        self,
        start_date: str | None = None,
        end_date: str | None = None,
        page: int = 1,
        page_size: int = 100,
    ):
        query = select(DailyTotalSale)
        if start_date:
            query = query.where(DailyTotalSale.date >= date.fromisoformat(start_date))
        if end_date:
            query = query.where(DailyTotalSale.date <= date.fromisoformat(end_date))
        query = query.order_by(DailyTotalSale.date)
        query = query.offset((page - 1) * page_size).limit(page_size)
        result = await self._session.execute(query)
        return result.scalars().all()

    async def get_daily_category_sales(
        self,
        category: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        page: int = 1,
        page_size: int = 100,
    ):
        query = select(DailyCategorySale)
        if category:
            query = query.where(DailyCategorySale.category == category)
        if start_date:
            query = query.where(DailyCategorySale.date >= date.fromisoformat(start_date))
        if end_date:
            query = query.where(DailyCategorySale.date <= date.fromisoformat(end_date))
        query = query.order_by(DailyCategorySale.date)
        query = query.offset((page - 1) * page_size).limit(page_size)
        result = await self._session.execute(query)
        return result.scalars().all()

    async def get_items(self):
        query = select(Item.name, Category.name).outerjoin(Category).order_by(Item.name)
        result = await self._session.execute(query)
        return result.all()

    async def get_categories(self):
        query = (
            select(DailyCategorySale.category)
            .distinct()
            .order_by(DailyCategorySale.category)
        )
        result = await self._session.execute(query)
        return [row[0] for row in result.all()]

    async def get_sales_dataframe(self):
        import pandas as pd

        result = await self._session.execute(text(SALES_JOIN_SQL))
        rows = result.fetchall()
        if not rows:
            return pd.DataFrame(columns=["Date", "Item", "Quantity_Sold"])
        df = pd.DataFrame(
            [tuple(row) for row in rows], columns=["Date", "Item", "Quantity_Sold"]
        )
        df["Date"] = pd.to_datetime(df["Date"])
        return df

    async def get_item_volumes(self):
        query = (
            select(
                Item.name,
                func.sum(DailyItemSale.quantity_sold).label("total_vol"),
            )
            .join(DailyItemSale, Item.id == DailyItemSale.item_id)
            .group_by(Item.id, Item.name)
            .order_by(text("total_vol DESC"))
        )
        result = await self._session.execute(query)
        return result.all()

    async def get_top_items(self, n: int = 20):
        query = (
            select(
                Item.name,
                func.sum(DailyItemSale.quantity_sold).label("total_qty"),
            )
            .join(DailyItemSale, Item.id == DailyItemSale.item_id)
            .group_by(Item.id, Item.name)
            .order_by(text("total_qty DESC"))
            .limit(n)
        )
        result = await self._session.execute(query)
        return result.all()


SALES_JOIN_SQL = """
    SELECT dis.date, i.name as item, dis.quantity_sold
    FROM daily_item_sales dis
    JOIN items i ON dis.item_id = i.id
"""
