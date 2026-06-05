from __future__ import annotations

from datetime import date

from sqlalchemy import select, func, text

from app.db.models import (
    Category,
    Item,
    DailyItemSale,
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

    async def get_items(self):
        query = select(Item.name, Category.name).outerjoin(Category).order_by(Item.name)
        result = await self._session.execute(query)
        return result.all()

    async def get_categories(self):
        query = select(Category.name).order_by(Category.name)
        result = await self._session.execute(query)
        return [row[0] for row in result.all()]

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


