from __future__ import annotations

from pydantic import BaseModel


class DailySale(BaseModel):
    date: str
    item: str
    quantity_sold: float
    category: str | None = None


class DailySalePage(BaseModel):
    data: list[DailySale]
    total: int
    page: int
    page_size: int


class DailyTotalSale(BaseModel):
    date: str
    quantity: float
    net_sales: float
    gross_sales: float
    unique_items: int
    unique_categories: int | None = None


class DailyCategorySale(BaseModel):
    date: str
    category: str
    quantity: float
    net_sales: float
    gross_sales: float
    unique_items: int


class ItemInfo(BaseModel):
    name: str
    category: str | None = None
