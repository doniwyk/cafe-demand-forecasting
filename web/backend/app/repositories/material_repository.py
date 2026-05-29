from __future__ import annotations

from datetime import date

from sqlalchemy import select

from app.db.models import RawMaterialRequirement
from app.repositories import BaseRepository


class MaterialRepository(BaseRepository):
    async def get_daily_materials(
        self,
        material: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        page: int = 1,
        page_size: int = 100,
    ):
        query = select(RawMaterialRequirement)
        if material:
            query = query.where(RawMaterialRequirement.raw_material.ilike(f"%{material}%"))
        if start_date:
            query = query.where(RawMaterialRequirement.date >= date.fromisoformat(start_date))
        if end_date:
            query = query.where(RawMaterialRequirement.date <= date.fromisoformat(end_date))
        query = query.order_by(RawMaterialRequirement.date, RawMaterialRequirement.raw_material)
        return await self._paginate(query, page, page_size)
