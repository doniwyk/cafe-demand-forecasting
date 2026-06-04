from __future__ import annotations

from sqlalchemy import select, text, update

from app.db.models import (
    ModelRun,
    ModelRunClassMetric,
    ModelRunTopItem,
    Forecast,
    ItemABC,
)
from app.repositories import BaseRepository


class ForecastRepository(BaseRepository):
    async def get_active_run(self, model_type: str | None = None):
        query = select(ModelRun).where(ModelRun.is_active == True)
        if model_type:
            query = query.where(ModelRun.model_type == model_type)
        query = query.order_by(ModelRun.trained_at.desc()).limit(1)
        return (await self._session.execute(query)).scalar_one_or_none()

    async def deactivate_model_type(self, model_type: str):
        await self._session.execute(
            update(ModelRun)
            .where(ModelRun.model_type == model_type)
            .values(is_active=False)
        )

    async def create_model_run(self, **kwargs) -> ModelRun:
        run = ModelRun(**kwargs)
        self._session.add(run)
        await self._session.flush()
        return run

    async def get_class_metrics(self, model_run_id: int):
        query = select(ModelRunClassMetric).where(
            ModelRunClassMetric.model_run_id == model_run_id
        )
        return (await self._session.execute(query)).scalars().all()

    async def get_top_items(self, model_run_id: int):
        query = select(ModelRunTopItem).where(ModelRunTopItem.model_run_id == model_run_id)
        return (await self._session.execute(query)).scalars().all()

    async def get_item_abc(self):
        query = select(ItemABC).order_by(ItemABC.cumulative_pct)
        return (await self._session.execute(query)).scalars().all()

    async def get_inactive_run_ids(self):
        query = select(ModelRun.id).where(ModelRun.is_active == False)
        return (await self._session.execute(query)).scalars().all()

    async def get_items_with_models(self, model_run_id: int) -> set[str]:
        run = await self._session.get(ModelRun, model_run_id)
        if not run or not run.items_with_models:
            return set()
        import json
        try:
            return {
                str(item).strip().lower()
                for item in json.loads(run.items_with_models or "[]")
                if str(item).strip()
            }
        except (TypeError, json.JSONDecodeError):
            return set()

    async def get_sales_dataframe(self, items: list[str] | None = None):
        import pandas as pd

        if items:
            placeholders = ", ".join(f":item_{i}" for i in range(len(items)))
            query = text(
                f"SELECT dis.date, i.name as item, dis.quantity_sold "
                f"FROM daily_item_sales dis JOIN items i ON dis.item_id = i.id "
                f"WHERE i.name IN ({placeholders})"
            )
            params = {f"item_{i}": item for i, item in enumerate(items)}
            result = await self._session.execute(query, params)
        else:
            query = text(SALES_JOIN_SQL)
            result = await self._session.execute(query)
        rows = result.fetchall()
        if not rows:
            return pd.DataFrame(columns=["Date", "Item", "Quantity_Sold"])
        df = pd.DataFrame(
            [tuple(row) for row in rows], columns=["Date", "Item", "Quantity_Sold"]
        )
        df["Date"] = pd.to_datetime(df["Date"])
        return df


SALES_JOIN_SQL = """
    SELECT dis.date, i.name as item, dis.quantity_sold
    FROM daily_item_sales dis
    JOIN items i ON dis.item_id = i.id
"""
