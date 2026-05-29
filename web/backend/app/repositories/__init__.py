from __future__ import annotations

from typing import Any, Sequence

from sqlalchemy import Select, func, select
from sqlalchemy.ext.asyncio import AsyncSession


class BaseRepository:
    def __init__(self, session: AsyncSession):
        self._session = session

    async def _count(self, query: Select) -> int:
        subq = query.subquery()
        count_q = select(func.count()).select_from(subq)
        result = await self._session.execute(count_q)
        return result.scalar() or 0

    async def _paginate(
        self, query: Select, page: int, page_size: int
    ) -> tuple[Sequence[Any], int]:
        total = await self._count(query)
        query = query.offset((page - 1) * page_size).limit(page_size)
        result = await self._session.execute(query)
        return result.unique().all(), total
