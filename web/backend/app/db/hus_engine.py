from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.config import HUS_DB_URL, HUS_DB_SYNC_URL

async_hus_engine = create_async_engine(
    HUS_DB_URL.replace("postgresql://", "postgresql+asyncpg://"),
    echo=False,
    pool_size=5,
    max_overflow=10,
)

hus_sync_engine = create_engine(
    HUS_DB_SYNC_URL, echo=False, pool_size=5, max_overflow=10
)

hus_async_session = async_sessionmaker(
    async_hus_engine, class_=AsyncSession, expire_on_commit=False
)
hus_sync_session = sessionmaker(bind=hus_sync_engine, expire_on_commit=False)
