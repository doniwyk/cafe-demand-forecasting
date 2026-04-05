from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.config import DATABASE_URL

async_engine = create_async_engine(
    DATABASE_URL, echo=False, pool_size=10, max_overflow=20
)

sync_database_url = DATABASE_URL.replace("postgresql+asyncpg://", "postgresql://")
sync_engine = create_engine(
    sync_database_url, echo=False, pool_size=10, max_overflow=20
)

async_session = async_sessionmaker(
    async_engine, class_=AsyncSession, expire_on_commit=False
)
sync_session = sessionmaker(bind=sync_engine, expire_on_commit=False)

engine = async_engine  # alias for backward compatibility
