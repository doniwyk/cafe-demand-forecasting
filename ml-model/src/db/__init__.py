import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(
    Path(__file__).resolve().parent.parent.parent.parent / "web" / "backend" / ".env"
)

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase


class Base(DeclarativeBase):
    pass


DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5432/cafe_forecasting",
)

engine = create_engine(DATABASE_URL, echo=False, pool_size=5, max_overflow=10)
SessionLocal = sessionmaker(engine, autocommit=False, autoflush=False)


def get_sync_url() -> str:
    url = DATABASE_URL
    if url.startswith("postgresql+asyncpg"):
        url = url.replace("postgresql+asyncpg", "postgresql")
    return url
