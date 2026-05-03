"""Seed the default admin user."""
from __future__ import annotations

import asyncio
import sys

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

sys.path.insert(0, str(__file__).rsplit("/", 3)[0])

from app.db.engine import async_session  # noqa: E402
from app.models.user import User  # noqa: E402
from app.services.auth import auth_service  # noqa: E402


async def seed():
    async with async_session() as session:
        session: AsyncSession
        result = await session.execute(select(User).where(User.email == "manager@cafe.com"))
        existing = result.scalar_one_or_none()
        if existing:
            print("User already exists, skipping.")
            return

        user = User(
            email="manager@cafe.com",
            name="Cafe Manager",
            hashed_password=auth_service.hash_password("password"),
            avatar="/avatars/cafe.jpg",
        )
        session.add(user)
        await session.commit()
        print(f"Created user: {user.email}")


if __name__ == "__main__":
    asyncio.run(seed())
