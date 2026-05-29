from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel, EmailStr
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.deps import get_session, get_current_user
from app.services.auth import auth_service

router = APIRouter(prefix="/api/auth", tags=["auth"])


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: dict


class UserResponse(BaseModel):
    id: int
    email: str
    name: str
    avatar: str | None = None


@router.post("/login", response_model=LoginResponse)
async def login(body: LoginRequest, session: AsyncSession = Depends(get_session)):
    user = await auth_service.authenticate(session, body.email, body.password)
    if user is None:
        from fastapi import HTTPException, status
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid email or password")
    token = auth_service.create_access_token(user.id)
    return LoginResponse(
        access_token=token,
        user={"id": user.id, "email": user.email, "name": user.name, "avatar": user.avatar},
    )


@router.get("/me", response_model=UserResponse)
async def get_me(user=Depends(get_current_user)):
    return user
