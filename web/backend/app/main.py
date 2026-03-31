from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path

from app.routers import (
    sales_router,
    forecasts_router,
    materials_router,
    analytics_router,
)
from app.config import STATIC_DIR

app = FastAPI(
    title="Cafe Supply Forecasting API",
    description="API for cafe sales forecasting, raw material requirements, and analytics",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(sales_router)
app.include_router(forecasts_router)
app.include_router(materials_router)
app.include_router(analytics_router)


@app.get("/api/health")
def health_check():
    return {"status": "ok"}


if STATIC_DIR.exists():
    app.mount("/assets", StaticFiles(directory=STATIC_DIR / "assets"), name="assets")

    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        file = STATIC_DIR / full_path
        if file.is_file():
            return FileResponse(file)
        return FileResponse(STATIC_DIR / "index.html")
