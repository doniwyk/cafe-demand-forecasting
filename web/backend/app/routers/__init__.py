from app.routers.sales import router as sales_router
from app.routers.forecasts import router as forecasts_router
from app.routers.materials import router as materials_router
from app.routers.analytics import router as analytics_router

__all__ = ["sales_router", "forecasts_router", "materials_router", "analytics_router"]
