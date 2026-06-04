from __future__ import annotations

import asyncio
from datetime import date

import pandas as pd
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.material import DailyMaterialRequirement, MaterialRequirementPage
from app.ml.engine import generate_forecast
from app.repositories.forecast_repository import ForecastRepository
from app.services.forecast_service import _forecast_cache


async def get_material_forecast(
    session: AsyncSession,
    material: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    page: int = 1,
    page_size: int = 100,
    model_type: str | None = None,
) -> MaterialRequirementPage:
    from app.config import MENU_BOM_PATH, CONDIMENT_BOM_PATH
    from src.models.raw_materials import RawMaterialProcessor

    model_type = model_type or "xgboost"
    repo = ForecastRepository(session)
    df = await repo.get_sales_dataframe()

    if df.empty:
        return MaterialRequirementPage(data=[], total=0, page=page, page_size=page_size)

    from app.services.forecast_service import filter_sales_to_training_cutoff
    df = await filter_sales_to_training_cutoff(session, df, model_type)

    cache_key = model_type
    cached = _forecast_cache.get(cache_key)

    if cached is not None:
        item_forecast_df = cached.copy()
    else:
        def _run_forecast():
            return generate_forecast(df, weeks=12, model_type=model_type)
        item_forecast_df = await asyncio.to_thread(_run_forecast)
        _forecast_cache[cache_key] = item_forecast_df.copy()

    forecast_df = item_forecast_df[["Date", "Item", "Predicted"]].rename(
        columns={"Predicted": "Quantity"}
    )
    forecast_df["Date"] = pd.to_datetime(forecast_df["Date"]).dt.date

    processor = RawMaterialProcessor(
        menu_bom_path=MENU_BOM_PATH,
        condiment_bom_path=CONDIMENT_BOM_PATH,
    )
    requirements = processor.compute_material_requirements(forecast_df)

    unit_map = _build_unit_map(MENU_BOM_PATH, CONDIMENT_BOM_PATH)

    if material:
        requirements = requirements[
            requirements["Raw_Material"].str.contains(material, case=False, na=False)
        ]
    if start_date:
        requirements = requirements[requirements["Date"] >= date.fromisoformat(start_date)]
    if end_date:
        requirements = requirements[requirements["Date"] <= date.fromisoformat(end_date)]

    aggregated = (
        requirements.groupby("Raw_Material", as_index=False)["Quantity_Required"]
        .sum()
        .sort_values("Quantity_Required", ascending=False)
    )
    total = len(aggregated)
    paginated = aggregated.iloc[(page - 1) * page_size : page * page_size]

    return MaterialRequirementPage(
        data=[
            DailyMaterialRequirement(
                date="",
                raw_material=str(row["Raw_Material"]),
                quantity_required=float(row["Quantity_Required"]),
                unit=unit_map.get(str(row["Raw_Material"]).strip().lower(), ""),
            )
            for _, row in paginated.iterrows()
        ],
        total=total,
        page=page,
        page_size=page_size,
    )


def _build_unit_map(menu_bom_path, condiment_bom_path) -> dict[str, str]:
    unit_map = {}
    try:
        menu = pd.read_csv(menu_bom_path)
        for _, row in menu.iterrows():
            unit_map[str(row["Bahan"]).strip().lower()] = str(row["Unit"]).strip()
        cond = pd.read_csv(condiment_bom_path)
        for _, row in cond.iterrows():
            unit_map[str(row["Sub_Ingredient"]).strip().lower()] = str(row["Sub_Unit"]).strip()
    except Exception:
        pass
    return unit_map
