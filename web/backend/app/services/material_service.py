from __future__ import annotations

import asyncio
from datetime import date

import pandas as pd
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.material import DailyMaterialRequirement, MaterialRequirementPage
from app.ml.engine import generate_forecast
from app.repositories.forecast_repository import ForecastRepository
import app.services.forecast_service as fc_svc


async def get_material_forecast(
    session: AsyncSession,
    material: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    page: int = 1,
    page_size: int = 100,
) -> MaterialRequirementPage:
    from app.config import MENU_BOM_PATH, CONDIMENT_BOM_PATH
    from app.ml.raw_materials import RawMaterialProcessor

    repo = ForecastRepository(session)
    df = await repo.get_sales_dataframe()

    if df.empty:
        return MaterialRequirementPage(data=[], total=0, page=page, page_size=page_size)

    from app.services.forecast_service import filter_sales_to_training_cutoff
    df = await filter_sales_to_training_cutoff(session, df)

    cached = fc_svc._forecast_cache

    if cached is not None:
        item_forecast_df = cached.copy()
    else:
        def _run_forecast():
            return generate_forecast(df, weeks=12)
        item_forecast_df = await asyncio.to_thread(_run_forecast)
        fc_svc._forecast_cache = item_forecast_df.copy()

    forecast_df = item_forecast_df[["Date", "Item", "Predicted"]].rename(
        columns={"Predicted": "Quantity"}
    )
    forecast_df["Date"] = pd.to_datetime(forecast_df["Date"]).dt.date

    processor = RawMaterialProcessor(
        menu_bom_path=MENU_BOM_PATH,
        condiment_bom_path=CONDIMENT_BOM_PATH,
    )
    requirements = processor.compute_material_requirements(forecast_df)

    unit_map = _build_unit_map()

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


_unit_map_cache: dict[str, str] | None = None


def _build_unit_map() -> dict[str, str]:
    global _unit_map_cache
    if _unit_map_cache is not None:
        return _unit_map_cache
    from app.config import MENU_BOM_PATH, CONDIMENT_BOM_PATH

    unit_map = {}
    try:
        menu = pd.read_csv(MENU_BOM_PATH)
        for _, row in menu.iterrows():
            unit_map[str(row["Bahan"]).strip().lower()] = str(row["Unit"]).strip()
        cond = pd.read_csv(CONDIMENT_BOM_PATH)
        for _, row in cond.iterrows():
            unit_map[str(row["Sub_Ingredient"]).strip().lower()] = str(row["Sub_Unit"]).strip()
    except Exception:
        pass
    _unit_map_cache = unit_map
    return unit_map


async def export_material_csv(
    session: AsyncSession,
    material: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> str:
    from app.config import MENU_BOM_PATH, CONDIMENT_BOM_PATH
    from app.ml.raw_materials import RawMaterialProcessor

    repo = ForecastRepository(session)
    df = await repo.get_sales_dataframe()
    if df.empty:
        return "Material,Total Quantity Required,Unit\n"

    from app.services.forecast_service import filter_sales_to_training_cutoff
    df = await filter_sales_to_training_cutoff(session, df)

    cached = fc_svc._forecast_cache
    if cached is not None:
        item_forecast_df = cached.copy()
    else:
        def _run_forecast():
            return generate_forecast(df, weeks=12)
        item_forecast_df = await asyncio.to_thread(_run_forecast)
        fc_svc._forecast_cache = item_forecast_df.copy()

    forecast_df = item_forecast_df[["Date", "Item", "Predicted"]].rename(
        columns={"Predicted": "Quantity"}
    )
    forecast_df["Date"] = pd.to_datetime(forecast_df["Date"]).dt.date

    processor = RawMaterialProcessor(
        menu_bom_path=MENU_BOM_PATH,
        condiment_bom_path=CONDIMENT_BOM_PATH,
    )
    requirements = processor.compute_material_requirements(forecast_df)

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

    unit_map = _build_unit_map()
    lines = ["Material,Total Quantity Required,Unit"]
    for _, row in aggregated.iterrows():
        name = str(row["Raw_Material"])
        qty = round(float(row["Quantity_Required"]), 2)
        unit = unit_map.get(name.strip().lower(), "")
        lines.append(f"{name},{qty},{unit}")
    return "\n".join(lines)
