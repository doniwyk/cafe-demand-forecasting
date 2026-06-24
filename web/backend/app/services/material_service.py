from __future__ import annotations

import asyncio
from datetime import date

import pandas as pd
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.material import DailyMaterialRequirement, MaterialRequirementPage
from app.ml.engine import generate_forecast
from app.repositories.forecast_repository import ForecastRepository
import app.services.forecast_service as fc_svc


async def _load_bom_from_db(session: AsyncSession) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    from sqlalchemy import text

    result = await session.execute(text("SELECT type, item_name, ingredient, quantity, unit FROM menu_bom"))
    menu_rows = result.fetchall()
    menu_df = pd.DataFrame(menu_rows, columns=["Tipe", "Item", "Bahan", "Qty", "Unit"])

    result = await session.execute(text(
        "SELECT item_name, total_qty, total_unit, sub_ingredient, qty_per_unit, sub_unit FROM condiment_bom"
    ))
    condiment_rows = result.fetchall()
    condiment_df = pd.DataFrame(condiment_rows, columns=[
        "Condiment", "Condiment_Qty", "Condiment_Unit", "Sub_Ingredient", "Qty_per_condiment_unit", "Sub_Unit"
    ])

    unit_map = {}
    for _, row in menu_df.iterrows():
        unit_map[str(row["Bahan"]).strip().lower()] = str(row["Unit"]).strip()
    for _, row in condiment_df.iterrows():
        unit_map[str(row["Sub_Ingredient"]).strip().lower()] = str(row["Sub_Unit"]).strip()
    return menu_df, condiment_df, unit_map


async def get_material_forecast(
    session: AsyncSession,
    material: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    page: int = 1,
    page_size: int = 100,
) -> MaterialRequirementPage:
    from app.ml.raw_materials import RawMaterialProcessor

    repo = ForecastRepository(session)
    df = await repo.get_sales_dataframe()

    if df.empty:
        return MaterialRequirementPage(data=[], total=0, page=page, page_size=page_size)

    from app.services.forecast_service import filter_sales_to_training_cutoff
    df = await filter_sales_to_training_cutoff(session, df)

    cached = fc_svc._forecast_cache
    needed_weeks = max(12, ((date.today() - date(2026, 4, 1)).days // 7) + 8)

    if cached is not None:
        cached_max = cached["Date"].max()
        if end_date and pd.Timestamp(end_date) > cached_max:
            cached = None
        elif (date.today() - cached_max.date()).days > 14:
            cached = None

    if cached is not None:
        item_forecast_df = cached.copy()
    else:
        def _run_forecast():
            return generate_forecast(df, weeks=needed_weeks)
        item_forecast_df = await asyncio.to_thread(_run_forecast)
        fc_svc._forecast_cache = item_forecast_df.copy()

    forecast_df = item_forecast_df[["Date", "Item", "Predicted"]].rename(
        columns={"Predicted": "Quantity"}
    )
    forecast_df["Date"] = pd.to_datetime(forecast_df["Date"]).dt.date

    menu_df, condiment_df, unit_map = await _load_bom_from_db(session)
    processor = RawMaterialProcessor(menu_bom_df=menu_df, condiment_bom_df=condiment_df)
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


async def export_material_csv(
    session: AsyncSession,
    material: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> str:
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

    menu_df, condiment_df, unit_map = await _load_bom_from_db(session)
    processor = RawMaterialProcessor(menu_bom_df=menu_df, condiment_bom_df=condiment_df)
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

    lines = ["Material,Total Quantity Required,Unit"]
    for _, row in aggregated.iterrows():
        name = str(row["Raw_Material"])
        qty = round(float(row["Quantity_Required"]), 2)
        unit = unit_map.get(name.strip().lower(), "")
        lines.append(f"{name},{qty},{unit}")
    return "\n".join(lines)
