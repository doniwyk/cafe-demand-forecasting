from __future__ import annotations

import asyncio
from datetime import date

import pandas as pd
from sqlalchemy import select, func, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import RawMaterialRequirement
from app.models.material import DailyMaterialRequirement, MaterialRequirementPage


async def get_daily_materials(
    session: AsyncSession,
    material: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    page: int = 1,
    page_size: int = 100,
) -> MaterialRequirementPage:
    query = select(RawMaterialRequirement)
    if material:
        query = query.where(RawMaterialRequirement.raw_material.ilike(f"%{material}%"))
    if start_date:
        query = query.where(
            RawMaterialRequirement.date >= date.fromisoformat(start_date)
        )
    if end_date:
        query = query.where(RawMaterialRequirement.date <= date.fromisoformat(end_date))

    count_q = select(func.count()).select_from(query.subquery())
    total = (await session.execute(count_q)).scalar() or 0

    query = query.order_by(
        RawMaterialRequirement.date, RawMaterialRequirement.raw_material
    )
    query = query.offset((page - 1) * page_size).limit(page_size)

    result = await session.execute(query)
    rows = result.scalars().all()

    return MaterialRequirementPage(
        data=[
            DailyMaterialRequirement(
                date=str(row.date),
                raw_material=row.raw_material,
                quantity_required=row.quantity_required,
            )
            for row in rows
        ],
        total=total,
        page=page,
        page_size=page_size,
    )


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
    from app.ml.engine import generate_forecast
    from app.services.forecast_service import _forecast_cache
    from src.models.raw_materials import RawMaterialProcessor

    model_type = model_type or "xgboost"
    sales_q = text(
        "SELECT dis.date, i.name as item, dis.quantity_sold "
        "FROM daily_item_sales dis JOIN items i ON dis.item_id = i.id"
    )
    result = await session.execute(sales_q)
    rows = result.fetchall()

    if not rows:
        return MaterialRequirementPage(data=[], total=0, page=page, page_size=page_size)

    df = pd.DataFrame(
        [tuple(row) for row in rows], columns=["Date", "Item", "Quantity_Sold"]
    )
    df["Date"] = pd.to_datetime(df["Date"])

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
