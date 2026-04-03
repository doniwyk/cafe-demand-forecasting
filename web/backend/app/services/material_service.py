from __future__ import annotations

from datetime import date

from sqlalchemy import select, func
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
) -> MaterialRequirementPage:
    import pandas as pd
    from sqlalchemy import text

    from app.config import MENU_BOM_PATH, CONDIMENT_BOM_PATH, CLEANED_SALES_PATH
    from src.models.raw_materials import RawMaterialProcessor

    forecast_q = text(
        "SELECT f.date, i.name as item, f.quantity_predicted as quantity "
        "FROM forecasts f JOIN items i ON f.item_id = i.id "
        "JOIN model_runs mr ON f.model_run_id = mr.id "
        "WHERE mr.is_active = TRUE"
    )
    result = await session.execute(forecast_q)
    rows = result.fetchall()

    if not rows:
        return MaterialRequirementPage(data=[], total=0, page=page, page_size=page_size)

    forecast_df = pd.DataFrame(rows, columns=["Date", "Item", "Quantity"])
    forecast_df["Date"] = pd.to_datetime(forecast_df["Date"]).dt.date

    processor = RawMaterialProcessor(
        sales_path=CLEANED_SALES_PATH,
        menu_bom_path=MENU_BOM_PATH,
        condiment_bom_path=CONDIMENT_BOM_PATH,
    )
    requirements = processor.compute_material_requirements(forecast_df)

    if material:
        requirements = requirements[
            requirements["Raw_Material"].str.contains(material, case=False, na=False)
        ]
    if start_date:
        requirements = requirements[requirements["Date"] >= start_date]
    if end_date:
        requirements = requirements[requirements["Date"] <= end_date]

    total = len(requirements)
    requirements = requirements.iloc[(page - 1) * page_size : page * page_size]

    return MaterialRequirementPage(
        data=[
            DailyMaterialRequirement(
                date=str(row["Date"]),
                raw_material=str(row["Raw_Material"]),
                quantity_required=float(row["Quantity_Required"]),
            )
            for _, row in requirements.iterrows()
        ],
        total=total,
        page=page,
        page_size=page_size,
    )
