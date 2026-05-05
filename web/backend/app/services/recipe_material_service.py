from __future__ import annotations

import asyncio
import pandas as pd
from datetime import date
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.engine import async_session, sync_session
from app.models.material import DailyMaterialRequirement, MaterialRequirementPage


async def get_daily_material_forecast(
    start_date: str | None = None,
    end_date: str | None = None,
    page: int = 1,
    page_size: int = 100,
) -> MaterialRequirementPage:
    from app.ml.engine import generate_forecast

    async with async_session() as session:
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

    def _run_forecast():
        return generate_forecast(df, weeks=12)

    item_forecast_df = await asyncio.to_thread(_run_forecast)

    forecast_df = item_forecast_df[["Date", "Item", "Predicted"]].rename(
        columns={"Predicted": "Quantity"}
    )
    forecast_df["Date"] = pd.to_datetime(forecast_df["Date"]).dt.date
    forecast_df["Quantity"] = pd.to_numeric(
        forecast_df["Quantity"], errors="coerce"
    ).fillna(0)

    if start_date:
        forecast_df = forecast_df[forecast_df["Date"] >= date.fromisoformat(start_date)]
    if end_date:
        forecast_df = forecast_df[forecast_df["Date"] <= date.fromisoformat(end_date)]

    material_requirements = _map_forecast_to_materials(forecast_df)

    total = len(material_requirements)
    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size
    paginated = material_requirements.iloc[start_idx:end_idx]

    return MaterialRequirementPage(
        data=[
            DailyMaterialRequirement(
                date=str(row["Date"]),
                raw_material=str(row["Raw_Material"]),
                quantity_required=float(row["Quantity_Required"]),
            )
            for _, row in paginated.iterrows()
        ],
        total=total,
        page=page,
        page_size=page_size,
    )


def _map_forecast_to_materials(forecast_df: pd.DataFrame) -> pd.DataFrame:
    with sync_session() as session:
        recipe_query = text(
            """
            SELECT 
                p.name as product,
                v.name as variant,
                m.name as material,
                COALESCE(m.unit_id, 1) as unit_id,
                pri.quantity as recipe_qty,
                c.name as condiment
            FROM product_recipe_ingredients pri
            JOIN products p ON pri.product_id = p.id
            LEFT JOIN product_variants v ON pri.variant_id = v.id
            LEFT JOIN materials m ON pri.material_id = m.id
            LEFT JOIN condiments c ON pri.condiment_id = c.id
            WHERE p.is_active = true
            """
        )
        recipe_result = session.execute(recipe_query)
        recipe_rows = recipe_result.fetchall()

    recipe_df = pd.DataFrame(
        recipe_rows,
        columns=[
            "Product",
            "Variant",
            "Material",
            "Unit_ID",
            "Recipe_Qty",
            "Condiment",
        ],
    )

    recipe_df["Recipe_Qty"] = pd.to_numeric(
        recipe_df["Recipe_Qty"], errors="coerce"
    ).fillna(0)

    recipe_df["Item_Name"] = recipe_df.apply(
        lambda row: (
            f"{row['Product']} {row['Variant']}".strip()
            if row["Variant"]
            else row["Product"]
        ),
        axis=1,
    )

    material_requirements = []

    for _, forecast_row in forecast_df.iterrows():
        date = forecast_row["Date"]
        item = forecast_row["Item"]
        qty = forecast_row["Quantity"]

        matching_recipes = recipe_df[recipe_df["Item_Name"].str.lower() == item.lower()]

        if matching_recipes.empty:
            matching_recipes = recipe_df[
                recipe_df["Product"].str.lower() == item.lower()
            ]

        if matching_recipes.empty:
            continue

        for _, recipe in matching_recipes.iterrows():
            if pd.notna(recipe["Material"]) and recipe["Material"]:
                material_requirements.append(
                    {
                        "Date": date,
                        "Raw_Material": recipe["Material"],
                        "Quantity_Required": recipe["Recipe_Qty"] * qty,
                    }
                )

            if pd.notna(recipe["Condiment"]) and recipe["Condiment"]:
                cond_qty = _expand_condiment(
                    recipe["Condiment"], recipe["Recipe_Qty"] * qty
                )
                for mat, mqty in cond_qty.items():
                    material_requirements.append(
                        {"Date": date, "Raw_Material": mat, "Quantity_Required": mqty}
                    )

    result_df = pd.DataFrame(material_requirements)

    if result_df.empty:
        return result_df

    result_df = result_df.groupby(["Date", "Raw_Material"], as_index=False)[
        "Quantity_Required"
    ].sum()
    result_df = result_df.sort_values(["Date", "Raw_Material"]).reset_index(drop=True)

    return result_df


def _expand_condiment(condiment_name: str, quantity: float) -> dict:
    with sync_session() as session:
        cond_query = text(
            """
            SELECT m.name, cb.quantity, c.batch_quantity
            FROM condiment_ingredients cb
            JOIN condiments c ON cb.condiment_id = c.id
            LEFT JOIN materials m ON cb.material_id = m.id
            WHERE c.name = :condiment_name
            """
        )
        result = session.execute(cond_query, {"condiment_name": condiment_name})
        rows = result.fetchall()

        if not rows:
            return {condiment_name: quantity}

        base_qty = float(rows[0][2]) if rows[0][2] else 1.0
        scaling = quantity / base_qty

        materials = {}
        for row in rows:
            if row[0]:
                sub_qty = float(row[1]) * scaling
                materials[row[0]] = materials.get(row[0], 0) + sub_qty

        return materials
