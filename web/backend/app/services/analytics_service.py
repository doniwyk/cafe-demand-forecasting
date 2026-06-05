from __future__ import annotations

import asyncio

import pandas as pd
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.analytics import (
    ABCItem,
    ABCAnalysisResponse,
)
from app.repositories.sales_repository import SalesRepository
from app.repositories.forecast_repository import ForecastRepository
from app.ml.engine import generate_forecast


async def get_abc_analysis(
    session: AsyncSession,
    model_type: str | None = None,
) -> ABCAnalysisResponse:
    if model_type:
        repo = ForecastRepository(session)
        df = await repo.get_sales_dataframe()
        if df.empty:
            return ABCAnalysisResponse(class_metrics={}, classifications=[])

        from app.services.forecast_service import filter_sales_to_training_cutoff
        df = await filter_sales_to_training_cutoff(session, df, model_type)

        def _run():
            return generate_forecast(df, weeks=12, model_type=model_type)

        forecast_df = await asyncio.to_thread(_run)

        item_vol = (
            forecast_df.groupby("Item")["Predicted"]
            .sum()
            .reset_index()
            .sort_values("Predicted", ascending=False)
        )
        rows = [
            type("Row", (), {"name": row["Item"], "total_vol": row["Predicted"]})
            for _, row in item_vol.iterrows()
        ]
    else:
        repo = SalesRepository(session)
        rows = await repo.get_item_volumes()

    if not rows:
        return ABCAnalysisResponse(class_metrics={}, classifications=[])

    return _compute_abc_classification(rows)


async def get_metrics(
    session: AsyncSession, model_type: str | None = None
) -> dict:
    repo = ForecastRepository(session)
    run = await repo.get_active_run(model_type)
    if run is None:
        return {
            "r2": 0, "wmape": 0, "mae": 0, "rmse": 0,
            "median_period_accuracy": 0, "periods_within_20pct": 0, "periods_within_50pct": 0,
        }
    return {
        "r2": run.r2 or 0,
        "wmape": run.wmape or 0,
        "mae": run.mae or 0,
        "rmse": run.rmse or 0,
        "median_period_accuracy": run.median_period_accuracy or 0,
        "periods_within_20pct": run.periods_within_20pct or 0,
        "periods_within_50pct": run.periods_within_50pct or 0,
    }


async def get_top_items(session: AsyncSession, n: int = 20) -> list[dict]:
    repo = SalesRepository(session)
    rows = await repo.get_top_items(n)
    return [
        {"item": row.name, "total_quantity": float(row.total_qty)}
        for row in rows
    ]


def _compute_abc_classification(rows) -> ABCAnalysisResponse:
    total = sum(r.total_vol for r in rows)
    cumulative = 0
    class_metrics = {
        "A": {"n_items": 0, "total_volume": 0, "pct_volume": 0},
        "B": {"n_items": 0, "total_volume": 0, "pct_volume": 0},
        "C": {"n_items": 0, "total_volume": 0, "pct_volume": 0},
    }
    classifications = []

    for r in rows:
        cumulative += r.total_vol
        pct = cumulative / total
        abc = "A" if pct <= 0.70 else ("B" if pct <= 0.90 else "C")
        class_metrics[abc]["n_items"] += 1
        class_metrics[abc]["total_volume"] += r.total_vol
        classifications.append(
            ABCItem(
                item=r.name,
                vol=float(r.total_vol),
                cum=float(cumulative),
                pct=float(pct),
                class_label=abc,
            )
        )

    for cls in class_metrics.values():
        cls["pct_volume"] = round(cls["total_volume"] / total * 100, 1) if total else 0

    return ABCAnalysisResponse(
        class_metrics=class_metrics,
        classifications=classifications,
    )
