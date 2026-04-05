from __future__ import annotations

from datetime import datetime
from re import search

from sqlalchemy import select, func, text, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import ItemABC, AssociationRule, Item, DailyItemSale
from app.models.analytics import ABCItem, ABCAnalysisResponse, AssociationRule


async def get_abc_analysis(session: AsyncSession) -> ABCAnalysisResponse:
    item_vol_q = (
        select(
            Item.name,
            func.sum(DailyItemSale.quantity_sold).label("total_vol"),
        )
        .join(DailyItemSale, Item.id == DailyItemSale.item_id)
        .group_by(Item.id, Item.name)
        .order_by(text("total_vol DESC"))
    )
    result = await session.execute(item_vol_q)
    rows = result.all()

    if not rows:
        return ABCAnalysisResponse(class_metrics={}, classifications=[])

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


async def get_metrics(session: AsyncSession, model_type: str | None = None) -> dict:
    from app.db.models import ModelRun

    run_q = select(ModelRun).where(ModelRun.is_active == True)
    if model_type:
        run_q = run_q.where(ModelRun.model_type == model_type)
    run_q = run_q.order_by(ModelRun.trained_at.desc()).limit(1)
    run = (await session.execute(run_q)).scalar_one_or_none()
    if run is None:
        return {"r2": 0, "wmape": 0, "mae": 0, "volume_accuracy": 0}
    return {
        "r2": run.r2 or 0,
        "wmape": run.wmape or 0,
        "mae": run.mae or 0,
        "volume_accuracy": run.volume_accuracy or 0,
    }


async def get_top_items(session: AsyncSession, n: int = 20) -> list[dict]:
    query = (
        select(
            Item.name,
            func.sum(DailyItemSale.quantity_sold).label("total_qty"),
        )
        .join(DailyItemSale, Item.id == DailyItemSale.item_id)
        .group_by(Item.id, Item.name)
        .order_by(text("total_qty DESC"))
        .limit(n)
    )
    result = await session.execute(query)
    return [
        {"item": row.name, "total_quantity": float(row.total_qty)}
        for row in result.all()
    ]


async def get_association_rules(
    session: AsyncSession,
    min_confidence: float = 0.3,
    min_lift: float = 1.0,
) -> list[AssociationRule]:
    query = (
        select(AssociationRule)
        .where(AssociationRule.confidence >= min_confidence)
        .where(AssociationRule.lift >= min_lift)
        .order_by(AssociationRule.lift.desc())
        .limit(100)
    )
    result = await session.execute(query)
    rows = result.scalars().all()
    return [
        AssociationRule(
            antecedents=row.antecedents,
            consequents=row.consequents,
            support=row.support,
            confidence=row.confidence,
            lift=row.lift,
        )
        for row in rows
    ]
