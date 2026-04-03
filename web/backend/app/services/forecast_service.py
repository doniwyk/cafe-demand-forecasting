from __future__ import annotations

from datetime import date

from sqlalchemy import select, func, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import (
    ModelRun,
    ModelRunClassMetric,
    ModelRunTopItem,
    Forecast,
    Item,
)
from app.models.forecast import (
    ForecastRecord,
    ForecastPage,
    ForecastSummary,
    ModelMetrics,
    ClassMetrics,
    TopItem,
    PredictRequest,
    PredictResponse,
)
from app.ml.engine import (
    generate_forecast,
    run_train_and_evaluate,
)


async def get_forecasts(
    session: AsyncSession,
    item: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    page: int = 1,
    page_size: int = 100,
) -> ForecastPage:
    query = (
        select(Forecast, Item.name)
        .join(Item)
        .join(ModelRun, onclause=Forecast.model_run_id == ModelRun.id)
        .where(ModelRun.is_active == True)
    )

    if item:
        query = query.where(Item.name == item)
    if start_date:
        query = query.where(Forecast.date >= date.fromisoformat(start_date))
    if end_date:
        query = query.where(Forecast.date <= date.fromisoformat(end_date))

    count_q = select(func.count()).select_from(query.subquery())
    total = (await session.execute(count_q)).scalar() or 0

    query = query.order_by(Forecast.date, Item.name)
    query = query.offset((page - 1) * page_size).limit(page_size)

    result = await session.execute(query)
    rows = result.all()

    return ForecastPage(
        data=[
            ForecastRecord(
                date=str(row.Forecast.date),
                item=row.name,
                quantity_sold=row.Forecast.quantity_predicted,
            )
            for row in rows
        ],
        total=total,
        page=page,
        page_size=page_size,
    )


async def get_forecast_summary(session: AsyncSession) -> ForecastSummary:
    run_q = (
        select(ModelRun)
        .where(ModelRun.is_active == True)
        .order_by(ModelRun.trained_at.desc())
        .limit(1)
    )
    run = (await session.execute(run_q)).scalar_one_or_none()
    if run is None:
        return ForecastSummary(
            global_metrics=ModelMetrics(r2=0, wmape=0, mae=0, volume_accuracy=0),
            class_metrics={},
            top_items=[],
        )

    class_q = select(ModelRunClassMetric).where(
        ModelRunClassMetric.model_run_id == run.id
    )
    class_rows = (await session.execute(class_q)).scalars().all()
    class_metrics = {
        row.abc_class: ClassMetrics(
            n_items=row.n_items, wmape=row.wmape, volume_accuracy=row.volume_accuracy
        )
        for row in class_rows
    }

    top_q = select(ModelRunTopItem).where(ModelRunTopItem.model_run_id == run.id)
    top_rows = (await session.execute(top_q)).scalars().all()
    top_items = [
        TopItem(
            item=row.item_name,
            quantity_sold=row.quantity_sold,
            predicted=row.predicted,
            accuracy_pct=row.accuracy_pct,
        )
        for row in top_rows
    ]

    return ForecastSummary(
        global_metrics=ModelMetrics(
            r2=run.r2 or 0,
            wmape=run.wmape or 0,
            mae=run.mae or 0,
            volume_accuracy=run.volume_accuracy or 0,
        ),
        class_metrics=class_metrics,
        top_items=top_items,
    )


async def predict_items(request: PredictRequest) -> PredictResponse:
    import pandas as pd
    from sqlalchemy import text

    from app.db.engine import async_session

    async with async_session() as session:
        query = text(
            "SELECT dis.date, i.name as item, dis.quantity_sold FROM daily_item_sales dis JOIN items i ON dis.item_id = i.id"
        )
        if request.items:
            placeholders = ", ".join(f":item_{i}" for i in range(len(request.items)))
            query = text(
                f"SELECT dis.date, i.name as item, dis.quantity_sold FROM daily_item_sales dis JOIN items i ON dis.item_id = i.id WHERE i.name IN ({placeholders})"
            )
            params = {f"item_{i}": item for i, item in enumerate(request.items)}
            result = await session.execute(query, params)
        else:
            result = await session.execute(query)
        rows = result.fetchall()

    df = pd.DataFrame(rows, columns=["Date", "Item", "Quantity_Sold"])
    df["Date"] = pd.to_datetime(df["Date"])
    result = generate_forecast(df, weeks=request.weeks)
    return PredictResponse(
        data=[
            ForecastRecord(
                date=str(row["Date"]),
                item=str(row["Item"]),
                quantity_sold=float(row["Predicted"]),
            )
            for _, row in result.iterrows()
        ],
        total=len(result),
    )


async def retrain(session: AsyncSession) -> dict:
    import pandas as pd
    from sqlalchemy import text

    query = text(
        "SELECT dis.date, i.name as item, dis.quantity_sold FROM daily_item_sales dis JOIN items i ON dis.item_id = i.id"
    )
    result = await session.execute(query)
    rows = result.fetchall()
    df = pd.DataFrame(rows, columns=["Date", "Item", "Quantity_Sold"])
    df["Date"] = pd.to_datetime(df["Date"])

    analysis = run_train_and_evaluate(df)
    return {
        "status": "success",
        "global_metrics": analysis["global_metrics"],
        "class_metrics": analysis["class_metrics"],
    }
