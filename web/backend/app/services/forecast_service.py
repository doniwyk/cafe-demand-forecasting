from __future__ import annotations

import asyncio
from datetime import date

import pandas as pd
from sqlalchemy import select, func, update, text
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
    model_type: str | None = None,
) -> ForecastPage:
    model_type = model_type or "xgboost"

    query = text(
        "SELECT dis.date, i.name as item, dis.quantity_sold "
        "FROM daily_item_sales dis JOIN items i ON dis.item_id = i.id"
    )
    result = await session.execute(query)
    rows = result.fetchall()

    if not rows:
        return ForecastPage(data=[], total=0, page=page, page_size=page_size)

    df = pd.DataFrame(
        [tuple(row) for row in rows], columns=["Date", "Item", "Quantity_Sold"]
    )
    df["Date"] = pd.to_datetime(df["Date"])

    if item:
        df = df[df["Item"] == item]

    if df.empty:
        return ForecastPage(data=[], total=0, page=page, page_size=page_size)

    def _run_forecast():
        return generate_forecast(df, weeks=12, model_type=model_type)

    result_df = await asyncio.to_thread(_run_forecast)

    if start_date:
        result_df = result_df[pd.to_datetime(result_df["Date"]) >= pd.to_datetime(start_date)]
    if end_date:
        result_df = result_df[pd.to_datetime(result_df["Date"]) <= pd.to_datetime(end_date)]

    result_df = result_df.sort_values(["Date", "Item"])
    total = len(result_df)

    result_df = result_df.iloc[(page - 1) * page_size : page * page_size]

    return ForecastPage(
        data=[
            ForecastRecord(
                date=str(pd.to_datetime(row["Date"]).date()),
                item=str(row["Item"]),
                quantity_sold=float(row["Predicted"]),
            )
            for _, row in result_df.iterrows()
        ],
        total=total,
        page=page,
        page_size=page_size,
    )


async def get_forecast_summary(
    session: AsyncSession, model_type: str | None = None
) -> ForecastSummary:
    run_q = select(ModelRun).where(ModelRun.is_active == True)
    if model_type:
        run_q = run_q.where(ModelRun.model_type == model_type)
    run_q = run_q.order_by(ModelRun.trained_at.desc()).limit(1)
    run = (await session.execute(run_q)).scalar_one_or_none()
    if run is None:
        return ForecastSummary(
            global_metrics=ModelMetrics(r2=0, wmape=0, mae=0),
            class_metrics={},
            top_items=[],
        )

    class_q = select(ModelRunClassMetric).where(
        ModelRunClassMetric.model_run_id == run.id
    )
    class_rows = (await session.execute(class_q)).scalars().all()
    class_metrics = {
        row.abc_class: ClassMetrics(
            n_items=row.n_items,
            wmape=row.wmape,
            median_period_accuracy=row.median_period_accuracy or row.volume_accuracy or 0,
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
            median_period_accuracy=run.median_period_accuracy or run.volume_accuracy or 0,
            periods_within_20pct=run.periods_within_20pct or 0,
            periods_within_50pct=run.periods_within_50pct or 0,
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

    df = pd.DataFrame(
        [tuple(row) for row in rows], columns=["Date", "Item", "Quantity_Sold"]
    )
    df["Date"] = pd.to_datetime(df["Date"])

    def _run_forecast():
        return generate_forecast(df, weeks=request.weeks, model_type=request.model_type)

    result = await asyncio.to_thread(_run_forecast)
    return PredictResponse(
        data=[
            ForecastRecord(
                date=str(pd.to_datetime(row["Date"]).date()),
                item=str(row["Item"]),
                quantity_sold=float(row["Predicted"]),
            )
            for _, row in result.iterrows()
        ],
        total=len(result),
    )


async def retrain(session: AsyncSession, model_type: str = "xgboost") -> dict:
    import pandas as pd
    import json as _json
    from datetime import datetime as _dt
    from sqlalchemy import text, update

    from app.db.models import (
        ModelRun,
        ModelRunClassMetric,
        ModelRunTopItem,
        Forecast,
    )

    query = text(
        "SELECT dis.date, i.name as item, dis.quantity_sold FROM daily_item_sales dis JOIN items i ON dis.item_id = i.id"
    )
    result = await session.execute(query)
    rows = result.fetchall()
    df = pd.DataFrame(
        [tuple(row) for row in rows], columns=["Date", "Item", "Quantity_Sold"]
    )
    df["Date"] = pd.to_datetime(df["Date"])

    from app.ml.engine import (
        run_train_and_evaluate,
        generate_forecast,
        _METADATA_FILE,
        ML_MODELS_DIR,
    )

    analysis = run_train_and_evaluate(df, model_type=model_type)

    await session.execute(
        update(ModelRun)
        .where(ModelRun.model_type == model_type)
        .values(is_active=False)
    )

    meta_path = ML_MODELS_DIR / _METADATA_FILE.get(model_type, "model_metadata.json")
    meta = {}
    if meta_path.exists():
        with open(meta_path) as f:
            meta = _json.load(f)

    gm = analysis["global_metrics"]
    run = ModelRun(
        model_type=model_type,
        trained_at=_dt.now(),
        n_item_models=meta.get("n_item_models"),
        n_records=meta.get("n_records"),
        date_range_start=pd.to_datetime(meta["date_range"][0]).date()
        if meta.get("date_range")
        else None,
        date_range_end=pd.to_datetime(meta["date_range"][1]).date()
        if meta.get("date_range")
        else None,
        r2=gm.get("r2"),
        wmape=gm.get("wmape"),
        mae=gm.get("mae"),
        median_period_accuracy=gm.get("median_period_accuracy"),
        periods_within_20pct=gm.get("periods_within_20pct"),
        periods_within_50pct=gm.get("periods_within_50pct"),
        features=_json.dumps(meta.get("features", [])),
        items_with_models=_json.dumps(meta.get("items_with_models", [])),
        is_active=True,
    )
    session.add(run)
    await session.flush()

    for cls, cm in analysis.get("class_metrics", {}).items():
        session.add(
            ModelRunClassMetric(
                model_run_id=run.id,
                abc_class=cls,
                n_items=cm.get("n_items", 0),
                wmape=cm.get("wmape", 0),
                median_period_accuracy=cm.get("median_period_acc", 0),
            )
        )

    for t in analysis.get("top_items", []):
        session.add(
            ModelRunTopItem(
                model_run_id=run.id,
                item_name=t["Item"],
                quantity_sold=t["Quantity_Sold"],
                predicted=t["Predicted"],
                accuracy_pct=t["accuracy_pct"],
            )
        )

    forecast_result = generate_forecast(df, weeks=12, model_type=model_type)

    item_rows = await session.execute(text("SELECT id, name FROM items"))
    item_name_to_id = {row[1]: row[0] for row in item_rows.fetchall()}

    for _, row in forecast_result.iterrows():
        item_name = str(row["Item"])
        item_id = item_name_to_id.get(item_name)
        if item_id is None:
            continue
        session.add(
            Forecast(
                model_run_id=run.id,
                item_id=item_id,
                date=pd.to_datetime(row["Date"]).date(),
                quantity_predicted=float(row["Predicted"]),
            )
        )

    await session.commit()
    print(f"Model run saved to DB (type={model_type}, id={run.id})")

    return {
        "status": "success",
        "global_metrics": analysis["global_metrics"],
        "class_metrics": analysis["class_metrics"],
    }
