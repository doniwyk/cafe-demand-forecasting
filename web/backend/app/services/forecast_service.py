from __future__ import annotations

import asyncio
from datetime import date

import pandas as pd
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.engine import async_session
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
from app.ml.engine import generate_forecast
from app.repositories.forecast_repository import ForecastRepository

_forecast_cache: dict[str, pd.DataFrame] = {}


async def filter_sales_to_training_cutoff(
    session: AsyncSession, df: pd.DataFrame, model_type: str
) -> pd.DataFrame:
    from app.repositories.forecast_repository import ForecastRepository

    repo = ForecastRepository(session)
    run = await repo.get_active_run(model_type)
    if run and run.date_range_end:
        cutoff = pd.to_datetime(run.date_range_end)
        df = df[df["Date"] <= cutoff]
    return df


def invalidate_forecast_cache(model_type: str | None = None):
    if model_type:
        _forecast_cache.pop(model_type, None)
    else:
        _forecast_cache.clear()


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
    repo = ForecastRepository(session)
    df = await repo.get_sales_dataframe()

    if df.empty:
        return ForecastPage(data=[], total=0, page=page, page_size=page_size)

    df = await filter_sales_to_training_cutoff(session, df, model_type)

    df = _resample_daily(df)

    if df.empty:
        return ForecastPage(data=[], total=0, page=page, page_size=page_size)

    forecast_weeks = _compute_forecast_weeks(df, end_date)

    result_df = await _get_or_generate_forecast(df, forecast_weeks, model_type)
    result_df = _filter_forecast(result_df, start_date, end_date, item)

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
    repo = ForecastRepository(session)
    run = await repo.get_active_run(model_type)
    if run is None:
        return ForecastSummary(
            global_metrics=ModelMetrics(r2=0, wmape=0, mae=0, rmse=0),
            class_metrics={},
            top_items=[],
        )

    class_rows = await repo.get_class_metrics(run.id)
    class_metrics = {
        row.abc_class: ClassMetrics(
            n_items=row.n_items,
            wmape=row.wmape,
            median_period_accuracy=row.median_period_accuracy or row.volume_accuracy or 0,
        )
        for row in class_rows
    }

    top_rows = await repo.get_top_items(run.id)
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
            rmse=run.rmse or 0,
            median_period_accuracy=run.median_period_accuracy or run.volume_accuracy or 0,
            periods_within_20pct=run.periods_within_20pct or 0,
            periods_within_50pct=run.periods_within_50pct or 0,
        ),
        class_metrics=class_metrics,
        top_items=top_items,
    )


async def predict_items(request: PredictRequest) -> PredictResponse:
    async with async_session() as session:
        repo = ForecastRepository(session)
        df = await repo.get_sales_dataframe(items=request.items)

    df = _resample_daily(df)

    def _run():
        return generate_forecast(df, weeks=request.weeks, model_type=request.model_type)

    result = await asyncio.to_thread(_run)
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


def _resample_daily(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    return (
        df.set_index("Date")
        .groupby("Item")
        .resample("D")["Quantity_Sold"]
        .sum()
        .fillna(0)
        .reset_index()
    )


def _compute_forecast_weeks(df: pd.DataFrame, end_date: str | None) -> int:
    if end_date:
        end_dt = pd.to_datetime(end_date)
        days_needed = max(14, (end_dt - df["Date"].max()).days + 7)
        return int(days_needed / 7) + 1
    return 4


def _filter_forecast(
    result_df: pd.DataFrame,
    start_date: str | None,
    end_date: str | None,
    item: str | None,
) -> pd.DataFrame:
    if result_df.empty:
        return result_df

    if start_date and end_date:
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        forecast_first = result_df["Date"].min() if len(result_df) > 0 else None
        if forecast_first and end_dt < forecast_first:
            return result_df
        result_df = result_df[
            (pd.to_datetime(result_df["Date"]) >= start_dt)
            & (pd.to_datetime(result_df["Date"]) <= end_dt)
        ]
    elif start_date:
        start_dt = pd.to_datetime(start_date)
        result_df = result_df[pd.to_datetime(result_df["Date"]) >= start_dt]
    elif end_date:
        end_dt = pd.to_datetime(end_date)
        result_df = result_df[pd.to_datetime(result_df["Date"]) <= end_dt]

    if item:
        result_df = result_df[result_df["Item"] == item]

    return result_df.sort_values(["Date", "Item"])


async def _get_or_generate_forecast(
    df: pd.DataFrame, forecast_weeks: int, model_type: str
) -> pd.DataFrame:
    cache_key = model_type
    cached = _forecast_cache.get(cache_key)

    if cached is not None:
        cache_max = cached["Date"].max()
        needed_max = df["Date"].max() + pd.Timedelta(weeks=forecast_weeks)
        if cache_max >= needed_max:
            return cached.copy()

    def _run():
        return generate_forecast(df, weeks=max(forecast_weeks, 8), model_type=model_type)

    result_df = await asyncio.to_thread(_run)
    _forecast_cache[cache_key] = result_df.copy()
    return result_df
