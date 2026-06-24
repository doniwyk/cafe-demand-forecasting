from __future__ import annotations

import asyncio

import pandas as pd
from sqlalchemy.ext.asyncio import AsyncSession

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

_forecast_cache: pd.DataFrame | None = None


async def filter_sales_to_training_cutoff(
    session: AsyncSession, df: pd.DataFrame
) -> pd.DataFrame:
    repo = ForecastRepository(session)
    run = await repo.get_active_run()
    if run and run.date_range_end:
        cutoff = pd.to_datetime(run.date_range_end)
        df = df[df["Date"] <= cutoff]
    return df


def invalidate_forecast_cache():
    global _forecast_cache
    _forecast_cache = None


async def get_forecasts(
    session: AsyncSession,
    item: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    page: int = 1,
    page_size: int = 100,
) -> ForecastPage:
    repo = ForecastRepository(session)
    original_df = await repo.get_sales_dataframe()

    if original_df.empty:
        return ForecastPage(data=[], total=0, page=page, page_size=page_size)

    df = original_df.copy()
    df = await filter_sales_to_training_cutoff(session, df)
    df = _resample_daily(df)

    if df.empty:
        return ForecastPage(data=[], total=0, page=page, page_size=page_size)

    forecast_weeks = _compute_forecast_weeks(df, end_date)
    result_df = await _get_or_generate_forecast(df, forecast_weeks)
    result_df = _filter_forecast(result_df, start_date, end_date, item)

    actual_df = original_df.copy()
    if start_date:
        actual_df = actual_df[actual_df["Date"] >= pd.to_datetime(start_date)]
    if end_date:
        actual_df = actual_df[actual_df["Date"] <= pd.to_datetime(end_date)]
    if item:
        actual_df = actual_df[actual_df["Item"] == item]

    pred_data = result_df[["Date", "Item", "Predicted", "Error_Std", "Buffer", "Supply"]].copy()
    actual_data = actual_df[["Date", "Item", "Quantity_Sold"]].copy()

    merged = pd.merge(
        pred_data,
        actual_data,
        on=["Date", "Item"],
        how="outer",
    )
    merged["Predicted"] = merged["Predicted"].fillna(0)
    merged["Quantity_Sold"] = merged["Quantity_Sold"].fillna(0)
    merged["Error_Std"] = merged["Error_Std"].fillna(0)
    merged["Buffer"] = merged["Buffer"].fillna(0)
    merged["Supply"] = merged["Supply"].fillna(0)
    merged = merged.sort_values(["Date", "Item"])

    total = len(merged)
    merged = merged.iloc[(page - 1) * page_size : page * page_size]

    return ForecastPage(
        data=[
            ForecastRecord(
                date=str(pd.to_datetime(row["Date"]).date()),
                item=str(row["Item"]),
                quantity_sold=float(row["Predicted"]),
                actual=float(row["Quantity_Sold"]),
                error_std=float(row["Error_Std"]),
                buffer=float(row["Buffer"]),
                supply=float(row["Supply"]),
            )
            for _, row in merged.iterrows()
        ],
        total=total,
        page=page,
        page_size=page_size,
    )


async def get_forecast_summary(
    session: AsyncSession,
) -> ForecastSummary:
    repo = ForecastRepository(session)
    run = await repo.get_active_run()
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
            r2=row.r2 or 0,
            mae=row.mae or 0,
            rmse=row.rmse or 0,
            median_period_accuracy=row.median_period_accuracy or row.volume_accuracy or 0,
            periods_within_20pct=row.periods_within_20pct or 0,
            periods_within_50pct=row.periods_within_50pct or 0,
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
        latest_training_date=str(run.date_range_end) if run.date_range_end else None,
    )


async def predict_items(request: PredictRequest) -> PredictResponse:
    from app.db.engine import async_session
    async with async_session() as session:
        repo = ForecastRepository(session)
        df = await repo.get_sales_dataframe(items=request.items)

    df = _resample_daily(df)

    def _run():
        return generate_forecast(df, weeks=request.weeks)

    result = await asyncio.to_thread(_run)
    return PredictResponse(
        data=[
            ForecastRecord(
                date=str(pd.to_datetime(row["Date"]).date()),
                item=str(row["Item"]),
                quantity_sold=float(row["Predicted"]),
                error_std=float(row.get("Error_Std", 0)),
                buffer=float(row.get("Buffer", 0)),
                supply=float(row.get("Supply", 0)),
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
    df: pd.DataFrame, forecast_weeks: int
) -> pd.DataFrame:
    global _forecast_cache

    cached = _forecast_cache
    if cached is not None:
        cache_max = cached["Date"].max()
        needed_max = df["Date"].max() + pd.Timedelta(weeks=forecast_weeks)
        if cache_max >= needed_max:
            return cached.copy()

    def _run():
        return generate_forecast(df, weeks=max(forecast_weeks, 8))

    result_df = await asyncio.to_thread(_run)
    _forecast_cache = result_df.copy()
    return result_df
