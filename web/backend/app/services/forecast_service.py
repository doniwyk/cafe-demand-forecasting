import pandas as pd

from app.config import FORECAST_PATH, DAILY_ITEM_SALES_PATH
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
    run_evaluate,
    generate_forecast,
    run_train_and_evaluate,
    get_model_metadata,
)


def get_forecasts(
    item: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    page: int = 1,
    page_size: int = 100,
) -> ForecastPage:
    df = pd.read_csv(FORECAST_PATH)
    if item:
        df = df[df["Item"] == item]
    if start_date:
        df = df[df["Date"] >= start_date]
    if end_date:
        df = df[df["Date"] <= end_date]
    total = len(df)
    df = df.iloc[(page - 1) * page_size : page * page_size]
    return ForecastPage(
        data=[
            ForecastRecord(
                date=str(row["Date"]),
                item=str(row["Item"]),
                quantity_sold=float(row["Quantity_Sold"]),
            )
            for _, row in df.iterrows()
        ],
        total=total,
        page=page,
        page_size=page_size,
    )


def get_forecast_summary() -> ForecastSummary:
    df_daily = pd.read_csv(DAILY_ITEM_SALES_PATH)
    analysis = run_evaluate(df_daily)

    class_metrics = {}
    for cls, metrics in analysis["class_metrics"].items():
        class_metrics[cls] = ClassMetrics(
            n_items=metrics["n_items"],
            wmape=metrics["wmape"],
            volume_accuracy=metrics["volume_accuracy"],
        )

    top_items = [
        TopItem(
            item=t["Item"],
            quantity_sold=float(t["Quantity_Sold"]),
            predicted=float(t["Predicted"]),
            accuracy_pct=float(t["accuracy_pct"]),
        )
        for t in analysis["top_items"]
    ]

    gm = analysis["global_metrics"]
    return ForecastSummary(
        global_metrics=ModelMetrics(
            r2=gm["r2"],
            wmape=gm["wmape"],
            mae=gm["mae"],
            volume_accuracy=gm["volume_accuracy"],
        ),
        class_metrics=class_metrics,
        top_items=top_items,
    )


def predict_items(request: PredictRequest) -> PredictResponse:
    df_daily = pd.read_csv(DAILY_ITEM_SALES_PATH)
    if request.items:
        df_daily = df_daily[df_daily["Item"].isin(request.items)]
    result = generate_forecast(df_daily, weeks=request.weeks)
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


def retrain() -> dict:
    df_daily = pd.read_csv(DAILY_ITEM_SALES_PATH)
    analysis = run_train_and_evaluate(df_daily)
    metadata = get_model_metadata()
    return {
        "status": "success",
        "global_metrics": analysis["global_metrics"],
        "class_metrics": analysis["class_metrics"],
        "model_metadata": metadata,
    }
