import pandas as pd

from app.config import (
    DAILY_ITEM_SALES_PATH,
    DAILY_CATEGORY_SALES_PATH,
    DAILY_TOTAL_SALES_PATH,
    MENU_BOM_PATH,
)
from app.models.sales import (
    DailySale,
    DailySalePage,
    DailyTotalSale,
    DailyCategorySale,
    ItemInfo,
)


_df_cache: dict[str, pd.DataFrame] = {}


def _load(path) -> pd.DataFrame:
    key = str(path)
    if key not in _df_cache:
        _df_cache[key] = pd.read_csv(path)
    return _df_cache[key]


def get_daily_sales(
    item: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    page: int = 1,
    page_size: int = 100,
) -> DailySalePage:
    df = _load(DAILY_ITEM_SALES_PATH)
    df = _apply_filters(df, item, start_date, end_date)
    total = len(df)
    df = df.iloc[(page - 1) * page_size : page * page_size]
    return DailySalePage(
        data=[
            DailySale(
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


def get_daily_total_sales(
    start_date: str | None = None,
    end_date: str | None = None,
    page: int = 1,
    page_size: int = 100,
) -> list[DailyTotalSale]:
    df = _load(DAILY_TOTAL_SALES_PATH)
    df = _apply_date_filters(df, start_date, end_date)
    df = df.iloc[(page - 1) * page_size : page * page_size]
    return [
        DailyTotalSale(
            date=str(row["Date"]),
            quantity=float(row["Quantity"]),
            net_sales=float(row["Net sales"]),
            gross_sales=float(row["Gross sales"]),
            unique_items=int(row["UniqueItemCount"]),
        )
        for _, row in df.iterrows()
    ]


def get_daily_category_sales(
    category: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    page: int = 1,
    page_size: int = 100,
) -> list[DailyCategorySale]:
    df = _load(DAILY_CATEGORY_SALES_PATH)
    if category:
        df = df[df["Category"] == category]
    df = _apply_date_filters(df, start_date, end_date)
    df = df.iloc[(page - 1) * page_size : page * page_size]
    return [
        DailyCategorySale(
            date=str(row["Date"]),
            category=str(row["Category"]),
            quantity=float(row["Quantity"]),
            net_sales=float(row["Net sales"]),
            gross_sales=float(row["Gross sales"]),
            unique_items=int(row["UniqueItemCount"]),
        )
        for _, row in df.iterrows()
    ]


def get_items() -> list[ItemInfo]:
    df = _load(DAILY_ITEM_SALES_PATH)
    bom_df = pd.read_csv(MENU_BOM_PATH)
    category_map = dict(zip(bom_df["Item"].str.strip(), bom_df["Tipe"].str.strip()))
    unique_items = sorted(df["Item"].unique())
    return [
        ItemInfo(name=item, category=category_map.get(item)) for item in unique_items
    ]


def get_categories() -> list[str]:
    df = _load(DAILY_CATEGORY_SALES_PATH)
    return sorted(df["Category"].dropna().unique())


def _apply_filters(df, item, start_date, end_date) -> pd.DataFrame:
    if item:
        df = df[df["Item"] == item]
    return _apply_date_filters(df, start_date, end_date)


def _apply_date_filters(df, start_date, end_date) -> pd.DataFrame:
    if start_date:
        df = df[df["Date"] >= start_date]
    if end_date:
        df = df[df["Date"] <= end_date]
    return df
