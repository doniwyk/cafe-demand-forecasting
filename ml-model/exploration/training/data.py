"""Shared data loading and preprocessing for training pipelines."""
from __future__ import annotations

import os
import pandas as pd
import numpy as np
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import FEATURE_COLUMNS, SALES_FORECASTING_DIR, DISCONTINUED_ITEMS
from features import create_features

MIN_TRAIN_RECORDS = 60
VAL_RATIO = 0.15
N_FOLDS = 3
TRAIN_MONTHS = 24  # None = use all data, or set to 12/24 for last 1/2 years

HUS_DB_URL = os.getenv("HUS_DB_URL", "postgresql://user:password@localhost:5432/hus_db")
CAFE_DB_URL = os.getenv(
    "CAFE_DB_URL",
    "postgresql://postgres:postgres@localhost:5433/cafe_forecasting",
)

SKIP_PREFIXES = [
    "Add ",
    "Filter",
    "FIlter",
    "V60",
    "Harum Jasmine Tea",
    "Cookies Redvelvet",
    "Lotus Cheesecake",
    "Strawberry Cheesecake",
    "Kopi Susu Bersemi",
]


def _should_skip(item_name: str) -> bool:
    for prefix in SKIP_PREFIXES:
        if item_name.startswith(prefix):
            return True
    return False


def load_from_hus_db() -> pd.DataFrame | None:
    try:
        import psycopg2
    except ImportError:
        print("psycopg2 not installed, cannot connect to hus_db")
        return None

    try:
        conn = psycopg2.connect(HUS_DB_URL)
    except Exception as e:
        print(f"Cannot connect to hus_db: {e}")
        return None

    print(f"Connected to hus_db, fetching sales data...")

    since = "2022-01-01"
    query = """
        SELECT
            DATE(o.created_at) as sale_date,
            oi.product_name_snapshot,
            oi.variant_name_snapshot,
            SUM(oi.quantity) as total_qty
        FROM order_items oi
        JOIN orders o ON oi.order_id = o.id
        WHERE o.status = 'PAID'
          AND o.created_at >= %s
        GROUP BY DATE(o.created_at), oi.product_name_snapshot, oi.variant_name_snapshot
        ORDER BY sale_date
    """
    cur = conn.cursor()
    cur.execute(query, (since,))
    rows = cur.fetchall()
    cur.close()
    conn.close()

    print(f"Fetched {len(rows)} rows from hus_db (since {since})")

    if not rows:
        return None

    matched_rows = []
    skipped = {}
    for sale_date, product_name, variant_name, qty in rows:
        item = f"{product_name} {variant_name or ''}".strip()
        if _should_skip(item):
            skipped[item] = skipped.get(item, 0) + int(qty)
            continue
        matched_rows.append({"Date": sale_date, "Item": item, "Quantity_Sold": int(qty)})

    if skipped:
        print(f"Skipped {sum(skipped.values())} units across {len(skipped)} products")

    df = pd.DataFrame(matched_rows)
    df["Date"] = pd.to_datetime(df["Date"])
    return df


def load_from_cafe_db() -> pd.DataFrame | None:
    try:
        import psycopg2
    except ImportError:
        return None

    try:
        conn = psycopg2.connect(CAFE_DB_URL)
    except Exception:
        return None

    query = """
        SELECT d.date, i.name AS item, d.quantity_sold
        FROM daily_item_sales d
        JOIN items i ON d.item_id = i.id
        ORDER BY d.date, i.name
    """
    cur = conn.cursor()
    cur.execute(query)
    rows = cur.fetchall()
    col_names = [desc[0] for desc in cur.description]
    cur.close()
    conn.close()

    if not rows:
        return None

    df = pd.DataFrame(rows, columns=col_names)
    df.rename(columns={"date": "Date_Only", "item": "Item", "quantity_sold": "Quantity_Sold"}, inplace=True)

    df = df[~df["Item"].apply(_should_skip)].copy()
    df = df[~df["Item"].isin(DISCONTINUED_ITEMS)].copy()

    return df


def load_and_prep_data(filepath: str | Path) -> pd.DataFrame:
    csv_path = SALES_FORECASTING_DIR / "daily_item_sales.csv"
    print(f"Loading data from: {csv_path}")
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    date_col = "Date_Only" if "Date_Only" in df.columns else "Date"
    qty_col = "Quantity" if "Quantity" in df.columns else "Quantity_Sold"
    df["Date"] = pd.to_datetime(df[date_col])
    df["Quantity_Sold"] = df[qty_col]

    df["Date"] = pd.to_datetime(df["Date_Only"])
    df["Quantity_Sold"] = df["Quantity_Sold"].astype(int)

    df = df[~df["Item"].str.strip().str.lower().str.startswith("add")]

    df_freq = (
        df.set_index("Date")
        .groupby("Item")
        .resample("D")["Quantity_Sold"]
        .sum()
        .reset_index()
    )

    print(f"Aggregated to daily: {len(df_freq):,} observations")
    print(f"Date range: {df_freq['Date'].min().date()} to {df_freq['Date'].max().date()}")
    return df_freq


def load_data() -> pd.DataFrame:
    cafe_df = load_from_cafe_db()
    if cafe_df is not None:
        print("Using cafe_db data")
        cafe_df["Date"] = pd.to_datetime(cafe_df["Date_Only"])
        return cafe_df

    hus_df = load_from_hus_db()
    if hus_df is not None:
        print("Using hus_db data")
        hus_df["Date"] = pd.to_datetime(hus_df["Date_Only"])
        return hus_df

    csv_path = SALES_FORECASTING_DIR / "daily_item_sales.csv"
    print(f"No database connection, falling back to CSV: {csv_path}")
    return load_and_prep_data(csv_path)


def create_full_grid(df: pd.DataFrame) -> pd.DataFrame:
    """Create full item×date grid, filling missing days with 0 sales."""
    items = df["Item"].unique()
    dates = pd.date_range(df["Date"].min(), df["Date"].max(), freq="D")
    grid = pd.MultiIndex.from_product([items, dates], names=["Item", "Date"]).to_frame(index=False)
    df_full = grid.merge(df, on=["Item", "Date"], how="left")
    df_full["Quantity_Sold"] = df_full["Quantity_Sold"].fillna(0).astype(int)
    print(f"Full grid: {len(items)} items × {len(dates)} dates = {len(df_full):,} rows "
          f"(was {len(df):,}, added {len(df_full) - len(df):,} zero-sale rows)")
    return df_full


def prepare_features(df: pd.DataFrame, use_full_grid: bool = True) -> pd.DataFrame:
    if use_full_grid:
        df = create_full_grid(df)
    df_feat = create_features(df)
    return df_feat


def filter_recent_data(df: pd.DataFrame, months: int | None = None) -> pd.DataFrame:
    """Filter data to recent N months if specified."""
    if months is None:
        return df

    max_date = df["Date"].max()
    cutoff_date = max_date - pd.DateOffset(months=months)
    df_filtered = df[df["Date"] >= cutoff_date].copy()
    print(f"Filtered to last {months} months: {df_filtered['Date'].min().date()} to {df_filtered['Date'].max().date()}")
    return df_filtered


def get_feature_columns(df_feat: pd.DataFrame) -> list:
    return [f for f in FEATURE_COLUMNS if f in df_feat.columns]


def split_train_val(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_parts = []
    val_parts = []
    for item in df["Item"].unique():
        item_df = df[df["Item"] == item].sort_values("Date")
        n_val = max(1, int(len(item_df) * VAL_RATIO))
        train_parts.append(item_df.iloc[: len(item_df) - n_val])
        val_parts.append(item_df.iloc[len(item_df) - n_val :])
    return pd.concat(train_parts, ignore_index=True), pd.concat(val_parts, ignore_index=True)


def time_series_cv(df: pd.DataFrame, n_folds: int = N_FOLDS) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """Expanding window time series cross-validation."""
    dates = sorted(df["Date"].unique())
    n_dates = len(dates)
    fold_size = n_dates // (n_folds + 1)

    folds = []
    for i in range(n_folds):
        train_end = fold_size * (i + 1)
        val_end = min(fold_size * (i + 2), n_dates)

        train_dates = dates[:train_end]
        val_dates = dates[train_end:val_end]

        train_df = df[df["Date"].isin(train_dates)]
        val_df = df[df["Date"].isin(val_dates)]

        folds.append((train_df, val_df))

    return folds


def compute_dow_factors(df: pd.DataFrame) -> dict:
    dow_pattern = (
        df.groupby(["Item", df["Date"].dt.weekday])["Quantity_Sold"]
        .mean()
        .reset_index()
    )
    item_avg = (
        df.groupby("Item")["Quantity_Sold"]
        .mean()
        .reset_index()
        .rename(columns={"Quantity_Sold": "item_avg"})
    )
    dow_pattern = dow_pattern.merge(item_avg, on="Item")
    dow_pattern["dow_factor"] = dow_pattern["Quantity_Sold"] / dow_pattern["item_avg"]
    return (
        dow_pattern.pivot(index="Item", columns="Date", values="dow_factor")
        .fillna(1.0)
        .to_dict("index")
    )
