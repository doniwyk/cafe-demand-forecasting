"""
Data Exploration
================
Explore the raw sales data structure, patterns, and distributions.
Fetches directly from hus_db (POS) when available, falls back to CSV.
Generates plots to reports/figures/.

Run from ml-model/: python exploration/eda/data_exploration.py
"""

import os
import sys
from pathlib import Path
from datetime import date, timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures" / "data_exploration"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import SALES_FORECASTING_DIR, DISCONTINUED_ITEMS

SEPARATOR = "=" * 80

HUS_DB_URL = os.getenv("HUS_DB_URL", "postgresql://user:password@localhost:5432/hus_db")

SKIP_PREFIXES = [
    "Add ", "Filter", "FIlter", "V60", "Harum Jasmine Tea",
    "Cookies Redvelvet", "Lotus Cheesecake", "Strawberry Cheesecake",
    "Kopi Susu Bersemi",
]
FALLBACK_VARIANT_MAP = {
    "Kopi Susu Husgendam": "Kopi Susu Husgendam Ice",
    "Cappucino": "Cappucino Ice",
}


def _should_skip(name: str) -> bool:
    return any(name.startswith(p) for p in SKIP_PREFIXES)


def _match_item(product_name: str, variant_name: str | None, cafe_items: set) -> str | None:
    product_name = (product_name or "").strip()
    variant_name = (variant_name or "").strip() if variant_name else None

    combined = f"{product_name} {variant_name}".strip() if variant_name else product_name
    if combined in cafe_items:
        return combined

    if product_name in cafe_items:
        return product_name

    if product_name in FALLBACK_VARIANT_MAP:
        fallback = FALLBACK_VARIANT_MAP[product_name]
        if fallback in cafe_items:
            return fallback

    return None


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

    csv_path = SALES_FORECASTING_DIR / "daily_item_sales.csv"
    if csv_path.exists():
        csv_df = pd.read_csv(csv_path)
        csv_df.columns = csv_df.columns.str.strip()
        date_col = "Date_Only" if "Date_Only" in csv_df.columns else "Date"
        last_date = pd.to_datetime(csv_df[date_col]).max()
        since = (last_date + timedelta(days=1)).strftime("%Y-%m-%d")
        cafe_items = set(csv_df["Item"].str.strip().unique())
    else:
        since = "2022-01-01"
        cafe_items = set()

    cur = conn.cursor()
    cur.execute("""
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
    """, (since,))
    rows = cur.fetchall()
    cur.close()
    conn.close()

    print(f"Fetched {len(rows)} rows from hus_db (since {since})")

    if not rows:
        if csv_path.exists():
            print("No new data, using CSV")
            return None
        print("No data in hus_db and no CSV found")
        return None

    matched_rows = []
    skipped = {}
    for sale_date, product_name, variant_name, qty in rows:
        item = _match_item(product_name, variant_name, cafe_items)
        if item is None:
            key = f"{product_name} {variant_name or ''}".strip()
            skipped[key] = skipped.get(key, 0) + int(qty)
            continue
        matched_rows.append({"Date": sale_date, "Item": item, "Quantity_Sold": int(qty)})

    print(f"Matched: {len(matched_rows)} rows")
    if skipped:
        print(f"Skipped: {sum(skipped.values())} units ({len(skipped)} products)")

    hus_df = pd.DataFrame(matched_rows)
    hus_df["Date"] = pd.to_datetime(hus_df["Date"])

    if csv_path.exists():
        csv_df = pd.read_csv(csv_path)
        csv_df.columns = csv_df.columns.str.strip()
        date_col = "Date_Only" if "Date_Only" in csv_df.columns else "Date"
        csv_df["Date"] = pd.to_datetime(csv_df[date_col])
        csv_df = csv_df.rename(columns={"Quantity": "Quantity_Sold"})
        csv_df = csv_df[~csv_df["Item"].str.strip().str.lower().str.startswith("add")]

        if "Category" in csv_df.columns:
            cat_map = csv_df.drop_duplicates("Item").set_index("Item")["Category"].to_dict()
            hus_df["Category"] = hus_df["Item"].map(cat_map)

        combined = pd.concat([csv_df, hus_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["Date", "Item"], keep="last")
        combined = combined.sort_values(["Date", "Item"]).reset_index(drop=True)
        print(f"Combined: {len(combined)} rows ({csv_df.shape[0]} CSV + {len(hus_df)} new)")
        return combined
    else:
        return hus_df


def load_data() -> pd.DataFrame:
    hus_df = load_from_hus_db()
    if hus_df is not None:
        return hus_df

    filepath = SALES_FORECASTING_DIR / "daily_item_sales.csv"
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()

    date_col = "Date_Only" if "Date_Only" in df.columns else "Date"
    df["Date"] = pd.to_datetime(df[date_col])
    df = df.rename(columns={"Quantity": "Quantity_Sold"})

    df = df[~df["Item"].str.strip().str.lower().str.startswith("add")]
    return df


def section_dataset_overview(df: pd.DataFrame):
    print(f"\n{SEPARATOR}")
    print("DATASET OVERVIEW")
    print(SEPARATOR)

    date_range_days = (df["Date"].max() - df["Date"].min()).days + 1
    unique_dates = df["Date"].dt.date.nunique()

    print(f"Total rows:             {len(df):,}")
    print(f"Date range:             {df['Date'].min().date()} -> {df['Date'].max().date()}")
    print(f"Span (calendar days):   {date_range_days}")
    print(f"Days with data:         {unique_dates}")
    print(f"Missing days:           {date_range_days - unique_dates}")
    print(f"Unique items:           {df['Item'].nunique()}")
    print(f"Total quantity sold:    {df['Quantity_Sold'].sum():,.0f}")
    print(f"Avg qty per row:        {df['Quantity_Sold'].mean():.2f}")
    print(f"\nColumns: {list(df.columns)}")


def section_temporal_patterns(df: pd.DataFrame):
    print(f"\n{SEPARATOR}")
    print("TEMPORAL PATTERNS")
    print(SEPARATOR)

    daily = df.groupby("Date")["Quantity_Sold"].sum()

    print(f"\nDaily aggregate stats:")
    print(f"  Mean:   {daily.mean():.1f}")
    print(f"  Median: {daily.median():.1f}")
    print(f"  Std:    {daily.std():.1f}")
    print(f"  Min:    {daily.min():.0f} ({daily.idxmin().date()})")
    print(f"  Max:    {daily.max():.0f} ({daily.idxmax().date()})")

    df["DOW"] = df["Date"].dt.dayofweek
    dow_avg = df.groupby("DOW")["Quantity_Sold"].mean()
    dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    print(f"\nAverage quantity by day of week:")
    for d in range(7):
        bar = "#" * int(dow_avg.get(d, 0) / dow_avg.max() * 30)
        print(f"  {dow_names[d]}: {dow_avg.get(d, 0):6.1f}  {bar}")

    df["Month"] = df["Date"].dt.month
    month_avg = df.groupby("Month")["Quantity_Sold"].mean()
    print(f"\nAverage quantity by month:")
    for m in range(1, 13):
        val = month_avg.get(m, 0)
        bar = "#" * int(val / month_avg.max() * 30)
        print(f"  {m:2d}: {val:6.1f}  {bar}")

    df["Year"] = df["Date"].dt.year
    year_avg = df.groupby("Year")["Quantity_Sold"].sum()
    print(f"\nTotal quantity by year:")
    for y, v in year_avg.items():
        print(f"  {y}: {v:,.0f}")


def section_item_analysis(df: pd.DataFrame):
    print(f"\n{SEPARATOR}")
    print("ITEM ANALYSIS")
    print(SEPARATOR)

    item_vol = df.groupby("Item")["Quantity_Sold"].agg(["sum", "mean", "count", "std"])
    item_vol.columns = ["total_qty", "avg_qty", "n_days", "std_qty"]
    item_vol = item_vol.sort_values("total_qty", ascending=False)
    item_vol["cv"] = item_vol["std_qty"] / item_vol["avg_qty"]

    print(f"\nTop 15 items by total volume:")
    for i, (name, row) in enumerate(item_vol.head(15).iterrows(), 1):
        print(f"  {i:2d}. {name:<35s}  total={row['total_qty']:6.0f}  avg={row['avg_qty']:.1f}/day  cv={row['cv']:.2f}")

    print(f"\nBottom 10 items by total volume:")
    for i, (name, row) in enumerate(item_vol.tail(10).iterrows(), 1):
        print(f"  {i:2d}. {name:<35s}  total={row['total_qty']:6.0f}  avg={row['avg_qty']:.1f}/day  cv={row['cv']:.2f}")

    n_sparse = (item_vol["n_days"] < 100).sum()
    n_volatile = (item_vol["cv"] > 1.5).sum()
    print(f"\nSparse items (< 100 days of data): {n_sparse}")
    print(f"Highly volatile (CV > 1.5):        {n_volatile}")

    if DISCONTINUED_ITEMS:
        print(f"Discontinued items:                {len(DISCONTINUED_ITEMS)}")
        for item in DISCONTINUED_ITEMS:
            if item in item_vol.index:
                row = item_vol.loc[item]
                print(f"  - {item}: total={row['total_qty']:.0f}")


def section_missing_data(df: pd.DataFrame):
    print(f"\n{SEPARATOR}")
    print("MISSING DATA ANALYSIS")
    print(SEPARATOR)

    date_range = pd.date_range(df["Date"].min(), df["Date"].max())
    dates_in_data = set(df["Date"].dt.date)
    missing_dates = sorted(set(date_range.date) - dates_in_data)

    print(f"Expected days: {len(date_range)}")
    print(f"Days with data: {len(dates_in_data)}")
    print(f"Missing dates:  {len(missing_dates)}")

    if missing_dates:
        print(f"\nMissing dates:")
        for d in missing_dates:
            print(f"  {d}")


def section_zero_days(df: pd.DataFrame):
    print(f"\n{SEPARATOR}")
    print("ZERO-QUANTITY DAYS")
    print(SEPARATOR)

    daily = df.groupby(["Item", "Date"])["Quantity_Sold"].sum().reset_index()
    zero_days = daily[daily["Quantity_Sold"] == 0]
    total_day_items = len(daily)
    n_zero = len(zero_days)

    print(f"Total day-item combinations: {total_day_items}")
    print(f"Zero-quantity days:          {n_zero} ({n_zero/total_day_items*100:.1f}%)")

    if n_zero > 0:
        item_zeros = zero_days.groupby("Item").size().sort_values(ascending=False)
        print(f"\nItems with most zero days:")
        for item, count in item_zeros.head(10).items():
            total_days = daily[daily["Item"] == item].shape[0]
            print(f"  {item:<35s}  {count:3d} zero days / {total_days} total ({count/total_days*100:.1f}%)")


def section_quantity_distribution(df: pd.DataFrame):
    print(f"\n{SEPARATOR}")
    print("QUANTITY DISTRIBUTION")
    print(SEPARATOR)

    qty = df["Quantity_Sold"]
    print(f"\n  Mean: {qty.mean():.2f}  |  Median: {qty.median():.0f}  |  Std: {qty.std():.2f}")
    print(f"  Min: {qty.min()}  |  Max: {qty.max()}  |  P75: {qty.quantile(0.75):.0f}  |  P90: {qty.quantile(0.90):.0f}")
    print(f"  Skewness: {qty.skew():.2f}  |  Kurtosis: {qty.kurtosis():.2f}")

    print(f"\n  Histogram (item-day rows):")
    print(f"  {'Qty':>4s}  {'Count':>7s}  {'Pct':>6s}  {'Cumul':>6s}")
    print(f"  {'-'*30}")

    cumul = 0
    for val in range(0, 12):
        count = (qty == val).sum()
        pct = count / len(qty) * 100
        cumul += pct
        bar = "#" * int(pct / 2)
        print(f"  {val:>4d}  {count:>7d}  {pct:>5.1f}%  {cumul:>5.1f}%  {bar}")

    count_11plus = (qty >= 11).sum()
    pct_11plus = count_11plus / len(qty) * 100
    cumul += pct_11plus
    print(f"  {'11+':>4s}  {count_11plus:>7d}  {pct_11plus:>5.1f}%  {cumul:>5.1f}%")

    n_1_to_3 = ((qty >= 1) & (qty <= 3)).sum()
    print(f"\n  Key insight: {(qty <= 3).sum()/len(qty)*100:.1f}% of rows sell 1-3 cups/day (avg {qty.mean():.2f})")
    print(f"  This means even a 1-cup error produces 47% wMAPE (1/{qty.mean():.2f}={1/qty.mean()*100:.0f}%)")
    print(f"  wMAPE is structurally inflated by small denominators — MAE (cups) is the honest metric.")

    item_avg = df.groupby("Item")["Quantity_Sold"].mean()
    print(f"\n  Per-item daily averages:")
    print(f"    Items avg 1-3 cups/day: {((item_avg >= 1) & (item_avg <= 3)).sum()}/{len(item_avg)} ({((item_avg >= 1) & (item_avg <= 3)).sum()/len(item_avg)*100:.0f}%)")
    print(f"    Items avg 3-5 cups/day: {((item_avg > 3) & (item_avg <= 5)).sum()}/{len(item_avg)}")
    print(f"    Items avg > 5 cups/day: {(item_avg > 5).sum()}/{len(item_avg)}")
    print(f"    Top item: {item_avg.idxmax()} ({item_avg.max():.2f} cups/day)")
    print(f"    Bottom item: {item_avg.idxmin()} ({item_avg.min():.2f} cups/day)")


def section_data_quality(df: pd.DataFrame):
    print(f"\n{SEPARATOR}")
    print("DATA QUALITY CHECKS")
    print(SEPARATOR)

    print(f"\nDuplicate (Date, Item) rows: {df.duplicated(subset=['Date', 'Item']).sum()}")

    print(f"\nQuantity_Sold stats:")
    print(f"  Negative values: {(df['Quantity_Sold'] < 0).sum()}")
    print(f"  Zero values:     {(df['Quantity_Sold'] == 0).sum()}")
    print(f"  NaN values:      {df['Quantity_Sold'].isna().sum()}")

    print(f"\nItem name issues:")
    items = df["Item"].unique()
    leading_spaces = [i for i in items if i != i.strip()]
    empty_names = [i for i in items if not i.strip()]
    print(f"  Leading/trailing spaces: {len(leading_spaces)}")
    print(f"  Empty names:             {len(empty_names)}")


def plot_daily_sales(df: pd.DataFrame):
    daily = df.groupby("Date")["Quantity_Sold"].sum()
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(daily.index, daily.values, linewidth=0.8, alpha=0.8)
    ax.set_title("Daily Total Sales Quantity", fontsize=14)
    ax.set_ylabel("Total Quantity Sold")
    ax.set_xlabel("")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "01_daily_sales.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: 01_daily_sales.png")


def plot_monthly_sales(df: pd.DataFrame):
    df["YearMonth"] = df["Date"].dt.to_period("M")
    monthly = df.groupby("YearMonth")["Quantity_Sold"].sum()
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(range(len(monthly)), monthly.values, width=0.8, alpha=0.8)
    ax.set_xticks(range(0, len(monthly), max(1, len(monthly) // 12)))
    ax.set_xticklabels(
        [str(m) for i, m in enumerate(monthly.index) if i % max(1, len(monthly) // 12) == 0],
        rotation=45, ha="right"
    )
    ax.set_title("Monthly Total Sales Quantity", fontsize=14)
    ax.set_ylabel("Total Quantity Sold")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "02_monthly_sales.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: 02_monthly_sales.png")


def plot_day_of_week(df: pd.DataFrame):
    dow_avg = df.groupby(df["Date"].dt.dayofweek)["Quantity_Sold"].mean()
    dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(dow_names, [dow_avg.get(i, 0) for i in range(7)], alpha=0.8)
    ax.set_title("Average Daily Sales by Day of Week", fontsize=14)
    ax.set_ylabel("Average Quantity Sold")
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.02, f"{h:.1f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "03_day_of_week.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: 03_day_of_week.png")


def plot_top_items(df: pd.DataFrame, n: int = 15):
    item_vol = df.groupby("Item")["Quantity_Sold"].sum().sort_values(ascending=False).head(n)
    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(range(n), item_vol.values[::-1], alpha=0.8)
    ax.set_yticks(range(n))
    ax.set_yticklabels(item_vol.index[::-1], fontsize=9)
    ax.set_title(f"Top {n} Items by Total Sales Volume", fontsize=14)
    ax.set_xlabel("Total Quantity Sold")
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 50, bar.get_y() + bar.get_height() / 2, f"{w:,.0f}", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "04_top_items.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: 04_top_items.png")


def plot_zero_pct_by_item(df: pd.DataFrame):
    daily = df.groupby(["Item", "Date"])["Quantity_Sold"].sum().reset_index()
    item_stats = daily.groupby("Item").agg(
        total_days=("Date", "count"),
        zero_days=("Quantity_Sold", lambda x: (x == 0).sum())
    )
    item_stats["zero_pct"] = item_stats["zero_days"] / item_stats["total_days"] * 100
    item_stats = item_stats.sort_values("zero_pct", ascending=True)
    top_items = item_stats.tail(20)

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = ["#e74c3c" if p > 70 else "#f39c12" if p > 50 else "#27ae60" for p in top_items["zero_pct"]]
    ax.barh(range(len(top_items)), top_items["zero_pct"], color=colors, alpha=0.8)
    ax.set_yticks(range(len(top_items)))
    ax.set_yticklabels(top_items.index, fontsize=8)
    ax.set_xlabel("Zero-Days (%)")
    ax.set_title("Zero-Days Percentage by Item (Top 20)", fontsize=14)
    ax.axvline(x=50, color="gray", linestyle="--", alpha=0.5, label="50%")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "05_zero_pct_by_item.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: 05_zero_pct_by_item.png")


def plot_quantity_distribution(df: pd.DataFrame):
    qty = df["Quantity_Sold"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    counts = qty.value_counts().sort_index()
    top = counts.head(11).copy()
    top.index = top.index.astype(int)
    if (qty >= 11).any():
        top.loc["11+"] = (qty >= 11).sum()

    colors = ["#e74c3c" if i >= 6 else "#3498db" for i in range(len(top))]
    axes[0].bar(range(len(top)), top.values, color=colors, alpha=0.8)
    axes[0].set_xticks(range(len(top)))
    axes[0].set_xticklabels(top.index)
    axes[0].set_xlabel("Quantity Sold (cups/day)")
    axes[0].set_ylabel("Number of Rows")
    axes[0].set_title("Quantity Distribution (Item-Day Rows)")

    cumul_pct = top.values.cumsum() / len(qty) * 100
    axes[1].plot(range(len(top)), cumul_pct, "o-", color="#2c3e50", linewidth=2, markersize=6)
    axes[1].axhline(y=86, color="#e74c3c", linestyle="--", alpha=0.5, label="86% (qty<=3)")
    axes[1].axhline(y=95, color="#f39c12", linestyle="--", alpha=0.5, label="95% (qty<=5)")
    axes[1].set_xticks(range(len(top)))
    axes[1].set_xticklabels(top.index)
    axes[1].set_xlabel("Quantity Sold (cups/day)")
    axes[1].set_ylabel("Cumulative %")
    axes[1].set_title("Cumulative Distribution")
    axes[1].set_ylim(0, 105)
    axes[1].legend()

    fig.suptitle(f"Daily Item Sales: avg={qty.mean():.2f}, median={qty.median():.0f} | 86% sell 1-3 cups",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "06_quantity_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: 06_quantity_distribution.png")


def generate_plots(df: pd.DataFrame):
    print(f"\n{SEPARATOR}")
    print("GENERATING PLOTS")
    print(SEPARATOR)

    plot_daily_sales(df)
    plot_monthly_sales(df)
    plot_day_of_week(df)
    plot_top_items(df)
    plot_zero_pct_by_item(df)
    plot_quantity_distribution(df)

    print(f"\nAll plots saved to: {FIGURES_DIR}")


def main():
    print("CAFE SUPPLY DATA EXPLORATION")
    print("=" * 80)

    df = load_data()
    print(f"Loaded {len(df):,} rows")

    section_dataset_overview(df)
    section_temporal_patterns(df)
    section_item_analysis(df)
    section_missing_data(df)
    section_zero_days(df)
    section_quantity_distribution(df)
    section_data_quality(df)
    generate_plots(df)

    print(f"\n{SEPARATOR}")
    print("EXPLORATION COMPLETE")
    print(SEPARATOR)


if __name__ == "__main__":
    main()
