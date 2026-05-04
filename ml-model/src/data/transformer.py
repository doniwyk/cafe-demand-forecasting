from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict

import pandas as pd

TEMPORAL_COLUMNS = [
    "Date_Only", "Year", "Month", "Day",
    "DayOfWeek", "DayOfWeekName", "WeekOfYear", "Quarter",
    "DayOfYear", "IsWeekend", "MonthProgress",
]


class SalesDataTransformer:
    def __init__(self, sales_path: str | Path, menu_bom_path: str | Path):
        self.sales_path = Path(sales_path)
        self.menu_bom_path = Path(menu_bom_path)

        print("Loading data...")
        self.sales_df = pd.read_csv(self.sales_path)
        self.menu_bom_df = pd.read_csv(self.menu_bom_path)

        self.item_category_map = self._build_item_category_map()
        self.category_list = sorted(self.menu_bom_df["Tipe"].dropna().unique())

        self.stats = {
            "original_transactions": len(self.sales_df),
            "unique_items": self.sales_df["Item"].nunique(),
            "date_range_start": None,
            "date_range_end": None,
            "total_days": 0,
            "daily_records_created": 0,
        }

    def _build_item_category_map(self) -> Dict[str, str]:
        return {
            row["Item"].strip(): row["Tipe"].strip()
            for _, row in self.menu_bom_df.iterrows()
            if pd.notna(row["Item"]) and pd.notna(row["Tipe"])
        }

    def _prepare_data(self) -> pd.DataFrame:
        print("Preparing data...")

        df = self.sales_df.copy()
        df["Date"] = pd.to_datetime(df["Date"])
        df["Date_Only"] = df["Date"].dt.date
        df["Item"] = df["Item"].str.strip()
        df = df[df["Quantity"] > 0]

        self.stats["date_range_start"] = df["Date_Only"].min()
        self.stats["date_range_end"] = df["Date_Only"].max()
        self.stats["total_days"] = (
            self.stats["date_range_end"] - self.stats["date_range_start"]
        ).days + 1

        print(
            f"Data prepared: {len(df)} transactions from "
            f"{self.stats['date_range_start']} to {self.stats['date_range_end']}"
        )
        return df

    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        print("Adding temporal features...")
        date = pd.to_datetime(df["Date_Only"])

        df["Year"] = date.dt.year
        df["Month"] = date.dt.month
        df["Day"] = date.dt.day
        df["DayOfWeek"] = date.dt.dayofweek
        df["DayOfWeekName"] = date.dt.day_name()
        df["WeekOfYear"] = date.dt.isocalendar().week.astype(int)
        df["Quarter"] = date.dt.quarter
        df["DayOfYear"] = date.dt.dayofyear
        df["IsWeekend"] = (date.dt.dayofweek >= 5).astype(int)
        df["MonthProgress"] = date.dt.day / date.dt.days_in_month

        return df

    def _add_category(self, df: pd.DataFrame) -> pd.DataFrame:
        df["Category"] = df["Item"].map(self.item_category_map).fillna("Unknown")
        uncategorized = (df["Category"] == "Unknown").sum()
        if uncategorized:
            print(f"Warning: {uncategorized} rows could not be categorized")
        return df

    def _get_temporal_lookup(self, df: pd.DataFrame) -> pd.DataFrame:
        return df[TEMPORAL_COLUMNS].drop_duplicates()

    def aggregate_daily_sales(self) -> pd.DataFrame:
        print("Aggregating daily sales...")

        df = self._prepare_data()
        df = self._add_temporal_features(df)
        df = self._add_category(df)

        daily = (
            df.groupby(["Date_Only", "Item", "Category"])
            .agg({"Quantity": "sum", "Net sales": "sum", "Gross sales": "sum"})
            .reset_index()
        )
        daily = daily.merge(self._get_temporal_lookup(df), on="Date_Only", how="left")
        daily = daily.sort_values(["Date_Only", "Item"]).reset_index(drop=True)

        self.stats["daily_records_created"] = len(daily)
        print(f"Daily item records: {len(daily)}")
        return daily

    def create_category_aggregates(self, daily_sales: pd.DataFrame) -> pd.DataFrame:
        print("Aggregating daily category sales...")

        cat = (
            daily_sales.groupby(["Date_Only", "Category"])
            .agg({"Quantity": "sum", "Net sales": "sum", "Gross sales": "sum", "Item": "count"})
            .rename(columns={"Item": "UniqueItemCount"})
            .reset_index()
        )
        cat = cat.merge(self._get_temporal_lookup(daily_sales), on="Date_Only", how="left")
        cat = cat.sort_values(["Date_Only", "Category"]).reset_index(drop=True)
        return cat

    def create_total_daily_sales(self, daily_sales: pd.DataFrame) -> pd.DataFrame:
        print("Aggregating daily total sales...")

        total = (
            daily_sales.groupby("Date_Only")
            .agg({
                "Quantity": "sum",
                "Net sales": "sum",
                "Gross sales": "sum",
                "Item": "count",
                "Category": "nunique",
            })
            .rename(columns={"Item": "UniqueItemCount", "Category": "UniqueCategoryCount"})
            .reset_index()
        )
        total = total.merge(self._get_temporal_lookup(daily_sales), on="Date_Only", how="left")
        total = total.sort_values("Date_Only").reset_index(drop=True)
        return total

    def save_transformed_data(
        self,
        daily_sales: pd.DataFrame,
        category_sales: pd.DataFrame,
        total_sales: pd.DataFrame,
        output_dir: str | Path,
    ):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        daily_sales.to_csv(output_dir / "daily_item_sales.csv", index=False)
        category_sales.to_csv(output_dir / "daily_category_sales.csv", index=False)
        total_sales.to_csv(output_dir / "daily_total_sales.csv", index=False)

        metadata = {
            "transformation_date": datetime.now().isoformat(),
            "statistics": self.stats,
            "categories": self.category_list,
            "top_items": daily_sales.groupby("Item")["Quantity"]
            .sum().sort_values(ascending=False).head(20).to_dict(),
            "data_periods": {
                "start_date": str(self.stats["date_range_start"]),
                "end_date": str(self.stats["date_range_end"]),
                "total_days": self.stats["total_days"],
            },
        }
        with open(output_dir / "transformation_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2, default=str)

        print(f"Saved to: {output_dir}")
        print(f"  daily_item_sales.csv: {len(daily_sales)} records")
        print(f"  daily_category_sales.csv: {len(category_sales)} records")
        print(f"  daily_total_sales.csv: {len(total_sales)} records")

    def print_summary(self):
        print("\n" + "=" * 80)
        print("TRANSFORMATION SUMMARY")
        print("=" * 80)
        print(f"\n  Transactions: {self.stats['original_transactions']:,}")
        print(f"  Unique items: {self.stats['unique_items']:,}")
        print(f"  Date range: {self.stats['date_range_start']} to {self.stats['date_range_end']} ({self.stats['total_days']} days)")
        print(f"  Daily item records: {self.stats['daily_records_created']:,}")
        print(f"\n  Categories: {', '.join(self.category_list)}")
        print("=" * 80)
