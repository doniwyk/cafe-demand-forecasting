from __future__ import annotations

import pandas as pd
import numpy as np
import os
import json
from typing import Dict, List, Set, Optional
from datetime import datetime, timedelta
from pathlib import Path

from src.utils.config import PROCESSED_DIR


class SalesDataTransformer:
    def __init__(self, sales_path: str | Path, menu_bom_path: str | Path):
        self.sales_path = Path(sales_path)
        self.menu_bom_path = Path(menu_bom_path)

        print("Loading data...")
        self.sales_df = pd.read_csv(self.sales_path)
        self.menu_bom_df = pd.read_csv(self.menu_bom_path)

        self.item_category_map = self._build_item_category_map()
        self.category_list = sorted(self.menu_bom_df["Tipe"].unique())

        self.stats = {
            "original_transactions": len(self.sales_df),
            "unique_items": self.sales_df["Item"].nunique(),
            "date_range_start": None,
            "date_range_end": None,
            "total_days": 0,
            "daily_records_created": 0,
        }

    def _build_item_category_map(self) -> Dict[str, str]:
        item_category_map = {}
        for _, row in self.menu_bom_df.iterrows():
            item_name = row["Item"].strip()
            category = row["Tipe"].strip()
            item_category_map[item_name] = category
        return item_category_map

    def _clean_and_prepare_data(self) -> pd.DataFrame:
        print("Cleaning and preparing data...")

        df = self.sales_df.copy()
        df["Date"] = pd.to_datetime(df["Date"])
        df["Date_Only"] = df["Date"].dt.date
        df["Item"] = df["Item"].str.strip()

        df["Item"] = df["Item"].str.replace(
            r"^espresso bon-bon$", "espresso", case=False, regex=True
        )

        df = df[df["Quantity"] > 0]

        initial_count = len(df)
        df = df[~df["Item"].str.lower().str.startswith("add")]
        filtered_count = initial_count - len(df)
        if filtered_count > 0:
            print(
                f"Filtered out {filtered_count} transactions with 'add' prefix (modifiers)"
            )

        initial_count = len(df)
        df = df[~df["Item"].str.lower().str.contains("cheese cake")]
        filtered_count = initial_count - len(df)
        if filtered_count > 0:
            print(
                f"Filtered out {filtered_count} transactions for discontinued item 'cheese cake'"
            )

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

        df["Date"] = pd.to_datetime(df["Date_Only"])

        df["Year"] = df["Date"].dt.year
        df["Month"] = df["Date"].dt.month
        df["Day"] = df["Date"].dt.day
        df["DayOfWeek"] = df["Date"].dt.dayofweek
        df["DayOfWeekName"] = df["Date"].dt.day_name()
        df["WeekOfYear"] = df["Date"].dt.isocalendar().week
        df["Quarter"] = df["Date"].dt.quarter
        df["DayOfYear"] = df["Date"].dt.dayofyear
        df["IsWeekend"] = (df["DayOfWeek"] >= 5).astype(int)
        df["MonthProgress"] = df["Day"] / df["Date"].dt.days_in_month

        return df

    def _add_category_features(self, df: pd.DataFrame) -> pd.DataFrame:
        print("Adding category features...")

        df["Category"] = df["Item"].map(self.item_category_map)

        uncategorized = df["Category"].isna().sum()
        if uncategorized > 0:
            print(f"Warning: {uncategorized} items could not be categorized")
            df["Category"] = df["Category"].fillna("Unknown")

        return df

    def aggregate_daily_sales(self) -> pd.DataFrame:
        print("Aggregating daily sales...")

        clean_df = self._clean_and_prepare_data()
        clean_df = self._add_temporal_features(clean_df)
        clean_df = self._add_category_features(clean_df)

        daily_sales = (
            clean_df.groupby(["Date_Only", "Item", "Category"])
            .agg(
                {
                    "Quantity": "sum",
                    "Net sales": "sum",
                    "Gross sales": "sum",
                }
            )
            .reset_index()
        )

        temp_features = clean_df[
            [
                "Date_Only",
                "Date",
                "Year",
                "Month",
                "Day",
                "DayOfWeek",
                "DayOfWeekName",
                "WeekOfYear",
                "Quarter",
                "DayOfYear",
                "IsWeekend",
                "MonthProgress",
            ]
        ].drop_duplicates()

        daily_sales = daily_sales.merge(temp_features, on="Date_Only", how="left")
        daily_sales = daily_sales.sort_values(["Date_Only", "Item"]).reset_index(
            drop=True
        )

        self.stats["daily_records_created"] = len(daily_sales)
        print(f"Daily aggregation complete: {len(daily_sales)} daily records created")

        return daily_sales

    def create_category_aggregates(self, daily_sales: pd.DataFrame) -> pd.DataFrame:
        print("Creating category-level aggregates...")

        category_sales = (
            daily_sales.groupby(["Date_Only", "Category"])
            .agg(
                {
                    "Quantity": "sum",
                    "Net sales": "sum",
                    "Gross sales": "sum",
                    "Item": "count",
                }
            )
            .rename(columns={"Item": "UniqueItemCount"})
            .reset_index()
        )

        temp_features = daily_sales[
            [
                "Date_Only",
                "Date",
                "Year",
                "Month",
                "Day",
                "DayOfWeek",
                "DayOfWeekName",
                "WeekOfYear",
                "Quarter",
                "DayOfYear",
                "IsWeekend",
                "MonthProgress",
            ]
        ].drop_duplicates()

        category_sales = category_sales.merge(temp_features, on="Date_Only", how="left")
        category_sales = category_sales.sort_values(
            ["Date_Only", "Category"]
        ).reset_index(drop=True)

        return category_sales

    def create_total_daily_sales(self, daily_sales: pd.DataFrame) -> pd.DataFrame:
        print("Creating total daily sales...")

        total_sales = (
            daily_sales.groupby("Date_Only")
            .agg(
                {
                    "Quantity": "sum",
                    "Net sales": "sum",
                    "Gross sales": "sum",
                    "Item": "count",
                    "Category": "nunique",
                }
            )
            .rename(
                columns={"Item": "UniqueItemCount", "Category": "UniqueCategoryCount"}
            )
            .reset_index()
        )

        temp_features = daily_sales[
            [
                "Date_Only",
                "Date",
                "Year",
                "Month",
                "Day",
                "DayOfWeek",
                "DayOfWeekName",
                "WeekOfYear",
                "Quarter",
                "DayOfYear",
                "IsWeekend",
                "MonthProgress",
            ]
        ].drop_duplicates()

        total_sales = total_sales.merge(temp_features, on="Date_Only", how="left")
        total_sales = total_sales.sort_values("Date_Only").reset_index(drop=True)

        return total_sales

    def save_transformed_data(
        self,
        daily_sales: pd.DataFrame,
        category_sales: pd.DataFrame,
        total_sales: pd.DataFrame,
        output_dir: str | Path,
    ):
        print("Saving transformed data...")

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
            .sum()
            .sort_values(ascending=False)
            .head(20)
            .to_dict(),
            "data_periods": {
                "start_date": str(self.stats["date_range_start"]),
                "end_date": str(self.stats["date_range_end"]),
                "total_days": self.stats["total_days"],
            },
        }

        with open(output_dir / "transformation_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2, default=str)

        print(f"Data saved to: {output_dir}")
        print(f"  - daily_item_sales.csv: {len(daily_sales)} records")
        print(f"  - daily_category_sales.csv: {len(category_sales)} records")
        print(f"  - daily_total_sales.csv: {len(total_sales)} records")

    def print_summary(self):
        print("\n" + "=" * 80)
        print("TRANSFORMATION SUMMARY")
        print("=" * 80)

        print(f"\nOriginal Data:")
        print(f"  - Total transactions: {self.stats['original_transactions']:,}")
        print(f"  - Unique items: {self.stats['unique_items']:,}")

        print(f"\nDate Range:")
        print(f"  - Start: {self.stats['date_range_start']}")
        print(f"  - End: {self.stats['date_range_end']}")
        print(f"  - Total days: {self.stats['total_days']}")

        print(f"\nTransformed Data:")
        print(f"  - Daily item records: {self.stats['daily_records_created']:,}")

        print(f"\nCategories:")
        for i, category in enumerate(self.category_list, 1):
            print(f"  {i}. {category}")

        print("=" * 80)
