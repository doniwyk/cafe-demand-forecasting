import pandas as pd
from typing import Dict, List, Set, Tuple, Optional
from pathlib import Path

from src.utils.config import PROCESSED_DIR


RENAME_MAP = {
    "lemon tea hot": "Lemon Hot",
    "lemon tea ice": "Lemon Ice",
    "hot tea ajeng": "Ajeng Hot",
    "ice tea ajeng": "Ajeng Ice",
    "hot tea anindya": "Anindya Hot",
    "ice tea anindya": "Anindya Ice",
    "kopi susu panas": "Kopi Susu Hot",
    "ice cappucino": "Cappucino Ice",
    "latte": "Latte Hot",
    "ice latte": "Latte Ice",
    "picolo": "Piccolo",
    "long black hot": "Black Hot",
    "long black ice": "Black Ice",
    "red velvet": "Red Velvet Ice",
    "susu coklat": "Coklat Hot",
    "v60 hacienda natural": "v60",
    "v60 argopuro": "v60",
    "v60 finca": "v60",
    "mie goreng telur": "Mie Goreng",
    "mie rebus telur": "Mie Rebus",
    "nasi goreng djawa": "Nasi Goreng Jawa",
    "nasi ayam daun jeruk": "Ayam Daun Jeruk",
    "nasi ayam rempah": "Ayam Rempah",
    "ayam mentega (paket)": "Ayam Mentega",
    "ayam curry (paket)": "Ayam Curry",
    "ayam dauh jeruk (paket)": "Ayam Daun Jeruk",
    "cireng isi": "Cireng",
    "pannacotta": "Panacotta",
    "add vanilla ice cream": "Add Ice Cream Vanilla",
    "add telur ceplok": "Add Telur",
    "add telur dadar": "Add Telur",
    "add caramel topping": "Add Topping Caramel",
}

PACKAGE_MAP = {
    "mie goreng telur": [("Mie Goreng", 1.0), ("Add Telur", 1.0)],
    "mie rebus telur": [("Mie Rebus", 1.0), ("Add Telur", 1.0)],
    "paket ayam teriyaki + lemon tea": [("Ayam Teriyaki", 1.0), ("Lemon Ice", 1.0)],
    "paket ayam curry + lemon tea": [("Ayam Curry", 1.0), ("Lemon Ice", 1.0)],
}


class SalesDataCleaner:
    def __init__(self, sales_path: str | Path, menu_bom_path: str | Path):
        self.sales_path = Path(sales_path)
        self.menu_bom_path = Path(menu_bom_path)
        self.sales_df = pd.read_csv(self.sales_path)
        self.menu_bom_df = pd.read_csv(self.menu_bom_path)

        self.rename_map = RENAME_MAP
        self.package_items = PACKAGE_MAP
        self.active_items = self._get_active_items()

        self.stats = {
            "original_records": len(self.sales_df),
            "renamed_records": 0,
            "expanded_packages": 0,
            "expanded_items": 0,
            "discontinued_items": 0,
            "removed_records": 0,
        }

    def _get_active_items(self) -> Set[str]:
        active_items = set()
        for item_name in self.menu_bom_df["Item"].unique():
            normalized = item_name.strip().lower()
            active_items.add(normalized)
        return active_items

    def _normalize_item_name(self, item_name: str) -> str:
        if pd.isna(item_name):
            return ""
        return str(item_name).strip().lower()

    def standardize_names(self, df: pd.DataFrame) -> pd.DataFrame:
        print("\n" + "-" * 80)
        print("STEP 1: STANDARDIZING ITEM NAMES")
        print("-" * 80)

        standardized_df = df.copy()

        renamed_count = 0
        for old_name, new_name in self.rename_map.items():
            mask = standardized_df["Item"].str.lower() == old_name.lower()
            if mask.any():
                count = mask.sum()
                standardized_df.loc[mask, "Item"] = new_name
                renamed_count += count
                print(f'  Renamed: "{old_name}" -> "{new_name}" ({count} records)')

        self.stats["renamed_records"] = renamed_count

        expanded_rows = []
        rows_to_remove = []

        for idx, row in standardized_df.iterrows():
            item_lower = str(row["Item"]).lower()

            if item_lower in self.package_items:
                rows_to_remove.append(idx)
                components = self.package_items[item_lower]
                original_qty = row["Quantity"]

                for component_item, qty_multiplier in components:
                    new_row = row.copy()
                    new_row["Item"] = component_item
                    new_row["Quantity"] = original_qty * qty_multiplier
                    expanded_rows.append(new_row)

                print(
                    f'  Expanded: "{row["Item"]}" -> {len(components)} items ({original_qty} qty each)'
                )

        if rows_to_remove:
            standardized_df = standardized_df.drop(rows_to_remove)
            self.stats["expanded_packages"] = len(rows_to_remove)

        if expanded_rows:
            expanded_df = pd.DataFrame(expanded_rows)
            standardized_df = pd.concat(
                [standardized_df, expanded_df], ignore_index=True
            )
            self.stats["expanded_items"] = len(expanded_rows)

        standardized_df = standardized_df.sort_values("Date").reset_index(drop=True)

        print(f"\nStandardization Summary:")
        print(f"  Simple renames: {renamed_count} records")
        print(
            f"  Package expansions: {len(rows_to_remove)} packages -> {len(expanded_rows)} items"
        )
        print(f"  Total records: {len(df)} -> {len(standardized_df)}")

        return standardized_df

    def identify_discontinued_items(
        self, df: pd.DataFrame
    ) -> Tuple[Set[str], pd.DataFrame, pd.DataFrame]:
        print("\n" + "-" * 80)
        print("STEP 2: IDENTIFYING DISCONTINUED ITEMS")
        print("-" * 80)

        sales_items = df["Item"].dropna().unique()
        discontinued_items = set()
        valid_items = set()

        for item in sales_items:
            normalized = self._normalize_item_name(item)
            if normalized and normalized not in self.active_items:
                discontinued_items.add(item)
            else:
                valid_items.add(item)

        self.stats["discontinued_items"] = len(discontinued_items)

        discontinued_stats = []
        for item in discontinued_items:
            item_data = df[df["Item"] == item]
            stats = {
                "Item": item,
                "Total_Quantity_Sold": item_data["Quantity"].sum(),
                "Total_Transactions": len(item_data),
                "First_Sale_Date": item_data["Date"].min(),
                "Last_Sale_Date": item_data["Date"].max(),
            }
            discontinued_stats.append(stats)

        discontinued_df = pd.DataFrame(discontinued_stats)

        if not discontinued_df.empty:
            discontinued_df = discontinued_df.sort_values(
                "Total_Quantity_Sold", ascending=False
            )

        print(f"\nValid items (found in BOM): {len(valid_items)}")
        print(f"Discontinued items (NOT in BOM): {len(discontinued_items)}")

        return discontinued_items, discontinued_df, df

    def remove_discontinued_items(
        self, df: pd.DataFrame, discontinued_items: Set[str]
    ) -> pd.DataFrame:
        cleaned_df = df[~df["Item"].isin(discontinued_items)].copy()
        self.stats["removed_records"] = len(df) - len(cleaned_df)
        return cleaned_df

    def save_cleaned_data(self, cleaned_df: pd.DataFrame, output_path: str | Path):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cleaned_df.to_csv(output_path, index=False)
        print(f"\nCleaned data saved to: {output_path}")
        print(f"  Total records: {len(cleaned_df)}")


def print_discontinued_report(
    discontinued_items: Set[str], discontinued_df: pd.DataFrame
):
    if not discontinued_items:
        print("\nNo discontinued items found!")
        print("  All items in sales data exist in the current menu BOM.")
        return

    print(f"\nTotal discontinued items: {len(discontinued_items)}")
    print(
        f"Total transactions affected: {discontinued_df['Total_Transactions'].sum():.0f}"
    )
    print(f"Total quantity sold: {discontinued_df['Total_Quantity_Sold'].sum():.0f}")

    print("\n" + "-" * 80)
    print("DISCONTINUED ITEMS LIST (Top 20):")
    print("-" * 80)
    print(f"{'Item':<40} {'Qty Sold':<12} {'Transactions':<15} {'Last Sale'}")
    print("-" * 80)

    for idx, row in discontinued_df.head(100).iterrows():
        item_name = row["Item"][:39]
        qty_sold = f"{row['Total_Quantity_Sold']:.0f}"
        transactions = f"{row['Total_Transactions']:.0f}"
        last_sale = str(row["Last_Sale_Date"])[:10]
        print(f"{item_name:<40} {qty_sold:<12} {transactions:<15} {last_sale}")

    if len(discontinued_df) > 20:
        print(f"\n... and {len(discontinued_df) - 20} more items")


def print_final_summary(stats: Dict):
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)

    print(f"\nOriginal records: {stats['original_records']:,}")
    print(f"\nStandardization:")
    print(f"  - Renamed records: {stats['renamed_records']:,}")
    print(
        f"  - Expanded packages: {stats['expanded_packages']:,} -> {stats['expanded_items']:,} items"
    )

    print(f"\nDiscontinued items:")
    print(f"  - Items found: {stats['discontinued_items']:,}")
    print(f"  - Records removed: {stats['removed_records']:,}")

    final_records = (
        stats["original_records"] + stats["expanded_items"] - stats["removed_records"]
    )
    print(f"\nFinal records: {final_records:,}")
    print("=" * 80)
