from __future__ import annotations

import pandas as pd
from typing import Dict, List, Set, Tuple, Optional
from pathlib import Path

from src.utils.config import PROCESSED_DIR


RENAME_MAP = {
    "lemon tea hot": "Lemon Tea Hot",
    "lemon tea ice": "Lemon Tea Ice",
    "lemon ice": "Lemon Tea Ice",
    "lemon hot": "Lemon Tea Hot",
    "lemon tea": "Lemon Tea Hot",
    "milk tea hot": "Mewangi Milk Tea Hot",
    "milk tea ice": "Mewangi Milk Tea Ice",
    "mewangi milk tea": "Mewangi Milk Tea Hot",
    "lychee tea hot": "Lychee Tea Hot",
    "lychee tea ice": "Lychee Tea Ice",
    "lychee tea": "Lychee Tea Hot",
    "kopi susu panas": "Kopi Susu Husgendam Hot",
    "kopi susu hot": "Kopi Susu Husgendam Hot",
    "kopi susu ice": "Kopi Susu Husgendam Ice",
    "kopi susu bersemi": "Kopi Susu Husgendam Hot",
    "kopi susu bersemi ice": "Kopi Susu Husgendam Ice",
    "kopi susu bersemi hot": "Kopi Susu Husgendam Hot",
    "kopi susu husgendam": "Kopi Susu Husgendam Hot",
    "husgendam hot": "Kopi Susu Husgendam Hot",
    "husgendam ice": "Kopi Susu Husgendam Ice",
    "husgen platter": "Husgendam Platter",
    "husgendam platter": "Husgendam Platter",
    "husgendam toast": "Husgendam Platter",
    "husgendam nugget": "Husgendam Platter",
    "long black hot": "Black Hot",
    "long black ice": "Black Ice",
    "americano hot": "Black Hot",
    "americano ice": "Black Ice",
    "ice cappucino": "Cappucino Ice",
    "latte": "Latte Hot",
    "ice latte": "Latte Ice",
    "susu coklat": "Chocolate Hot",
    "coklat hot": "Chocolate Hot",
    "coklat ice": "Chocolate Ice",
    "coklat": "Chocolate Hot",
    "tubruk susu": "Tubruk",
    "picolo": "Espresso",
    "piccolo": "Espresso",
    "v60 hacienda natural": "Filter",
    "v60 argopuro": "Filter",
    "v60 finca": "Filter",
    "v60 - ijen classic washed hot": "Filter",
    "v60 - ijen classic washed ice": "Filter",
    "v60 - strawberry twist hot": "Filter",
    "v60 - strawberry twist ice": "Filter",
    "filter - ijen classic washed hot": "Filter",
    "filter - ijen classic washed ice": "Filter",
    "filter - semendo liberika hsn hot": "Filter",
    "filter - semendo liberika hsn ice": "Filter",
    "filter - sinapeul black honey hot": "Filter",
    "filter - sinapeul black honey ice": "Filter",
    "filter - strawberry twist hot": "Filter",
    "filter - strawberry twist ice": "Filter",
    "mie goreng telur": "Mie Goreng",
    "mie rebus telur": "Mie Rebus",
    "nasi goreng djawa": "Nasi Goreng Jawa",
    "nasi ayam daun jeruk": "Nasi Ayam Daun Jeruk",
    "ayam daun jeruk": "Nasi Ayam Daun Jeruk",
    "ayam mentega (paket)": "Nasi Ayam Mentega",
    "ayam curry (paket)": "Nasi Ayam Curry",
    "ayam dauh jeruk (paket)": "Nasi Ayam Daun Jeruk",
    "ayam curry": "Nasi Ayam Curry",
    "ayam mentega": "Nasi Ayam Mentega",
    "ayam chili padi": "Nasi Ayam Chili Padi",
    "cireng isi": "Cireng",
    "kentang goreng": "Kentang",
    "kentang": "Kentang",
    "pisang goreng": "Pisang Goreng Madu",
    "pisang goreng aren": "Pisang Goreng Madu",
    "pisang goreng madu": "Pisang Goreng Madu",
    "puspha matcha": "Puspa Matcha",
    "puspa matcha": "Puspa Matcha",
    "original cookie": "Cookies Original",
    "cookies original": "Cookies Original",
    "waffle vanilla": "Candana Vanilla",
    "waffle strawberry": "Kirana Strawberry",
    "waffle matcha": "Puspa Matcha",
    "memukau (for him)": "Memukau",
    "menawan (for her)": "Menawan",
    "cheese cake": "New York Cheesecake",
    "cheesecake": "New York Cheesecake",
    "pannacotta": "New York Cheesecake",
    "panacotta": "New York Cheesecake",
    "air mineral besar": "Air Mineral",
    "filter strw": "Filter",
    "filter ethiopia": "Filter",
    "filter kenya": "Filter",
    "filter flores mewangi": "Filter",
    "filter kamojang kembang": "Filter",
    "filter coffe abhimanyu": "Filter",
}

class SalesDataCleaner:
    def __init__(self, sales_path: str | Path, menu_bom_path: str | Path):
        self.sales_path = Path(sales_path)
        self.menu_bom_path = Path(menu_bom_path)
        self.sales_df = pd.read_csv(self.sales_path)
        self.menu_bom_df = pd.read_csv(self.menu_bom_path)

        self.rename_map = RENAME_MAP
        self.active_items = self._get_active_items()
        self.bom_name_map = self._get_bom_name_map()

        self.stats = {
            "original_records": len(self.sales_df),
            "renamed_records": 0,
            "discontinued_items": 0,
            "removed_records": 0,
        }

    def _get_active_items(self) -> Set[str]:
        active_items = set()
        for item_name in self.menu_bom_df["Item"].unique():
            normalized = item_name.strip().lower()
            active_items.add(normalized)
        return active_items

    def _get_bom_name_map(self) -> Dict[str, str]:
        name_map = {}
        for item_name in self.menu_bom_df["Item"].dropna().unique():
            normalized = item_name.strip().lower()
            name_map[normalized] = item_name.strip()
        return name_map

    def _try_suffix_match(self, item_name: str) -> Optional[str]:
        base = item_name.strip().lower()

        hot_key = base + " hot"
        ice_key = base + " ice"

        hot_match = hot_key in self.active_items
        ice_match = ice_key in self.active_items
        base_match = base in self.active_items

        if hot_match and not ice_match:
            return self.bom_name_map[hot_key]
        if ice_match and not hot_match:
            return self.bom_name_map[ice_key]
        if hot_match and ice_match:
            if item_name.strip().lower().endswith(" ice"):
                return self.bom_name_map[ice_key]
            return self.bom_name_map[hot_key]
        if base_match:
            return self.bom_name_map[base]
        return None

    def _normalize_item_name(self, item_name: str) -> str:
        if pd.isna(item_name):
            return ""
        return str(item_name).strip().lower()

    def _apply_modifiers(self, df: pd.DataFrame) -> pd.DataFrame:
        if "Modifiers applied" not in df.columns:
            return df

        mod_col = df["Modifiers applied"].astype(str).str.strip()
        has_hot = mod_col == "Hot"
        has_ice = mod_col == "Ice"

        item_lower = df["Item"].astype(str).str.lower().str.strip()
        already_hot = item_lower.str.endswith(" hot")
        already_ice = item_lower.str.endswith(" ice")

        needs_hot = has_hot & ~already_hot & ~already_ice
        needs_ice = has_ice & ~already_hot & ~already_ice

        count_hot = needs_hot.sum()
        count_ice = needs_ice.sum()

        df.loc[needs_hot, "Item"] = df.loc[needs_hot, "Item"].str.strip() + " Hot"
        df.loc[needs_ice, "Item"] = df.loc[needs_ice, "Item"].str.strip() + " Ice"

        if count_hot or count_ice:
            print(f"  Applied modifiers: {count_hot} Hot, {count_ice} Ice")

        return df

    def standardize_names(self, df: pd.DataFrame) -> pd.DataFrame:
        print("\n" + "-" * 80)
        print("STEP 1: STANDARDIZING ITEM NAMES")
        print("-" * 80)

        standardized_df = df.copy()

        standardized_df = self._apply_modifiers(standardized_df)

        renamed_count = 0
        for old_name, new_name in self.rename_map.items():
            mask = standardized_df["Item"].str.lower() == old_name.lower()
            if mask.any():
                count = mask.sum()
                standardized_df.loc[mask, "Item"] = new_name
                renamed_count += count
                print(f'  Renamed: "{old_name}" -> "{new_name}" ({count} records)')

        self.stats["renamed_records"] = renamed_count

        bom_names = {
            item.strip().lower(): item.strip()
            for item in self.menu_bom_df["Item"].dropna().unique()
        }
        case_fixed = 0
        for idx, row in standardized_df.iterrows():
            item = str(row["Item"]).strip()
            bom_match = bom_names.get(item.lower())
            if bom_match and bom_match != item:
                standardized_df.at[idx, "Item"] = bom_match
                case_fixed += 1

        if case_fixed:
            print(f"  Case-normalized: {case_fixed} records to match BOM names")

        standardized_df = standardized_df.sort_values("Date").reset_index(drop=True)

        v60_mask = standardized_df["Item"].str.lower().str.contains("v60")
        v60_count = v60_mask.sum()
        if v60_count:
            standardized_df.loc[v60_mask, "Item"] = "Filter"
            print(f"  V60/Filter normalized: {v60_count} records -> Filter")

        suffix_map = {}
        for item in standardized_df["Item"].dropna().unique():
            item_str = str(item).strip()
            if item_str.lower() not in self.active_items:
                matched = self._try_suffix_match(item_str)
                if matched:
                    suffix_map[item_str] = matched

        suffix_renamed = 0
        if suffix_map:
            mask = standardized_df["Item"].isin(suffix_map.keys())
            suffix_renamed = mask.sum()
            standardized_df["Item"] = standardized_df["Item"].replace(suffix_map)
            print(f"  Suffix-matched: {suffix_renamed} records to BOM Hot/Ice variants")
            for old, new in sorted(suffix_map.items()):
                print(f'    "{old}" -> "{new}"')

        print(f"\nStandardization Summary:")
        print(f"  Simple renames: {renamed_count} records")
        if v60_count:
            print(f"  V60 normalized: {v60_count} records")
        if suffix_renamed:
            print(f"  Suffix-matched: {suffix_renamed} records")
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

    print(f"\nDiscontinued items:")
    print(f"  - Items found: {stats['discontinued_items']:,}")
    print(f"  - Records removed: {stats['removed_records']:,}")

    final_records = stats["original_records"] - stats["removed_records"]
    print(f"\nFinal records: {final_records:,}")
    print("=" * 80)
