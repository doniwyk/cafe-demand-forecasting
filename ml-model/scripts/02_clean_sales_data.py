"""
Sales Data Cleaner Script

Cleans and standardizes sales data to match the menu BOM.

Usage:
    python scripts/02_clean_sales_data.py
    python scripts/02_clean_sales_data.py --remove
    python scripts/02_clean_sales_data.py --no-remove
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.cleaner import (
    SalesDataCleaner,
    print_discontinued_report,
    print_final_summary,
)
from src.utils.config import PROCESSED_DIR, BOM_DIR


def get_user_confirmation(auto_mode=None) -> bool:
    if auto_mode is True:
        print("\n[AUTO-REMOVE MODE] Automatically removing discontinued items...")
        return True
    elif auto_mode is False:
        print("\n[KEEP-ALL MODE] Keeping all items, including discontinued ones...")
        return False

    print("\n" + "?" * 80)
    print("REMOVAL CONFIRMATION")
    print("?" * 80)

    try:
        while True:
            response = (
                input(
                    "\nDo you want to remove discontinued items from sales data? (yes/no): "
                )
                .strip()
                .lower()
            )
            if response in ["yes", "y"]:
                return True
            elif response in ["no", "n"]:
                return False
            else:
                print('Invalid input. Please enter "yes" or "no".')
    except (EOFError, KeyboardInterrupt):
        print("\n\nInterrupted. No changes will be made.")
        return False


def main():
    auto_mode = None
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        if arg in ["--remove", "-r", "--yes", "-y"]:
            auto_mode = True
        elif arg in ["--no-remove", "-n", "--no", "--keep-all"]:
            auto_mode = False
        elif arg in ["--help", "-h"]:
            print(__doc__)
            return

    sales_path = PROCESSED_DIR / "sales_data.csv"
    menu_bom_path = BOM_DIR / "menu_bom.csv"
    output_path = PROCESSED_DIR / "sales_data_cleaned.csv"

    print("=" * 80)
    print("SALES DATA CLEANER")
    print("=" * 80)
    print(f"\nSales data: {sales_path}")
    print(f"Menu BOM: {menu_bom_path}")
    print(f"Output: {output_path}")

    cleaner = SalesDataCleaner(sales_path, menu_bom_path)

    print(f"\nOriginal records: {len(cleaner.sales_df):,}")

    standardized_df = cleaner.standardize_names(cleaner.sales_df)

    discontinued_items, discontinued_df, current_df = (
        cleaner.identify_discontinued_items(standardized_df)
    )

    print_discontinued_report(discontinued_items, discontinued_df)

    print("\n" + "-" * 80)
    print("STEP 3: HANDLING DISCONTINUED ITEMS")
    print("-" * 80)

    final_df = current_df

    if discontinued_items:
        if get_user_confirmation(auto_mode):
            print("\nRemoving discontinued items...")
            final_df = cleaner.remove_discontinued_items(current_df, discontinued_items)
            print(f"Removed {cleaner.stats['removed_records']:,} records")
        else:
            print("\nKeeping all items (including discontinued ones)")
            cleaner.stats["removed_records"] = 0
    else:
        print("\nNo discontinued items to remove")

    print("\n" + "-" * 80)
    print("STEP 4: SAVING CLEANED DATA")
    print("-" * 80)
    cleaner.save_cleaned_data(final_df, output_path)

    print_final_summary(cleaner.stats)

    print("\nSales data cleaning complete!\n")


if __name__ == "__main__":
    main()
