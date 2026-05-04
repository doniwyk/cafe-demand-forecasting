"""
Sales Data Transformation for Forecasting

Transforms cleaned sales data into daily aggregated format suitable for forecasting.

Usage:
    python scripts/03_transform_sales.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.transformer import SalesDataTransformer
from src.utils.config import PROCESSED_DIR, BOM_DIR


def main():
    sales_path = PROCESSED_DIR / "sales_data_cleaned.csv"
    menu_bom_path = BOM_DIR / "menu_bom.csv"
    output_dir = PROCESSED_DIR / "sales_forecasting"

    print("=" * 80)
    print("SALES DATA TRANSFORMATION FOR FORECASTING")
    print("=" * 80)
    print(f"\nSales data: {sales_path}")
    print(f"Menu BOM: {menu_bom_path}")
    print(f"Output directory: {output_dir}")

    if not sales_path.exists():
        print(f"\nError: Sales data file not found: {sales_path}")
        print("Please run the sales data cleaning script first.")
        return

    if not menu_bom_path.exists():
        print(f"\nError: Menu BOM file not found: {menu_bom_path}")
        return

    try:
        transformer = SalesDataTransformer(sales_path, menu_bom_path)

        daily_sales = transformer.aggregate_daily_sales()
        category_sales = transformer.create_category_aggregates(daily_sales)
        total_sales = transformer.create_total_daily_sales(daily_sales)

        transformer.save_transformed_data(
            daily_sales, category_sales, total_sales, output_dir
        )
        transformer.print_summary()

        print("\nSales data transformation complete!")

    except Exception as e:
        print(f"\nError during transformation: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
