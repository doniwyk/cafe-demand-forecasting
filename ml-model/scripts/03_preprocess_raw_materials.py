"""
Sales Data Transformation for Forecasting

Transforms raw sales data into daily aggregated format suitable for sales forecasting.

Usage:
    python scripts/03_preprocess_raw_materials.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models.raw_materials import RawMaterialProcessor
from src.utils.config import PROCESSED_DIR, BOM_DIR


def main():
    sales_path = PROCESSED_DIR / "sales_data_cleaned.csv"
    menu_bom_path = BOM_DIR / "menu_bom.csv"
    condiment_bom_path = BOM_DIR / "condiment_bom.csv"
    output_path = PROCESSED_DIR / "daily_raw_material_requirements.csv"

    print("=" * 80)
    print("RAW MATERIAL REQUIREMENTS PROCESSOR")
    print("=" * 80)
    print(f"\nSales data: {sales_path}")
    print(f"Menu BOM: {menu_bom_path}")
    print(f"Condiment BOM: {condiment_bom_path}")

    processor = RawMaterialProcessor(sales_path, menu_bom_path, condiment_bom_path)
    processor.save_results(output_path)


if __name__ == "__main__":
    main()
