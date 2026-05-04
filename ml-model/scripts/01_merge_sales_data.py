"""
Sales Data Merger Script

Merges two sales CSV files with different languages (Indonesian and English)
into a single standardized dataset.

Usage:
    python scripts/01_merge_sales_data.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.merger import merge_sales_files
from src.utils.config import SALES_DIR, PROCESSED_DIR


def main():
    file_paths = [
        SALES_DIR / "receipts-by-item-2022-01-01-2025-06-30.csv",
        SALES_DIR / "receipts-by-item-2025-05-01-2025-09-25.csv",
        SALES_DIR / "receipts-by-item-2025-09-26-2026-03-31.csv",
    ]
    output_path = PROCESSED_DIR / "sales_data.csv"

    print("=" * 80)
    print("SALES DATA MERGER")
    print("=" * 80)

    merged_data = merge_sales_files(file_paths, output_path)

    if merged_data is not None:
        print("\nFirst few rows of merged data:")
        print(merged_data.head())

        print("\nData summary:")
        print(merged_data.info())


if __name__ == "__main__":
    main()
