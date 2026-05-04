"""
Forecast to Raw Material Requirements

Converts item-level sales forecast into raw material procurement needs
using the menu BOM and condiment BOM.

Usage:
    python scripts/05_forecast_to_materials.py
    python scripts/05_forecast_to_materials.py -f weekly
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse

import pandas as pd

from src.models.raw_materials import RawMaterialProcessor
from src.utils.config import BOM_DIR, PREDICTIONS_DIR


def main():
    parser = argparse.ArgumentParser(
        prog="05_forecast_to_materials.py",
        description="Convert item forecast to raw material procurement needs",
    )
    parser.add_argument(
        "-f",
        "--frequency",
        choices=["daily", "weekly"],
        default="weekly",
        help="Forecast frequency (default: weekly)",
    )
    args = parser.parse_args()

    forecast_path = PREDICTIONS_DIR / args.frequency / "3_month_forecasts.csv"
    menu_bom_path = BOM_DIR / "menu_bom.csv"
    condiment_bom_path = BOM_DIR / "condiment_bom.csv"
    output_path = PREDICTIONS_DIR / args.frequency / "raw_material_forecast.csv"

    if not forecast_path.exists():
        print(f"Error: Forecast file not found: {forecast_path}")
        print("Run 04_forecast.py first to generate forecasts.")
        sys.exit(1)

    print("=" * 80)
    print(f"FORECAST → RAW MATERIAL REQUIREMENTS ({args.frequency.upper()})")
    print("=" * 80)
    print(f"\nForecast:  {forecast_path}")
    print(f"Menu BOM:  {menu_bom_path}")
    print(f"Condiment: {condiment_bom_path}")

    forecast_df = pd.read_csv(forecast_path)
    forecast_df.columns = forecast_df.columns.str.strip()

    print(f"\nForecast data: {len(forecast_df)} rows")
    print(f"  Items: {forecast_df['Item'].nunique()}")
    print(f"  Date range: {forecast_df['Date'].min()} to {forecast_df['Date'].max()}")
    print(f"  Total predicted qty: {forecast_df['Quantity_Sold'].sum():,.0f}")

    processor = RawMaterialProcessor(
        menu_bom_path=menu_bom_path,
        condiment_bom_path=condiment_bom_path,
    )

    print("\nMapping forecast to raw materials via BOM...")

    forecast_df["Date"] = pd.to_datetime(forecast_df["Date"]).dt.date

    daily_requirements: dict[tuple, float] = {}
    for _, row in forecast_df.iterrows():
        date = row["Date"]
        item = row["Item"]
        quantity = row["Quantity_Sold"]

        if pd.isna(quantity) or quantity <= 0:
            continue

        raw_materials = processor._get_item_raw_materials(item, quantity)

        for material, material_qty in raw_materials.items():
            normalized_material = processor._normalize_material_name(material)
            key = (date, normalized_material)
            daily_requirements[key] = daily_requirements.get(key, 0) + material_qty

    material_df = pd.DataFrame(
        [
            {"Date": date, "Raw_Material": material, "Quantity_Required": qty}
            for (date, material), qty in daily_requirements.items()
        ]
    )
    material_df = material_df.sort_values(["Date", "Raw_Material"]).reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    material_df.to_csv(output_path, index=False)

    print(f"\nResults saved to: {output_path}")
    print(f"  Unique raw materials: {material_df['Raw_Material'].nunique()}")
    print(f"  Total rows: {len(material_df)}")

    print("\nTop 20 raw materials by total forecasted quantity:")
    top_materials = (
        material_df.groupby("Raw_Material")["Quantity_Required"]
        .sum()
        .sort_values(ascending=False)
        .head(20)
    )
    for material, qty in top_materials.items():
        print(f"  {material:<40} {qty:>12,.1f}")

    print("\nDone!")


if __name__ == "__main__":
    main()
