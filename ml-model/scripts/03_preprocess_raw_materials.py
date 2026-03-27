"""
Raw Material Requirements Preprocessing Script

This script processes sales data and BOM (Bill of Materials) files to calculate
daily raw material requirements. It handles hierarchical BOM structures where:
1. Menu items can be composed of raw materials and condiments
2. Condiments can be composed of raw materials and other condiments
3. The script recursively expands condiments to calculate total raw material needs
"""

import pandas as pd
import os
from datetime import datetime
from typing import Dict, Tuple


class RawMaterialProcessor:
    """Process sales data and BOM files to calculate raw material requirements"""

    def __init__(self, sales_path: str, menu_bom_path: str, condiment_bom_path: str):
        """
        Initialize the processor with file paths

        Args:
            sales_path: Path to sales data CSV
            menu_bom_path: Path to menu BOM CSV
            condiment_bom_path: Path to condiment BOM CSV
        """
        self.sales_df = pd.read_csv(sales_path)
        self.menu_bom_df = pd.read_csv(menu_bom_path)
        self.condiment_bom_df = pd.read_csv(condiment_bom_path)

        # Normalize item names in BOMs for matching
        self.menu_bom_df['Item_normalized'] = self.menu_bom_df['Item'].str.strip().str.lower()

        # Build item name mapping for fuzzy matching
        self.item_name_map = self._build_item_name_map()

        # Build condiment lookup dictionary for fast access
        self.condiment_dict = self._build_condiment_dict()

        # Cache for memoization of condiment expansion
        self.expansion_cache: Dict[Tuple[str, float], Dict[str, float]] = {}

    def _build_item_name_map(self) -> Dict[str, str]:
        """
        Build a dictionary mapping normalized item names to original names in BOM

        Returns:
            Dictionary with normalized names as keys and original BOM names as values
        """
        name_map = {}
        for item_name in self.menu_bom_df['Item'].unique():
            normalized = item_name.strip().lower()
            name_map[normalized] = item_name
        return name_map

    def _build_condiment_dict(self) -> Dict[str, pd.DataFrame]:
        """
        Build a dictionary mapping condiment names to their ingredients

        Returns:
            Dictionary with condiment names as keys and DataFrames of ingredients as values
        """
        condiment_dict = {}
        for condiment_name in self.condiment_bom_df['Condiment'].unique():
            condiment_dict[condiment_name] = self.condiment_bom_df[
                self.condiment_bom_df['Condiment'] == condiment_name
            ].copy()
        return condiment_dict

    def _is_condiment(self, ingredient_name: str) -> bool:
        """
        Check if an ingredient is a condiment (needs further expansion)

        Args:
            ingredient_name: Name of the ingredient to check

        Returns:
            True if ingredient is a condiment, False otherwise
        """
        return ingredient_name in self.condiment_dict

    def _expand_condiment(self, condiment_name: str, quantity: float,
                         condiment_unit: str) -> Dict[str, float]:
        """
        Recursively expand a condiment into raw materials

        Args:
            condiment_name: Name of the condiment to expand
            quantity: Quantity of condiment needed
            condiment_unit: Unit of the condiment

        Returns:
            Dictionary mapping raw material names to quantities
        """
        # Check cache first
        cache_key = (condiment_name, quantity)
        if cache_key in self.expansion_cache:
            return self.expansion_cache[cache_key].copy()

        raw_materials = {}

        if condiment_name not in self.condiment_dict:
            # Not a condiment, treat as raw material
            raw_materials[condiment_name] = quantity
            return raw_materials

        # Get condiment recipe
        condiment_recipe = self.condiment_dict[condiment_name]

        # Get the base quantity that the recipe makes
        base_condiment_qty = condiment_recipe['Condiment_Qty'].iloc[0]

        # Calculate scaling factor
        scaling_factor = quantity / base_condiment_qty

        # Expand each sub-ingredient
        for _, row in condiment_recipe.iterrows():
            sub_ingredient = row['Sub_Ingredient']
            sub_qty = row['Qty_per_condiment_unit']
            sub_unit = row['Sub_Unit']

            # Convert quantity to float
            try:
                sub_qty = float(sub_qty)
            except (ValueError, TypeError):
                print(f"Warning: Invalid quantity '{sub_qty}' for sub-ingredient '{sub_ingredient}' in condiment '{condiment_name}'. Skipping.")
                continue

            # Calculate actual quantity needed
            actual_qty = sub_qty * scaling_factor

            # Recursively expand if it's another condiment
            if self._is_condiment(sub_ingredient):
                sub_materials = self._expand_condiment(sub_ingredient, actual_qty, sub_unit)
                # Aggregate the results
                for material, material_qty in sub_materials.items():
                    raw_materials[material] = raw_materials.get(material, 0) + material_qty
            else:
                # It's a raw material
                raw_materials[sub_ingredient] = raw_materials.get(sub_ingredient, 0) + actual_qty

        # Cache the result
        self.expansion_cache[cache_key] = raw_materials.copy()

        return raw_materials

    def _get_item_raw_materials(self, item_name: str, quantity: float) -> Dict[str, float]:
        """
        Get all raw materials needed for a menu item

        Args:
            item_name: Name of the menu item
            quantity: Quantity of items sold

        Returns:
            Dictionary mapping raw material names to quantities
        """
        raw_materials = {}

        # Try to normalize item name for matching
        normalized_name = item_name.strip().lower()

        # Check if we have a mapping for this normalized name
        if normalized_name in self.item_name_map:
            matched_name = self.item_name_map[normalized_name]
            item_recipe = self.menu_bom_df[self.menu_bom_df['Item'] == matched_name]
        else:
            # Try exact match
            item_recipe = self.menu_bom_df[self.menu_bom_df['Item'] == item_name]

        if item_recipe.empty:
            # Item not found in BOM, skip with warning
            print(f"Warning: Item '{item_name}' not found in menu BOM")
            return raw_materials

        # Process each ingredient in the recipe
        for _, row in item_recipe.iterrows():
            ingredient = row['Bahan']
            ingredient_qty = row['Qty']
            ingredient_unit = row['Unit']

            # Convert quantity to float to handle string values
            try:
                ingredient_qty = float(ingredient_qty)
            except (ValueError, TypeError):
                print(f"Warning: Invalid quantity '{ingredient_qty}' for ingredient '{ingredient}' in item '{item_name}'. Skipping.")
                continue

            # Calculate total quantity needed
            total_qty = ingredient_qty * quantity

            # Check if ingredient is a condiment and needs expansion
            if self._is_condiment(ingredient):
                sub_materials = self._expand_condiment(ingredient, total_qty, ingredient_unit)
                # Aggregate the results
                for material, material_qty in sub_materials.items():
                    raw_materials[material] = raw_materials.get(material, 0) + material_qty
            else:
                # It's a raw material
                raw_materials[ingredient] = raw_materials.get(ingredient, 0) + total_qty

        return raw_materials

    def _normalize_material_name(self, material_name: str) -> str:
        """
        Normalize material names to handle case inconsistencies

        Args:
            material_name: Raw material name to normalize

        Returns:
            Normalized material name with consistent casing
        """
        # Strip whitespace
        normalized = material_name.strip()

        # Handle common acronyms that should be uppercase
        acronyms = ['SKM', 'BSJ', 'SP']
        upper_name = normalized.upper()
        if upper_name in acronyms:
            return upper_name

        # For other materials, use title case for consistency
        return normalized.title()

    def process_sales_data(self) -> pd.DataFrame:
        """
        Process sales data to calculate daily raw material requirements

        Returns:
            DataFrame with columns: Date, Raw_Material, Quantity_Required
        """
        # Convert Date column to datetime and extract just the date
        self.sales_df['Date'] = pd.to_datetime(self.sales_df['Date']).dt.date

        # Dictionary to store daily raw material requirements
        # Structure: {(date, raw_material): quantity}
        daily_requirements = {}

        # Process each sales transaction
        total_rows = len(self.sales_df)
        for idx, row in self.sales_df.iterrows():
            if idx % 100 == 0:
                print(f"Processing row {idx + 1}/{total_rows}...")

            date = row['Date']
            item = row['Item']
            quantity = row['Quantity']

            # Skip if quantity is invalid
            if pd.isna(quantity) or quantity <= 0:
                continue

            # Get raw materials for this item
            raw_materials = self._get_item_raw_materials(item, quantity)

            # Aggregate by date and raw material (with normalized names)
            for material, material_qty in raw_materials.items():
                normalized_material = self._normalize_material_name(material)
                key = (date, normalized_material)
                daily_requirements[key] = daily_requirements.get(key, 0) + material_qty

        # Convert to DataFrame
        result_data = []
        for (date, material), qty in daily_requirements.items():
            result_data.append({
                'Date': date,
                'Raw_Material': material,
                'Quantity_Required': qty
            })

        result_df = pd.DataFrame(result_data)

        # Sort by date and raw material
        result_df = result_df.sort_values(['Date', 'Raw_Material']).reset_index(drop=True)

        return result_df

    def save_results(self, output_path: str):
        """
        Process sales data and save results to CSV

        Args:
            output_path: Path where the output CSV should be saved
        """
        print("Starting raw material requirements processing...")
        result_df = self.process_sales_data()

        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Save to CSV
        result_df.to_csv(output_path, index=False)
        print(f"\nProcessing complete!")
        print(f"Results saved to: {output_path}")
        print(f"Total unique dates: {result_df['Date'].nunique()}")
        print(f"Total unique raw materials: {result_df['Raw_Material'].nunique()}")
        print(f"Total rows in output: {len(result_df)}")


def main():
    """Main execution function"""
    # Define file paths
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sales_path = os.path.join(base_path, 'data', 'processed', 'sales_data_cleaned.csv')
    menu_bom_path = os.path.join(base_path, 'data', 'raw', 'bom', 'menu_bom.csv')
    condiment_bom_path = os.path.join(base_path, 'data', 'raw', 'bom', 'condiment_bom.csv')
    output_path = os.path.join(base_path, 'data', 'processed', 'daily_raw_material_requirements.csv')

    # Initialize processor
    processor = RawMaterialProcessor(sales_path, menu_bom_path, condiment_bom_path)

    # Process and save
    processor.save_results(output_path)


if __name__ == "__main__":
    main()
