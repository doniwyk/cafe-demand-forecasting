"""
Item Name Standardization Script

This script standardizes item names in sales data to match the menu BOM nomenclature.
It handles:
1. Simple renames (e.g., "lemon tea hot" → "Lemon Hot")
2. Variant consolidation (e.g., multiple V60 variants → "v60")
3. Package deals that need to be split into multiple items
4. Verification against menu BOM to ensure all items are valid
"""

import pandas as pd
import os
from typing import Dict, List, Tuple


class ItemNameStandardizer:
    """Standardize item names in sales data to match menu BOM"""

    def __init__(self, sales_path: str, menu_bom_path: str):
        """
        Initialize the standardizer with file paths

        Args:
            sales_path: Path to sales data CSV
            menu_bom_path: Path to menu BOM CSV
        """
        self.sales_path = sales_path
        self.menu_bom_path = menu_bom_path
        self.sales_df = pd.read_csv(sales_path)
        self.menu_bom_df = pd.read_csv(menu_bom_path)

        # Build rename mapping
        self.rename_map = self._build_rename_mapping()
        self.package_items = self._build_package_mapping()

    def _build_rename_mapping(self) -> Dict[str, str]:
        """
        Build mapping of old names to standardized names

        Returns:
            Dictionary mapping old item names to new standardized names
        """
        # Define rename mappings (sales name → BOM name)
        rename_map = {
            # Tea items
            'lemon tea hot': 'Lemon Hot',
            'lemon tea ice': 'Lemon Ice',
            'hot tea ajeng': 'Ajeng Hot',
            'ice tea ajeng': 'Ajeng Ice',
            'hot tea anindya': 'Anindya Hot',
            'ice tea anindya': 'Anindya Ice',

            # Coffee items
            'kopi susu panas': 'Kopi Susu Hot',
            'ice cappucino': 'Cappucino Ice',
            'latte': 'Latte Hot',
            'ice latte': 'Latte Ice',
            'picolo': 'Piccolo',
            'long black hot': 'Black Hot',
            'long black ice': 'Black Ice',

            # Milk-based items
            'red velvet': 'Red Velvet Ice',
            'susu coklat': 'Coklat Hot',

            # V60 variants (all consolidate to v60)
            'v60 hacienda natural': 'v60',
            'v60 argopuro': 'v60',
            'v60 finca': 'v60',

            # Noodle items
            'mie goreng telur': 'Mie Goreng',  # Note: Will add telur separately
            'mie rebus telur': 'Mie Rebus',    # Note: Will add telur separately

            # Rice items
            'nasi goreng djawa': 'Nasi Goreng Jawa',
            'nasi ayam daun jeruk': 'Ayam Daun Jeruk',
            'nasi ayam rempah': 'Ayam Rempah',

            # Main course items
            'ayam mentega (paket)': 'Ayam Mentega',
            'ayam curry (paket)': 'Ayam Curry',
            'ayam dauh jeruk (paket)': 'Ayam Daun Jeruk',

            # Snacks
            'cireng isi': 'Cireng',

            # Desserts
            'pannacotta': 'Panacotta',

            # Additional items
            'add vanilla ice cream': 'Add Ice Cream Vanilla',
            'add telur ceplok': 'Add Telur',
            'add telur dadar': 'Add Telur',
            'add caramel topping': 'Add Topping Caramel',
        }

        return rename_map

    def _build_package_mapping(self) -> Dict[str, List[Tuple[str, float]]]:
        """
        Build mapping for package deals that need to be split

        Returns:
            Dictionary mapping package names to list of (item, quantity multiplier) tuples
        """
        package_map = {
            # Mie with telur - add telur as separate item
            'mie goreng telur': [
                ('Mie Goreng', 1.0),
                ('Add Telur', 1.0)
            ],
            'mie rebus telur': [
                ('Mie Rebus', 1.0),
                ('Add Telur', 1.0)
            ],
            # Package deals
            'paket ayam teriyaki + lemon tea': [
                ('Ayam Teriyaki', 1.0),
                ('Lemon Ice', 1.0)
            ],
            'paket ayam curry + lemon tea': [
                ('Ayam Curry', 1.0),
                ('Lemon Ice', 1.0)
            ],
        }

        return package_map

    def standardize_names(self) -> pd.DataFrame:
        """
        Standardize item names in sales data

        Returns:
            DataFrame with standardized item names
        """
        print('Standardizing item names...')

        # Create a copy to work with
        standardized_df = self.sales_df.copy()

        # Track changes
        renamed_count = 0
        expanded_count = 0

        # Apply simple renames first
        for old_name, new_name in self.rename_map.items():
            # Case-insensitive matching
            mask = standardized_df['Item'].str.lower() == old_name.lower()
            if mask.any():
                count = mask.sum()
                standardized_df.loc[mask, 'Item'] = new_name
                renamed_count += count
                print(f'  Renamed: "{old_name}" → "{new_name}" ({count} records)')

        # Handle package items that need to be split
        expanded_rows = []
        rows_to_remove = []

        for idx, row in standardized_df.iterrows():
            item_lower = str(row['Item']).lower()

            if item_lower in self.package_items:
                # Mark this row for removal
                rows_to_remove.append(idx)

                # Create new rows for each component
                components = self.package_items[item_lower]
                original_qty = row['Quantity']

                for component_item, qty_multiplier in components:
                    new_row = row.copy()
                    new_row['Item'] = component_item
                    new_row['Quantity'] = original_qty * qty_multiplier
                    expanded_rows.append(new_row)

                expanded_count += 1
                print(f'  Expanded: "{row["Item"]}" → {len(components)} items ({original_qty} qty each)')

        # Remove package items
        if rows_to_remove:
            standardized_df = standardized_df.drop(rows_to_remove)

        # Add expanded rows
        if expanded_rows:
            expanded_df = pd.DataFrame(expanded_rows)
            standardized_df = pd.concat([standardized_df, expanded_df], ignore_index=True)

        # Sort by date
        standardized_df = standardized_df.sort_values('Date').reset_index(drop=True)

        print(f'\nStandardization complete:')
        print(f'  Simple renames: {renamed_count} records')
        print(f'  Package expansions: {expanded_count} packages → {len(expanded_rows)} items')
        print(f'  Total records: {len(self.sales_df)} → {len(standardized_df)}')

        return standardized_df

    def verify_against_bom(self, standardized_df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """
        Verify that all items in standardized sales data exist in menu BOM

        Args:
            standardized_df: DataFrame with standardized item names

        Returns:
            Tuple of (list of valid items, list of items not found in BOM)
        """
        # Get unique items from BOM (normalize for comparison)
        bom_items = set()
        for item in self.menu_bom_df['Item'].unique():
            bom_items.add(item.strip().lower())

        # Get unique items from sales
        sales_items = standardized_df['Item'].dropna().unique()

        valid_items = []
        invalid_items = []

        for item in sales_items:
            if item.strip().lower() in bom_items:
                valid_items.append(item)
            else:
                invalid_items.append(item)

        return valid_items, invalid_items

    def save_standardized_data(self, standardized_df: pd.DataFrame, output_path: str):
        """
        Save standardized sales data to CSV

        Args:
            standardized_df: Standardized DataFrame
            output_path: Path where CSV should be saved
        """
        # Create output directory if needed
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Save to CSV
        standardized_df.to_csv(output_path, index=False)

        print(f'\n✓ Standardized data saved to: {output_path}')
        print(f'  Total records: {len(standardized_df)}')


def print_verification_report(valid_items: List[str], invalid_items: List[str]):
    """
    Print verification report

    Args:
        valid_items: List of items that match BOM
        invalid_items: List of items not found in BOM
    """
    print('\n' + '=' * 80)
    print('VERIFICATION REPORT')
    print('=' * 80)

    print(f'\n✓ Valid items (found in BOM): {len(valid_items)}')

    if invalid_items:
        print(f'\n✗ Items NOT found in BOM: {len(invalid_items)}')
        print('-' * 80)
        for item in sorted(invalid_items):
            print(f'  - {item}')
        print('-' * 80)
        print('\n⚠ WARNING: Some items in sales data are not in the menu BOM!')
        print('  You may need to add these items to the BOM or create additional rename mappings.')
    else:
        print('\n✓ All items successfully matched to menu BOM!')

    print('=' * 80)


def main():
    """Main execution function"""
    # Define file paths
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sales_path = os.path.join(base_path, 'data', 'processed', 'sales_data.csv')
    menu_bom_path = os.path.join(base_path, 'data', 'raw', 'bom', 'menu_bom.csv')
    output_path = os.path.join(base_path, 'data', 'processed', 'sales_data.csv')

    print('=' * 80)
    print('ITEM NAME STANDARDIZATION')
    print('=' * 80)
    print(f'\nSales data: {sales_path}')
    print(f'Menu BOM: {menu_bom_path}')
    print(f'Output: {output_path}')

    # Initialize standardizer
    standardizer = ItemNameStandardizer(sales_path, menu_bom_path)

    print(f'\nOriginal records: {len(standardizer.sales_df)}')

    # Standardize names
    print('\n' + '-' * 80)
    standardized_df = standardizer.standardize_names()

    # Verify against BOM
    print('\n' + '-' * 80)
    print('Verifying items against menu BOM...')
    valid_items, invalid_items = standardizer.verify_against_bom(standardized_df)

    # Print verification report
    print_verification_report(valid_items, invalid_items)

    # Save standardized data
    standardizer.save_standardized_data(standardized_df, output_path)

    print('\n✓ Standardization complete!\n')


if __name__ == '__main__':
    main()
