"""
Sales Data Cleaner Script

This script cleans and standardizes sales data to match the menu BOM. It:
1. Standardizes item names (renames, variant consolidation)
2. Expands package deals into individual items
3. Identifies discontinued/invalid items not in BOM
4. Optionally removes discontinued items
5. Saves cleaned and standardized sales data

Usage:
    python clean_sales_data.py                  # Interactive mode
    python clean_sales_data.py --remove         # Auto-remove discontinued items
    python clean_sales_data.py --no-remove      # Standardize only, keep all items
    python clean_sales_data.py --help           # Show this help
"""

import pandas as pd
import os
import sys
from typing import Dict, List, Set, Tuple, Optional


class SalesDataCleaner:
    """Clean and standardize sales data based on menu BOM"""

    def __init__(self, sales_path: str, menu_bom_path: str):
        """
        Initialize the cleaner with file paths

        Args:
            sales_path: Path to sales data CSV
            menu_bom_path: Path to menu BOM CSV
        """
        self.sales_path = sales_path
        self.menu_bom_path = menu_bom_path
        self.sales_df = pd.read_csv(sales_path)
        self.menu_bom_df = pd.read_csv(menu_bom_path)

        # Build mappings
        self.rename_map = self._build_rename_mapping()
        self.package_items = self._build_package_mapping()
        self.active_items = self._get_active_items()

        # Track statistics
        self.stats = {
            'original_records': len(self.sales_df),
            'renamed_records': 0,
            'expanded_packages': 0,
            'expanded_items': 0,
            'discontinued_items': 0,
            'removed_records': 0
        }

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
            'mie goreng telur': 'Mie Goreng',
            'mie rebus telur': 'Mie Rebus',

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

    def _get_active_items(self) -> Set[str]:
        """
        Get set of active items from menu BOM

        Returns:
            Set of normalized active item names
        """
        active_items = set()

        # Get unique items from menu BOM
        for item_name in self.menu_bom_df['Item'].unique():
            # Normalize by stripping whitespace and converting to lowercase
            normalized = item_name.strip().lower()
            active_items.add(normalized)

        return active_items

    def _normalize_item_name(self, item_name: str) -> str:
        """
        Normalize item name for comparison

        Args:
            item_name: Original item name

        Returns:
            Normalized item name (lowercase, stripped)
        """
        if pd.isna(item_name):
            return ''
        return str(item_name).strip().lower()

    def standardize_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize item names in sales data

        Args:
            df: DataFrame to standardize

        Returns:
            DataFrame with standardized item names
        """
        print('\n' + '-' * 80)
        print('STEP 1: STANDARDIZING ITEM NAMES')
        print('-' * 80)

        # Create a copy to work with
        standardized_df = df.copy()

        # Apply simple renames first
        renamed_count = 0
        for old_name, new_name in self.rename_map.items():
            # Case-insensitive matching
            mask = standardized_df['Item'].str.lower() == old_name.lower()
            if mask.any():
                count = mask.sum()
                standardized_df.loc[mask, 'Item'] = new_name
                renamed_count += count
                print(f'  ✓ Renamed: "{old_name}" → "{new_name}" ({count} records)')

        self.stats['renamed_records'] = renamed_count

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

                print(f'  ✓ Expanded: "{row["Item"]}" → {len(components)} items ({original_qty} qty each)')

        # Remove package items
        if rows_to_remove:
            standardized_df = standardized_df.drop(rows_to_remove)
            self.stats['expanded_packages'] = len(rows_to_remove)

        # Add expanded rows
        if expanded_rows:
            expanded_df = pd.DataFrame(expanded_rows)
            standardized_df = pd.concat([standardized_df, expanded_df], ignore_index=True)
            self.stats['expanded_items'] = len(expanded_rows)

        # Sort by date
        standardized_df = standardized_df.sort_values('Date').reset_index(drop=True)

        print(f'\nStandardization Summary:')
        print(f'  Simple renames: {renamed_count} records')
        print(f'  Package expansions: {len(rows_to_remove)} packages → {len(expanded_rows)} items')
        print(f'  Total records: {len(df)} → {len(standardized_df)}')

        return standardized_df

    def identify_discontinued_items(self, df: pd.DataFrame) -> Tuple[Set[str], pd.DataFrame, pd.DataFrame]:
        """
        Identify items in sales data that are not in menu BOM

        Args:
            df: DataFrame to check

        Returns:
            Tuple of (set of discontinued item names, DataFrame with discontinued items stats, DataFrame with valid items)
        """
        print('\n' + '-' * 80)
        print('STEP 2: IDENTIFYING DISCONTINUED ITEMS')
        print('-' * 80)

        # Get all unique items from sales data
        sales_items = df['Item'].dropna().unique()

        # Find discontinued items (in sales but not in menu BOM)
        discontinued_items = set()
        valid_items = set()

        for item in sales_items:
            normalized = self._normalize_item_name(item)
            if normalized and normalized not in self.active_items:
                discontinued_items.add(item)  # Store original name for display
            else:
                valid_items.add(item)

        self.stats['discontinued_items'] = len(discontinued_items)

        # Create statistics DataFrame for discontinued items
        discontinued_stats = []
        for item in discontinued_items:
            item_data = df[df['Item'] == item]
            stats = {
                'Item': item,
                'Total_Quantity_Sold': item_data['Quantity'].sum(),
                'Total_Transactions': len(item_data),
                'First_Sale_Date': item_data['Date'].min(),
                'Last_Sale_Date': item_data['Date'].max()
            }
            discontinued_stats.append(stats)

        discontinued_df = pd.DataFrame(discontinued_stats)

        # Sort by total quantity sold (descending)
        if not discontinued_df.empty:
            discontinued_df = discontinued_df.sort_values('Total_Quantity_Sold', ascending=False)

        print(f'\n✓ Valid items (found in BOM): {len(valid_items)}')
        print(f'✗ Discontinued items (NOT in BOM): {len(discontinued_items)}')

        return discontinued_items, discontinued_df, df

    def remove_discontinued_items(self, df: pd.DataFrame, discontinued_items: Set[str]) -> pd.DataFrame:
        """
        Remove discontinued items from sales data

        Args:
            df: DataFrame to clean
            discontinued_items: Set of discontinued item names to remove

        Returns:
            Cleaned DataFrame with discontinued items removed
        """
        # Filter out discontinued items
        cleaned_df = df[~df['Item'].isin(discontinued_items)].copy()

        self.stats['removed_records'] = len(df) - len(cleaned_df)

        return cleaned_df

    def save_cleaned_data(self, cleaned_df: pd.DataFrame, output_path: str):
        """
        Save cleaned sales data to CSV

        Args:
            cleaned_df: Cleaned DataFrame
            output_path: Path where cleaned CSV should be saved
        """
        # Create output directory if needed
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Save to CSV
        cleaned_df.to_csv(output_path, index=False)

        print(f'\n✓ Cleaned data saved to: {output_path}')
        print(f'  Total records: {len(cleaned_df)}')


def print_discontinued_report(discontinued_items: Set[str], discontinued_df: pd.DataFrame):
    """
    Print detailed report of discontinued items

    Args:
        discontinued_items: Set of discontinued item names
        discontinued_df: DataFrame with discontinued items statistics
    """
    if not discontinued_items:
        print('\n✓ No discontinued items found!')
        print('  All items in sales data exist in the current menu BOM.')
        return

    print(f'\nTotal discontinued items: {len(discontinued_items)}')
    print(f'Total transactions affected: {discontinued_df["Total_Transactions"].sum():.0f}')
    print(f'Total quantity sold: {discontinued_df["Total_Quantity_Sold"].sum():.0f}')

    print('\n' + '-' * 80)
    print('DISCONTINUED ITEMS LIST (Top 20):')
    print('-' * 80)

    # Print table header
    print(f'{"Item":<40} {"Qty Sold":<12} {"Transactions":<15} {"Last Sale"}')
    print('-' * 80)

    # Print top 20 discontinued items
    for idx, row in discontinued_df.head(100).iterrows():
        item_name = row['Item'][:39]  # Truncate if too long
        qty_sold = f"{row['Total_Quantity_Sold']:.0f}"
        transactions = f"{row['Total_Transactions']:.0f}"
        last_sale = str(row['Last_Sale_Date'])[:10]

        print(f'{item_name:<40} {qty_sold:<12} {transactions:<15} {last_sale}')

    if len(discontinued_df) > 20:
        print(f'\n... and {len(discontinued_df) - 20} more items')


def get_user_confirmation(auto_mode: Optional[bool] = None) -> bool:
    """
    Ask user for confirmation to remove discontinued items

    Args:
        auto_mode: If True, auto-confirm. If False, auto-reject. If None, ask user.

    Returns:
        True if user confirms, False otherwise
    """
    # If auto mode is specified, use it
    if auto_mode is True:
        print('\n[AUTO-REMOVE MODE] Automatically removing discontinued items...')
        return True
    elif auto_mode is False:
        print('\n[KEEP-ALL MODE] Keeping all items, including discontinued ones...')
        return False

    # Interactive mode
    print('\n' + '?' * 80)
    print('REMOVAL CONFIRMATION')
    print('?' * 80)

    try:
        while True:
            response = input('\nDo you want to remove discontinued items from sales data? (yes/no): ').strip().lower()

            if response in ['yes', 'y']:
                return True
            elif response in ['no', 'n']:
                return False
            else:
                print('Invalid input. Please enter "yes" or "no".')
    except (EOFError, KeyboardInterrupt):
        print('\n\nInterrupted. No changes will be made.')
        return False


def print_final_summary(stats: Dict):
    """
    Print final summary of cleaning process

    Args:
        stats: Dictionary with cleaning statistics
    """
    print('\n' + '=' * 80)
    print('FINAL SUMMARY')
    print('=' * 80)

    print(f'\nOriginal records: {stats["original_records"]:,}')
    print(f'\nStandardization:')
    print(f'  - Renamed records: {stats["renamed_records"]:,}')
    print(f'  - Expanded packages: {stats["expanded_packages"]:,} → {stats["expanded_items"]:,} items')

    print(f'\nDiscontinued items:')
    print(f'  - Items found: {stats["discontinued_items"]:,}')
    print(f'  - Records removed: {stats["removed_records"]:,}')

    final_records = stats["original_records"] + stats["expanded_items"] - stats["removed_records"]
    print(f'\nFinal records: {final_records:,}')
    print('=' * 80)


def main():
    """Main execution function"""
    # Parse command line arguments
    auto_mode = None
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        if arg in ['--remove', '-r', '--yes', '-y']:
            auto_mode = True
        elif arg in ['--no-remove', '-n', '--no', '--keep-all']:
            auto_mode = False
        elif arg in ['--help', '-h']:
            print(__doc__)
            return
        else:
            print(f'Unknown argument: {sys.argv[1]}')
            print('Use --help to see usage information')
            return

    # Define file paths
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sales_path = os.path.join(base_path, 'data', 'processed', 'sales_data.csv')
    menu_bom_path = os.path.join(base_path, 'data', 'raw', 'bom', 'menu_bom.csv')
    output_path = os.path.join(base_path, 'data', 'processed', 'sales_data_cleaned.csv')

    print('=' * 80)
    print('SALES DATA CLEANER')
    print('=' * 80)
    print(f'\nSales data: {sales_path}')
    print(f'Menu BOM: {menu_bom_path}')
    print(f'Output: {output_path}')

    # Initialize cleaner
    cleaner = SalesDataCleaner(sales_path, menu_bom_path)

    print(f'\nOriginal records: {len(cleaner.sales_df):,}')

    # Step 1: Standardize names
    standardized_df = cleaner.standardize_names(cleaner.sales_df)

    # Step 2: Identify discontinued items
    discontinued_items, discontinued_df, current_df = cleaner.identify_discontinued_items(standardized_df)

    # Print discontinued items report
    print_discontinued_report(discontinued_items, discontinued_df)

    # Step 3: Handle discontinued items
    print('\n' + '-' * 80)
    print('STEP 3: HANDLING DISCONTINUED ITEMS')
    print('-' * 80)

    final_df = current_df

    if discontinued_items:
        if get_user_confirmation(auto_mode):
            print('\nRemoving discontinued items...')
            final_df = cleaner.remove_discontinued_items(current_df, discontinued_items)
            print(f'✓ Removed {cleaner.stats["removed_records"]:,} records')
        else:
            print('\n✗ Keeping all items (including discontinued ones)')
            cleaner.stats['removed_records'] = 0
    else:
        print('\n✓ No discontinued items to remove')

    # Save cleaned data
    print('\n' + '-' * 80)
    print('STEP 4: SAVING CLEANED DATA')
    print('-' * 80)
    cleaner.save_cleaned_data(final_df, output_path)

    # Print final summary
    print_final_summary(cleaner.stats)

    print('\n✓ Sales data cleaning complete!\n')


if __name__ == '__main__':
    main()
