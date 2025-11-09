"""
Discontinued Items Cleaner Script

This script identifies items in sales data that are not present in the menu BOM
(discontinued products) and provides an interactive option to remove them. It:
1. Compares sales items against current menu BOM
2. Reports discontinued items with statistics
3. Interactively asks user whether to remove discontinued items
4. Saves cleaned sales data if removal is confirmed

Usage:
    python clean_discontinued_items.py              # Interactive mode
    python clean_discontinued_items.py --remove     # Auto-remove discontinued items
    python clean_discontinued_items.py --no-remove  # Only report, don't remove
"""

import pandas as pd
import os
import sys
from typing import List, Set, Tuple, Optional


class DiscontinuedItemsCleaner:
    """Clean discontinued items from sales data based on menu BOM"""

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

        # Get unique items from menu BOM (current active items)
        self.active_items = self._get_active_items()

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

    def identify_discontinued_items(self) -> Tuple[Set[str], pd.DataFrame]:
        """
        Identify items in sales data that are not in menu BOM

        Returns:
            Tuple of (set of discontinued item names, DataFrame with discontinued items stats)
        """
        # Get all unique items from sales data
        sales_items = self.sales_df['Item'].dropna().unique()

        # Find discontinued items (in sales but not in menu BOM)
        discontinued_items = set()

        for item in sales_items:
            normalized = self._normalize_item_name(item)
            if normalized and normalized not in self.active_items:
                discontinued_items.add(item)  # Store original name for display

        # Create statistics DataFrame
        discontinued_stats = []
        for item in discontinued_items:
            item_data = self.sales_df[self.sales_df['Item'] == item]
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

        return discontinued_items, discontinued_df

    def remove_discontinued_items(self, discontinued_items: Set[str]) -> pd.DataFrame:
        """
        Remove discontinued items from sales data

        Args:
            discontinued_items: Set of discontinued item names to remove

        Returns:
            Cleaned DataFrame with discontinued items removed
        """
        # Filter out discontinued items
        cleaned_df = self.sales_df[~self.sales_df['Item'].isin(discontinued_items)].copy()

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
        print(f'  Total records remaining: {len(cleaned_df)}')


def print_discontinued_report(discontinued_items: Set[str], discontinued_df: pd.DataFrame):
    """
    Print detailed report of discontinued items

    Args:
        discontinued_items: Set of discontinued item names
        discontinued_df: DataFrame with discontinued items statistics
    """
    print('\n' + '=' * 80)
    print('DISCONTINUED ITEMS REPORT')
    print('=' * 80)

    if not discontinued_items:
        print('\n✓ No discontinued items found!')
        print('  All items in sales data exist in the current menu BOM.')
        return

    print(f'\nTotal discontinued items found: {len(discontinued_items)}')
    print(f'Total transactions affected: {discontinued_df["Total_Transactions"].sum():.0f}')
    print(f'Total quantity sold: {discontinued_df["Total_Quantity_Sold"].sum():.0f}')

    print('\n' + '-' * 80)
    print('DISCONTINUED ITEMS LIST:')
    print('-' * 80)

    # Print table header
    print(f'{"Item":<35} {"Qty Sold":<12} {"Transactions":<15} {"Last Sale"}')
    print('-' * 80)

    # Print each discontinued item
    for _, row in discontinued_df.iterrows():
        item_name = row['Item'][:34]  # Truncate if too long
        qty_sold = f"{row['Total_Quantity_Sold']:.0f}"
        transactions = f"{row['Total_Transactions']:.0f}"
        last_sale = str(row['Last_Sale_Date'])[:10]

        print(f'{item_name:<35} {qty_sold:<12} {transactions:<15} {last_sale}')

    print('=' * 80)


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
        print('\n[REPORT-ONLY MODE] Skipping removal...')
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


def main():
    """Main execution function"""
    # Parse command line arguments
    auto_mode = None
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        if arg in ['--remove', '-r', '--yes', '-y']:
            auto_mode = True
        elif arg in ['--no-remove', '-n', '--no', '--report-only']:
            auto_mode = False
        elif arg in ['--help', '-h']:
            print(__doc__)
            return

    # Define file paths
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sales_path = os.path.join(base_path, 'data', 'processed', 'sales_data.csv')
    menu_bom_path = os.path.join(base_path, 'data', 'raw', 'bom', 'menu_bom.csv')
    output_path = os.path.join(base_path, 'data', 'processed', 'sales_data.csv')

    print('Starting discontinued items analysis...')
    print(f'Sales data: {sales_path}')
    print(f'Menu BOM: {menu_bom_path}')

    # Initialize cleaner
    cleaner = DiscontinuedItemsCleaner(sales_path, menu_bom_path)

    # Get initial record count
    initial_count = len(cleaner.sales_df)
    print(f'\nTotal records in sales data: {initial_count}')

    # Identify discontinued items
    discontinued_items, discontinued_df = cleaner.identify_discontinued_items()

    # Print report
    print_discontinued_report(discontinued_items, discontinued_df)

    # If there are discontinued items, ask user what to do
    if discontinued_items:
        if get_user_confirmation(auto_mode):
            print('\nRemoving discontinued items...')

            # Remove discontinued items
            cleaned_df = cleaner.remove_discontinued_items(discontinued_items)

            # Calculate removal statistics
            removed_count = initial_count - len(cleaned_df)
            print(f'\n✓ Removed {removed_count} records ({removed_count/initial_count*100:.1f}% of total)')

            # Save cleaned data
            cleaner.save_cleaned_data(cleaned_df, output_path)

            print('\n✓ Sales data has been cleaned successfully!')

        else:
            print('\n✗ No changes made. Sales data remains unchanged.')
    else:
        print('\nNo action needed. All items are active.')


if __name__ == '__main__':
    main()

