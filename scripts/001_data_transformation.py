"""
Sales Data Transformation for Forecasting

This script transforms raw sales data into daily aggregated format suitable for sales forecasting.
It:
1. Loads and cleans sales data
2. Aggregates sales by date and menu item
3. Adds temporal features (day of week, month, etc.)
4. Categorizes menu items based on BOM
5. Saves transformed data in forecasting-ready format

Usage:
    python 001_data_transformation.py
"""

import pandas as pd
import numpy as np
import os
import sys
from typing import Dict, List, Set, Optional
from datetime import datetime, timedelta
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')


class SalesDataTransformer:
    """Transform sales data for forecasting purposes"""

    def __init__(self, sales_path: str, menu_bom_path: str):
        """
        Initialize the transformer with file paths

        Args:
            sales_path: Path to cleaned sales data CSV
            menu_bom_path: Path to menu BOM CSV
        """
        self.sales_path = sales_path
        self.menu_bom_path = menu_bom_path
        
        # Load data
        print("Loading data...")
        self.sales_df = pd.read_csv(sales_path)
        self.menu_bom_df = pd.read_csv(menu_bom_path)
        
        # Build mappings
        self.item_category_map = self._build_item_category_map()
        self.category_list = sorted(self.menu_bom_df['Tipe'].unique())
        
        # Statistics
        self.stats = {
            'original_transactions': len(self.sales_df),
            'unique_items': self.sales_df['Item'].nunique(),
            'date_range_start': None,
            'date_range_end': None,
            'total_days': 0,
            'daily_records_created': 0
        }

    def _build_item_category_map(self) -> Dict[str, str]:
        """
        Build mapping from item name to category

        Returns:
            Dictionary mapping item names to categories
        """
        item_category_map = {}
        
        for _, row in self.menu_bom_df.iterrows():
            item_name = row['Item'].strip()
            category = row['Tipe'].strip()
            item_category_map[item_name] = category
            
        return item_category_map

    def _clean_and_prepare_data(self) -> pd.DataFrame:
        """
        Clean and prepare sales data for aggregation

        Returns:
            Cleaned DataFrame
        """
        print("Cleaning and preparing data...")
        
        # Make a copy to avoid modifying original
        df = self.sales_df.copy()
        
        # Convert Date column to datetime
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Extract date part (remove time)
        df['Date_Only'] = df['Date'].dt.date
        
        # Clean Item names
        df['Item'] = df['Item'].str.strip()
        
        # Consolidate espresso variants - treat 'espresso bon-bon' as 'espresso'
        df['Item'] = df['Item'].str.replace(r'^espresso bon-bon$', 'espresso', case=False, regex=True)
        
        # Filter out invalid quantities
        df = df[df['Quantity'] > 0]
        
        # Filter out items with 'add' prefix (case insensitive) - these are modifiers
        initial_count = len(df)
        df = df[~df['Item'].str.lower().str.startswith('add')]
        filtered_count = initial_count - len(df)
        if filtered_count > 0:
            print(f"Filtered out {filtered_count} transactions with 'add' prefix (modifiers)")
        
        # Filter out discontinued items - cheese cake
        initial_count = len(df)
        df = df[~df['Item'].str.lower().str.contains('cheese cake')]
        filtered_count = initial_count - len(df)
        if filtered_count > 0:
            print(f"Filtered out {filtered_count} transactions for discontinued item 'cheese cake'")
        
        # Filter out returns/refunds (negative quantities already handled above)
        
        # Update statistics
        self.stats['date_range_start'] = df['Date_Only'].min()
        self.stats['date_range_end'] = df['Date_Only'].max()
        self.stats['total_days'] = (self.stats['date_range_end'] - self.stats['date_range_start']).days + 1
        
        print(f"Data prepared: {len(df)} transactions from {self.stats['date_range_start']} to {self.stats['date_range_end']}")
        
        return df

    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add temporal features for forecasting

        Args:
            df: DataFrame with Date_Only column

        Returns:
            DataFrame with added temporal features
        """
        print("Adding temporal features...")
        
        # Convert Date_Only back to datetime for feature extraction
        df['Date'] = pd.to_datetime(df['Date_Only'])
        
        # Basic temporal features
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        df['Day'] = df['Date'].dt.day
        df['DayOfWeek'] = df['Date'].dt.dayofweek  # 0=Monday, 6=Sunday
        df['DayOfWeekName'] = df['Date'].dt.day_name()
        df['WeekOfYear'] = df['Date'].dt.isocalendar().week
        df['Quarter'] = df['Date'].dt.quarter
        df['DayOfYear'] = df['Date'].dt.dayofyear
        
        # Weekend indicator
        df['IsWeekend'] = (df['DayOfWeek'] >= 5).astype(int)
        
        # Month progress (0-1, where 0.5 is middle of month)
        df['MonthProgress'] = df['Day'] / df['Date'].dt.days_in_month
        
        return df

    def _add_category_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add category information to items

        Args:
            df: DataFrame with Item column

        Returns:
            DataFrame with category information
        """
        print("Adding category features...")
        
        # Map items to categories
        df['Category'] = df['Item'].map(self.item_category_map)
        
        # Count items without category
        uncategorized = df['Category'].isna().sum()
        if uncategorized > 0:
            print(f"Warning: {uncategorized} items could not be categorized")
            df['Category'] = df['Category'].fillna('Unknown')
        
        return df

    def aggregate_daily_sales(self) -> pd.DataFrame:
        """
        Aggregate sales data to daily level

        Returns:
            DataFrame with daily aggregated sales
        """
        print("Aggregating daily sales...")
        
        # Clean and prepare data
        clean_df = self._clean_and_prepare_data()
        
        # Add features
        clean_df = self._add_temporal_features(clean_df)
        clean_df = self._add_category_features(clean_df)
        
        # Aggregate by date and item
        daily_sales = clean_df.groupby(['Date_Only', 'Item', 'Category']).agg({
            'Quantity': 'sum',
            'Net sales': 'sum',
            'Gross sales': 'sum'
        }).reset_index()
        
        # Add temporal features back
        temp_features = clean_df[['Date_Only', 'Date', 'Year', 'Month', 'Day', 
                                 'DayOfWeek', 'DayOfWeekName', 'WeekOfYear', 
                                 'Quarter', 'DayOfYear', 'IsWeekend', 'MonthProgress']].drop_duplicates()
        
        # Merge temporal features
        daily_sales = daily_sales.merge(temp_features, on='Date_Only', how='left')
        
        # Sort by date and item
        daily_sales = daily_sales.sort_values(['Date_Only', 'Item']).reset_index(drop=True)
        
        # Update statistics
        self.stats['daily_records_created'] = len(daily_sales)
        
        print(f"Daily aggregation complete: {len(daily_sales)} daily records created")
        
        return daily_sales

    def create_category_aggregates(self, daily_sales: pd.DataFrame) -> pd.DataFrame:
        """
        Create category-level daily aggregates

        Args:
            daily_sales: Daily sales data

        Returns:
            DataFrame with category-level aggregates
        """
        print("Creating category-level aggregates...")
        
        # Aggregate by date and category
        category_sales = daily_sales.groupby(['Date_Only', 'Category']).agg({
            'Quantity': 'sum',
            'Net sales': 'sum',
            'Gross sales': 'sum',
            'Item': 'count'  # Number of unique items in category
        }).rename(columns={'Item': 'UniqueItemCount'}).reset_index()
        
        # Add temporal features
        temp_features = daily_sales[['Date_Only', 'Date', 'Year', 'Month', 'Day', 
                                 'DayOfWeek', 'DayOfWeekName', 'WeekOfYear', 
                                 'Quarter', 'DayOfYear', 'IsWeekend', 'MonthProgress']].drop_duplicates()
        
        category_sales = category_sales.merge(temp_features, on='Date_Only', how='left')
        
        # Sort by date and category
        category_sales = category_sales.sort_values(['Date_Only', 'Category']).reset_index(drop=True)
        
        return category_sales

    def create_total_daily_sales(self, daily_sales: pd.DataFrame) -> pd.DataFrame:
        """
        Create total daily sales (all items combined)

        Args:
            daily_sales: Daily sales data

        Returns:
            DataFrame with total daily sales
        """
        print("Creating total daily sales...")
        
        # Aggregate by date only
        total_sales = daily_sales.groupby('Date_Only').agg({
            'Quantity': 'sum',
            'Net sales': 'sum',
            'Gross sales': 'sum',
            'Item': 'count',  # Number of unique items sold
            'Category': 'nunique'  # Number of categories represented
        }).rename(columns={
            'Item': 'UniqueItemCount',
            'Category': 'UniqueCategoryCount'
        }).reset_index()
        
        # Add temporal features
        temp_features = daily_sales[['Date_Only', 'Date', 'Year', 'Month', 'Day', 
                                 'DayOfWeek', 'DayOfWeekName', 'WeekOfYear', 
                                 'Quarter', 'DayOfYear', 'IsWeekend', 'MonthProgress']].drop_duplicates()
        
        total_sales = total_sales.merge(temp_features, on='Date_Only', how='left')
        
        # Sort by date
        total_sales = total_sales.sort_values('Date_Only').reset_index(drop=True)
        
        return total_sales

    def save_transformed_data(self, daily_sales: pd.DataFrame, 
                            category_sales: pd.DataFrame, 
                            total_sales: pd.DataFrame,
                            output_dir: str):
        """
        Save all transformed datasets

        Args:
            daily_sales: Daily item-level sales
            category_sales: Daily category-level sales
            total_sales: Total daily sales
            output_dir: Directory to save files
        """
        print("Saving transformed data...")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Save datasets
        daily_sales.to_csv(os.path.join(output_dir, 'daily_item_sales.csv'), index=False)
        category_sales.to_csv(os.path.join(output_dir, 'daily_category_sales.csv'), index=False)
        total_sales.to_csv(os.path.join(output_dir, 'daily_total_sales.csv'), index=False)
        
        # Save metadata
        metadata = {
            'transformation_date': datetime.now().isoformat(),
            'statistics': self.stats,
            'categories': self.category_list,
            'top_items': daily_sales.groupby('Item')['Quantity'].sum().sort_values(ascending=False).head(20).to_dict(),
            'data_periods': {
                'start_date': str(self.stats['date_range_start']),
                'end_date': str(self.stats['date_range_end']),
                'total_days': self.stats['total_days']
            }
        }
        
        import json
        with open(os.path.join(output_dir, 'transformation_metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        print(f"Data saved to: {output_dir}")
        print(f"  - daily_item_sales.csv: {len(daily_sales)} records")
        print(f"  - daily_category_sales.csv: {len(category_sales)} records")
        print(f"  - daily_total_sales.csv: {len(total_sales)} records")

    def print_summary(self):
        """Print transformation summary"""
        print("\n" + "=" * 80)
        print("TRANSFORMATION SUMMARY")
        print("=" * 80)
        
        print(f"\nOriginal Data:")
        print(f"  - Total transactions: {self.stats['original_transactions']:,}")
        print(f"  - Unique items: {self.stats['unique_items']:,}")
        
        print(f"\nDate Range:")
        print(f"  - Start: {self.stats['date_range_start']}")
        print(f"  - End: {self.stats['date_range_end']}")
        print(f"  - Total days: {self.stats['total_days']}")
        
        print(f"\nTransformed Data:")
        print(f"  - Daily item records: {self.stats['daily_records_created']:,}")
        
        print(f"\nCategories:")
        for i, category in enumerate(self.category_list, 1):
            print(f"  {i}. {category}")
        
        print("=" * 80)


def main():
    """Main execution function"""
    # Define file paths
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sales_path = os.path.join(base_path, 'data', 'processed', 'sales_data_cleaned.csv')
    menu_bom_path = os.path.join(base_path, 'data', 'raw', 'bom', 'menu_bom.csv')
    output_dir = os.path.join(base_path, 'data', 'processed', 'sales_forecasting')
    
    print('=' * 80)
    print('SALES DATA TRANSFORMATION FOR FORECASTING')
    print('=' * 80)
    print(f'\nSales data: {sales_path}')
    print(f'Menu BOM: {menu_bom_path}')
    print(f'Output directory: {output_dir}')
    
    # Check if input files exist
    if not os.path.exists(sales_path):
        print(f"\nError: Sales data file not found: {sales_path}")
        print("Please run the sales data cleaning script first.")
        return
    
    if not os.path.exists(menu_bom_path):
        print(f"\nError: Menu BOM file not found: {menu_bom_path}")
        return
    
    try:
        # Initialize transformer
        transformer = SalesDataTransformer(sales_path, menu_bom_path)
        
        # Transform data
        daily_sales = transformer.aggregate_daily_sales()
        category_sales = transformer.create_category_aggregates(daily_sales)
        total_sales = transformer.create_total_daily_sales(daily_sales)
        
        # Save results
        transformer.save_transformed_data(daily_sales, category_sales, total_sales, output_dir)
        
        # Print summary
        transformer.print_summary()
        
        print('\n✓ Sales data transformation complete!')
        
    except Exception as e:
        print(f"\nError during transformation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()