import pandas as pd
import numpy as np
import os

def translate_indonesian_to_english(df_indonesian):
    """Translate Indonesian column names and values to English"""
    
    # Column mapping from Indonesian to English
    column_mapping = {
        'Tanggal': 'Date',
        'Nomor struk': 'Receipt number',
        'Jenis struk': 'Receipt type',
        'Kategori': 'Category',
        'SKU': 'SKU',
        'Barang': 'Item',
        'Varian': 'Variant',
        'Pemodifikasi diterapkan': 'Modifiers applied',
        'Kuantitas': 'Quantity',
        'Penjualan Kotor': 'Gross sales',
        'Diskon': 'Discounts',
        'Penjualan bersih': 'Net sales',
        'Harga pokok': 'Cost of goods',
        'Laba kotor': 'Gross profit',
        'Pajak': 'Taxes',
        'Jenis pesanan': 'Dining option',
        'POS': 'POS',
        'Toko': 'Store',
        'Nama Kasir': 'Cashier name',
        'Nama Pelanggan': 'Customer name',
        'Kontak Pelanggan': 'Customer contacts',
        'Komentar': 'Comment',
        'Keadaan': 'Status'
    }
    
    # Rename columns
    df_indonesian = df_indonesian.rename(columns=column_mapping)
    
    # Translate specific values
    value_mappings = {
        'Receipt type': {
            'Penjualan': 'Sale'
        },
        'Dining option': {
            'Makan di tempat': 'Dine in'
        },
        'Status': {
            'Ditutup': 'Closed'
        }
    }
    
    # Apply value translations
    for column, mapping in value_mappings.items():
        if column in df_indonesian.columns:
            df_indonesian[column] = df_indonesian[column].replace(mapping)
    
    return df_indonesian

def parse_date(date_str):
    """Parse different date formats to standard datetime"""
    if pd.isna(date_str):
        return pd.NaT
    
    date_str = str(date_str).strip()
    
    # Try different date formats
    formats_to_try = [
        '%d/%m/%y %H.%M',  # Indonesian format: 13/05/25 23.43
        '%d/%m/%y %H:%M',  # Alternative Indonesian format
        '%m/%d/%y %I:%M %p',  # English format: 9/25/25 10:07 PM
        '%m/%d/%Y %I:%M %p',  # English format with 4-digit year
    ]
    
    for fmt in formats_to_try:
        try:
            return pd.to_datetime(date_str, format=fmt)
        except:
            continue
    
    # If none of the specific formats work, try pandas auto-detection
    try:
        return pd.to_datetime(date_str)
    except:
        return pd.NaT

def clean_numeric_columns(df):
    """Clean and convert numeric columns"""
    numeric_columns = [
        'Quantity', 'Gross sales', 'Discounts', 'Net sales', 
        'Cost of goods', 'Gross profit', 'Taxes'
    ]
    
    for col in numeric_columns:
        if col in df.columns:
            # Remove any non-numeric characters except decimal point
            df[col] = df[col].astype(str).str.replace(r'[^\d.]', '', regex=True)
            # Convert to numeric, errors='coerce' will turn invalid values to NaN
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df

def merge_sales_files(file1_path, file2_path, output_path):
    """Merge two sales CSV files with different languages"""
    
    print(f"Reading file 1: {file1_path}")
    print(f"Reading file 2: {file2_path}")
    
    # Read both files
    try:
        # Indonesian file (first file)
        df1 = pd.read_csv(file1_path, sep=';')
        print(f"File 1 loaded successfully. Shape: {df1.shape}")
        
        # English file (second file)
        df2 = pd.read_csv(file2_path, sep=',')
        print(f"File 2 loaded successfully. Shape: {df2.shape}")
        
    except Exception as e:
        print(f"Error reading files: {e}")
        return
    
    # Translate Indonesian file to English
    print("Translating Indonesian file to English...")
    df1_translated = translate_indonesian_to_english(df1)
    
    # Clean numeric columns in both dataframes
    print("Cleaning numeric columns...")
    df1_translated = clean_numeric_columns(df1_translated)
    df2 = clean_numeric_columns(df2)
    
    # Parse dates in both dataframes
    print("Parsing dates...")
    df1_translated['Date'] = df1_translated['Date'].apply(parse_date)
    df2['Date'] = df2['Date'].apply(parse_date)
    
    # Ensure both dataframes have the same columns
    all_columns = set(df1_translated.columns) | set(df2.columns)
    
    for col in all_columns:
        if col not in df1_translated.columns:
            df1_translated[col] = np.nan
        if col not in df2.columns:
            df2[col] = np.nan
    
    # Reorder columns to match English file order
    column_order = [
        'Date', 'Receipt number', 'Receipt type', 'Category', 'SKU', 'Item', 
        'Variant', 'Modifiers applied', 'Quantity', 'Gross sales', 'Discounts', 
        'Net sales', 'Cost of goods', 'Gross profit', 'Taxes', 'Dining option', 
        'POS', 'Store', 'Cashier name', 'Customer name', 'Customer contacts', 
        'Comment', 'Status'
    ]
    
    df1_translated = df1_translated[column_order]
    df2 = df2[column_order]
    
    # Combine the dataframes
    print("Combining dataframes...")
    combined_df = pd.concat([df1_translated, df2], ignore_index=True)
    
    # Sort by date
    print("Sorting by date...")
    combined_df = combined_df.sort_values('Date', na_position='last')
    
    # Reset index
    combined_df = combined_df.reset_index(drop=True)
    
    # Save the merged file
    print(f"Saving merged file to: {output_path}")
    combined_df.to_csv(output_path, index=False)
    
    print(f"Merge completed successfully!")
    print(f"Total records: {len(combined_df)}")
    print(f"Date range: {combined_df['Date'].min()} to {combined_df['Date'].max()}")
    
    return combined_df

if __name__ == "__main__":
    # Define file paths
    file1_path = "data/raw/sales/receipts-by-item-2022-01-01-2025-06-30.csv"
    file2_path = "data/raw/sales/receipts-by-item-2025-05-01-2025-09-25.csv"
    output_path = "data/processed/sales_data.csv"
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Merge the files
    merged_data = merge_sales_files(file1_path, file2_path, output_path)
    
    if merged_data is not None:
        print("\nFirst few rows of merged data:")
        print(merged_data.head())
        
        print("\nData summary:")
        print(merged_data.info())