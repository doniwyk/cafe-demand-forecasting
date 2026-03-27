import pandas as pd
import numpy as np
from typing import Optional
from pathlib import Path

from src.utils.config import PROCESSED_DIR


COLUMN_MAPPING = {
    "Tanggal": "Date",
    "Nomor struk": "Receipt number",
    "Jenis struk": "Receipt type",
    "Kategori": "Category",
    "SKU": "SKU",
    "Barang": "Item",
    "Varian": "Variant",
    "Pemodifikasi diterapkan": "Modifiers applied",
    "Kuantitas": "Quantity",
    "Penjualan Kotor": "Gross sales",
    "Diskon": "Discounts",
    "Penjualan bersih": "Net sales",
    "Harga pokok": "Cost of goods",
    "Laba kotor": "Gross profit",
    "Pajak": "Taxes",
    "Jenis pesanan": "Dining option",
    "POS": "POS",
    "Toko": "Store",
    "Nama Kasir": "Cashier name",
    "Nama Pelanggan": "Customer name",
    "Kontak Pelanggan": "Customer contacts",
    "Komentar": "Comment",
    "Keadaan": "Status",
}

VALUE_MAPPINGS = {
    "Receipt type": {"Penjualan": "Sale"},
    "Dining option": {"Makan di tempat": "Dine in"},
    "Status": {"Ditutup": "Closed"},
}

NUMERIC_COLUMNS = [
    "Quantity",
    "Gross sales",
    "Discounts",
    "Net sales",
    "Cost of goods",
    "Gross profit",
    "Taxes",
]

COLUMN_ORDER = [
    "Date",
    "Receipt number",
    "Receipt type",
    "Category",
    "SKU",
    "Item",
    "Variant",
    "Modifiers applied",
    "Quantity",
    "Gross sales",
    "Discounts",
    "Net sales",
    "Cost of goods",
    "Gross profit",
    "Taxes",
    "Dining option",
    "POS",
    "Store",
    "Cashier name",
    "Customer name",
    "Customer contacts",
    "Comment",
    "Status",
]

DATE_FORMATS = [
    "%d/%m/%y %H.%M",
    "%d/%m/%y %H:%M",
    "%m/%d/%y %I:%M %p",
    "%m/%d/%Y %I:%M %p",
]


def translate_indonesian_to_english(df_indonesian: pd.DataFrame) -> pd.DataFrame:
    df_indonesian = df_indonesian.rename(columns=COLUMN_MAPPING)

    for column, mapping in VALUE_MAPPINGS.items():
        if column in df_indonesian.columns:
            df_indonesian[column] = df_indonesian[column].replace(mapping)

    return df_indonesian


def parse_date(date_str) -> pd.Timestamp:
    if pd.isna(date_str):
        return pd.NaT

    date_str = str(date_str).strip()

    for fmt in DATE_FORMATS:
        try:
            return pd.to_datetime(date_str, format=fmt)
        except Exception:
            continue

    try:
        return pd.to_datetime(date_str)
    except Exception:
        return pd.NaT


def clean_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(r"[^\d.]", "", regex=True)
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def merge_sales_files(
    file1_path: str | Path,
    file2_path: str | Path,
    output_path: str | Path,
) -> Optional[pd.DataFrame]:
    print(f"Reading file 1: {file1_path}")
    print(f"Reading file 2: {file2_path}")

    try:
        df1 = pd.read_csv(file1_path, sep=";")
        print(f"File 1 loaded successfully. Shape: {df1.shape}")

        df2 = pd.read_csv(file2_path, sep=",")
        print(f"File 2 loaded successfully. Shape: {df2.shape}")
    except Exception as e:
        print(f"Error reading files: {e}")
        return None

    print("Translating Indonesian file to English...")
    df1_translated = translate_indonesian_to_english(df1)

    print("Cleaning numeric columns...")
    df1_translated = clean_numeric_columns(df1_translated)
    df2 = clean_numeric_columns(df2)

    print("Parsing dates...")
    df1_translated["Date"] = df1_translated["Date"].apply(parse_date)
    df2["Date"] = df2["Date"].apply(parse_date)

    all_columns = set(df1_translated.columns) | set(df2.columns)
    for col in all_columns:
        if col not in df1_translated.columns:
            df1_translated[col] = np.nan
        if col not in df2.columns:
            df2[col] = np.nan

    df1_translated = df1_translated[COLUMN_ORDER]
    df2 = df2[COLUMN_ORDER]

    print("Combining dataframes...")
    combined_df = pd.concat([df1_translated, df2], ignore_index=True)

    print("Sorting by date...")
    combined_df = combined_df.sort_values("Date", na_position="last")
    combined_df = combined_df.reset_index(drop=True)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined_df.to_csv(output_path, index=False)

    print(f"Merge completed successfully!")
    print(f"Total records: {len(combined_df)}")
    print(f"Date range: {combined_df['Date'].min()} to {combined_df['Date'].max()}")

    return combined_df
