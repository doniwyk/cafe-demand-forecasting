from __future__ import annotations

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
    file_paths: list[str | Path],
    output_path: str | Path,
) -> Optional[pd.DataFrame]:
    if len(file_paths) < 2:
        print("Error: At least 2 file paths are required")
        return None

    dfs = []
    for i, fp in enumerate(file_paths):
        print(f"Reading file {i + 1}: {fp}")
        try:
            sep = ";" if i == 0 else ","
            df = pd.read_csv(fp, sep=sep)
            print(f"File {i + 1} loaded successfully. Shape: {df.shape}")
            dfs.append(df)
        except Exception as e:
            print(f"Error reading file {i + 1}: {e}")
            return None

    print(f"Translating file 1 (Indonesian) to English...")
    dfs[0] = translate_indonesian_to_english(dfs[0])

    print("Cleaning numeric columns...")
    dfs[0] = clean_numeric_columns(dfs[0])
    for i in range(1, len(dfs)):
        dfs[i] = clean_numeric_columns(dfs[i])

    print("Parsing dates...")
    dfs[0]["Date"] = dfs[0]["Date"].apply(parse_date)
    for i in range(1, len(dfs)):
        dfs[i]["Date"] = dfs[i]["Date"].apply(parse_date)

    all_columns = set().union(*[set(df.columns) for df in dfs])
    for i in range(len(dfs)):
        for col in all_columns:
            if col not in dfs[i].columns:
                dfs[i][col] = np.nan
        dfs[i] = dfs[i][COLUMN_ORDER]

    print("Combining dataframes...")
    combined_df = pd.concat(dfs, ignore_index=True)

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
