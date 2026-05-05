"""Seed the database with CSV data from ml-model/data/processed/."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from sqlalchemy import text

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.db.engine import sync_session
from app.config import ML_DATA_DIR, ML_RAW_DIR, ML_PROCESSED_DIR

SALES_FORECASTING_DIR = ML_PROCESSED_DIR / "sales_forecasting"
from app.db.models import (
    Category, Item, DailyItemSale, DailyCategorySale,
    DailyTotalSale, BomRecipe, CondimentRecipe, SaleCleaned,
)

PROCESSED_DIR = ML_DATA_DIR / "processed"
RAW_DIR = ML_RAW_DIR


def _upsert_category(session, name: str) -> int | None:
    if not name:
        return None
    existing = session.execute(
        text("SELECT id FROM categories WHERE name = :name"), {"name": name}
    ).fetchone()
    if existing:
        return existing[0]
    result = session.execute(
        text("INSERT INTO categories (name) VALUES (:name) RETURNING id"),
        {"name": name},
    )
    return result.scalar()


def _upsert_item(session, name: str, category_id: int | None = None) -> int | None:
    if not name:
        return None
    existing = session.execute(
        text("SELECT id FROM items WHERE name = :name"), {"name": name}
    ).fetchone()
    if existing:
        return existing[0]
    result = session.execute(
        text("INSERT INTO items (name, category_id) VALUES (:name, :cat_id) RETURNING id"),
        {"name": name, "cat_id": category_id},
    )
    return result.scalar()


def seed_categories_and_items(session, items_csv: Path):
    print("Seeding categories and items...")
    df = pd.read_csv(items_csv)
    df.columns = df.columns.str.strip()

    item_to_cat: dict[str, str] = {}
    cleaned_csv = PROCESSED_DIR / "sales_data_cleaned.csv"
    if cleaned_csv.exists():
        df_clean = pd.read_csv(cleaned_csv, low_memory=False)
        df_clean.columns = df_clean.columns.str.strip()
        if "Item" in df_clean.columns and "Category" in df_clean.columns:
            for _, row in df_clean.dropna(subset=["Item", "Category"]).iterrows():
                item_to_cat[str(row["Item"])] = str(row["Category"])

    categories: set[str] = set(item_to_cat.values())
    if not categories:
        cat_sales_csv = PROCESSED_DIR / "daily_category_sales_seed.csv"
        if cat_sales_csv.exists():
            df_cat = pd.read_csv(cat_sales_csv)
            df_cat.columns = df_cat.columns.str.strip()
            if "Category" in df_cat.columns:
                categories = set(df_cat["Category"].dropna().unique())

    cat_map: dict[str, int] = {}
    for cat in sorted(categories):
        cat_id = _upsert_category(session, cat)
        if cat_id:
            cat_map[cat] = cat_id

    items = df["Item"].dropna().unique()
    for item_name in items:
        cat_name = item_to_cat.get(item_name)
        cat_id = cat_map.get(cat_name) if cat_name else None
        _upsert_item(session, item_name, cat_id)

    session.flush()
    print(f"  Categories: {len(cat_map)}, Items: {len(items)}")


def seed_daily_item_sales(session, csv_path: Path):
    print("Seeding daily_item_sales...")
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    date_col = "Date_Only" if "Date_Only" in df.columns else "Date"
    qty_col = "Quantity" if "Quantity" in df.columns else "Quantity_Sold"
    df["date"] = pd.to_datetime(df[date_col]).dt.date
    df["qty"] = df[qty_col].astype(float)

    item_rows = session.execute(text("SELECT id, name FROM items")).fetchall()
    item_map = {r[1]: r[0] for r in item_rows}

    inserted = 0
    for _, row in df.iterrows():
        item_id = item_map.get(row["Item"])
        if item_id is None:
            continue
        existing = session.execute(
            text("SELECT 1 FROM daily_item_sales WHERE date = :d AND item_id = :iid"),
            {"d": row["date"], "iid": item_id},
        ).fetchone()
        if existing:
            continue
        session.execute(
            text("INSERT INTO daily_item_sales (date, item_id, quantity_sold) VALUES (:d, :iid, :qty)"),
            {"d": row["date"], "iid": item_id, "qty": row["qty"]},
        )
        inserted += 1

    session.flush()
    print(f"  daily_item_sales: {inserted} rows")


def seed_daily_category_sales(session, csv_path: Path):
    print("Seeding daily_category_sales...")
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    df["date"] = pd.to_datetime(df["Date"]).dt.date

    inserted = 0
    for _, row in df.iterrows():
        existing = session.execute(
            text("SELECT 1 FROM daily_category_sales WHERE date = :d AND category = :c"),
            {"d": row["date"], "c": row["Category"]},
        ).fetchone()
        if existing:
            continue
        session.execute(
            text(
                "INSERT INTO daily_category_sales (date, category, quantity, net_sales, gross_sales, unique_items) "
                "VALUES (:d, :c, :q, :ns, :gs, :ui)"
            ),
            {
                "d": row["date"], "c": row["Category"],
                "q": float(row["Quantity"]), "ns": float(row["Net sales"]),
                "gs": float(row["Gross sales"]), "ui": int(row["UniqueItemCount"]),
            },
        )
        inserted += 1

    session.flush()
    print(f"  daily_category_sales: {inserted} rows")


def seed_daily_total_sales(session, csv_path: Path):
    print("Seeding daily_total_sales...")
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    df["date"] = pd.to_datetime(df["Date"]).dt.date

    inserted = 0
    for _, row in df.iterrows():
        existing = session.execute(
            text("SELECT 1 FROM daily_total_sales WHERE date = :d"),
            {"d": row["date"]},
        ).fetchone()
        if existing:
            continue
        session.execute(
            text(
                "INSERT INTO daily_total_sales (date, quantity, net_sales, gross_sales, unique_items) "
                "VALUES (:d, :q, :ns, :gs, :ui)"
            ),
            {
                "d": row["date"], "q": float(row["Quantity"]),
                "ns": float(row["Net sales"]), "gs": float(row["Gross sales"]),
                "ui": int(row["UniqueItemCount"]),
            },
        )
        inserted += 1

    session.flush()
    print(f"  daily_total_sales: {inserted} rows")


def seed_bom_recipes(session, csv_path: Path):
    print("Seeding bom_recipes...")
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    item_rows = session.execute(text("SELECT id, name FROM items")).fetchall()
    item_map = {r[1]: r[0] for r in item_rows}
    cat_rows = session.execute(text("SELECT id, name FROM categories")).fetchall()
    cat_map = {r[1]: r[0] for r in cat_rows}

    inserted = 0
    for _, row in df.iterrows():
        session.execute(
            text(
                "INSERT INTO bom_recipes (category_name, item_name, ingredient, quantity, unit, category_id, item_id) "
                "VALUES (:cn, :in, :ig, :q, :u, :cid, :iid)"
            ),
            {
                "cn": row["Tipe"], "in": row["Item"], "ig": row["Bahan"],
                "q": float(row["Qty"]), "u": row["Unit"],
                "cid": cat_map.get(row["Tipe"]),
                "iid": item_map.get(row["Item"]),
            },
        )
        inserted += 1

    session.flush()
    print(f"  bom_recipes: {inserted} rows")


def seed_condiment_recipes(session, csv_path: Path):
    print("Seeding condiment_recipes...")
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    inserted = 0
    for _, row in df.iterrows():
        session.execute(
            text(
                "INSERT INTO condiment_recipes (condiment, condiment_qty, condiment_unit, "
                "sub_ingredient, qty_per_unit, sub_unit) "
                "VALUES (:c, :cq, :cu, :si, :qp, :su)"
            ),
            {
                "c": row["Condiment"], "cq": float(row["Condiment_Qty"]),
                "cu": row["Condiment_Unit"], "si": row["Sub_Ingredient"],
                "qp": float(row["Qty_per_condiment_unit"]), "su": row["Sub_Unit"],
            },
        )
        inserted += 1

    session.flush()
    print(f"  condiment_recipes: {inserted} rows")


def seed_sales_cleaned(session, csv_path: Path):
    print("Seeding sales_cleaned...")
    df = pd.read_csv(csv_path, low_memory=False)
    df.columns = df.columns.str.strip()

    col_map = {
        "Date": "date", "Receipt number": "receipt_number", "Receipt type": "receipt_type",
        "Category": "category", "SKU": "sku", "Item": "item", "Variant": "variant",
        "Modifiers applied": "modifiers_applied", "Quantity": "quantity",
        "Gross sales": "gross_sales", "Discounts": "discounts", "Net sales": "net_sales",
        "Cost of goods": "cost_of_goods", "Gross profit": "gross_profit", "Taxes": "taxes",
        "Dining option": "dining_option", "POS": "pos", "Store": "store",
        "Cashier name": "cashier_name", "Customer name": "customer_name", "Status": "status",
    }
    df = df.rename(columns=col_map)
    df["date"] = pd.to_datetime(df["date"])

    numeric_cols = [
        "quantity", "gross_sales", "discounts", "net_sales",
        "cost_of_goods", "gross_profit", "taxes",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    batch_size = 5000
    total = 0
    for start in range(0, len(df), batch_size):
        batch = df.iloc[start : start + batch_size]
        for _, row in batch.iterrows():
            session.execute(
                text(
                    "INSERT INTO sales_cleaned (date, receipt_number, receipt_type, category, "
                    "sku, item, variant, modifiers_applied, quantity, gross_sales, discounts, "
                    "net_sales, cost_of_goods, gross_profit, taxes, dining_option, pos, store, "
                    "cashier_name, customer_name, status) VALUES "
                    "(:d, :rn, :rt, :cat, :sku, :item, :var, :mod, :qty, :gs, :disc, :ns, "
                    ":cog, :gp, :tax, :do, :pos, :store, :cn, :cust, :st)"
                ),
                {
                    "d": row.get("date"), "rn": row.get("receipt_number"),
                    "rt": row.get("receipt_type"), "cat": row.get("category"),
                    "sku": row.get("sku"), "item": row.get("item"),
                    "var": row.get("variant"), "mod": row.get("modifiers_applied"),
                    "qty": row.get("quantity"), "gs": row.get("gross_sales"),
                    "disc": row.get("discounts"), "ns": row.get("net_sales"),
                    "cog": row.get("cost_of_goods"), "gp": row.get("gross_profit"),
                    "tax": row.get("taxes"), "do": row.get("dining_option"),
                    "pos": row.get("pos"), "store": row.get("store"),
                    "cn": row.get("cashier_name"), "cust": row.get("customer_name"),
                    "st": row.get("status"),
                },
            )
            total += 1
        session.flush()
        print(f"  sales_cleaned: {total} rows...")

    print(f"  sales_cleaned: {total} total rows")


def seed():
    session = sync_session()

    try:
        seed_categories_and_items(session, SALES_FORECASTING_DIR / "daily_item_sales.csv")

        if (SALES_FORECASTING_DIR / "daily_item_sales.csv").exists():
            seed_daily_item_sales(session, SALES_FORECASTING_DIR / "daily_item_sales.csv")

        if (PROCESSED_DIR / "daily_category_sales_seed.csv").exists():
            seed_daily_category_sales(session, PROCESSED_DIR / "daily_category_sales_seed.csv")

        if (PROCESSED_DIR / "daily_total_sales_seed.csv").exists():
            seed_daily_total_sales(session, PROCESSED_DIR / "daily_total_sales_seed.csv")

        if (RAW_DIR / "bom" / "menu_bom.csv").exists():
            seed_bom_recipes(session, RAW_DIR / "bom" / "menu_bom.csv")

        if (RAW_DIR / "bom" / "condiment_bom.csv").exists():
            seed_condiment_recipes(session, RAW_DIR / "bom" / "condiment_bom.csv")

        if (PROCESSED_DIR / "sales_data_cleaned.csv").exists():
            seed_sales_cleaned(session, PROCESSED_DIR / "sales_data_cleaned.csv")

        session.commit()
        print("\nDatabase seeded successfully!")
    except Exception as e:
        session.rollback()
        print(f"Error seeding database: {e}")
        raise
    finally:
        session.close()


if __name__ == "__main__":
    seed()
