"""
Seed PostgreSQL database from existing CSV files.

Usage:
    cd ml-model
    python scripts/seed_database.py
    python scripts/seed_database.py --skip-sales-cleaned   # Skip the large ~55K row table
    python scripts/seed_database.py --truncate             # Clear all tables first
"""

import sys
import os
import json
import argparse
from datetime import datetime, date

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.db import SessionLocal, Base, get_sync_url
from src.db.models import (
    Category,
    Item,
    BomRecipe,
    CondimentRecipe,
    SaleCleaned,
    DailyItemSale,
    DailyCategorySale,
    DailyTotalSale,
    ModelRun,
    ModelRunClassMetric,
    ModelRunTopItem,
    Forecast,
    ItemABC,
    AssociationRule,
    RawMaterialRequirement,
)
from src.utils.config import (
    PROCESSED_DIR,
    PREDICTIONS_DIR,
    RAW_DIR,
    MODELS_DIR,
    BOM_DIR,
)

CHUNK_SIZE = 5000


def parse_number(val):
    if pd.isna(val):
        return None
    s = str(val).strip().replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return None


def seed_categories(session):
    bom_df = pd.read_csv(BOM_DIR / "menu_bom.csv")
    categories = bom_df["Tipe"].dropna().str.strip().unique()

    cat_map = {}
    for cat_name in sorted(categories):
        cat = Category(name=cat_name)
        session.add(cat)
        session.flush()
        cat_map[cat_name] = cat.id
        print(f"  Category: {cat_name} (id={cat.id})")

    session.commit()
    return cat_map


def seed_items(session, cat_map):
    bom_df = pd.read_csv(BOM_DIR / "menu_bom.csv")
    item_names = bom_df["Item"].dropna().str.strip().unique()

    item_map = {}
    for item_name in sorted(item_names):
        cat_name = bom_df[bom_df["Item"].str.strip() == item_name]["Tipe"].iloc[0]
        cat_id = cat_map.get(cat_name.strip())
        item = Item(name=item_name, category_id=cat_id)
        session.add(item)
        session.flush()
        item_map[item_name] = item.id

    session.commit()
    print(f"  Seeded {len(item_map)} items")
    return item_map


def seed_bom_recipes(session, cat_map, item_map):
    bom_df = pd.read_csv(BOM_DIR / "menu_bom.csv")
    bom_df.columns = bom_df.columns.str.strip()

    for _, row in bom_df.iterrows():
        item_name = str(row["Item"]).strip()
        cat_name = str(row["Tipe"]).strip()
        qty = _safe_float(row["Qty"])
        if qty is None:
            continue
        recipe = BomRecipe(
            category_name=cat_name,
            item_name=item_name,
            ingredient=str(row["Bahan"]).strip(),
            quantity=_safe_float(row["Qty"]),
            unit=str(row["Unit"]).strip(),
            category_id=cat_map.get(cat_name),
            item_id=item_map.get(item_name),
        )
        session.add(recipe)

    session.commit()
    print(f"  Seeded {len(bom_df)} BOM recipes")


def seed_condiment_recipes(session):
    cond_df = pd.read_csv(BOM_DIR / "condiment_bom.csv")
    cond_df.columns = cond_df.columns.str.strip()

    for _, row in cond_df.iterrows():
        qty_per_unit = _safe_float(row["Qty_per_condiment_unit"])
        if qty_per_unit is None:
            continue
        recipe = CondimentRecipe(
            condiment=str(row["Condiment"]).strip(),
            condiment_qty=_safe_float(row["Condiment_Qty"]),
            condiment_unit=str(row["Condiment_Unit"]).strip(),
            sub_ingredient=str(row["Sub_Ingredient"]).strip(),
            qty_per_unit=_safe_float(row["Qty_per_condiment_unit"]),
            sub_unit=str(row["Sub_Unit"]).strip(),
        )
        session.add(recipe)

    session.commit()
    print(f"  Seeded {len(cond_df)} condiment recipes")


def seed_sales_cleaned(session, skip=False):
    if skip:
        print("  Skipped (use --no-skip-sales-cleaned to include)")
        return

    filepath = PROCESSED_DIR / "sales_data_cleaned.csv"
    if not filepath.exists():
        print(f"  File not found: {filepath}")
        return

    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()
    print(f"  Seeding {len(df)} cleaned sales rows...")

    for i in range(0, len(df), CHUNK_SIZE):
        chunk = df.iloc[i : i + CHUNK_SIZE]
        rows = []
        for _, row in chunk.iterrows():
            dt = pd.to_datetime(row.get("Date"))
            rows.append(
                SaleCleaned(
                    date=dt.to_pydatetime() if pd.notna(dt) else None,
                    receipt_number=str(row.get("Receipt number", "")).strip() or None,
                    receipt_type=str(row.get("Receipt type", "")).strip() or None,
                    category=str(row.get("Category", "")).strip() or None,
                    sku=str(row.get("SKU", "")).strip() or None,
                    item=str(row.get("Item", "")).strip() or None,
                    variant=str(row.get("Variant", "")).strip() or None,
                    modifiers_applied=str(row.get("Modifiers applied", "")).strip()
                    or None,
                    quantity=_safe_float(row.get("Quantity")),
                    gross_sales=_safe_float(row.get("Gross sales")),
                    discounts=_safe_float(row.get("Discounts")),
                    net_sales=_safe_float(row.get("Net sales")),
                    cost_of_goods=_safe_float(row.get("Cost of goods")),
                    gross_profit=_safe_float(row.get("Gross profit")),
                    taxes=_safe_float(row.get("Taxes")),
                    dining_option=str(row.get("Dining option", "")).strip() or None,
                    pos=str(row.get("POS", "")).strip() or None,
                    store=str(row.get("Store", "")).strip() or None,
                    cashier_name=str(row.get("Cashier name", "")).strip() or None,
                    customer_name=str(row.get("Customer name", "")).strip() or None,
                    status=str(row.get("Status", "")).strip() or None,
                )
            )
        session.bulk_save_objects(rows)
        if (i + CHUNK_SIZE) % (CHUNK_SIZE * 10) == 0 or i + CHUNK_SIZE >= len(df):
            session.commit()
            print(f"    ... {min(i + CHUNK_SIZE, len(df))}/{len(df)}")

    print(f"  Done: {len(df)} rows")


def seed_daily_item_sales(session, item_map):
    filepath = PROCESSED_DIR / "daily_item_sales.csv"
    if not filepath.exists():
        print(f"  File not found: {filepath}")
        return

    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()
    missing = []
    count = 0
    for _, row in df.iterrows():
        item_name = str(row["Item"]).strip()
        item_id = item_map.get(item_name)
        if item_id is None:
            missing.append(item_name)
            continue
        d = pd.to_datetime(row["Date"]).date()
        session.add(
            DailyItemSale(
                date=d,
                item_id=item_id,
                quantity_sold=float(row["Quantity_Sold"]),
            )
        )
        count += 1
    session.commit()
    print(f"  Seeded {count} daily item sales rows")
    if missing:
        print(f"  Warning: {len(set(missing))} items not in DB: {set(missing)[:5]}...")


def seed_daily_category_sales(session):
    filepath = PROCESSED_DIR / "daily_category_sales.csv"
    if not filepath.exists():
        print(f"  File not found: {filepath}")
        return

    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()
    for _, row in df.iterrows():
        session.add(
            DailyCategorySale(
                date=pd.to_datetime(row["Date"]).date(),
                category=str(row["Category"]).strip(),
                quantity=float(row["Quantity"]),
                net_sales=float(row["Net sales"]),
                gross_sales=float(row["Gross sales"]),
                unique_items=int(row["UniqueItemCount"]),
            )
        )
    session.commit()
    print(f"  Seeded {len(df)} daily category sales rows")


def seed_daily_total_sales(session):
    filepath = PROCESSED_DIR / "daily_total_sales.csv"
    if not filepath.exists():
        print(f"  File not found: {filepath}")
        return

    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()
    for _, row in df.iterrows():
        session.add(
            DailyTotalSale(
                date=pd.to_datetime(row["Date"]).date(),
                quantity=float(row["Quantity"]),
                net_sales=float(row["Net sales"]),
                gross_sales=float(row["Gross sales"]),
                unique_items=int(row["UniqueItemCount"]),
            )
        )
    session.commit()
    print(f"  Seeded {len(df)} daily total sales rows")


def seed_model_runs(session, item_map):
    meta_path = MODELS_DIR / "model_metadata.json"
    summary_path = PREDICTIONS_DIR / "forecast_summary.json"

    if not meta_path.exists() or not summary_path.exists():
        print("  Model metadata or forecast summary not found, skipping")
        return

    with open(meta_path) as f:
        meta = json.load(f)
    with open(summary_path) as f:
        summary = json.load(f)

    gm = summary["global_metrics"]
    run = ModelRun(
        model_type="xgboost",
        trained_at=datetime.now(),
        n_item_models=meta.get("n_item_models"),
        n_records=meta.get("n_records"),
        date_range_start=pd.to_datetime(meta["date_range"][0]).date()
        if meta.get("date_range")
        else None,
        date_range_end=pd.to_datetime(meta["date_range"][1]).date()
        if meta.get("date_range")
        else None,
        r2=gm.get("r2"),
        wmape=gm.get("wmape"),
        mae=gm.get("mae"),
        volume_accuracy=gm.get("volume_accuracy"),
        features=json.dumps(meta.get("features", [])),
        items_with_models=json.dumps(meta.get("items_with_models", [])),
        is_active=True,
    )
    session.add(run)
    session.flush()

    for cls, cm in summary.get("class_metrics", {}).items():
        session.add(
            ModelRunClassMetric(
                model_run_id=run.id,
                abc_class=cls,
                n_items=cm["n_items"],
                wmape=cm["wmape"],
                volume_accuracy=cm["volume_accuracy"],
            )
        )

    for t in summary.get("top_items", []):
        session.add(
            ModelRunTopItem(
                model_run_id=run.id,
                item_name=t["Item"],
                quantity_sold=t["Quantity_Sold"],
                predicted=t["Predicted"],
                accuracy_pct=t["accuracy_pct"],
            )
        )

    session.commit()
    print(f"  Seeded model run (id={run.id})")

    return run


def seed_forecasts(session, model_run, item_map):
    if model_run is None:
        print("  No model run, skipping forecasts")
        return

    filepath = PREDICTIONS_DIR / "3_month_forecasts.csv"
    if not filepath.exists():
        print(f"  File not found: {filepath}")
        return

    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()
    count = 0
    for _, row in df.iterrows():
        item_name = str(row["Item"]).strip()
        item_id = item_map.get(item_name)
        if item_id is None:
            continue
        session.add(
            Forecast(
                model_run_id=model_run.id,
                item_id=item_id,
                date=pd.to_datetime(row["Date"]).date(),
                quantity_predicted=float(row["Quantity_Sold"]),
            )
        )
        count += 1
    session.commit()
    print(f"  Seeded {count} forecast rows")


def seed_association_rules(session):
    filepath = PROCESSED_DIR / "association_rules_fpgrowth.csv"
    if not filepath.exists():
        print(f"  File not found: {filepath}")
        return

    import re

    df = pd.read_csv(filepath)
    for _, row in df.iterrows():

        def clean_fs(val):
            m = re.search(r"'([^']+)'\}", str(val))
            return m.group(1) if m else str(val)

        session.add(
            AssociationRule(
                antecedents=clean_fs(row.get("antecedents", "")),
                consequents=clean_fs(row.get("consequents", "")),
                support=float(row.get("support", 0)),
                confidence=float(row.get("confidence", 0)),
                lift=float(row.get("lift", 0)),
                representativity=_safe_float(row.get("representativity")),
                leverage=_safe_float(row.get("leverage")),
                conviction=_safe_float(row.get("conviction")),
                zhangs_metric=_safe_float(row.get("zhangs_metric")),
                jaccard=_safe_float(row.get("jaccard")),
            )
        )
    session.commit()
    print(f"  Seeded {len(df)} association rules")


def seed_item_abc(session, item_map):
    from src.evaluation.metrics import classify_abc

    filepath = PROCESSED_DIR / "daily_item_sales.csv"
    if not filepath.exists():
        print(f"  File not found: {filepath}")
        return

    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()
    abc_df = classify_abc(df, volume_col="Quantity_Sold")

    count = 0
    for idx, row in abc_df.iterrows():
        item_id = item_map.get(str(idx))
        if item_id is None:
            continue
        session.add(
            ItemABC(
                item_id=item_id,
                total_volume=float(row["Vol"]),
                cumulative_volume=float(row["Cum"]),
                cumulative_pct=float(row["Pct"]),
                abc_class=str(row["Class"]),
                computed_at=datetime.now(),
            )
        )
        count += 1
    session.commit()
    print(f"  Seeded {count} ABC classifications")


def _safe_float(val):
    if val is None or pd.isna(val):
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def truncate_all(session):
    tables = [
        "forecasts",
        "model_run_top_items",
        "model_run_class_metrics",
        "model_runs",
        "item_abc",
        "raw_material_requirements",
        "association_rules",
        "daily_item_sales",
        "daily_category_sales",
        "daily_total_sales",
        "sales_cleaned",
        "condiment_recipes",
        "bom_recipes",
        "items",
        "categories",
    ]
    for table in tables:
        session.execute(text(f"TRUNCATE TABLE {table} CASCADE"))
    session.commit()
    print("All tables truncated.")


def main():
    parser = argparse.ArgumentParser(description="Seed PostgreSQL from CSV files")
    parser.add_argument(
        "--skip-sales-cleaned",
        action="store_true",
        help="Skip large sales_cleaned table",
    )
    parser.add_argument(
        "--truncate", action="store_true", help="Clear all tables before seeding"
    )
    args = parser.parse_args()

    print("Creating tables...")
    sync_url = get_sync_url()
    sync_engine = create_engine(sync_url, echo=False, pool_size=5, max_overflow=10)

    from sqlalchemy.inspection import inspect

    inspector = inspect(sync_engine)
    tables = inspector.get_table_names()

    if tables:
        print(f"Tables already exist: {tables}. Skipping create_all.")
    else:
        Base.metadata.create_all(sync_engine)

    SessionLocal = sessionmaker(sync_engine, autocommit=False, autoflush=False)
    session = SessionLocal()
    try:
        if args.truncate:
            truncate_all(session)

        print("\n1. Seeding categories...")
        cat_map = seed_categories(session)

        print("\n2. Seeding items...")
        item_map = seed_items(session, cat_map)

        print("\n3. Seeding BOM recipes...")
        seed_bom_recipes(session, cat_map, item_map)

        print("\n4. Seeding condiment recipes...")
        seed_condiment_recipes(session)

        print("\n5. Seeding cleaned sales...")
        seed_sales_cleaned(session, skip=args.skip_sales_cleaned)

        print("\n6. Seeding daily item sales...")
        seed_daily_item_sales(session, item_map)

        print("\n7. Seeding daily category sales...")
        seed_daily_category_sales(session)

        print("\n8. Seeding daily total sales...")
        seed_daily_total_sales(session)

        print("\n9. Seeding model runs...")
        model_run = seed_model_runs(session, item_map)

        print("\n10. Seeding forecasts...")
        seed_forecasts(session, model_run, item_map)

        print("\n11. Seeding ABC classifications...")
        seed_item_abc(session, item_map)

        print("\n12. Seeding association rules...")
        seed_association_rules(session)

        print("\nDone! Database seeded successfully.")

    finally:
        session.close()


if __name__ == "__main__":
    main()
