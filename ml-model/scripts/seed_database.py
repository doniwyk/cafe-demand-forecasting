"""
Seed PostgreSQL database from existing CSV files.

Only seeds tables that the web/backend actually uses:
  categories, items, bom_recipes, daily_item_sales,
  model_runs, model_run_class_metrics, model_run_top_items

All inserts use ON CONFLICT DO NOTHING for idempotent re-runs.

Usage:
    cd ml-model
    python scripts/seed_database.py
    python scripts/seed_database.py --skip-sales-cleaned
"""

import sys
import os
import json
import argparse
from datetime import datetime

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.dialects.postgresql import insert as pg_insert

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.db import get_sync_url
from src.db.models import (
    Category,
    Item,
    BomRecipe,
    DailyItemSale,
    ModelRun,
    ModelRunClassMetric,
    ModelRunTopItem,
)
from src.utils.config import (
    PROCESSED_DIR,
    PREDICTIONS_DIR,
    MODELS_DIR,
    BOM_DIR,
)

CHUNK_SIZE = 5000


def _safe_float(val):
    if val is None or pd.isna(val):
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def seed_categories(session):
    bom_df = pd.read_csv(BOM_DIR / "menu_bom.csv")
    categories = bom_df["Tipe"].dropna().str.strip().unique()

    cat_map = {}
    for cat_name in sorted(categories):
        stmt = pg_insert(Category).values(name=cat_name).on_conflict_do_nothing()
        result = session.execute(stmt)
        session.flush()
        if result.rowcount == 0:
            cat = session.query(Category).filter_by(name=cat_name).first()
            cat_map[cat_name] = cat.id
        else:
            cat_map[cat_name] = result.inserted_primary_key[0]

    session.commit()
    print(f"  Seeded {len(cat_map)} categories")
    return cat_map


def seed_items(session, cat_map):
    bom_df = pd.read_csv(BOM_DIR / "menu_bom.csv")
    item_names = bom_df["Item"].dropna().str.strip().unique()

    item_map = {}
    for item_name in sorted(item_names):
        cat_name = bom_df[bom_df["Item"].str.strip() == item_name]["Tipe"].iloc[0]
        cat_id = cat_map.get(cat_name.strip())
        stmt = (
            pg_insert(Item)
            .values(name=item_name, category_id=cat_id)
            .on_conflict_do_nothing()
        )
        result = session.execute(stmt)
        session.flush()
        if result.rowcount == 0:
            item = session.query(Item).filter_by(name=item_name).first()
            item_map[item_name] = item.id
        else:
            item_map[item_name] = result.inserted_primary_key[0]

    session.commit()
    print(f"  Seeded {len(item_map)} items")
    return item_map


def seed_bom_recipes(session, cat_map, item_map):
    bom_df = pd.read_csv(BOM_DIR / "menu_bom.csv")
    bom_df.columns = bom_df.columns.str.strip()

    existing = set(
        session.query(BomRecipe.item_name, BomRecipe.ingredient).all()
    )

    count = 0
    skipped = 0
    for _, row in bom_df.iterrows():
        qty = _safe_float(row["Qty"])
        if qty is None:
            continue
        item_name = str(row["Item"]).strip()
        cat_name = str(row["Tipe"]).strip()
        ingredient = str(row["Bahan"]).strip()
        if (item_name, ingredient) in existing:
            skipped += 1
            continue
        stmt = pg_insert(BomRecipe).values(
            category_name=cat_name,
            item_name=item_name,
            ingredient=ingredient,
            quantity=qty,
            unit=str(row["Unit"]).strip(),
            category_id=cat_map.get(cat_name),
            item_id=item_map.get(item_name),
        ).on_conflict_do_nothing()
        session.execute(stmt)
        count += 1

    session.commit()
    print(f"  Seeded {count} BOM recipes ({skipped} skipped)")


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
        stmt = pg_insert(DailyItemSale).values(
            date=d,
            item_id=item_id,
            quantity_sold=float(row["Quantity_Sold"]),
        ).on_conflict_do_nothing()
        session.execute(stmt)
        count += 1
    session.commit()
    print(f"  Seeded {count} daily item sales rows")
    if missing:
        print(f"  Warning: {len(set(missing))} items not in DB: {set(missing)[:5]}...")


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


def truncate_all(session):
    tables = [
        "model_run_top_items",
        "model_run_class_metrics",
        "model_runs",
        "daily_item_sales",
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
        "--truncate", action="store_true", help="Clear all tables before seeding"
    )
    args = parser.parse_args()

    sync_url = get_sync_url()
    sync_engine = create_engine(sync_url, echo=False, pool_size=5, max_overflow=10)
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

        print("\n4. Seeding daily item sales...")
        seed_daily_item_sales(session, item_map)

        print("\n5. Seeding model runs...")
        model_run = seed_model_runs(session, item_map)

        print("\nDone! Database seeded successfully.")

    finally:
        session.close()


if __name__ == "__main__":
    main()
