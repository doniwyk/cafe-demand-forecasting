"""
Pull condiment, material, and item BOM data from hus_db and generate CSV files
for the ML pipeline.

Connects to the running hus_db Docker container (port 5432) and extracts:
- menu_bom.csv: Product recipes (product -> materials/condiments)
- condiment_bom.csv: Condiment sub-recipes (condiment -> raw materials)

Usage:
    cd ml-model
    python scripts/00_pull_from_hus_db.py
"""

import sys
import os
import csv

import psycopg2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.config import BOM_DIR

HUS_DB = {
    "host": "localhost",
    "port": 5432,
    "user": "user",
    "password": "password",
    "dbname": "hus_db",
}


def get_connection():
    return psycopg2.connect(**HUS_DB)


def pull_condiment_bom(conn):
    print("\n--- Condiment BOM ---")
    with conn.cursor() as cur:
        cur.execute("""
            SELECT
                c.name AS condiment,
                c.batch_quantity,
                u.code AS condiment_unit,
                COALESCE(m.name, cc.name) AS sub_ingredient,
                ci.quantity AS qty_per_unit,
                mu.code AS sub_unit
            FROM condiment_ingredients ci
            JOIN condiments c ON c.id = ci.condiment_id
            LEFT JOIN units u ON u.id = c.unit_id
            LEFT JOIN materials m ON m.id = ci.material_id
            LEFT JOIN condiments cc ON cc.id = ci.child_condiment_id
            LEFT JOIN units mu ON mu.id = (
                CASE WHEN ci.material_id IS NOT NULL THEN m.unit_id ELSE NULL END
            )
            ORDER BY c.name, sub_ingredient
        """)
        rows = cur.fetchall()
    print(f"  {len(rows)} rows")
    return rows


def pull_menu_bom(conn):
    print("\n--- Menu BOM ---")
    with conn.cursor() as cur:
        cur.execute("""
            SELECT
                COALESCE(cat.name, 'Uncategorized') AS tipe,
                CASE
                    WHEN pv.id IS NOT NULL AND pv.name != 'default'
                    THEN p.name || ' ' || pv.name
                    ELSE p.name
                END AS item,
                COALESCE(m.name, c.name) AS bahan,
                pri.quantity AS qty,
                COALESCE(mu.code, cu.code, '') AS unit
            FROM product_recipe_ingredients pri
            JOIN products p ON p.id = pri.product_id
            LEFT JOIN product_variants pv ON pv.id = pri.variant_id
            LEFT JOIN materials m ON m.id = pri.material_id
            LEFT JOIN condiments c ON c.id = pri.condiment_id
            LEFT JOIN product_category_links pcl ON pcl.product_id = p.id
            LEFT JOIN categories cat ON cat.id = pcl.category_id
            LEFT JOIN units mu ON mu.id = m.unit_id
            LEFT JOIN units cu ON cu.id = c.unit_id
            WHERE p.is_active = true
              AND (m.is_active = true OR m.id IS NULL)
              AND (pv.id IS NULL OR pv.is_active = true)
            ORDER BY tipe, item, bahan
        """)
        rows = cur.fetchall()
    print(f"  {len(rows)} rows")
    return rows


def save_menu_bom(rows, output_path):
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Tipe", "Item", "Bahan", "Qty", "Unit"])
        for row in rows:
            writer.writerow(row)
    print(f"  Saved: {output_path}")


def save_condiment_bom(rows, output_path):
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Condiment", "Condiment_Qty", "Condiment_Unit",
            "Sub_Ingredient", "Qty_per_condiment_unit", "Sub_Unit",
        ])
        for row in rows:
            writer.writerow(row)
    print(f"  Saved: {output_path}")


def main():
    print("=" * 80)
    print("PULL BOM DATA FROM HUS_DB")
    print("=" * 80)
    print(f"  Host: {HUS_DB['host']}:{HUS_DB['port']}")
    print(f"  DB:   {HUS_DB['dbname']}")

    conn = get_connection()
    try:
        menu_bom_rows = pull_menu_bom(conn)
        condiment_bom_rows = pull_condiment_bom(conn)

        BOM_DIR.mkdir(parents=True, exist_ok=True)

        save_menu_bom(menu_bom_rows, BOM_DIR / "menu_bom.csv")
        save_condiment_bom(condiment_bom_rows, BOM_DIR / "condiment_bom.csv")

        print("\nDone!")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
