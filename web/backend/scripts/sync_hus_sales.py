"""
Sync sales data from hus_db POS system into cafe_forecasting daily_item_sales.

Usage:
    python scripts/sync_hus_sales.py                          # Only matching products
    python scripts/sync_hus_sales.py --include-new            # Add new products too
    python scripts/sync_hus_sales.py --since 2026-04-01       # From specific date
"""
from __future__ import annotations

import argparse
import os
from datetime import date, timedelta

import psycopg2
import psycopg2.extras

HUS_DB_URL = os.getenv("HUS_DB_URL", "postgresql://user:password@localhost:5432/hus_db")
CAFE_DB_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5432/cafe_forecasting",
)


def _cafe_url():
    url = CAFE_DB_URL
    if url.startswith("postgresql+asyncpg"):
        url = "postgresql" + url[len("postgresql+asyncpg"):]
    return url


# Manual fallback: products that appear without variant snapshot but have variants
FALLBACK_VARIANT_MAP = {
    "Kopi Susu Husgendam": "Kopi Susu Husgendam Ice",
    "Cappucino": "Cappucino Ice",
}

# Products to skip (add-ons, different product lines, V60 filter coffees)
SKIP_PREFIXES = [
    "Add ",
    "Filter",
    "FIlter",
    "V60",
    "Harum Jasmine Tea",
    "Cookies Redvelvet",
    "Lotus Cheesecake",
    "Strawberry Cheesecake",
    "Kopi Susu Bersemi",
]


def _should_skip(product_name: str) -> bool:
    for prefix in SKIP_PREFIXES:
        if product_name.startswith(prefix):
            return True
    return False


def _get_cafe_items(cur) -> dict[str, int]:
    """Return {item_name: item_id} from cafe_forecasting.items."""
    cur.execute("SELECT id, name FROM items")
    return {row[1].strip(): row[0] for row in cur.fetchall()}


def _get_hus_products(cur) -> dict[int, tuple[str, list[str]]]:
    """Return {product_id: (name, [variant_names])} from hus_db."""
    cur.execute("""
        SELECT p.id, p.name, p.has_variants
        FROM products p WHERE p.is_active = true
    """)
    products = {row[0]: (row[1], []) for row in cur.fetchall()}
    cur.execute("""
        SELECT pv.product_id, pv.name
        FROM product_variants pv WHERE pv.is_active = true
        ORDER BY pv.product_id, pv.name
    """)
    for row in cur.fetchall():
        pid, vname = row
        if pid in products:
            products[pid][1].append(vname)
    return products


def _match_item(
    product_name: str, variant_name: str | None, cafe_items: dict[str, int]
) -> int | None:
    """Return item_id if this product+variant matches a cafe item, else None."""
    combined = (product_name + " " + variant_name).strip() if variant_name else product_name.strip()

    if combined in cafe_items:
        return cafe_items[combined]

    no_variant = product_name.strip()
    if no_variant in cafe_items:
        return cafe_items[no_variant]

    if no_variant in FALLBACK_VARIANT_MAP:
        fallback = FALLBACK_VARIANT_MAP[no_variant]
        if fallback in cafe_items:
            return cafe_items[fallback]

    return None


def _add_new_product(
    cur,
    product_name: str,
    variant_name: str | None,
    cafe_items: dict[str, int],
) -> int | None:
    """Add a new product to cafe_forecasting. Returns the new item_id."""
    item_name = (product_name + " " + variant_name).strip() if variant_name else product_name.strip()
    item_name = item_name.strip()

    if _should_skip(item_name):
        return None

    if item_name in cafe_items:
        return cafe_items[item_name]

    cur.execute(
        "INSERT INTO items (name) VALUES (%s) ON CONFLICT (name) DO UPDATE SET name = EXCLUDED.name RETURNING id",
        (item_name,),
    )
    new_id = cur.fetchone()[0]
    cafe_items[item_name] = new_id
    return new_id


def sync_sales(since: str | None = None, include_new: bool = False) -> dict:
    """Sync sales data from hus_db. Returns {added_rows, skipped_units, new_products}."""
    if since is None:
        since = (date.today() - timedelta(days=90)).isoformat()

    hus_conn = psycopg2.connect(HUS_DB_URL)
    cafe_conn = psycopg2.connect(_cafe_url())

    try:
        hus_cur = hus_conn.cursor()
        cafe_cur = cafe_conn.cursor()

        cafe_items = _get_cafe_items(cafe_cur)

        hus_cur.execute("""
            SELECT
                DATE(o.created_at) as sale_date,
                oi.product_name_snapshot,
                oi.variant_name_snapshot,
                SUM(oi.quantity) as total_qty
            FROM order_items oi
            JOIN orders o ON oi.order_id = o.id
            WHERE o.status = 'PAID'
              AND o.created_at >= %s
            GROUP BY DATE(o.created_at), oi.product_name_snapshot, oi.variant_name_snapshot
            ORDER BY sale_date
        """, (since,))
        rows = hus_cur.fetchall()

        inserted = 0
        skipped = {}
        new_products = []

        for sale_date, product_name, variant_name, qty in rows:
            product_name = product_name.strip() if product_name else ""
            variant_name = variant_name.strip() if variant_name else None

            item_id = _match_item(product_name, variant_name, cafe_items)

            if item_id is None and include_new:
                item_id = _add_new_product(cafe_cur, product_name, variant_name, cafe_items)
                if item_id:
                    new_products.append(
                        (product_name + " " + (variant_name or "")).strip()
                    )

            if item_id is None:
                key = (product_name + " " + (variant_name or "")).strip()
                skipped[key] = skipped.get(key, 0) + int(qty)
                continue

            cafe_cur.execute(
                """INSERT INTO daily_item_sales (date, item_id, quantity_sold)
                   VALUES (%s, %s, %s)
                   ON CONFLICT (date, item_id)
                   DO UPDATE SET quantity_sold = EXCLUDED.quantity_sold""",
                (sale_date, item_id, int(qty)),
            )
            inserted += 1

        cafe_conn.commit()
        hus_conn.commit()

        return {
            "inserted": inserted,
            "skipped_units": sum(skipped.values()),
            "skipped_products": len(skipped),
            "new_products": len(set(new_products)),
            "new_product_names": sorted(set(new_products)),
            "skipped_names": sorted(skipped.keys()),
        }

    finally:
        hus_cur.close()
        cafe_cur.close()
        hus_conn.close()
        cafe_conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sync hus_db sales into cafe_forecasting")
    parser.add_argument(
        "--since",
        default=(date.today() - timedelta(days=90)).isoformat(),
        help="Start date (default: 90 days ago)",
    )
    parser.add_argument(
        "--include-new",
        action="store_true",
        help="Add new products from hus_db to cafe_forecasting.items",
    )
    args = parser.parse_args()

    result = sync_sales(since=args.since, include_new=args.include_new)
    print(f"Inserted: {result['inserted']} rows")
    print(f"Skipped: {result['skipped_units']} units across {result['skipped_products']} products")
    if result["new_products"]:
        print(f"New products added: {result['new_products']}")
        for name in result["new_product_names"]:
            print(f"  + {name}")
    if result["skipped_names"]:
        print(f"Skipped products: {result['skipped_names']}")
