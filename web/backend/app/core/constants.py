from __future__ import annotations

MODEL_TYPES = frozenset({"xgboost", "random_forest", "sarimax", "prophet"})

DEFAULT_MODEL_TYPE = "xgboost"

DAILY_SALES_JOIN_SQL = """
    SELECT dis.date, i.name as item, dis.quantity_sold
    FROM daily_item_sales dis
    JOIN items i ON dis.item_id = i.id
"""
