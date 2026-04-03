"""create initial schema

Revision ID: 0001
Revises:
Create Date: 2026-04-03
"""

from alembic import op
import sqlalchemy as sa

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "categories",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("name", sa.String(100), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("name"),
    )

    op.create_table(
        "items",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("name", sa.String(100), nullable=False),
        sa.Column("category_id", sa.Integer(), nullable=True),
        sa.ForeignKeyConstraint(["category_id"], ["categories.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("name"),
    )
    op.create_index("ix_items_category_id", "items", ["category_id"])

    op.create_table(
        "bom_recipes",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("category_name", sa.String(100), nullable=False),
        sa.Column("item_name", sa.String(100), nullable=False),
        sa.Column("ingredient", sa.String(200), nullable=False),
        sa.Column("quantity", sa.Float(), nullable=False),
        sa.Column("unit", sa.String(50), nullable=False),
        sa.Column("category_id", sa.Integer(), nullable=True),
        sa.Column("item_id", sa.Integer(), nullable=True),
        sa.ForeignKeyConstraint(["category_id"], ["categories.id"]),
        sa.ForeignKeyConstraint(["item_id"], ["items.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_bom_item", "bom_recipes", ["item_name"])
    op.create_index("ix_bom_ingredient", "bom_recipes", ["ingredient"])

    op.create_table(
        "condiment_recipes",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("condiment", sa.String(200), nullable=False),
        sa.Column("condiment_qty", sa.Float(), nullable=False),
        sa.Column("condiment_unit", sa.String(50), nullable=False),
        sa.Column("sub_ingredient", sa.String(200), nullable=False),
        sa.Column("qty_per_unit", sa.Float(), nullable=False),
        sa.Column("sub_unit", sa.String(50), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_condiment", "condiment_recipes", ["condiment"])
    op.create_index("ix_condiment_sub", "condiment_recipes", ["sub_ingredient"])

    op.create_table(
        "daily_total_sales",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("date", sa.Date(), nullable=False),
        sa.Column("quantity", sa.Float(), nullable=False),
        sa.Column("net_sales", sa.Float(), nullable=False),
        sa.Column("gross_sales", sa.Float(), nullable=False),
        sa.Column("unique_items", sa.Integer(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("date"),
    )

    op.create_table(
        "sales_cleaned",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("date", sa.DateTime(), nullable=False),
        sa.Column("receipt_number", sa.String(100), nullable=True),
        sa.Column("receipt_type", sa.String(50), nullable=True),
        sa.Column("category", sa.String(100), nullable=True),
        sa.Column("sku", sa.String(50), nullable=True),
        sa.Column("item", sa.String(100), nullable=False),
        sa.Column("variant", sa.String(100), nullable=True),
        sa.Column("modifiers_applied", sa.Text(), nullable=True),
        sa.Column("quantity", sa.Float(), nullable=True),
        sa.Column("gross_sales", sa.Float(), nullable=True),
        sa.Column("discounts", sa.Float(), nullable=True),
        sa.Column("net_sales", sa.Float(), nullable=True),
        sa.Column("cost_of_goods", sa.Float(), nullable=True),
        sa.Column("gross_profit", sa.Float(), nullable=True),
        sa.Column("taxes", sa.Float(), nullable=True),
        sa.Column("dining_option", sa.String(50), nullable=True),
        sa.Column("pos", sa.String(50), nullable=True),
        sa.Column("store", sa.String(100), nullable=True),
        sa.Column("cashier_name", sa.String(100), nullable=True),
        sa.Column("customer_name", sa.String(100), nullable=True),
        sa.Column("status", sa.String(50), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_sales_cleaned_date_item", "sales_cleaned", ["date", "item"])
    op.create_index("ix_sales_cleaned_date", "sales_cleaned", ["date"])

    op.create_table(
        "model_runs",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("model_type", sa.String(50), nullable=False),
        sa.Column("trained_at", sa.DateTime(), nullable=False),
        sa.Column("n_item_models", sa.Integer(), nullable=True),
        sa.Column("n_records", sa.Integer(), nullable=True),
        sa.Column("date_range_start", sa.Date(), nullable=True),
        sa.Column("date_range_end", sa.Date(), nullable=True),
        sa.Column("r2", sa.Float(), nullable=True),
        sa.Column("wmape", sa.Float(), nullable=True),
        sa.Column("mae", sa.Float(), nullable=True),
        sa.Column("volume_accuracy", sa.Float(), nullable=True),
        sa.Column("features", sa.Text(), nullable=True),
        sa.Column("items_with_models", sa.Text(), nullable=True),
        sa.Column("params", sa.Text(), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_model_runs_type_active", "model_runs", ["model_type", "is_active"]
    )

    op.create_table(
        "daily_category_sales",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("date", sa.Date(), nullable=False),
        sa.Column("category", sa.String(100), nullable=False),
        sa.Column("quantity", sa.Float(), nullable=False),
        sa.Column("net_sales", sa.Float(), nullable=False),
        sa.Column("gross_sales", sa.Float(), nullable=False),
        sa.Column("unique_items", sa.Integer(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "date", "category", name="uq_daily_category_sales_date_category"
        ),
    )
    op.create_index("ix_daily_category_sales_date", "daily_category_sales", ["date"])

    op.create_table(
        "model_run_class_metrics",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("model_run_id", sa.Integer(), nullable=False),
        sa.Column("abc_class", sa.String(1), nullable=False),
        sa.Column("n_items", sa.Integer(), nullable=False),
        sa.Column("wmape", sa.Float(), nullable=False),
        sa.Column("volume_accuracy", sa.Float(), nullable=False),
        sa.ForeignKeyConstraint(["model_run_id"], ["model_runs.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("model_run_id", "abc_class", name="uq_model_run_class"),
    )

    op.create_table(
        "model_run_top_items",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("model_run_id", sa.Integer(), nullable=False),
        sa.Column("item_name", sa.String(100), nullable=False),
        sa.Column("quantity_sold", sa.Float(), nullable=False),
        sa.Column("predicted", sa.Float(), nullable=False),
        sa.Column("accuracy_pct", sa.Float(), nullable=False),
        sa.ForeignKeyConstraint(["model_run_id"], ["model_runs.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("model_run_id", "item_name", name="uq_model_run_top_item"),
    )

    op.create_table(
        "association_rules",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("antecedents", sa.String(500), nullable=False),
        sa.Column("consequents", sa.String(500), nullable=False),
        sa.Column("support", sa.Float(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("lift", sa.Float(), nullable=False),
        sa.Column("representativity", sa.Float(), nullable=True),
        sa.Column("leverage", sa.Float(), nullable=True),
        sa.Column("conviction", sa.Float(), nullable=True),
        sa.Column("zhangs_metric", sa.Float(), nullable=True),
        sa.Column("jaccard", sa.Float(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_assoc_rules_confidence", "association_rules", ["confidence"])
    op.create_index("ix_assoc_rules_lift", "association_rules", ["lift"])

    op.create_table(
        "daily_item_sales",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("date", sa.Date(), nullable=False),
        sa.Column("item_id", sa.Integer(), nullable=False),
        sa.Column("quantity_sold", sa.Float(), nullable=False),
        sa.ForeignKeyConstraint(["item_id"], ["items.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("date", "item_id", name="uq_daily_item_sales_date_item"),
    )
    op.create_index("ix_daily_item_sales_date", "daily_item_sales", ["date"])

    op.create_table(
        "forecasts",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("model_run_id", sa.Integer(), nullable=False),
        sa.Column("item_id", sa.Integer(), nullable=False),
        sa.Column("date", sa.Date(), nullable=False),
        sa.Column("quantity_predicted", sa.Float(), nullable=False),
        sa.ForeignKeyConstraint(["model_run_id"], ["model_runs.id"]),
        sa.ForeignKeyConstraint(["item_id"], ["items.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "model_run_id", "item_id", "date", name="uq_forecast_run_item_date"
        ),
    )
    op.create_index("ix_forecasts_date", "forecasts", ["date"])
    op.create_index("ix_forecasts_item", "forecasts", ["item_id"])

    op.create_table(
        "item_abc",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("item_id", sa.Integer(), nullable=False),
        sa.Column("total_volume", sa.Float(), nullable=False),
        sa.Column("cumulative_volume", sa.Float(), nullable=False),
        sa.Column("cumulative_pct", sa.Float(), nullable=False),
        sa.Column("abc_class", sa.String(1), nullable=False),
        sa.Column("computed_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["item_id"], ["items.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("item_id"),
    )

    op.create_table(
        "raw_material_requirements",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("date", sa.Date(), nullable=False),
        sa.Column("raw_material", sa.String(200), nullable=False),
        sa.Column("quantity_required", sa.Float(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_raw_material_date_material",
        "raw_material_requirements",
        ["date", "raw_material"],
    )


def downgrade() -> None:
    op.drop_table("raw_material_requirements")
    op.drop_table("item_abc")
    op.drop_table("forecasts")
    op.drop_table("daily_item_sales")
    op.drop_table("association_rules")
    op.drop_table("model_run_top_items")
    op.drop_table("model_run_class_metrics")
    op.drop_table("daily_category_sales")
    op.drop_table("model_runs")
    op.drop_table("sales_cleaned")
    op.drop_table("daily_total_sales")
    op.drop_table("condiment_recipes")
    op.drop_table("bom_recipes")
    op.drop_table("items")
    op.drop_table("categories")
