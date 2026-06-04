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
        sa.Column("rmse", sa.Float(), nullable=True),
        sa.Column("volume_accuracy", sa.Float(), nullable=True),
        sa.Column("median_period_accuracy", sa.Float(), nullable=True),
        sa.Column("periods_within_20pct", sa.Float(), nullable=True),
        sa.Column("periods_within_50pct", sa.Float(), nullable=True),
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
        "model_run_class_metrics",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("model_run_id", sa.Integer(), nullable=False),
        sa.Column("abc_class", sa.String(1), nullable=False),
        sa.Column("n_items", sa.Integer(), nullable=False),
        sa.Column("wmape", sa.Float(), nullable=False),
        sa.Column("r2", sa.Float(), nullable=False, server_default="0"),
        sa.Column("mae", sa.Float(), nullable=False, server_default="0"),
        sa.Column("rmse", sa.Float(), nullable=False, server_default="0"),
        sa.Column("volume_accuracy", sa.Float(), nullable=False),
        sa.Column("median_period_accuracy", sa.Float(), nullable=True),
        sa.Column("periods_within_20pct", sa.Float(), nullable=True),
        sa.Column("periods_within_50pct", sa.Float(), nullable=True),
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
        "users",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("email", sa.String(255), nullable=False),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("hashed_password", sa.String(255), nullable=False),
        sa.Column("avatar", sa.String(500), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.Column("updated_at", sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("email"),
    )


def downgrade() -> None:
    op.drop_table("users")
    op.drop_table("model_run_top_items")
    op.drop_table("model_run_class_metrics")
    op.drop_table("daily_item_sales")
    op.drop_index("ix_model_runs_type_active", table_name="model_runs")
    op.drop_table("model_runs")
    op.drop_index("ix_bom_ingredient", table_name="bom_recipes")
    op.drop_index("ix_bom_item", table_name="bom_recipes")
    op.drop_table("bom_recipes")
    op.drop_index("ix_items_category_id", table_name="items")
    op.drop_table("items")
    op.drop_table("categories")
