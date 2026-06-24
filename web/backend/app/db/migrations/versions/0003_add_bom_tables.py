"""add menu_bom and condiment_bom tables

Revision ID: 0003
Revises: 0002
Create Date: 2026-06-25
"""

from alembic import op
import sqlalchemy as sa

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "menu_bom",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("type", sa.String(100), nullable=False),
        sa.Column("item_name", sa.String(100), nullable=False),
        sa.Column("ingredient", sa.String(200), nullable=False),
        sa.Column("quantity", sa.Float(), nullable=False),
        sa.Column("unit", sa.String(50), nullable=False),
    )
    op.create_table(
        "condiment_bom",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("item_name", sa.String(100), nullable=False),
        sa.Column("total_qty", sa.Float(), nullable=False),
        sa.Column("total_unit", sa.String(50), nullable=False),
        sa.Column("sub_ingredient", sa.String(200), nullable=False),
        sa.Column("qty_per_unit", sa.Float(), nullable=False),
        sa.Column("sub_unit", sa.String(50), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("condiment_bom")
    op.drop_table("menu_bom")
