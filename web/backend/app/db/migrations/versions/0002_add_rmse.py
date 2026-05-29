"""add rmse column to model_runs

Revision ID: 0002
Revises: 0001
Create Date: 2026-05-29
"""

from alembic import op
import sqlalchemy as sa

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("model_runs", sa.Column("rmse", sa.Float(), nullable=True))


def downgrade() -> None:
    op.drop_column("model_runs", "rmse")
