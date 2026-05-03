"""Add HUS recipe tables

Revision ID: 0002_add_hus_tables
Revises: 0001_initial_schema
Create Date: 2025-01-03

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = '0002_add_hus_tables'
down_revision: Union[str, None] = '0001'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'products',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('name', sa.String(length=200), nullable=False),
        sa.Column('is_active', sa.Boolean(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_products_name', 'products', ['name'])

    op.create_table(
        'product_variants',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('name', sa.String(length=200), nullable=False),
        sa.Column('product_id', sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(['product_id'], ['products.id']),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_product_variants_product_id', 'product_variants', ['product_id'])

    op.create_table(
        'materials',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('name', sa.String(length=200), nullable=False),
        sa.Column('unit_id', sa.Integer(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_materials_name', 'materials', ['name'])

    op.create_table(
        'condiments',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('name', sa.String(length=200), nullable=False),
        sa.Column('batch_quantity', sa.Float(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_condiments_name', 'condiments', ['name'])

    op.create_table(
        'product_recipe_ingredients',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('product_id', sa.Integer(), nullable=False),
        sa.Column('variant_id', sa.Integer(), nullable=True),
        sa.Column('material_id', sa.Integer(), nullable=True),
        sa.Column('condiment_id', sa.Integer(), nullable=True),
        sa.Column('quantity', sa.Float(), nullable=False),
        sa.ForeignKeyConstraint(['condiment_id'], ['condiments.id']),
        sa.ForeignKeyConstraint(['material_id'], ['materials.id']),
        sa.ForeignKeyConstraint(['product_id'], ['products.id']),
        sa.ForeignKeyConstraint(['variant_id'], ['product_variants.id']),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_product_recipe_product_id', 'product_recipe_ingredients', ['product_id'])
    op.create_index('ix_product_recipe_variant_id', 'product_recipe_ingredients', ['variant_id'])

    op.create_table(
        'condiment_ingredients',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('condiment_id', sa.Integer(), nullable=False),
        sa.Column('material_id', sa.Integer(), nullable=True),
        sa.Column('quantity', sa.Float(), nullable=False),
        sa.ForeignKeyConstraint(['condiment_id'], ['condiments.id']),
        sa.ForeignKeyConstraint(['material_id'], ['materials.id']),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_condiment_ingredients_condiment_id', 'condiment_ingredients', ['condiment_id'])


def downgrade() -> None:
    op.drop_index('ix_condiment_ingredients_condiment_id', table_name='condiment_ingredients')
    op.drop_table('condiment_ingredients')

    op.drop_index('ix_product_recipe_variant_id', table_name='product_recipe_ingredients')
    op.drop_index('ix_product_recipe_product_id', table_name='product_recipe_ingredients')
    op.drop_table('product_recipe_ingredients')

    op.drop_index('ix_condiments_name', table_name='condiments')
    op.drop_table('condiments')

    op.drop_index('ix_materials_name', table_name='materials')
    op.drop_table('materials')

    op.drop_index('ix_product_variants_product_id', table_name='product_variants')
    op.drop_table('product_variants')

    op.drop_index('ix_products_name', table_name='products')
    op.drop_table('products')
