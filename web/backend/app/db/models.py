from __future__ import annotations

from datetime import date, datetime
from typing import Optional

from sqlalchemy import (
    BigInteger,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    Index,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


from app.models.user import User


class Category(Base):
    __tablename__ = "categories"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)

    items: Mapped[list["Item"]] = relationship(back_populates="category", lazy="noload")
    bom_recipes: Mapped[list["BomRecipe"]] = relationship(
        back_populates="category_ref", lazy="noload"
    )


class Item(Base):
    __tablename__ = "items"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
    category_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("categories.id"), nullable=True
    )

    category: Mapped[Optional["Category"]] = relationship(
        back_populates="items", lazy="noload"
    )
    bom_recipes: Mapped[list["BomRecipe"]] = relationship(
        back_populates="item_ref", lazy="noload"
    )
    daily_item_sales: Mapped[list["DailyItemSale"]] = relationship(
        back_populates="item", lazy="noload"
    )

    __table_args__ = (Index("ix_items_category_id", "category_id"),)


class BomRecipe(Base):
    __tablename__ = "bom_recipes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    category_name: Mapped[str] = mapped_column(String(100), nullable=False)
    item_name: Mapped[str] = mapped_column(String(100), nullable=False)
    ingredient: Mapped[str] = mapped_column(String(200), nullable=False)
    quantity: Mapped[float] = mapped_column(Float, nullable=False)
    unit: Mapped[str] = mapped_column(String(50), nullable=False)

    category_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("categories.id"), nullable=True
    )
    item_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("items.id"), nullable=True
    )

    category_ref: Mapped[Optional["Category"]] = relationship(
        back_populates="bom_recipes", foreign_keys=[category_id], lazy="noload"
    )
    item_ref: Mapped[Optional["Item"]] = relationship(
        back_populates="bom_recipes", foreign_keys=[item_id], lazy="noload"
    )

    __table_args__ = (
        Index("ix_bom_item", "item_name"),
        Index("ix_bom_ingredient", "ingredient"),
    )


class DailyItemSale(Base):
    __tablename__ = "daily_item_sales"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    item_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("items.id"), nullable=False
    )
    quantity_sold: Mapped[float] = mapped_column(Float, nullable=False)

    item: Mapped["Item"] = relationship(
        back_populates="daily_item_sales", lazy="noload"
    )

    __table_args__ = (
        UniqueConstraint("date", "item_id", name="uq_daily_item_sales_date_item"),
        Index("ix_daily_item_sales_date", "date"),
    )


class ModelRun(Base):
    __tablename__ = "model_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model_type: Mapped[str] = mapped_column(String(50), nullable=False)
    trained_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    n_item_models: Mapped[Optional[int]] = mapped_column(Integer)
    n_records: Mapped[Optional[int]] = mapped_column(Integer)
    date_range_start: Mapped[Optional[date]] = mapped_column(Date)
    date_range_end: Mapped[Optional[date]] = mapped_column(Date)
    r2: Mapped[Optional[float]] = mapped_column(Float)
    wmape: Mapped[Optional[float]] = mapped_column(Float)
    mae: Mapped[Optional[float]] = mapped_column(Float)
    rmse: Mapped[Optional[float]] = mapped_column(Float)
    volume_accuracy: Mapped[Optional[float]] = mapped_column(Float)
    median_period_accuracy: Mapped[Optional[float]] = mapped_column(Float)
    periods_within_20pct: Mapped[Optional[float]] = mapped_column(Float)
    periods_within_50pct: Mapped[Optional[float]] = mapped_column(Float)
    features: Mapped[Optional[str]] = mapped_column(Text)
    items_with_models: Mapped[Optional[str]] = mapped_column(Text)
    params: Mapped[Optional[str]] = mapped_column(Text)
    is_active: Mapped[bool] = mapped_column(default=True)

    class_metrics: Mapped[list["ModelRunClassMetric"]] = relationship(
        back_populates="model_run", lazy="noload"
    )
    top_items: Mapped[list["ModelRunTopItem"]] = relationship(
        back_populates="model_run", lazy="noload"
    )

    __table_args__ = (Index("ix_model_runs_type_active", "model_type", "is_active"),)


class ModelRunClassMetric(Base):
    __tablename__ = "model_run_class_metrics"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model_run_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("model_runs.id"), nullable=False
    )
    abc_class: Mapped[str] = mapped_column(String(1), nullable=False)
    n_items: Mapped[int] = mapped_column(Integer, nullable=False)
    wmape: Mapped[float] = mapped_column(Float, nullable=False)
    r2: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    mae: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    rmse: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    volume_accuracy: Mapped[float] = mapped_column(Float, nullable=False)
    median_period_accuracy: Mapped[Optional[float]] = mapped_column(Float)
    periods_within_20pct: Mapped[Optional[float]] = mapped_column(Float)
    periods_within_50pct: Mapped[Optional[float]] = mapped_column(Float)

    model_run: Mapped["ModelRun"] = relationship(
        back_populates="class_metrics", lazy="noload"
    )

    __table_args__ = (
        UniqueConstraint("model_run_id", "abc_class", name="uq_model_run_class"),
    )


class ModelRunTopItem(Base):
    __tablename__ = "model_run_top_items"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model_run_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("model_runs.id"), nullable=False
    )
    item_name: Mapped[str] = mapped_column(String(100), nullable=False)
    quantity_sold: Mapped[float] = mapped_column(Float, nullable=False)
    predicted: Mapped[float] = mapped_column(Float, nullable=False)
    accuracy_pct: Mapped[float] = mapped_column(Float, nullable=False)

    model_run: Mapped["ModelRun"] = relationship(
        back_populates="top_items", lazy="noload"
    )

    __table_args__ = (
        UniqueConstraint("model_run_id", "item_name", name="uq_model_run_top_item"),
    )
