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

from src.db import Base


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
    forecasts: Mapped[list["Forecast"]] = relationship(
        back_populates="item", lazy="noload"
    )
    item_abc: Mapped[Optional["ItemABC"]] = relationship(
        back_populates="item", lazy="noload", uselist=False
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


class CondimentRecipe(Base):
    __tablename__ = "condiment_recipes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    condiment: Mapped[str] = mapped_column(String(200), nullable=False)
    condiment_qty: Mapped[float] = mapped_column(Float, nullable=False)
    condiment_unit: Mapped[str] = mapped_column(String(50), nullable=False)
    sub_ingredient: Mapped[str] = mapped_column(String(200), nullable=False)
    qty_per_unit: Mapped[float] = mapped_column(Float, nullable=False)
    sub_unit: Mapped[str] = mapped_column(String(50), nullable=False)

    __table_args__ = (
        Index("ix_condiment", "condiment"),
        Index("ix_condiment_sub", "sub_ingredient"),
    )


class SaleCleaned(Base):
    __tablename__ = "sales_cleaned"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    date: Mapped[datetime] = mapped_column(DateTime, nullable=False, index=True)
    receipt_number: Mapped[Optional[str]] = mapped_column(String(100))
    receipt_type: Mapped[Optional[str]] = mapped_column(String(50))
    category: Mapped[Optional[str]] = mapped_column(String(100))
    sku: Mapped[Optional[str]] = mapped_column(String(50))
    item: Mapped[Optional[str]] = mapped_column(String(100), nullable=False, index=True)
    variant: Mapped[Optional[str]] = mapped_column(String(100))
    modifiers_applied: Mapped[Optional[str]] = mapped_column(Text)
    quantity: Mapped[Optional[float]] = mapped_column(Float)
    gross_sales: Mapped[Optional[float]] = mapped_column(Float)
    discounts: Mapped[Optional[float]] = mapped_column(Float)
    net_sales: Mapped[Optional[float]] = mapped_column(Float)
    cost_of_goods: Mapped[Optional[float]] = mapped_column(Float)
    gross_profit: Mapped[Optional[float]] = mapped_column(Float)
    taxes: Mapped[Optional[float]] = mapped_column(Float)
    dining_option: Mapped[Optional[str]] = mapped_column(String(50))
    pos: Mapped[Optional[str]] = mapped_column(String(50))
    store: Mapped[Optional[str]] = mapped_column(String(100))
    cashier_name: Mapped[Optional[str]] = mapped_column(String(100))
    customer_name: Mapped[Optional[str]] = mapped_column(String(100))
    status: Mapped[Optional[str]] = mapped_column(String(50))

    __table_args__ = (Index("ix_sales_cleaned_date_item", "date", "item"),)


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


class DailyCategorySale(Base):
    __tablename__ = "daily_category_sales"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    category: Mapped[str] = mapped_column(String(100), nullable=False)
    quantity: Mapped[float] = mapped_column(Float, nullable=False)
    net_sales: Mapped[float] = mapped_column(Float, nullable=False)
    gross_sales: Mapped[float] = mapped_column(Float, nullable=False)
    unique_items: Mapped[int] = mapped_column(Integer, nullable=False)

    __table_args__ = (
        UniqueConstraint(
            "date", "category", name="uq_daily_category_sales_date_category"
        ),
        Index("ix_daily_category_sales_date", "date"),
    )


class DailyTotalSale(Base):
    __tablename__ = "daily_total_sales"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    date: Mapped[date] = mapped_column(
        Date, primary_key=False, unique=True, nullable=False
    )
    quantity: Mapped[float] = mapped_column(Float, nullable=False)
    net_sales: Mapped[float] = mapped_column(Float, nullable=False)
    gross_sales: Mapped[float] = mapped_column(Float, nullable=False)
    unique_items: Mapped[int] = mapped_column(Integer, nullable=False)


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
    volume_accuracy: Mapped[Optional[float]] = mapped_column(Float)
    features: Mapped[Optional[str]] = mapped_column(Text)
    items_with_models: Mapped[Optional[str]] = mapped_column(Text)
    params: Mapped[Optional[str]] = mapped_column(Text)
    is_active: Mapped[bool] = mapped_column(default=True)

    forecasts: Mapped[list["Forecast"]] = relationship(
        back_populates="model_run", lazy="noload"
    )
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
    volume_accuracy: Mapped[float] = mapped_column(Float, nullable=False)

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


class Forecast(Base):
    __tablename__ = "forecasts"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    model_run_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("model_runs.id"), nullable=False
    )
    item_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("items.id"), nullable=False
    )
    date: Mapped[date] = mapped_column(Date, nullable=False)
    quantity_predicted: Mapped[float] = mapped_column(Float, nullable=False)

    model_run: Mapped["ModelRun"] = relationship(
        back_populates="forecasts", lazy="noload"
    )
    item: Mapped["Item"] = relationship(back_populates="forecasts", lazy="noload")

    __table_args__ = (
        UniqueConstraint(
            "model_run_id", "item_id", "date", name="uq_forecast_run_item_date"
        ),
        Index("ix_forecasts_date", "date"),
        Index("ix_forecasts_item", "item_id"),
    )


class ItemABC(Base):
    __tablename__ = "item_abc"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    item_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("items.id"), unique=True, nullable=False
    )
    total_volume: Mapped[float] = mapped_column(Float, nullable=False)
    cumulative_volume: Mapped[float] = mapped_column(Float, nullable=False)
    cumulative_pct: Mapped[float] = mapped_column(Float, nullable=False)
    abc_class: Mapped[str] = mapped_column(String(1), nullable=False)
    computed_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)

    item: Mapped["Item"] = relationship(back_populates="item_abc", lazy="noload")


class AssociationRule(Base):
    __tablename__ = "association_rules"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    antecedents: Mapped[str] = mapped_column(String(500), nullable=False)
    consequents: Mapped[str] = mapped_column(String(500), nullable=False)
    support: Mapped[float] = mapped_column(Float, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    lift: Mapped[float] = mapped_column(Float, nullable=False)
    representativity: Mapped[Optional[float]] = mapped_column(Float)
    leverage: Mapped[Optional[float]] = mapped_column(Float)
    conviction: Mapped[Optional[float]] = mapped_column(Float)
    zhangs_metric: Mapped[Optional[float]] = mapped_column(Float)
    jaccard: Mapped[Optional[float]] = mapped_column(Float)

    __table_args__ = (
        Index("ix_assoc_rules_confidence", "confidence"),
        Index("ix_assoc_rules_lift", "lift"),
    )


class RawMaterialRequirement(Base):
    __tablename__ = "raw_material_requirements"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    raw_material: Mapped[str] = mapped_column(String(200), nullable=False)
    quantity_required: Mapped[float] = mapped_column(Float, nullable=False)

    __table_args__ = (Index("ix_raw_material_date_material", "date", "raw_material"),)
