"""Convert predicted item sales into raw material requirements (Bill of Materials).

Adapted from web/backend/app/ml/raw_materials.py to run standalone
without any web-framework dependencies.
"""
from __future__ import annotations

import pandas as pd
from pathlib import Path
from typing import Dict, Tuple

from config import MENU_BOM_PATH, CONDIMENT_BOM_PATH


class RawMaterialProcessor:
    def __init__(
        self,
        menu_bom_path: str | Path = MENU_BOM_PATH,
        condiment_bom_path: str | Path = CONDIMENT_BOM_PATH,
    ):
        self.menu_bom_path = Path(menu_bom_path)
        self.condiment_bom_path = Path(condiment_bom_path)

        self.menu_bom_df = pd.read_csv(self.menu_bom_path)
        self.condiment_bom_df = pd.read_csv(self.condiment_bom_path)

        self.menu_bom_df["Item_normalized"] = (
            self.menu_bom_df["Item"].str.strip().str.lower()
        )

        self.item_name_map = self._build_item_name_map()
        self.condiment_dict = self._build_condiment_dict()
        self.expansion_cache: Dict[Tuple[str, float], Dict[str, float]] = {}

        self._build_unit_map()

    def _build_item_name_map(self) -> Dict[str, str]:
        name_map = {}
        for item_name in self.menu_bom_df["Item"].unique():
            normalized = item_name.strip().lower()
            name_map[normalized] = item_name
        return name_map

    def _build_condiment_dict(self) -> Dict[str, pd.DataFrame]:
        condiment_dict = {}
        for condiment_name in self.condiment_bom_df["Condiment"].unique():
            condiment_dict[condiment_name] = self.condiment_bom_df[
                self.condiment_bom_df["Condiment"] == condiment_name
            ].copy()
        return condiment_dict

    def _is_condiment(self, ingredient_name: str) -> bool:
        return ingredient_name in self.condiment_dict

    def _expand_condiment(
        self, condiment_name: str, quantity: float, condiment_unit: str
    ) -> Dict[str, float]:
        cache_key = (condiment_name, quantity)
        if cache_key in self.expansion_cache:
            return self.expansion_cache[cache_key].copy()

        raw_materials = {}

        if condiment_name not in self.condiment_dict:
            raw_materials[condiment_name] = quantity
            return raw_materials

        condiment_recipe = self.condiment_dict[condiment_name]
        base_condiment_qty = condiment_recipe["Condiment_Qty"].iloc[0]
        scaling_factor = quantity / base_condiment_qty

        for _, row in condiment_recipe.iterrows():
            sub_ingredient = row["Sub_Ingredient"]
            sub_qty = row["Qty_per_condiment_unit"]
            sub_unit = row["Sub_Unit"]

            try:
                sub_qty = float(sub_qty)
            except (ValueError, TypeError):
                print(
                    f"Warning: Invalid quantity '{sub_qty}' for sub-ingredient "
                    f"'{sub_ingredient}' in condiment '{condiment_name}'. Skipping."
                )
                continue

            actual_qty = sub_qty * scaling_factor

            if self._is_condiment(sub_ingredient):
                sub_materials = self._expand_condiment(
                    sub_ingredient, actual_qty, sub_unit
                )
                for material, material_qty in sub_materials.items():
                    raw_materials[material] = (
                        raw_materials.get(material, 0) + material_qty
                    )
            else:
                raw_materials[sub_ingredient] = (
                    raw_materials.get(sub_ingredient, 0) + actual_qty
                )

        self.expansion_cache[cache_key] = raw_materials.copy()
        return raw_materials

    def _get_item_raw_materials(
        self, item_name: str, quantity: float
    ) -> Dict[str, float]:
        raw_materials = {}
        normalized_name = item_name.strip().lower()

        if normalized_name in self.item_name_map:
            matched_name = self.item_name_map[normalized_name]
            item_recipe = self.menu_bom_df[self.menu_bom_df["Item"] == matched_name]
        else:
            item_recipe = self.menu_bom_df[self.menu_bom_df["Item"] == item_name]

        if item_recipe.empty:
            print(f"Warning: Item '{item_name}' not found in menu BOM")
            return raw_materials

        for _, row in item_recipe.iterrows():
            ingredient = row["Bahan"]
            ingredient_qty = row["Qty"]
            ingredient_unit = row["Unit"]

            try:
                ingredient_qty = float(ingredient_qty)
            except (ValueError, TypeError):
                print(
                    f"Warning: Invalid quantity '{ingredient_qty}' for ingredient "
                    f"'{ingredient}' in item '{item_name}'. Skipping."
                )
                continue

            total_qty = ingredient_qty * quantity

            if self._is_condiment(ingredient):
                sub_materials = self._expand_condiment(
                    ingredient, total_qty, ingredient_unit
                )
                for material, material_qty in sub_materials.items():
                    raw_materials[material] = (
                        raw_materials.get(material, 0) + material_qty
                    )
            else:
                raw_materials[ingredient] = raw_materials.get(ingredient, 0) + total_qty

        return raw_materials

    def _normalize_material_name(self, material_name: str) -> str:
        normalized = material_name.strip()
        acronyms = ["SKM", "BSJ", "SP"]
        upper_name = normalized.upper()
        if upper_name in acronyms:
            return upper_name
        return normalized.title()

    def _build_unit_map(self):
        self.unit_map: Dict[str, str] = {}
        for _, row in self.menu_bom_df.iterrows():
            material = row["Bahan"].strip()
            unit = row["Unit"].strip()
            self.unit_map[material.lower()] = unit

        for _, row in self.condiment_bom_df.iterrows():
            sub = row["Sub_Ingredient"].strip()
            unit = row["Sub_Unit"].strip()
            if sub.lower() not in self.unit_map:
                self.unit_map[sub.lower()] = unit

    def get_unit(self, material_name: str) -> str:
        return self.unit_map.get(material_name.lower(), "")

    def compute_material_requirements(self, forecast_df: pd.DataFrame) -> pd.DataFrame:
        df = forecast_df.copy()
        df["Date"] = pd.to_datetime(df["Date"]).dt.date

        daily_requirements: dict[tuple, float] = {}
        total_rows = len(df)

        for idx, row in df.iterrows():
            if total_rows > 500 and idx % 100 == 0:
                print(f"  BOM processing row {idx + 1}/{total_rows}...")

            date = row["Date"]
            item = row["Item"]
            quantity = row["Predicted"]

            if pd.isna(quantity) or quantity <= 0:
                continue

            raw_materials = self._get_item_raw_materials(item, quantity)

            for material, material_qty in raw_materials.items():
                normalized_material = self._normalize_material_name(material)
                key = (date, normalized_material)
                daily_requirements[key] = daily_requirements.get(key, 0) + material_qty

        result_data = []
        for (date, material), qty in daily_requirements.items():
            result_data.append({
                "Date": date,
                "Raw_Material": material,
                "Quantity_Required": round(qty, 2),
                "Unit": self.get_unit(material),
            })

        result_df = pd.DataFrame(result_data)
        result_df = result_df.sort_values(["Date", "Raw_Material"]).reset_index(drop=True)

        return result_df

    def aggregate_by_material(self, daily_df: pd.DataFrame) -> pd.DataFrame:
        agg = (
            daily_df.groupby("Raw_Material")
            .agg(Total_Required=("Quantity_Required", "sum"), Unit=("Unit", "first"))
            .reset_index()
            .sort_values("Total_Required", ascending=False)
        )
        agg["Total_Required"] = agg["Total_Required"].round(2)
        return agg
