import pandas as pd

from app.config import (
    DAILY_RAW_MATERIAL_PATH,
    FORECAST_PATH,
    MENU_BOM_PATH,
    CONDIMENT_BOM_PATH,
    CLEANED_SALES_PATH,
)
from app.models.material import DailyMaterialRequirement, MaterialRequirementPage
from src.models.raw_materials import RawMaterialProcessor


_df_cache: dict[str, pd.DataFrame] = {}


def _load(path) -> pd.DataFrame:
    key = str(path)
    if key not in _df_cache:
        _df_cache[key] = pd.read_csv(path)
    return _df_cache[key]


def get_daily_materials(
    material: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    page: int = 1,
    page_size: int = 100,
) -> MaterialRequirementPage:
    df = _load(DAILY_RAW_MATERIAL_PATH)
    if material:
        df = df[df["Raw_Material"].str.contains(material, case=False, na=False)]
    if start_date:
        df = df[df["Date"] >= start_date]
    if end_date:
        df = df[df["Date"] <= end_date]
    total = len(df)
    df = df.iloc[(page - 1) * page_size : page * page_size]
    return MaterialRequirementPage(
        data=[
            DailyMaterialRequirement(
                date=str(row["Date"]),
                raw_material=str(row["Raw_Material"]),
                quantity_required=float(row["Quantity_Required"]),
            )
            for _, row in df.iterrows()
        ],
        total=total,
        page=page,
        page_size=page_size,
    )


def get_material_forecast(
    material: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    page: int = 1,
    page_size: int = 100,
) -> MaterialRequirementPage:
    forecast_df = _load(FORECAST_PATH)
    sales_df = forecast_df.rename(columns={"Quantity_Sold": "Quantity"})
    sales_df["Date"] = pd.to_datetime(sales_df["Date"]).dt.date

    processor = RawMaterialProcessor(
        sales_path=CLEANED_SALES_PATH,
        menu_bom_path=MENU_BOM_PATH,
        condiment_bom_path=CONDIMENT_BOM_PATH,
    )
    requirements = processor.compute_material_requirements(sales_df)

    if material:
        requirements = requirements[
            requirements["Raw_Material"].str.contains(material, case=False, na=False)
        ]
    if start_date:
        requirements = requirements[requirements["Date"] >= start_date]
    if end_date:
        requirements = requirements[requirements["Date"] <= end_date]

    total = len(requirements)
    requirements = requirements.iloc[(page - 1) * page_size : page * page_size]

    return MaterialRequirementPage(
        data=[
            DailyMaterialRequirement(
                date=str(row["Date"]),
                raw_material=str(row["Raw_Material"]),
                quantity_required=float(row["Quantity_Required"]),
            )
            for _, row in requirements.iterrows()
        ],
        total=total,
        page=page,
        page_size=page_size,
    )
