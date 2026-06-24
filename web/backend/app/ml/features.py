import pandas as pd
import numpy as np

TEMPORAL = [
    "DOW", "Is_Weekend", "Month", "Year", "WeekOfYear", "DayOfMonth",
    "Quarter", "MonthStart", "MonthEnd", "Is_Holiday_Season",
    "WeekOfMonth", "DaysFromStart", "DOW_Sin", "DOW_Cos", "Month_Sin", "Month_Cos",
]

RECENCY = ["Days_Since_Last_Sale", "Sales_Last_7D"]

LAGS = ["Lag_1", "Lag_7", "Lag_14", "Lag_28"]

CROSS = [
    "Day_Total_Qty", "Day_Total_Items_Sold", "Day_Total_Beverage",
    "Day_Total_Food", "Day_Total_Qty_7D",
]

ALL_FEATURE_COLS = TEMPORAL + RECENCY + LAGS + CROSS


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy().sort_values(["Item", "Date"]).reset_index(drop=True)
    items = sorted(data["Item"].unique())

    _add_temporal_features(data)
    _add_recency_features(data)
    _add_lag_features(data)
    _add_cross_item_features(data)

    na_cols = [
        "Days_Since_First_Sale", "Days_Since_Last_Sale", "Sales_Last_7D",
        "Lag_1", "Lag_7", "Lag_14", "Lag_28",
    ]
    for col in na_cols:
        if col in data.columns:
            data[col] = data[col].fillna(0)

    return data.replace([np.inf, -np.inf], 0)


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in ALL_FEATURE_COLS if c in df.columns]


def normalize_dow_factors(dow_factors: dict) -> dict:
    result = {}
    for item, factors in dow_factors.items():
        result[item] = {
            int(k) if isinstance(k, str) and k.isdigit()
            else int(k) if isinstance(k, (int, float))
            else k: v
            for k, v in factors.items()
        }
    return result


def _add_temporal_features(data: pd.DataFrame) -> None:
    d = data["Date"]
    data["DOW"] = d.dt.dayofweek
    data["Is_Weekend"] = (data["DOW"] >= 5).astype(int)
    data["Month"] = d.dt.month
    data["Year"] = d.dt.year
    data["WeekOfYear"] = d.dt.isocalendar().week.astype(int)
    data["DayOfMonth"] = d.dt.day
    data["Quarter"] = d.dt.quarter
    data["MonthStart"] = (data["DayOfMonth"] <= 7).astype(int)
    data["MonthEnd"] = (data["DayOfMonth"] >= 25).astype(int)
    data["Is_Holiday_Season"] = data["Month"].isin([12, 1]).astype(int)
    data["WeekOfMonth"] = ((data["DayOfMonth"] - 1) // 7 + 1).astype(int)
    data["DaysFromStart"] = (d - d.min()).dt.days
    data["DOW_Sin"] = np.sin(2 * np.pi * data["DOW"] / 7)
    data["DOW_Cos"] = np.cos(2 * np.pi * data["DOW"] / 7)
    data["Month_Sin"] = np.sin(2 * np.pi * data["Month"] / 12)
    data["Month_Cos"] = np.cos(2 * np.pi * data["Month"] / 12)


def _add_recency_features(data: pd.DataFrame) -> None:
    for item, grp in data.groupby("Item", sort=False):
        mask = data["Item"] == item
        qty = grp["Quantity_Sold"].values
        n = len(qty)

        days_since = np.full(n, 999, dtype=int)
        s7d = np.zeros(n, dtype=int)
        last = None
        for i in range(n):
            if qty[i] > 0:
                last = i
            if last is not None and i > 0:
                days_since[i] = min(i - last, 999)
            s7d[i] = int(sum(1 for j in range(max(0, i - 7), i) if qty[j] > 0))

        data.loc[mask, "Days_Since_Last_Sale"] = days_since
        data.loc[mask, "Sales_Last_7D"] = s7d.astype(int)


def _add_lag_features(data: pd.DataFrame) -> None:
    for item, grp in data.groupby("Item", sort=False):
        mask = data["Item"] == item
        qty = grp["Quantity_Sold"].values
        s = pd.Series(qty)
        data.loc[mask, "Lag_1"] = s.shift(1).fillna(0).values
        data.loc[mask, "Lag_7"] = s.shift(7).fillna(0).values
        data.loc[mask, "Lag_14"] = s.shift(14).fillna(0).values
        data.loc[mask, "Lag_28"] = s.shift(28).fillna(0).values


def _add_cross_item_features(data: pd.DataFrame) -> None:
    daily = data.groupby("Date").agg(
        Total_Qty=("Quantity_Sold", "sum"),
        Total_Items_Sold=("Quantity_Sold", lambda x: (x > 0).sum()),
    ).shift(1).fillna(0)

    daily["Total_Beverage"] = (
        data[data["Category"] == "beverage"]
        .groupby("Date")["Quantity_Sold"]
        .sum().shift(1).fillna(0)
    ) if "Category" in data.columns else 0

    daily["Total_Food"] = (
        data[data["Category"] == "food"]
        .groupby("Date")["Quantity_Sold"]
        .sum().shift(1).fillna(0)
    ) if "Category" in data.columns else 0

    daily["Total_Qty_7D"] = daily["Total_Qty"].rolling(7, min_periods=1).mean()

    for col in ["Total_Qty", "Total_Items_Sold", "Total_Beverage", "Total_Food", "Total_Qty_7D"]:
        data[f"Day_{col}"] = data["Date"].map(daily[col]).fillna(0)
