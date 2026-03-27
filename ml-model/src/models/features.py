import pandas as pd
import numpy as np

from src.utils.config import (
    INDONESIAN_HOLIDAYS,
    RAMADAN_RANGES,
    REBRANDING_DATE,
)


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data = data.sort_values("Date").reset_index(drop=True)

    data["Month"] = data["Date"].dt.month
    data["Week"] = data["Date"].dt.isocalendar().week.astype(int)
    data["Year"] = data["Date"].dt.year
    data["DOY"] = data["Date"].dt.dayofyear
    data["DOW"] = data["Date"].dt.weekday

    data["Sin_Week"] = np.sin(2 * np.pi * data["Week"] / 52)
    data["Cos_Week"] = np.cos(2 * np.pi * data["Week"] / 52)

    data["Weeks_Since_Start"] = (data["Date"] - data["Date"].min()).dt.days // 7

    data["Is_Payday_Week"] = data["Date"].dt.day.apply(
        lambda x: 1 if x <= 7 or x >= 25 else 0
    )
    data["Is_Weekend_Before_Payday"] = (
        (data["DOW"] >= 4) & (data["Date"].dt.day >= 25)
    ).astype(int)

    data["Is_Holiday"] = (
        data["Date"].dt.strftime("%Y-%m-%d").isin(INDONESIAN_HOLIDAYS).astype(int)
    )

    data["Is_Ramadan"] = 0
    for start, end in RAMADAN_RANGES:
        mask = (data["Date"] >= start) & (data["Date"] <= end)
        data.loc[mask, "Is_Ramadan"] = 1

    data["Is_Post_Rebranding"] = (data["Date"] >= REBRANDING_DATE).astype(int)
    data["Weeks_Since_Rebrand"] = (
        (data["Date"] - pd.to_datetime(REBRANDING_DATE)).dt.days / 7
    ).clip(lower=0)

    return data


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    data = add_calendar_features(df)
    data = data.sort_values(["Item", "Date"])
    data["Item_Code"] = data["Item"].astype("category").cat.codes

    for item in data["Item"].unique():
        mask = data["Item"] == item
        g = data.loc[mask, "Quantity_Sold"]

        data.loc[mask, "Lag_1"] = g.shift(1)
        data.loc[mask, "Lag_2"] = g.shift(2)
        data.loc[mask, "Lag_4"] = g.shift(4)

        shifted = g.shift(1)
        data.loc[mask, "Roll_Mean_4"] = shifted.rolling(4, min_periods=1).mean()
        data.loc[mask, "Roll_Mean_12"] = shifted.rolling(12, min_periods=1).mean()
        data.loc[mask, "Roll_Std_4"] = shifted.rolling(4, min_periods=1).std()
        data.loc[mask, "Roll_Q95_4"] = shifted.rolling(4, min_periods=1).quantile(0.95)

        data.loc[mask, "EWMA_4"] = shifted.ewm(span=4, adjust=False).mean()
        data.loc[mask, "EWMA_12"] = shifted.ewm(span=12, adjust=False).mean()

        data.loc[mask, "Diff_1"] = g.diff(1)
        data.loc[mask, "Accel_2"] = data.loc[mask, "Diff_1"].diff(1)

        recent = shifted.rolling(4).mean()
        older = g.shift(5).rolling(8).mean()
        data.loc[mask, "Recent_vs_Old_Trend"] = (recent / (older + 1)).clip(0, 10)

        pre_mean = g[data.loc[mask, "Date"] < REBRANDING_DATE].mean()
        post_mean = g[data.loc[mask, "Date"] >= REBRANDING_DATE].mean()
        surge = (post_mean / (pre_mean + 1)) if pd.notna(post_mean) else 1.0
        data.loc[mask, "Post_Rebrand_Surge_Ratio"] = surge

    data = data.fillna(0)
    data.replace([np.inf, -np.inf], 0, inplace=True)
    return data
