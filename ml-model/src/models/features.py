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


def create_features(df: pd.DataFrame, frequency: str = "weekly") -> pd.DataFrame:
    data = add_calendar_features(df)
    data = data.sort_values(["Item", "Date"])
    data["Item_Code"] = data["Item"].astype("category").cat.codes

    if frequency == "daily":
        lag_steps = [1, 7, 14]
        roll_short, roll_long = 7, 28
        roll_std = 7
        ewma_short, ewma_long = 7, 28
    else:
        lag_steps = [1, 2, 4]
        roll_short, roll_long = 4, 12
        roll_std = 4
        ewma_short, ewma_long = 4, 12

    for item in data["Item"].unique():
        mask = data["Item"] == item
        g = data.loc[mask, "Quantity_Sold"]

        for lag_n in lag_steps:
            data.loc[mask, f"Lag_{lag_n}"] = g.shift(lag_n)

        shifted = g.shift(1)
        data.loc[mask, f"Roll_Mean_{roll_short}"] = shifted.rolling(roll_short, min_periods=1).mean()
        data.loc[mask, f"Roll_Mean_{roll_long}"] = shifted.rolling(roll_long, min_periods=1).mean()
        data.loc[mask, f"Roll_Std_{roll_std}"] = shifted.rolling(roll_std, min_periods=1).std()
        data.loc[mask, f"Roll_Q95_{roll_short}"] = shifted.rolling(roll_short, min_periods=1).quantile(0.95)

        data.loc[mask, f"EWMA_{ewma_short}"] = shifted.ewm(span=ewma_short, adjust=False).mean()
        data.loc[mask, f"EWMA_{ewma_long}"] = shifted.ewm(span=ewma_long, adjust=False).mean()

        data.loc[mask, "Diff_1"] = g.diff(1)
        data.loc[mask, "Accel_2"] = data.loc[mask, "Diff_1"].diff(1)

        recent = shifted.rolling(roll_short).mean()
        older = g.shift(roll_short + 1).rolling(roll_long).mean()
        data.loc[mask, "Recent_vs_Old_Trend"] = (recent / (older + 1)).clip(0, 10)

        pre_mean = g[data.loc[mask, "Date"] < REBRANDING_DATE].mean()
        post_mean = g[data.loc[mask, "Date"] >= REBRANDING_DATE].mean()
        surge = (post_mean / (pre_mean + 1)) if pd.notna(post_mean) else 1.0
        data.loc[mask, "Post_Rebrand_Surge_Ratio"] = surge

    data = data.fillna(0)
    data.replace([np.inf, -np.inf], 0, inplace=True)
    return data
