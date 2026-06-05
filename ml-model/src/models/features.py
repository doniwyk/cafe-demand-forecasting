import pandas as pd
import numpy as np


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy().sort_values(["Item", "Date"]).reset_index(drop=True)

    roll_short, roll_long = 7, 28

    for item in data["Item"].unique():
        mask = data["Item"] == item
        g = data.loc[mask, "Quantity_Sold"]
        shifted = g.shift(1)

        data.loc[mask, "Lag_1"] = g.shift(1).values
        data.loc[mask, "Diff_1"] = g.diff(1).values
        data.loc[mask, "Accel_2"] = g.diff(1).diff(1).values

        g_lag1 = g.shift(1)
        g_lag4 = g.shift(4)
        data.loc[mask, "Seasonal_Strength"] = (g_lag1 / (g_lag4 + 1) - 1).values

        data.loc[mask, "Roll_Mean_7"] = shifted.rolling(roll_short, min_periods=1).mean().values
        data.loc[mask, "Roll_Mean_28"] = shifted.rolling(roll_long, min_periods=1).mean().values
        data.loc[mask, "Roll_Std_7"] = shifted.rolling(roll_short, min_periods=1).std().values
        data.loc[mask, "Roll_Q95_7"] = shifted.rolling(roll_short, min_periods=1).quantile(0.95).values

        data.loc[mask, "EWMA_7"] = shifted.ewm(span=roll_short, adjust=False).mean().values
        data.loc[mask, "EWMA_28"] = shifted.ewm(span=roll_long, adjust=False).mean().values

    data = data.fillna(0)
    data.replace([np.inf, -np.inf], 0, inplace=True)
    return data