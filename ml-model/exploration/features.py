"""Local feature engineering + helpers — no dependency on src/."""
from __future__ import annotations

import pandas as pd
import numpy as np


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build feature matrix from daily item sales."""
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

        roll7 = shifted.rolling(roll_short, min_periods=1).mean()
        roll28 = shifted.rolling(roll_long, min_periods=1).mean()

        data.loc[mask, "Trend_7"] = ((roll7 - roll28) / (roll28 + 1)).values

        recent3 = shifted.rolling(3, min_periods=1).mean()
        data.loc[mask, "Momentum_3"] = ((recent3 - roll7) / (roll7 + 1)).values

        data.loc[mask, "Price_Level"] = (shifted / (roll28 + 1)).values

        # Seasonal lags — explicit weekly, monthly, 6-month patterns
        lag7 = g.shift(7)
        lag14 = g.shift(14)
        lag28 = g.shift(28)
        lag182 = g.shift(182)

        data.loc[mask, "Lag_7"] = lag7.values
        data.loc[mask, "Lag_14"] = lag14.values
        data.loc[mask, "Lag_28"] = lag28.values
        data.loc[mask, "Lag_182"] = lag182.values

        # Seasonal ratios — is this week stronger/weaker than usual?
        data.loc[mask, "Weekly_Ratio"] = (lag7 / (lag28 + 1)).values
        data.loc[mask, "Monthly_Ratio"] = (lag28 / (lag182 + 1)).values
        data.loc[mask, "Seasonal_Diff"] = (lag7 - lag28).values

    data["DOW"] = data["Date"].dt.dayofweek
    data["Is_Weekend"] = (data["DOW"] >= 5).astype(int)

    data = data.fillna(0)
    data.replace([np.inf, -np.inf], 0, inplace=True)
    return data


def _split_train_val(df: pd.DataFrame, val_ratio: float = 0.15):
    """Per-item temporal train/val split."""
    train_parts: list[pd.DataFrame] = []
    val_parts: list[pd.DataFrame] = []
    for item in df["Item"].unique():
        item_df = df[df["Item"] == item].sort_values("Date")
        n_val = max(1, int(len(item_df) * val_ratio))
        train_parts.append(item_df.iloc[: len(item_df) - n_val])
        val_parts.append(item_df.iloc[len(item_df) - n_val :])
    return pd.concat(train_parts, ignore_index=True), pd.concat(val_parts, ignore_index=True)
