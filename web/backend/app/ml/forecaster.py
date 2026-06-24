from __future__ import annotations

import numpy as np
import pandas as pd


class Forecaster:
    """Recursive multi-day forecaster with frozen lags and updated recency.

    Separates state that must stay anchored on historical data (lags, DOW
    baselines) from state that evolves with each prediction (recency, cross-item
    totals).  This is the same approach used in exploration_v2/forecast.py.
    """

    def __init__(self, data: pd.DataFrame, max_window: int = 60):
        self.items = sorted(data["Item"].unique())
        self.max_window = max_window
        self._frozen_qty: dict[str, np.ndarray] = {}
        self._live_qty: dict[str, np.ndarray] = {}

        for item in self.items:
            grp = data[data["Item"] == item].sort_values("Date").tail(max_window)
            vals = grp["Quantity_Sold"].values.astype(float)
            self._frozen_qty[item] = vals
            self._live_qty[item] = vals.copy()

    def build_features(self, date: pd.Timestamp, cross: dict) -> pd.DataFrame:
        dow = date.dayofweek
        rows = []

        for item in self.items:
            qty_live = self._live_qty.get(item, np.array([0.0]))
            dsl = self._compute_days_since_last_sale(qty_live)
            s7d = int(sum(1 for q in qty_live[-7:] if q > 0))
            if dow >= 4:
                dsl = 0
                s7d = max(s7d, 1)

            fq = self._frozen_qty.get(item, np.array([0.0]))
            n = len(fq)
            lags = {
                "Lag_1": float(fq[-1]) if n >= 1 else 0.0,
                "Lag_7": float(fq[-7]) if n >= 7 else 0.0,
                "Lag_14": float(fq[-14]) if n >= 14 else 0.0,
                "Lag_28": float(fq[-28]) if n >= 28 else 0.0,
            }

            rows.append({
                "Item": item,
                "Quantity_Sold": float(qty_live[-1]) if len(qty_live) > 0 else 0.0,
                "DOW": dow,
                "Is_Weekend": 1 if dow >= 5 else 0,
                "Month": date.month,
                "Year": date.year,
                "WeekOfYear": date.isocalendar()[1],
                "DayOfMonth": date.day,
                "Quarter": (date.month - 1) // 3 + 1,
                "MonthStart": 1 if date.day <= 7 else 0,
                "MonthEnd": 1 if date.day >= 25 else 0,
                "Is_Holiday_Season": 1 if date.month in (12, 1) else 0,
                "WeekOfMonth": (date.day - 1) // 7 + 1,
                "DaysFromStart": (date - pd.Timestamp("2022-01-01")).days,
                "DOW_Sin": np.sin(2 * np.pi * dow / 7),
                "DOW_Cos": np.cos(2 * np.pi * dow / 7),
                "Month_Sin": np.sin(2 * np.pi * date.month / 12),
                "Month_Cos": np.cos(2 * np.pi * date.month / 12),
                "Days_Since_Last_Sale": float(dsl),
                "Sales_Last_7D": s7d,
                **lags,
                "Day_Total_Qty": cross.get("total_qty", 0.0),
                "Day_Total_Items_Sold": cross.get("total_items", 0.0),
                "Day_Total_Beverage": 0.0,
                "Day_Total_Food": 0.0,
                "Day_Total_Qty_7D": cross.get("total_qty_7d", 0.0),
                "Category": "unknown",
            })

        return pd.DataFrame(rows)

    def update(self, preds: np.ndarray) -> dict:
        for i, item in enumerate(self.items):
            self._live_qty[item] = np.append(self._live_qty[item], preds[i])
            if len(self._live_qty[item]) > self.max_window:
                self._live_qty[item] = self._live_qty[item][-self.max_window:]

        tq = float(np.sum(preds))
        return {
            "total_qty": tq,
            "total_items": float(np.sum(preds > 0.5)),
            "total_qty_7d": tq,
        }

    def update_cross_7d(self, cross: dict, totals: list[float]) -> dict:
        totals.append(cross["total_qty"])
        if len(totals) > 7:
            totals = totals[-7:]
        cross["total_qty_7d"] = float(np.mean(totals))
        return cross

    @staticmethod
    def _compute_days_since_last_sale(qty: np.ndarray) -> int:
        for i in range(len(qty) - 1, -1, -1):
            if qty[i] > 0:
                return len(qty) - 1 - i
        return 999
