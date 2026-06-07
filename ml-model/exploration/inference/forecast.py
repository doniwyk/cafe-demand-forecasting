"""Production inference for daily item sales forecasting.

Blended approach:
  - Fri/Sat: 70% DOW_P75 baseline + 30% quantile XGBoost
  - Weekdays: 60% DOW_Median baseline + 40% quantile XGBoost

DOW_P75 captures the upper range of weekend demand while avoiding the
extreme overprediction of P90. Backtested across 5 historical periods:
  - Overall MAE: ~1.4 | Fri/Sat MAE: ~3.4 (vs 4.8 with P90)
  - Slight overprediction bias is intentional for supply planning.
"""
from __future__ import annotations

import json
import os
import pickle
import numpy as np
import pandas as pd
from datetime import timedelta
from pathlib import Path
from xgboost import XGBRegressor

BASE_DIR = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(BASE_DIR))

CAFE_DB_URL = os.getenv(
    "CAFE_DB_URL",
    "postgresql://postgres:postgres@localhost:5433/cafe_forecasting",
)
MODELS_DIR = BASE_DIR / "models" / "exploration"

QUANTILE = 0.75
N_ESTIMATORS = 600
MAX_DEPTH = 5
LEARNING_RATE = 0.04
DOW_LOOKBACK_WEEKS = 12
FORECAST_HORIZON = 7
MIN_NONZERO_DAYS = 60

SKIP_PREFIXES = [
    "Add ", "Filter", "FIlter", "V60",
]
DISCONTINUED_ITEMS = ["Menawan"]

WEEKEND_BLEND_MODEL = 0.3
WEEKDAY_BLEND_MODEL = 0.4

FEATURE_COLS = [
    "Lag_7", "Lag_14", "Lag_28",
    "Roll_Mean_7", "Roll_Mean_28",
    "EWMA_7", "EWMA_28", "Trend_7",
    "DOW", "Is_Weekend",
    "DOW_Avg", "DOW_P75", "DOW_P90", "DOW_Std", "DOW_Median",
]


def _should_skip(item_name: str) -> bool:
    for prefix in SKIP_PREFIXES:
        if item_name.startswith(prefix):
            return True
    return item_name in DISCONTINUED_ITEMS


def load_all_items() -> pd.DataFrame:
    import psycopg2
    conn = psycopg2.connect(CAFE_DB_URL)
    query = """
        SELECT d.date, i.name AS item, d.quantity_sold
        FROM daily_item_sales d
        JOIN items i ON d.item_id = i.id
        ORDER BY i.name, d.date
    """
    df = pd.read_sql(query, conn)
    conn.close()
    df.rename(columns={"date": "Date", "item": "Item", "quantity_sold": "Quantity_Sold"}, inplace=True)
    df["Date"] = pd.to_datetime(df["Date"])
    df["Quantity_Sold"] = df["Quantity_Sold"].astype(int)
    df = df[~df["Item"].apply(_should_skip)].copy()
    print(f"Loaded {len(df):,} rows, {df['Item'].nunique()} items")
    return df


def load_item_data(item_name: str, df_all: pd.DataFrame | None = None) -> pd.DataFrame | None:
    if df_all is not None:
        df = df_all[df_all["Item"] == item_name].copy()
    else:
        import psycopg2
        conn = psycopg2.connect(CAFE_DB_URL)
        query = """
            SELECT d.date, d.quantity_sold
            FROM daily_item_sales d
            JOIN items i ON d.item_id = i.id
            WHERE i.name = %s
            ORDER BY d.date
        """
        df = pd.read_sql(query, conn, params=(item_name,))
        conn.close()
        df.rename(columns={"date": "Date", "quantity_sold": "Quantity_Sold"}, inplace=True)
        df["Date"] = pd.to_datetime(df["Date"])
        df["Quantity_Sold"] = df["Quantity_Sold"].astype(int)
    return df.reset_index(drop=True)


def compute_dow_stats(df: pd.DataFrame, lookback_weeks: int = DOW_LOOKBACK_WEEKS) -> pd.DataFrame:
    non_zero = df[df["Quantity_Sold"] > 0].copy()
    if len(non_zero) == 0:
        return pd.DataFrame({"DOW": range(7)})

    cutoff = non_zero["Date"].max() - pd.Timedelta(weeks=lookback_weeks)
    recent = non_zero[non_zero["Date"] >= cutoff]

    stats = recent.groupby(recent["Date"].dt.dayofweek)["Quantity_Sold"].agg(
        DOW_Avg="mean",
        DOW_P75=lambda x: x.quantile(0.75),
        DOW_P90=lambda x: x.quantile(0.90),
        DOW_Std="std",
        DOW_Median="median",
    ).reset_index()
    stats.columns = ["DOW", "DOW_Avg", "DOW_P75", "DOW_P90", "DOW_Std", "DOW_Median"]
    stats = stats.fillna(0)
    return stats


def build_item_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("Date").reset_index(drop=True)
    g = df["Quantity_Sold"]
    shifted = g.shift(1)

    for lag in [7, 14, 28, 182]:
        df[f"Lag_{lag}"] = g.shift(lag).values

    df["Roll_Mean_7"] = shifted.rolling(7, min_periods=1).mean().values
    df["Roll_Mean_28"] = shifted.rolling(28, min_periods=1).mean().values
    df["EWMA_7"] = shifted.ewm(span=7, adjust=False).mean().values
    df["EWMA_28"] = shifted.ewm(span=28, adjust=False).mean().values

    roll7 = shifted.rolling(7, min_periods=1).mean()
    roll28 = shifted.rolling(28, min_periods=1).mean()
    df["Trend_7"] = ((roll7 - roll28) / (roll28 + 1)).values

    df["Weekly_Ratio"] = (g.shift(7) / (g.shift(28) + 1)).values
    df["Seasonal_Diff"] = (g.shift(7) - g.shift(28)).values

    df["DOW"] = df["Date"].dt.dayofweek
    df["Is_Weekend"] = (df["DOW"] >= 5).astype(int)

    dow_stats = compute_dow_stats(df)
    df = df.merge(dow_stats, on="DOW", how="left")

    df = df.fillna(0)
    df.replace([np.inf, -np.inf], 0, inplace=True)
    return df


def train_model(df: pd.DataFrame, features: list) -> XGBRegressor:
    non_zero = df[df["Quantity_Sold"] > 0].copy()

    sample_weight = np.ones(len(non_zero))
    fri_sat_mask = non_zero["DOW"].isin([4, 5])
    sample_weight[fri_sat_mask] = 3.0

    model = XGBRegressor(
        objective="reg:quantileerror",
        quantile_alpha=QUANTILE,
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        learning_rate=LEARNING_RATE,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=1.0,
        reg_lambda=2.0,
        random_state=42,
    )
    model.fit(non_zero[features], non_zero["Quantity_Sold"], sample_weight=sample_weight, verbose=False)
    return model


def _dow_baseline(dow_stats: pd.DataFrame, dow: int) -> float:
    row = dow_stats[dow_stats["DOW"] == dow]
    if row.empty:
        return 3.0
    row = row.iloc[0]
    if dow in (4, 5):
        return row["DOW_P75"]
    return row["DOW_Median"]


def forecast_item(
    model: XGBRegressor,
    dow_stats: pd.DataFrame,
    df_hist: pd.DataFrame,
    features: list,
    n_days: int = FORECAST_HORIZON,
) -> pd.DataFrame:
    last_date = df_hist["Date"].max()
    forecast_dates = [last_date + timedelta(days=d) for d in range(1, n_days + 1)]

    all_rows = df_hist[["Date", "Quantity_Sold"]].copy()
    for fd in forecast_dates:
        all_rows = pd.concat(
            [all_rows, pd.DataFrame({"Date": [fd], "Quantity_Sold": [np.nan]})],
            ignore_index=True,
        )
    all_rows = all_rows.sort_values("Date").reset_index(drop=True)

    results = []
    for fd in forecast_dates:
        feat_df = build_item_features(all_rows.copy())
        row = feat_df[feat_df["Date"] == fd]
        if row.empty:
            continue

        model_pred = max(0, model.predict(row[features])[0])
        dow = fd.dayofweek
        baseline = _dow_baseline(dow_stats, dow)

        blend_w = WEEKEND_BLEND_MODEL if dow in (4, 5) else WEEKDAY_BLEND_MODEL
        blended = blend_w * model_pred + (1 - blend_w) * baseline

        results.append({
            "Date": fd,
            "DOW": dow,
            "DOW_Name": fd.day_name(),
            "Model": round(model_pred, 2),
            "Baseline": round(baseline, 2),
            "Predicted": round(blended, 2),
        })

        idx = all_rows[all_rows["Date"] == fd].index[0]
        all_rows.loc[idx, "Quantity_Sold"] = blended

    return pd.DataFrame(results)


def forecast_single(item_name: str, df_all: pd.DataFrame | None = None, n_days: int = FORECAST_HORIZON):
    df = load_item_data(item_name, df_all)
    if df is None or len(df) < MIN_NONZERO_DAYS:
        print(f"  Skipping '{item_name}': insufficient data ({len(df) if df is not None else 0} days)")
        return None

    nonzero = (df["Quantity_Sold"] > 0).sum()
    if nonzero < MIN_NONZERO_DAYS:
        print(f"  Skipping '{item_name}': only {nonzero} non-zero days")
        return None

    df_feat = build_item_features(df.copy())
    features = [f for f in FEATURE_COLS if f in df_feat.columns]
    dow_stats = compute_dow_stats(df)
    model = train_model(df_feat, features)

    return forecast_item(model, dow_stats, df, features, n_days)


def forecast_all(n_days: int = FORECAST_HORIZON) -> dict[str, pd.DataFrame]:
    df_all = load_all_items()

    items = sorted(df_all["Item"].unique())
    print(f"\nForecasting {len(items)} items...")

    results = {}
    skipped = []
    for idx, item in enumerate(items):
        if (idx + 1) % 10 == 0 or idx == 0:
            print(f"  [{idx + 1}/{len(items)}] {item}")

        result = forecast_single(item, df_all, n_days)
        if result is not None:
            results[item] = result
        else:
            skipped.append(item)

    print(f"\nForecasted {len(results)} items, skipped {len(skipped)}")
    if skipped:
        print(f"  Skipped: {skipped[:10]}{'...' if len(skipped) > 10 else ''}")

    return results


def print_forecast_table(results: dict[str, pd.DataFrame], top_n: int = 15):
    print("\n" + "=" * 80)
    print(f"FORECAST RESULTS (top {top_n} items by avg predicted daily sales)")
    print("=" * 80)

    avg_preds = {}
    for item, df in results.items():
        avg_preds[item] = df["Predicted"].mean()

    top_items = sorted(avg_preds, key=avg_preds.get, reverse=True)[:top_n]

    for item in top_items:
        df = results[item]
        print(f"\n--- {item} (avg: {avg_preds[item]:.1f}/day) ---")
        print(f"  {'Date':<12} {'DOW':<10} {'Model':>7} {'Baseline':>9} {'Predicted':>10}")
        print(f"  {'-'*50}")
        for _, r in df.iterrows():
            print(f"  {r['Date'].strftime('%Y-%m-%d'):<12} {r['DOW_Name']:<10} "
                  f"{r['Model']:>7.1f} {r['Baseline']:>9.1f} {r['Predicted']:>10.1f}")


def save_results(results: dict[str, pd.DataFrame]):
    output_dir = MODELS_DIR / "inference"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_forecasts = []
    for item, df in results.items():
        df_copy = df.copy()
        df_copy["Item"] = item
        all_forecasts.append(df_copy)

    if all_forecasts:
        combined = pd.concat(all_forecasts, ignore_index=True)
        combined = combined[["Item", "Date", "DOW", "DOW_Name", "Model", "Baseline", "Predicted"]]
        combined.to_csv(output_dir / "forecasts.csv", index=False)
        print(f"Saved {len(combined)} forecast rows to {output_dir / 'forecasts.csv'}")

    metadata = {
        "generated_at": pd.Timestamp.now().isoformat(),
        "n_items": len(results),
        "items": sorted(results.keys()),
        "forecast_horizon": FORECAST_HORIZON,
        "quantile": QUANTILE,
        "dow_lookback_weeks": DOW_LOOKBACK_WEEKS,
        "blend_weights": {
            "fri_sat_model": WEEKEND_BLEND_MODEL,
            "weekday_model": WEEKDAY_BLEND_MODEL,
        },
        "features": FEATURE_COLS,
    }
    with open(output_dir / "forecast_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)


def main():
    print("=" * 80)
    print("PRODUCTION INFERENCE: All Items")
    print(f"Quantile: {QUANTILE} | Horizon: {FORECAST_HORIZON} days | DOW lookback: {DOW_LOOKBACK_WEEKS} weeks")
    print(f"Blend: Fri/Sat={WEEKEND_BLEND_MODEL:.0%} model, Weekdays={WEEKDAY_BLEND_MODEL:.0%} model")
    print("=" * 80)

    results = forecast_all()
    print_forecast_table(results)
    save_results(results)

    return results


if __name__ == "__main__":
    main()
