"""Production inference for daily item sales forecasting.

Blended approach with 3 components:
  - Quantile XGBoost (tuned params, q=0.75)
  - Random Forest (tuned params)
  - DOW percentile baseline

Fri/Sat: XGB + RF average, blended with DOW_P75 baseline
Weekdays: XGB + RF average, blended with DOW_Median baseline

Backtested across 5 historical periods:
  - Overall MAE: ~1.4 | Fri/Sat MAE: ~1.5
  - Slight overprediction bias is intentional for supply planning.
"""
from __future__ import annotations

import json
import os
import numpy as np
import pandas as pd
from datetime import timedelta
from pathlib import Path
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor

BASE_DIR = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(BASE_DIR))

from config import MODELS_DIR
from inference.bom import RawMaterialProcessor

CAFE_DB_URL = os.getenv(
    "CAFE_DB_URL",
    "postgresql://postgres:postgres@localhost:5433/cafe_forecasting",
)

TUNING_DIR = MODELS_DIR / "exploration" / "tuning"
XGB_TUNING_FILE = TUNING_DIR / "quantile_best_params.json"
RF_TUNING_FILE = TUNING_DIR / "rf_best_params.json"
BLEND_TUNING_FILE = TUNING_DIR / "blend_best_params.json"

QUANTILE = 0.75
DOW_LOOKBACK_WEEKS = 12
FORECAST_HORIZON = 7
MIN_NONZERO_DAYS = 60
FRI_SAT_UPWEIGHT = 3.0

DEFAULT_XGB_PARAMS = {
    "n_estimators": 200,
    "max_depth": 3,
    "learning_rate": 0.02,
    "min_child_weight": 1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 1.0,
    "reg_lambda": 1.0,
}

DEFAULT_RF_PARAMS = {
    "n_estimators": 300,
    "max_depth": 7,
    "min_samples_split": 10,
    "min_samples_leaf": 1,
    "max_features": 1.0,
}


def _load_xgb_params() -> dict:
    if not hasattr(_load_xgb_params, "_cache"):
        if XGB_TUNING_FILE.exists():
            with open(XGB_TUNING_FILE) as f:
                tuned = json.load(f)
            _load_xgb_params._cache = tuned.get("params", {})
        else:
            _load_xgb_params._cache = DEFAULT_XGB_PARAMS
    return _load_xgb_params._cache


def _load_rf_params() -> dict:
    if not hasattr(_load_rf_params, "_cache"):
        if RF_TUNING_FILE.exists():
            with open(RF_TUNING_FILE) as f:
                data = json.load(f)
            params = data.get("params", data)
            params["random_state"] = 42
            params["n_jobs"] = -1
            _load_rf_params._cache = params
        else:
            _load_rf_params._cache = {**DEFAULT_RF_PARAMS, "random_state": 42, "n_jobs": -1}
    return _load_rf_params._cache

DEFAULT_BLEND_CONFIG = {
    "weekend_baseline": "P75",
    "weekend_model_w": 0.8,
    "weekday_model_w": 0.6,
    "rf_weight": 0.5,
}


def _load_blend_config() -> dict:
    if not hasattr(_load_blend_config, "_cache"):
        if BLEND_TUNING_FILE.exists():
            with open(BLEND_TUNING_FILE) as f:
                tuned = json.load(f)
            _load_blend_config._cache = tuned.get("best_config", DEFAULT_BLEND_CONFIG)
        else:
            _load_blend_config._cache = DEFAULT_BLEND_CONFIG
    return _load_blend_config._cache


SKIP_PREFIXES = [
    "Add ", "Filter", "FIlter", "V60",
]
DISCONTINUED_ITEMS = ["Menawan"]

FEATURE_COLS = [
    "Lag_7", "Lag_14", "Lag_28",
    "Roll_Mean_7", "Roll_Mean_28",
    "EWMA_7", "EWMA_28", "Trend_7",
    "Momentum",
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
        DOW_P95=lambda x: x.quantile(0.95),
        DOW_Std="std",
        DOW_Median="median",
    ).reset_index()
    stats.columns = ["DOW", "DOW_Avg", "DOW_P75", "DOW_P90", "DOW_P95", "DOW_Std", "DOW_Median"]
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

    df["Momentum"] = ((df["Roll_Mean_7"] - df["DOW_Avg"]) / (df["DOW_Avg"] + 1)).fillna(0)
    df.replace([np.inf, -np.inf], 0, inplace=True)

    df = df.fillna(0)
    df.replace([np.inf, -np.inf], 0, inplace=True)
    return df


def train_models(df: pd.DataFrame, features: list) -> tuple[XGBRegressor, RandomForestRegressor]:
    non_zero = df[df["Quantity_Sold"] > 0].copy()

    sample_weight = np.ones(len(non_zero))
    fri_sat_mask = non_zero["DOW"].isin([4, 5])
    sample_weight[fri_sat_mask] = FRI_SAT_UPWEIGHT

    xgb_params = _load_xgb_params()
    xgb = XGBRegressor(
        objective="reg:quantileerror",
        quantile_alpha=QUANTILE,
        random_state=42,
        **xgb_params,
    )
    xgb.fit(non_zero[features], non_zero["Quantity_Sold"], sample_weight=sample_weight, verbose=False)

    rf_params = _load_rf_params()
    rf = RandomForestRegressor(**rf_params)
    rf.fit(non_zero[features], non_zero["Quantity_Sold"], sample_weight=sample_weight)

    return xgb, rf


def compute_global_dow_stats(df_all: pd.DataFrame) -> pd.DataFrame:
    """DOW statistics pooled across ALL items as fallback baseline."""
    df_all = df_all[df_all["Quantity_Sold"] > 0].copy()
    if len(df_all) == 0:
        return pd.DataFrame({"DOW": range(7)}).fillna(0)
    cutoff = df_all["Date"].max() - pd.Timedelta(weeks=DOW_LOOKBACK_WEEKS)
    recent = df_all[df_all["Date"] >= cutoff]
    stats = recent.groupby(recent["Date"].dt.dayofweek)["Quantity_Sold"].agg(
        DOW_Avg="mean",
        DOW_P75=lambda x: x.quantile(0.75),
        DOW_P90=lambda x: x.quantile(0.90),
        DOW_P95=lambda x: x.quantile(0.95),
        DOW_Std="std",
        DOW_Median="median",
    ).reset_index()
    stats.columns = ["DOW", "DOW_Avg", "DOW_P75", "DOW_P90", "DOW_P95", "DOW_Std", "DOW_Median"]
    return stats.fillna(0)


def train_global_models(df_all: pd.DataFrame) -> tuple[XGBRegressor, RandomForestRegressor]:
    """Train models on ALL items pooled together as fallback for low-data items."""
    all_feat = []
    for item in df_all["Item"].unique():
        item_df = df_all[df_all["Item"] == item].copy()
        if len(item_df) < 2:
            continue
        feat_df = build_item_features(item_df)
        all_feat.append(feat_df)
    pooled = pd.concat(all_feat, ignore_index=True)
    pooled = pooled[pooled["Quantity_Sold"] > 0].copy()
    features = [f for f in FEATURE_COLS if f in pooled.columns]
    print(f"  Global model training: {len(pooled):,} rows from {df_all['Item'].nunique()} items")
    return train_models(pooled, features)


def _round_cups(value: float) -> int:
    fractional = value - int(value)
    if fractional >= 0.2:
        return int(np.ceil(value))
    return int(np.floor(value))


def _dow_baseline(dow_stats: pd.DataFrame, dow: int, stat: str = "P75") -> float:
    row = dow_stats[dow_stats["DOW"] == dow]
    if row.empty:
        return 3.0
    row = row.iloc[0]
    if dow in (4, 5):
        col = f"DOW_{stat}"
        return row[col] if col in row.index else row["DOW_P75"]
    return row["DOW_Median"]


def forecast_item(
    xgb: XGBRegressor,
    rf: RandomForestRegressor,
    dow_stats: pd.DataFrame,
    df_hist: pd.DataFrame,
    features: list,
    n_days: int = FORECAST_HORIZON,
) -> pd.DataFrame:
    blend_cfg = _load_blend_config()

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

        xgb_pred = max(0, xgb.predict(row[features])[0])
        rf_pred = max(0, rf.predict(row[features])[0])
        rf_w = blend_cfg.get("rf_weight", 0.0)
        model_pred = rf_w * rf_pred + (1 - rf_w) * xgb_pred

        dow = fd.dayofweek
        baseline = _dow_baseline(dow_stats, dow, blend_cfg.get("weekend_baseline", "P75"))

        blend_w = blend_cfg.get("weekend_model_w", 0.8) if dow in (4, 5) else blend_cfg.get("weekday_model_w", 0.6)
        blended = blend_w * model_pred + (1 - blend_w) * baseline

        results.append({
            "Date": fd,
            "DOW": dow,
            "DOW_Name": fd.day_name(),
            "XGB": round(xgb_pred, 2),
            "RF": round(rf_pred, 2),
            "Baseline": round(baseline, 2),
            "Predicted": round(blended, 2),
        })

        idx = all_rows[all_rows["Date"] == fd].index[0]
        all_rows.loc[idx, "Quantity_Sold"] = blended

    return pd.DataFrame(results)


def forecast_single(
    item_name: str,
    df_all: pd.DataFrame | None = None,
    n_days: int = FORECAST_HORIZON,
    global_models: tuple[XGBRegressor, RandomForestRegressor] | None = None,
    global_dow_stats: pd.DataFrame | None = None,
) -> pd.DataFrame | None:
    df = load_item_data(item_name, df_all)
    if df is None or len(df) == 0:
        print(f"  Skipping '{item_name}': no data")
        return None

    nonzero = (df["Quantity_Sold"] > 0).sum()

    if nonzero >= MIN_NONZERO_DAYS:
        df_feat = build_item_features(df.copy())
        features = [f for f in FEATURE_COLS if f in df_feat.columns]
        dow_stats = compute_dow_stats(df)
        xgb, rf = train_models(df_feat, features)
    elif global_models is not None:
        xgb, rf = global_models
        features = [f for f in FEATURE_COLS]
        dow_stats = compute_dow_stats(df)
        if dow_stats["DOW_Avg"].sum() == 0 and global_dow_stats is not None:
            dow_stats = global_dow_stats
        print(f"  Using global model for '{item_name}' ({nonzero} non-zero days)")
    else:
        print(f"  Skipping '{item_name}': only {nonzero} non-zero days")
        return None

    result = forecast_item(xgb, rf, dow_stats, df, features, n_days)
    result["Predicted"] = result["Predicted"].apply(_round_cups)
    return result


def forecast_all(n_days: int = FORECAST_HORIZON) -> dict[str, pd.DataFrame]:
    df_all = load_all_items()

    items = sorted(df_all["Item"].unique())
    print(f"\nForecasting {len(items)} items...")

    print("Training global fallback models...")
    global_xgb, global_rf = train_global_models(df_all)
    global_dow = compute_global_dow_stats(df_all)
    global_models = (global_xgb, global_rf)

    results = {}
    skipped = []
    for idx, item in enumerate(items):
        if (idx + 1) % 10 == 0 or idx == 0:
            print(f"  [{idx + 1}/{len(items)}] {item}")

        result = forecast_single(item, df_all, n_days, global_models, global_dow)
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
        print(f"  {'Date':<12} {'DOW':<10} {'XGB':>7} {'RF':>7} {'Baseline':>9} {'Predicted':>10}")
        print(f"  {'-'*55}")
        for _, r in df.iterrows():
            print(f"  {r['Date'].strftime('%Y-%m-%d'):<12} {r['DOW_Name']:<10} "
                  f"{r['XGB']:>7.1f} {r['RF']:>7.1f} {r['Baseline']:>9.1f} {r['Predicted']:>10d}")


def convert_forecast_to_bom(
    results: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convert forecasted item sales into raw material requirements.

    Returns:
        (daily_bom_df, aggregated_bom_df)
        - daily_bom_df: Date, Raw_Material, Quantity_Required, Unit
        - aggregated_bom_df: Raw_Material, Total_Required, Unit (summed across all dates)
    """
    all_forecasts = []
    for item, df in results.items():
        df_copy = df[["Date", "Predicted"]].copy()
        df_copy["Item"] = item
        all_forecasts.append(df_copy)

    if not all_forecasts:
        return pd.DataFrame(), pd.DataFrame()

    combined = pd.concat(all_forecasts, ignore_index=True)

    print(f"Converting forecasts to material requirements...")
    processor = RawMaterialProcessor()
    daily_bom = processor.compute_material_requirements(combined)

    if daily_bom.empty:
        return daily_bom, pd.DataFrame()

    agg_bom = processor.aggregate_by_material(daily_bom)
    return daily_bom, agg_bom


def print_bom_summary(agg_bom: pd.DataFrame, top_n: int = 20):
    if agg_bom.empty:
        return

    print("\n" + "=" * 80)
    print(f"TOP {top_n} RAW MATERIAL REQUIREMENTS (aggregated across all forecast days)")
    print("=" * 80)
    print(f"  {'Material':<30} {'Total Qty':>12} {'Unit':<10}")
    print(f"  {'-'*52}")
    for _, row in agg_bom.head(top_n).iterrows():
        print(f"  {row['Raw_Material']:<30} {row['Total_Required']:>12.1f} {row['Unit']:<10}")


def save_results(
    results: dict[str, pd.DataFrame],
    daily_bom: pd.DataFrame | None = None,
    agg_bom: pd.DataFrame | None = None,
):
    output_dir = MODELS_DIR / "inference"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_forecasts = []
    for item, df in results.items():
        df_copy = df.copy()
        df_copy["Item"] = item
        all_forecasts.append(df_copy)

    if all_forecasts:
        combined = pd.concat(all_forecasts, ignore_index=True)
        combined = combined[["Item", "Date", "DOW", "DOW_Name", "XGB", "RF", "Baseline", "Predicted"]]
        combined.to_csv(output_dir / "forecasts.csv", index=False)
        print(f"Saved {len(combined)} forecast rows to {output_dir / 'forecasts.csv'}")

    if daily_bom is not None and not daily_bom.empty:
        daily_bom.to_csv(output_dir / "bom_daily.csv", index=False)
        print(f"Saved {len(daily_bom)} daily BOM rows to {output_dir / 'bom_daily.csv'}")

    if agg_bom is not None and not agg_bom.empty:
        agg_bom.to_csv(output_dir / "bom_aggregated.csv", index=False)
        print(f"Saved {len(agg_bom)} aggregated BOM rows to {output_dir / 'bom_aggregated.csv'}")

    metadata = {
        "generated_at": pd.Timestamp.now().isoformat(),
        "n_items": len(results),
        "items": sorted(results.keys()),
        "forecast_horizon": FORECAST_HORIZON,
        "quantile": QUANTILE,
        "dow_lookback_weeks": DOW_LOOKBACK_WEEKS,
        "blend_config": _load_blend_config(),
        "features": FEATURE_COLS,
        "n_raw_materials": len(agg_bom) if agg_bom is not None and not agg_bom.empty else 0,
    }
    with open(output_dir / "forecast_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)


def main():
    blend_cfg = _load_blend_config()
    print(f"Forecast: quantile={QUANTILE} horizon={FORECAST_HORIZON}d blend={blend_cfg}")
    results = forecast_all()
    daily_bom, agg_bom = convert_forecast_to_bom(results)
    save_results(results, daily_bom, agg_bom)
    return results


if __name__ == "__main__":
    main()
