"""Backtesting validation for the blended forecasting approach.

Tests the forecast accuracy on historical data by:
1. Picking multiple test weeks (expanding window)
2. Training on data up to each test period
3. Forecasting the next 7 days
4. Comparing predictions vs actuals

Reports aggregate metrics by DOW and by item category.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from datetime import timedelta
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys_path = BASE_DIR.parent
import sys
sys.path.insert(0, str(sys_path))

from forecast import (
    load_all_items,
    build_item_features,
    compute_dow_stats,
    train_models,
    forecast_item,
    FEATURE_COLS,
    _dow_baseline,
    WEEKEND_BLEND_MODEL,
    WEEKDAY_BLEND_MODEL,
    QUANTILE,
    DOW_LOOKBACK_WEEKS,
    MIN_NONZERO_DAYS,
    _should_skip,
)

DOW_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

TEST_PERIODS = [
    ("2026-03-15", "2026-03-21"),
    ("2026-04-05", "2026-04-11"),
    ("2026-04-19", "2026-04-25"),
    ("2026-05-03", "2026-05-09"),
    ("2026-05-17", "2026-05-23"),
]


def backtest_item(
    item_name: str,
    df_item: pd.DataFrame,
    test_start: str,
    test_end: str,
) -> pd.DataFrame | None:
    test_start_ts = pd.Timestamp(test_start)
    test_end_ts = pd.Timestamp(test_end)

    train_df = df_item[df_item["Date"] < test_start_ts].copy()
    test_df = df_item[(df_item["Date"] >= test_start_ts) & (df_item["Date"] <= test_end_ts)].copy()

    nonzero_train = (train_df["Quantity_Sold"] > 0).sum()
    if nonzero_train < MIN_NONZERO_DAYS or len(test_df) == 0:
        return None

    dow_stats = compute_dow_stats(train_df)
    df_feat = build_item_features(train_df.copy())
    features = [f for f in FEATURE_COLS if f in df_feat.columns]

    try:
        xgb, rf = train_models(df_feat, features)
    except Exception:
        return None

    n_days = (test_end_ts - train_df["Date"].max()).days
    if n_days <= 0 or n_days > 14:
        return None

    forecast = forecast_item(xgb, rf, dow_stats, train_df, features, n_days=n_days)

    forecast = forecast[forecast["Date"].isin(test_df["Date"])]
    test_df = test_df[test_df["Date"].isin(forecast["Date"])]

    if len(forecast) == 0:
        return None

    merged = forecast[["Date", "DOW", "DOW_Name", "Predicted"]].merge(
        test_df[["Date", "Quantity_Sold"]].rename(columns={"Quantity_Sold": "Actual"}),
        on="Date",
    )

    merged["Error"] = merged["Predicted"] - merged["Actual"]
    merged["Abs_Error"] = merged["Error"].abs()
    merged["Pct_Error"] = np.where(
        merged["Actual"] > 0,
        merged["Abs_Error"] / merged["Actual"] * 100,
        0,
    )
    merged["Item"] = item_name
    merged["Period"] = f"{test_start} to {test_end}"

    return merged


def run_backtest(
    items_to_test: list[str] | None = None,
    test_periods: list[tuple[str, str]] | None = None,
) -> pd.DataFrame:
    df_all = load_all_items()
    periods = test_periods or TEST_PERIODS

    if items_to_test:
        items = items_to_test
    else:
        item_counts = df_all.groupby("Item")["Quantity_Sold"].apply(
            lambda x: (x > 0).sum()
        )
        items = item_counts[item_counts >= MIN_NONZERO_DAYS].index.tolist()
        items = [i for i in items if not _should_skip(i)]

    print(f"\nBacktesting {len(items)} items across {len(periods)} periods")
    print(f"Periods: {periods}")
    print()

    all_results = []
    for p_idx, (test_start, test_end) in enumerate(periods):
        print(f"Period {p_idx + 1}/{len(periods)}: {test_start} to {test_end}")
        period_results = []

        for item in items:
            df_item = df_all[df_all["Item"] == item].copy()
            result = backtest_item(item, df_item, test_start, test_end)
            if result is not None:
                period_results.append(result)

        if period_results:
            period_df = pd.concat(period_results, ignore_index=True)
            n_items = period_df["Item"].nunique()
            mae = period_df["Abs_Error"].mean()
            mape = period_df[period_df["Actual"] > 0]["Pct_Error"].mean()
            print(f"  {n_items} items | MAE: {mae:.2f} | MAPE: {mape:.1f}%")
            all_results.append(period_df)

    return pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()


def print_backtest_report(results: pd.DataFrame):
    print("\n" + "=" * 80)
    print("BACKTEST REPORT")
    print("=" * 80)

    if results.empty:
        print("No results to report.")
        return

    overall_mae = results["Abs_Error"].mean()
    overall_rmse = np.sqrt((results["Error"] ** 2).mean())
    nonzero = results[results["Actual"] > 0]
    overall_mape = nonzero["Pct_Error"].mean()

    print(f"\nOverall: {len(results)} predictions across {results['Item'].nunique()} items")
    print(f"  MAE:  {overall_mae:.2f}")
    print(f"  RMSE: {overall_rmse:.2f}")
    print(f"  MAPE: {overall_mape:.1f}%")

    print(f"\n--- By Day of Week ---")
    print(f"  {'DOW':<10} {'MAE':>7} {'RMSE':>7} {'MAPE':>8} {'N':>6}")
    print(f"  {'-'*42}")

    for dow in range(7):
        dow_data = results[results["DOW"] == dow]
        if len(dow_data) == 0:
            continue
        mae = dow_data["Abs_Error"].mean()
        rmse = np.sqrt((dow_data["Error"] ** 2).mean())
        nz = dow_data[dow_data["Actual"] > 0]
        mape = nz["Pct_Error"].mean() if len(nz) > 0 else 0
        print(f"  {DOW_NAMES[dow]:<10} {mae:>7.2f} {rmse:>7.2f} {mape:>7.1f}% {len(dow_data):>6}")

    fri_sat = results[results["DOW"].isin([4, 5])]
    if len(fri_sat) > 0:
        fri_sat_mae = fri_sat["Abs_Error"].mean()
        fri_sat_rmse = np.sqrt((fri_sat["Error"] ** 2).mean())
        nz_fs = fri_sat[fri_sat["Actual"] > 0]
        fri_sat_mape = nz_fs["Pct_Error"].mean() if len(nz_fs) > 0 else 0
        print(f"  {'Fri+Sat':<10} {fri_sat_mae:>7.2f} {fri_sat_rmse:>7.2f} {fri_sat_mape:>7.1f}% {len(fri_sat):>6}")

    print(f"\n--- By Period ---")
    print(f"  {'Period':<25} {'MAE':>7} {'RMSE':>7} {'MAPE':>8} {'N':>6}")
    print(f"  {'-'*55}")

    for period in results["Period"].unique():
        p_data = results[results["Period"] == period]
        mae = p_data["Abs_Error"].mean()
        rmse = np.sqrt((p_data["Error"] ** 2).mean())
        nz = p_data[p_data["Actual"] > 0]
        mape = nz["Pct_Error"].mean() if len(nz) > 0 else 0
        print(f"  {period:<25} {mae:>7.2f} {rmse:>7.2f} {mape:>7.1f}% {len(p_data):>6}")

    print(f"\n--- Top 15 Items by Volume (MAPE) ---")
    print(f"  {'Item':<35} {'Avg/Day':>8} {'MAE':>7} {'MAPE':>8} {'N':>4}")
    print(f"  {'-'*65}")

    item_stats = []
    for item in results["Item"].unique():
        i_data = results[results["Item"] == item]
        avg_qty = i_data["Actual"].mean()
        mae = i_data["Abs_Error"].mean()
        nz = i_data[i_data["Actual"] > 0]
        mape = nz["Pct_Error"].mean() if len(nz) > 0 else 0
        item_stats.append((item, avg_qty, mae, mape, len(i_data)))

    item_stats.sort(key=lambda x: x[1], reverse=True)
    for item, avg_qty, mae, mape, n in item_stats[:15]:
        print(f"  {item:<35} {avg_qty:>8.1f} {mae:>7.2f} {mape:>7.1f}% {n:>4}")

    print(f"\n--- Accuracy Buckets ---")
    within_20 = (nonzero["Pct_Error"] <= 20).mean() * 100
    within_50 = (nonzero["Pct_Error"] <= 50).mean() * 100
    within_100 = (nonzero["Pct_Error"] <= 100).mean() * 100
    print(f"  Within 20%:  {within_20:.1f}%")
    print(f"  Within 50%:  {within_50:.1f}%")
    print(f"  Within 100%: {within_100:.1f}%")

    bias = results["Error"].mean()
    print(f"\n  Mean Bias (Pred - Actual): {bias:+.2f} ({'overpredict' if bias > 0 else 'underpredict'})")


def main():
    print("=" * 80)
    print("BACKTESTING: Blended Forecast Approach")
    print(f"Quantile: {QUANTILE} | DOW lookback: {DOW_LOOKBACK_WEEKS}w")
    print(f"Blend: Fri/Sat={WEEKEND_BLEND_MODEL:.0%} model, Weekdays={WEEKDAY_BLEND_MODEL:.0%} model")
    print("=" * 80)

    results = run_backtest()
    print_backtest_report(results)

    output_dir = Path(__file__).resolve().parent.parent / "models" / "exploration" / "inference"
    output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_dir / "backtest_results.csv", index=False)
    print(f"\nSaved to {output_dir / 'backtest_results.csv'}")

    return results


if __name__ == "__main__":
    main()
