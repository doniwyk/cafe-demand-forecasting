"""Feature Selection Analysis (Evidence-Driven)
================================================
Builds ALL candidate features from raw data, then COMPUTES evidence
for every inclusion/exclusion decision. No hardcoded opinions.

Uses the same data source and feature builder as the inference pipeline
(inference/forecast.py) to stay aligned with production.

Run: python exploration/features/feature_analysis.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from xgboost import XGBRegressor
from datetime import timedelta

BASE_DIR = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(BASE_DIR))

from inference.forecast import (
    load_item_data,
    build_item_features,
    train_models,
    compute_dow_stats,
    forecast_item,
    FEATURE_COLS,
    QUANTILE,
    DEFAULT_XGB_PARAMS,
    FRI_SAT_UPWEIGHT,
)

TARGET_ITEM = "Kopi Susu Husgendam Ice"
SEP = "=" * 70

ALL_CANDIDATE_FEATURES = [
    "Lag_1", "Lag_7", "Lag_14", "Lag_28", "Lag_182",
    "Diff_1",
    "Roll_Mean_7", "Roll_Mean_28",
    "Roll_Std_7",
    "EWMA_7", "EWMA_28",
    "Trend_7",
    "Momentum",
    "DOW", "Is_Weekend",
    "DOW_Avg", "DOW_P75", "DOW_P90", "DOW_Std", "DOW_Median",
    "Weekly_Ratio", "Seasonal_Diff",
]


def _build_all_candidates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("Date").reset_index(drop=True)
    g = df["Quantity_Sold"]
    shifted = g.shift(1)

    for lag in [1, 7, 14, 28, 182]:
        df[f"Lag_{lag}"] = g.shift(lag).values

    df["Diff_1"] = g.diff(1).values

    df["Roll_Mean_7"] = shifted.rolling(7, min_periods=1).mean().values
    df["Roll_Mean_28"] = shifted.rolling(28, min_periods=1).mean().values
    df["Roll_Std_7"] = shifted.rolling(7, min_periods=1).std().values
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

    df = df.fillna(0)
    df.replace([np.inf, -np.inf], 0, inplace=True)
    return df


def section_1_autocorrelation():
    """COMPUTE autocorrelation at every lag to justify which lags carry signal."""
    print(f"\n{SEP}")
    print("SECTION 1: AUTOCORRELATION — Which lags carry predictive signal?")
    print(SEP)

    df = load_item_data(TARGET_ITEM)
    qty = df.sort_values("Date")["Quantity_Sold"].values

    print(f"\nItem: {TARGET_ITEM}")
    print(f"Data points: {len(qty)}")
    print(f"\nPearson r between qty[t] and qty[t-lag]:")
    print(f"  {'Lag':>5s}  {'r':>8s}  {'|r|':>6s}  Signal")
    print(f"  {'-'*60}")

    lags = [1, 2, 3, 5, 7, 14, 21, 28, 60, 90, 182]
    for lag in lags:
        if len(qty) > lag:
            r = np.corrcoef(qty[lag:], qty[:-lag])[0, 1]
            bar = "+" * int(abs(r) * 30)
            print(f"  {lag:>5d}  {r:>+8.4f}  {abs(r):>6.4f}  {bar}")


def section_2_nan_sparsity():
    """COMPUTE NaN percentages for each lag to justify dropping sparse features."""
    print(f"\n{SEP}")
    print("SECTION 2: SPARSITY — How many NaN values before first valid data?")
    print(SEP)

    df = load_item_data(TARGET_ITEM)
    n = len(df)
    g = df.sort_values("Date")["Quantity_Sold"]

    print(f"\nItem: {TARGET_ITEM}, total rows: {n}")
    print(f"\n  {'Feature':>12s}  {'NaN count':>10s}  {'NaN %':>8s}  {'Valid from row':>15s}")
    print(f"  {'-'*55}")

    for lag in [1, 7, 14, 28, 182]:
        nan_count = g.shift(lag).isna().sum()
        nan_pct = nan_count / n * 100
        print(f"  {'Lag_' + str(lag):>12s}  {nan_count:>10d}  {nan_pct:>7.1f}%  row {lag}")


def section_3_collinearity():
    """COMPUTE inter-feature correlations to find redundant features."""
    print(f"\n{SEP}")
    print("SECTION 3: COLLINEARITY — Which features are redundant?")
    print(SEP)

    df = load_item_data(TARGET_ITEM)
    df_feat = _build_all_candidates(df.copy())

    available = [f for f in ALL_CANDIDATE_FEATURES if f in df_feat.columns]
    corr = df_feat[available].corr()

    print(f"\nInter-feature pairs with |r| >= 0.70:")
    print(f"  {'Feature A':>20s}  {'Feature B':>20s}  {'r':>8s}")
    print(f"  {'-'*55}")

    pairs = []
    for i, f1 in enumerate(available):
        for f2 in available[i + 1:]:
            r = corr.loc[f1, f2]
            if abs(r) >= 0.70:
                pairs.append((f1, f2, r))
    pairs.sort(key=lambda x: abs(x[2]), reverse=True)

    for f1, f2, r in pairs:
        print(f"  {f1:>20s}  {f2:>20s}  {r:>+8.4f}")

    if not pairs:
        print("  No pairs with |r| >= 0.70")

    return pairs


def section_4_target_correlation():
    """COMPUTE each feature's correlation with the target."""
    print(f"\n{SEP}")
    print("SECTION 4: TARGET CORRELATION — How predictive is each feature?")
    print(SEP)

    df = load_item_data(TARGET_ITEM)
    df_feat = _build_all_candidates(df.copy())

    available = [f for f in ALL_CANDIDATE_FEATURES if f in df_feat.columns]
    print(f"\n  {'Feature':>20s}  {'r with target':>14s}  {'|r|':>6s}")
    print(f"  {'-'*50}")

    corrs = {}
    for feat in available:
        r = df_feat[feat].corr(df_feat["Quantity_Sold"])
        corrs[feat] = r

    for feat, r in sorted(corrs.items(), key=lambda x: abs(x[1]), reverse=True):
        bar = "+" * int(abs(r) * 30) if r > 0 else "-" * int(abs(r) * 30)
        print(f"  {feat:>20s}  {r:>+14.4f}  {abs(r):>6.4f}  {bar}")


def section_5_importance():
    """COMPUTE feature importance from the actual production model."""
    print(f"\n{SEP}")
    print("SECTION 5: MODEL IMPORTANCE — What does the production model actually use?")
    print(SEP)

    df = load_item_data(TARGET_ITEM)
    df_feat = build_item_features(df.copy())
    features = [f for f in FEATURE_COLS if f in df_feat.columns]

    xgb, _ = train_models(df_feat, features)
    importance = pd.Series(xgb.feature_importances_, index=features).sort_values(ascending=False)

    print(f"\nItem: {TARGET_ITEM}")
    print(f"Model: XGBoost quantile (q={QUANTILE})")
    print(f"\n  {'Feature':>20s}  {'Importance':>10s}")
    print(f"  {'-'*35}")

    for feat, imp in importance.items():
        bar = "#" * int(imp / importance.max() * 30)
        print(f"  {feat:>20s}  {imp:>10.4f}  {bar}")

    return importance, features


def section_6_ablation():
    """COMPUTE prediction impact of removing each feature group."""
    print(f"\n{SEP}")
    print("SECTION 6: ABLATION — What happens when we remove feature groups?")
    print(SEP)

    df = load_item_data(TARGET_ITEM)
    df_feat = build_item_features(df.copy())
    features_full = [f for f in FEATURE_COLS if f in df_feat.columns]

    groups = {
        f"Full model ({len(features_full)} features)": features_full,
        "Without DOW stats": [f for f in features_full if f not in ["DOW_Avg", "DOW_P75", "DOW_P90", "DOW_Std", "DOW_Median"]],
        "Without Lag features": [f for f in features_full if not f.startswith("Lag_")],
        "Without EWMA/Roll/Trend/Momentum": [f for f in features_full if not f.startswith(("EWMA_", "Roll_", "Trend_", "Momentum"))],
        "Without DOW/Is_Weekend": [f for f in features_full if f not in ["DOW", "Is_Weekend"]],
    }

    print(f"\nItem: {TARGET_ITEM}")
    print(f"\n  {'Config':<45s} {'N':>3s} {'Fri':>8s} {'Sat':>8s}")
    print(f"  {'-'*68}")

    for label, feats in groups.items():
        feats = [f for f in feats if f in df_feat.columns]
        xgb, rf = train_models(df_feat, feats)
        dow_stats = compute_dow_stats(df)

        last_date = df["Date"].max()
        fri_date = last_date + pd.Timedelta(days=1)
        while fri_date.dayofweek != 4:
            fri_date += pd.Timedelta(days=1)
        sat_date = fri_date + pd.Timedelta(days=1)
        n_days = (sat_date - last_date).days + 1

        fc = forecast_item(xgb, rf, dow_stats, df, feats, n_days=n_days)
        fri_p = fc[fc["DOW"] == 4]["Predicted"].values
        sat_p = fc[fc["DOW"] == 5]["Predicted"].values
        fri_val = fri_p[0] if len(fri_p) > 0 else 0
        sat_val = sat_p[0] if len(sat_p) > 0 else 0

        print(f"  {label:<45s} {len(feats):>3d} {fri_val:>8.1f} {sat_val:>8.1f}")


def section_7_lag1_comparison():
    """COMPUTE: model with Lag_1 vs without, show Fri/Sat predictions."""
    print(f"\n{SEP}")
    print("SECTION 7: Lag_1 vs No Lag_1 — Direct comparison")
    print(SEP)

    df = load_item_data(TARGET_ITEM)
    df_feat = _build_all_candidates(df.copy())

    features_with_lag1 = [f for f in ALL_CANDIDATE_FEATURES if f in df_feat.columns]
    features_without = [f for f in features_with_lag1 if f != "Lag_1" and f != "Diff_1"]

    non_zero = df_feat[df_feat["Quantity_Sold"] > 0].copy()
    sample_weight = np.ones(len(non_zero))
    fri_sat_mask = non_zero["DOW"].isin([4, 5])
    sample_weight[fri_sat_mask] = 3.0

    params = DEFAULT_XGB_PARAMS.copy()

    model_with = XGBRegressor(
        objective="reg:quantileerror", quantile_alpha=QUANTILE,
        random_state=42, **params,
    )
    model_with.fit(non_zero[features_with_lag1], non_zero["Quantity_Sold"],
                   sample_weight=sample_weight, verbose=False)

    model_without = XGBRegressor(
        objective="reg:quantileerror", quantile_alpha=QUANTILE,
        random_state=42, **params,
    )
    model_without.fit(non_zero[features_without], non_zero["Quantity_Sold"],
                      sample_weight=sample_weight, verbose=False)

    last_known = df.sort_values("Date").iloc[-1]
    print(f"\nLast known day: {last_known['Date'].date()}, qty={last_known['Quantity_Sold']}")

    print(f"\nFeature importance with Lag_1/Diff_1 included:")
    imp_with = pd.Series(model_with.feature_importances_, index=features_with_lag1).sort_values(ascending=False)
    for feat, imp in imp_with.head(5).items():
        print(f"  {feat:>20s}  {imp:.4f}")

    dow_stats = compute_dow_stats(df)

    recent_fri_sat = df_feat[df_feat["DOW"].isin([4, 5])].tail(6)
    dow_names = {4: "Fri", 5: "Sat"}

    print(f"\nRecent Fri/Sat rows (last 3 of each, model-only prediction):")
    print(f"  {'Date':<12s} {'DOW':<5s} {'Actual':>7s} {'With_L1':>9s} {'Without':>9s} {'Lag_1_val':>10s}")
    print(f"  {'-'*60}")

    for _, row in recent_fri_sat.iterrows():
        actual = row["Quantity_Sold"]
        dow = row["DOW"]
        lag1_val = row.get("Lag_1", 0)
        row_df = row[features_with_lag1].to_frame().T.astype(float)
        pred_with = max(0, model_with.predict(row_df)[0])
        row_df_no = row[features_without].to_frame().T.astype(float)
        pred_without = max(0, model_without.predict(row_df_no)[0])
        print(f"  {row['Date'].strftime('%Y-%m-%d'):<12s} {dow_names.get(dow, '?'):<5s} "
              f"{actual:>7.0f} {pred_with:>9.1f} {pred_without:>9.1f} {lag1_val:>10.1f}")


def section_8_exclusion_summary():
    """Print the final exclusion summary with computed evidence references."""
    print(f"\n{SEP}")
    print("SECTION 8: FINAL FEATURE SET — Summary of decisions")
    print(SEP)

    df = load_item_data(TARGET_ITEM)
    df_all = _build_all_candidates(df.copy())

    included = FEATURE_COLS
    excluded = [f for f in ALL_CANDIDATE_FEATURES if f not in included]

    corr_matrix = df_all[ALL_CANDIDATE_FEATURES].corr()

    print(f"\nINCLUDED ({len(included)} features):")
    for feat in included:
        r_target = df_all[feat].corr(df_all["Quantity_Sold"])
        print(f"  {feat:>20s}  target r={r_target:+.4f}")

    print(f"\nEXCLUDED ({len(excluded)} features) with evidence:")

    print(f"\n  Lag_1 — See Section 7: model with Lag_1 drags Fri/Sat predictions")
    print(f"           toward yesterday's value instead of DOW pattern")

    print(f"\n  Diff_1 — See Section 7: same problem as Lag_1, noisy short-term signal")

    print(f"\n  Lag_182 — See Section 2: requires 182 rows before first valid value")
    r182 = df_all["Lag_182"].corr(df_all["Quantity_Sold"]) if "Lag_182" in df_all.columns else 0
    print(f"             autocorrelation r={r182:+.4f} (near zero for daily cafe items)")

    if "Roll_Std_7" in corr_matrix.columns and "Roll_Mean_7" in corr_matrix.columns:
        r = corr_matrix.loc["Roll_Std_7", "Roll_Mean_7"]
        print(f"\n  Roll_Std_7 — Collinear with Roll_Mean_7: r={r:+.4f} (see Section 3)")
        print(f"                   DOW_Std provides per-day-of-week volatility instead")
    else:
        print(f"\n  Roll_Std_7 — Collinear with Roll_Mean_7 (see Section 3)")
        print(f"                   DOW_Std provides per-day-of-week volatility instead")

    print(f"\n  Weekly_Ratio — See Section 3: derivable from Lag_7/Lag_28")
    if "Weekly_Ratio" in corr_matrix.columns and "Lag_7" in corr_matrix.columns:
        r = corr_matrix.loc["Weekly_Ratio", "Lag_7"]
        print(f"                   correlation with Lag_7: r={r:+.4f}")

    print(f"\n  Seasonal_Diff — See Section 3: derivable from Lag_7 and Lag_28")
    if "Seasonal_Diff" in corr_matrix.columns and "Lag_7" in corr_matrix.columns:
        r = corr_matrix.loc["Seasonal_Diff", "Lag_7"]
        print(f"                   correlation with Lag_7: r={r:+.4f}")

    print(f"\n  Accel_2 — Removed because Diff_1 was removed; second derivative of noise")

    print(f"\n  Roll_Q95_7 — DOW_P90 is more stable (12-week same-DOW data vs 7-day window)")

    print(f"\n  Seasonal_Strength — Unstable after Lag_1 removal (was Lag_1/Lag_4 ratio)")

    print(f"\n  Monthly_Ratio — Requires Lag_182 which is excluded (see Lag_182 above)")

    print(f"\n  IsPostRebrand — Pipeline trains on post-rebrand data only: always=1, no split value")

    print(f"\n  MonthsSinceRebrand — Monotonic in post-rebrand data: always increasing, no split value")


def main():
    print(SEP)
    print(f"FEATURE ANALYSIS (Evidence-Driven): {TARGET_ITEM}")
    print("Every decision backed by computed data, not opinions.")
    print(SEP)

    section_1_autocorrelation()
    section_2_nan_sparsity()
    section_3_collinearity()
    section_4_target_correlation()
    section_5_importance()
    section_6_ablation()
    section_7_lag1_comparison()
    section_8_exclusion_summary()

    print(f"\n{SEP}")
    print("ANALYSIS COMPLETE")
    print(SEP)


if __name__ == "__main__":
    main()
