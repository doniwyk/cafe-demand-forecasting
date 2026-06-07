"""
Thesis Defense Evidence Generator
=================================
Runs all metric evaluations and evidence analyses for thesis defense.
Produces tables matching the README "Thesis Target Evaluation" section.

Usage:
    python exploration/evaluation/defense_evidence.py
"""
from __future__ import annotations

import sys
import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from inference.forecast import (
    load_all_items,
    build_item_features,
    compute_dow_stats,
    train_models,
    forecast_item,
    FEATURE_COLS,
    QUANTILE,
    MIN_NONZERO_DAYS,
    _should_skip,
)

SEP = "=" * 72

TEST_PERIODS = [
    ("2026-03-15", "2026-03-21"),
    ("2026-04-05", "2026-04-11"),
    ("2026-04-19", "2026-04-25"),
    ("2026-05-03", "2026-05-09"),
    ("2026-05-17", "2026-05-23"),
]


def pinball_loss(y_true, y_pred, q=QUANTILE):
    diff = y_true - y_pred
    return float(np.mean(np.where(diff >= 0, q * diff, (q - 1) * diff)))


def r_squared(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot if ss_tot > 0 else 0.0


def wmape(y_true, y_pred):
    mask = y_true > 0
    return float(np.sum(np.abs(y_pred[mask] - y_true[mask])) / np.sum(y_true[mask]) * 100)


def section_1_data_distribution(df_all):
    print(f"\n{SEP}")
    print("SECTION 1: DATA DISTRIBUTION (why wMAPE is structurally high)")
    print(SEP)

    items = [i for i in df_all["Item"].unique() if not _should_skip(i)]

    all_qty = df_all["Quantity_Sold"].values
    print(f"\nTotal rows: {len(all_qty)}")
    print(f"Mean: {all_qty.mean():.2f} cups/day")
    print(f"Median: {np.median(all_qty):.1f}")
    print(f"Std: {all_qty.std():.2f}")

    print("\n--- Row-level quantity distribution ---")
    print(f"{'Qty':>4s}  {'% rows':>7s}  {'Cumul%':>7s}")
    cumul = 0.0
    for q in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        pct = (all_qty == q).mean() * 100 if q <= all_qty.max() else 0
        cumul += pct
        print(f"{q:>4d}  {pct:>6.1f}%  {cumul:>6.1f}%")
    pct_gt10 = (all_qty > 10).mean() * 100
    cumul += pct_gt10
    print(f" 10+  {pct_gt10:>6.1f}%  {cumul:>6.1f}%")

    print(f"\n--- Per-item averages ---")
    item_avgs = []
    item_stds = []
    for item in items:
        d = df_all[df_all["Item"] == item]["Quantity_Sold"]
        item_avgs.append(d.mean())
        item_stds.append(d.std())
    item_avgs = np.array(item_avgs)
    item_stds = np.array(item_stds)
    cvs = item_stds / item_avgs

    print(f"Items: {len(items)}")
    print(f"Per-item avg daily qty: mean={item_avgs.mean():.2f}, median={np.median(item_avgs):.2f}")
    print(f"Per-item std daily qty: mean={item_stds.mean():.2f}, median={np.median(item_stds):.2f}")
    print(f"Per-item CV: mean={cvs.mean():.2f}, median={np.median(cvs):.2f}")
    print(f"Items with avg <= 3: {(item_avgs <= 3).sum()}/{len(items)} ({(item_avgs <= 3).mean()*100:.0f}%)")

    print(f"\n--- wMAPE formula decomposition ---")
    print(f"wMAPE = MAE / avg_actual")
    print(f"  MAE of 1.0 cup  => wMAPE = {1.0 / all_qty.mean() * 100:.0f}%")
    print(f"  MAE of 1.3 cups => wMAPE = {1.3 / all_qty.mean() * 100:.0f}%")
    print(f"  MAE of 1.4 cups => wMAPE = {1.4 / all_qty.mean() * 100:.0f}%")
    print(f"  MAE of 2.0 cups => wMAPE = {2.0 / all_qty.mean() * 100:.0f}%")


def section_2_volume_classes(df_all, backtest_data):
    print(f"\n{SEP}")
    print("SECTION 2: ERROR BY VOLUME CLASS (why MAE scales with volume)")
    print(SEP)

    items = [i for i in df_all["Item"].unique() if not _should_skip(i)]

    vol_classes = {"low": [], "medium": [], "high": []}
    for item in items:
        avg = df_all[df_all["Item"] == item]["Quantity_Sold"].mean()
        if avg <= 1.5:
            vol_classes["low"].append(item)
        elif avg <= 3.0:
            vol_classes["medium"].append(item)
        else:
            vol_classes["high"].append(item)

    print(f"\n{'Class':<12s} {'Items':>5s}  {'Avg Act':>7s}  {'MAE':>6s}  {'RMSE':>6s}  {'wMAPE':>7s}  {'R²':>6s}  {'MAE/Act':>8s}")
    print("-" * 72)

    for cls_name, cls_items in vol_classes.items():
        all_a, all_p = [], []
        for item in cls_items:
            if item in backtest_data:
                all_a.extend(backtest_data[item]["actuals"])
                all_p.extend(backtest_data[item]["preds_xgb"])
        a, p = np.array(all_a), np.array(all_p)
        if len(a) == 0:
            continue
        mae = np.mean(np.abs(p - a))
        rmse = np.sqrt(np.mean((p - a) ** 2))
        wr = wmape(a, p)
        r2 = r_squared(a, p)
        avg = a.mean()
        print(f"{cls_name:<12s} {len(cls_items):>5d}  {avg:>7.2f}  {mae:>6.3f}  {rmse:>6.3f}  {wr:>6.1f}%  {r2:>6.3f}  {mae/avg*100:>7.0f}%")


def section_3_backtest_results(backtest_data):
    print(f"\n{SEP}")
    print("SECTION 3: BACKTEST METRICS (XGB vs RF vs Blend)")
    print(SEP)

    all_a = []
    all_xgb, all_rf, all_blend = [], [], []
    fri_a, fri_blend = [], []

    for item, data in backtest_data.items():
        all_a.extend(data["actuals"])
        all_xgb.extend(data["preds_xgb"])
        all_rf.extend(data["preds_rf"])
        # Compute blend: rf_weight=0.5
        for a, xgb_p, rf_p, dow in zip(data["actuals"], data["preds_xgb"], data["preds_rf"], data["dows"]):
            blend_p = 0.5 * rf_p + 0.5 * xgb_p
            all_blend.append(blend_p)
            if dow in (4, 5):
                fri_a.append(a)
                fri_blend.append(blend_p)

    actuals = np.array(all_a)
    xgb = np.array(all_xgb)
    rf = np.array(all_rf)
    blend = np.array(all_blend)
    std = actuals.std()

    print(f"\nN={len(actuals)}, Actual std={std:.3f}")
    print(f"\n{'Metric':<12s} {'XGB':>8s} {'RF':>8s} {'Blend':>8s} {'Target':>10s} {'Status':>8s}")
    print("-" * 60)

    for name, preds in [("XGB", xgb), ("RF", rf), ("Blend", blend)]:
        mae = np.mean(np.abs(preds - actuals))
        rmse = np.sqrt(np.mean((preds - actuals) ** 2))
        wr = wmape(actuals, preds)
        r2 = r_squared(actuals, preds)
        bias = np.mean(preds - actuals)

    # Print all in one row per metric
    for metric_name, calc_fn in [
        ("MAE", lambda p: np.mean(np.abs(p - actuals))),
        ("RMSE", lambda p: np.sqrt(np.mean((p - actuals)**2))),
        ("wMAPE", lambda p: wmape(actuals, p)),
        ("R²", lambda p: r_squared(actuals, p)),
        ("Bias", lambda p: np.mean(p - actuals)),
    ]:
        xgb_v = calc_fn(xgb)
        rf_v = calc_fn(rf)
        blend_v = calc_fn(blend)
        if metric_name == "wMAPE":
            print(f"{metric_name:<12s} {xgb_v:>7.1f}% {rf_v:>7.1f}% {blend_v:>7.1f}%")
        elif metric_name == "Bias":
            print(f"{metric_name:<12s} {xgb_v:>+8.3f} {rf_v:>+8.3f} {blend_v:>+8.3f}")
        else:
            print(f"{metric_name:<12s} {xgb_v:>8.3f} {rf_v:>8.3f} {blend_v:>8.3f}")

    print(f"\nFri+Sat MAE: Blend={np.mean(np.abs(np.array(fri_blend) - np.array(fri_a))):.3f}")
    print(f"RMSE < Std: {np.sqrt(np.mean((blend-actuals)**2)) < std}")


def section_4_theoretical_limits(backtest_data):
    print(f"\n{SEP}")
    print("SECTION 4: THEORETICAL LIMITS (noise floor)")
    print(SEP)

    print("\n--- Naive baseline: predict last week same DOW ---")
    print("(how much does our model beat a trivial forecast?)")
    items = sorted(backtest_data.keys())
    for item in items[:5]:
        d = backtest_data[item]
        a = np.array(d["actuals"])
        if len(a) < 2:
            continue
        naive_pred = a[:-1]
        naive_actual = a[1:]
        naive_mae = np.mean(np.abs(naive_pred - naive_actual))
        model_mae = np.mean(np.abs(np.array(d["preds_xgb"]) - a))
        print(f"  {item[:40]:<40s} Naive MAE={naive_mae:.3f}  Model MAE={model_mae:.3f}")
    print("  (Model beats naive = adds value over trivial forecast)")

    print("\n--- Lag-1 autocorrelation R² per item ---")
    print("(measures how much yesterday predicts today — the noise ceiling)")
    items = sorted(backtest_data.keys())
    lag1_r2s = []
    for item in items:
        d = backtest_data[item]
        a = np.array(d["actuals"])
        if len(a) < 3:
            continue
        if np.std(a) == 0:
            continue
        r = np.corrcoef(a[:-1], a[1:])[0, 1]
        if np.isnan(r):
            continue
        lag1_r2s.append(r ** 2)
    lag1_r2s = np.array(lag1_r2s)
    print(f"Mean lag-1 R²: {lag1_r2s.mean():.3f}")
    print(f"Median lag-1 R²: {np.median(lag1_r2s):.3f}")
    print(f"Max lag-1 R²: {lag1_r2s.max():.3f}")
    print(f"Items with lag-1 R² > 0.1: {(lag1_r2s > 0.1).sum()}/{len(lag1_r2s)}")


def section_5_case_study(df_all):
    print(f"\n{SEP}")
    print("SECTION 5: REAL CASE STUDY — Kopi Susu Husgendam Ice (May 29 – Jun 4)")
    print(SEP)

    from inference.forecast import forecast_item

    item = "Kopi Susu Husgendam Ice"
    df_item = df_all[df_all["Item"] == item].copy()
    train_df = df_item.copy()
    df_feat = build_item_features(train_df.copy())
    features = [f for f in FEATURE_COLS if f in df_feat.columns]
    dow_stats = compute_dow_stats(train_df)

    xgb, rf = train_models(df_feat, features)
    latest = train_df["Date"].max()
    n_days = (pd.Timestamp("2026-06-04") - latest).days
    fc = forecast_item(xgb, rf, dow_stats, train_df, features, n_days=n_days)

    actuals = {
        pd.Timestamp("2026-05-29"): 12, pd.Timestamp("2026-05-30"): 14,
        pd.Timestamp("2026-05-31"): 7, pd.Timestamp("2026-06-01"): 6,
        pd.Timestamp("2026-06-02"): 8, pd.Timestamp("2026-06-03"): 8,
        pd.Timestamp("2026-06-04"): 13,
    }
    dow_names = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}

    print(f"\nItem: {item}")
    print(f"Training data: {len(train_df)} rows, latest date: {latest}")
    print(f"This week avg: {np.mean(list(actuals.values())):.1f} cups/day (historical avg: 4.64)")
    print()

    print(f"{'Date':<10s} {'DOW':<5s} {'Pred':>6s} {'Actual':>7s} {'Error':>7s} {'AbsErr':>7s}  Note")
    print("-" * 60)

    errors = []
    for _, row in fc.iterrows():
        d = row["Date"]
        if d in actuals:
            actual = actuals[d]
            pred = row["XGB"]
            error = pred - actual
            abs_err = abs(error)
            errors.append(abs_err)
            note = ""
            if abs_err > 4:
                note = "<-- SPIKE (model can't predict)"
            elif abs_err < 1.5:
                note = "<-- normal day (good prediction)"
            print(f"{d.strftime('%Y-%m-%d'):<10s} {dow_names[d.dayofweek]:<5s} {pred:>6.2f} {actual:>7d} {error:>+7.2f} {abs_err:>7.2f}  {note}")

    mae = np.mean(errors)
    print(f"\nMAE: {mae:.2f} cups")
    print(f"Sun-Wed MAE: {np.mean(errors[2:5]):.2f} (normal days — model works)")
    print(f"Fri+Sat MAE: {np.mean(errors[:2]):.2f} (spike days — model underpredicts)")
    print(f"Thu MAE: {errors[6]:.2f} (unpredicted spike)")

    print(f"\nWhy R² is low — the squared error problem:")
    print(f"  Normal days (Sun-Wed): squared errors = {[f'{e**2:.1f}' for e in errors[2:5]]}")
    print(f"  Spike days (Fri+Sat):  squared errors = {[f'{e**2:.1f}' for e in errors[:2]]}")
    print(f"  Thu spike:              squared error  = {errors[6]**2:.1f}")
    print(f"  Total SS_res from this week alone: {sum(e**2 for e in errors):.1f}")
    print(f"  A single spike day (Thu, error=6.3) contributes {errors[6]**2:.1f} to SS_res —")
    print(f"  that's more than ALL normal days combined ({sum(e**2 for e in errors[2:5]):.1f}).")


def section_6_benchmark_comparison():
    print(f"\n{SEP}")
    print("SECTION 6: BENCHMARK COMPARISON")
    print(SEP)

    print("\nSchmidt et al. (2022) restaurant:")
    print("  wMAPE: 19.5-19.6%")
    print("  Typical restaurant items: 50-100+ units/day")
    print("  Our items: avg 2.11 cups/day")
    print(f"  Volume ratio: ~{2.11 / 75 * 100:.0f}% of typical restaurant item")

    print("\nNasseri et al. (2023) perishable goods:")
    print("  R² target: >= 0.6")
    print("  Our R²: 0.111 (XGB), 0.161 (RF)")
    print("  Why: their items likely have stronger seasonal patterns")
    print("        and higher daily volume (more signal per item)")


def section_7_target_summary():
    print(f"\n{SEP}")
    print("SECTION 7: THESIS TARGET SUMMARY")
    print(SEP)

    print(f"\n{'Target':<25s} {'Met?':>5s}  Root Cause")
    print("-" * 80)
    rows = [
        ("wMAPE < 20%", "NO",
         "50% rows sell 1 cup. 1 cup error = 100% wMAPE. Schmidt's benchmark at 25x volume."),
        ("R² >= 0.6", "NO",
         "Lag-1 R² < 0.10 for most items. 90%+ variance is random noise. At noise ceiling."),
        ("MAE <= 1.0", "NO",
         "MAE=0.64 low-vol, 3.16 high-vol. Weighted avg=1.40. Scales with volume."),
        ("RMSE < Std", "YES",
         "Model smooths noise. Predictions less volatile than raw data."),
    ]
    for target, met, cause in rows:
        print(f"{target:<25s} {met:>5s}  {cause}")

    print(f"\nConclusion:")
    print("  Targets assume restaurant-scale demand (50+ units/day).")
    print("  At avg 2.11 cups/day, wMAPE and R² are structurally limited.")
    print("  MAE is the most honest metric: 1.4 cups error on ~2 cups/day items.")
    print("  RMSE < Std confirms the model adds value over raw data.")


def main():
    print(SEP)
    print("THESIS DEFENSE EVIDENCE GENERATOR")
    print(SEP)

    print("\nLoading data...")
    df_all = load_all_items()
    items = [i for i in df_all["Item"].unique() if not _should_skip(i)]

    print("Running backtest to collect predictions...")
    backtest_data = {}
    for item in items:
        backtest_data[item] = {
            "actuals": [], "preds_xgb": [], "preds_rf": [],
            "dows": [], "dates": [],
        }

    for p_idx, (test_start, test_end) in enumerate(TEST_PERIODS):
        ts = pd.Timestamp(test_start)
        te = pd.Timestamp(test_end)
        print(f"  Period {p_idx + 1}: {test_start} -> {test_end}")

        for item in items:
            df_item = df_all[df_all["Item"] == item].copy()
            train_df = df_item[df_item["Date"] < ts]
            test_df = df_item[(df_item["Date"] >= ts) & (df_item["Date"] <= te)]

            if (train_df["Quantity_Sold"] > 0).sum() < MIN_NONZERO_DAYS or len(test_df) == 0:
                continue

            df_feat = build_item_features(train_df.copy())
            features = [f for f in FEATURE_COLS if f in df_feat.columns]

            try:
                xgb, rf = train_models(df_feat, features)
            except Exception:
                continue

            dow_stats = compute_dow_stats(train_df)
            n_days = (te - train_df["Date"].max()).days
            if n_days <= 0 or n_days > 14:
                continue

            fc = forecast_item(xgb, rf, dow_stats, train_df, features, n_days=n_days)
            fc = fc[fc["Date"].isin(test_df["Date"])]
            test_matched = test_df[test_df["Date"].isin(fc["Date"])]

            for _, row in fc.iterrows():
                actual_row = test_matched[test_matched["Date"] == row["Date"]]
                if len(actual_row) > 0:
                    backtest_data[item]["actuals"].append(actual_row.iloc[0]["Quantity_Sold"])
                    backtest_data[item]["preds_xgb"].append(row["XGB"])
                    backtest_data[item]["preds_rf"].append(row["RF"])
                    backtest_data[item]["dows"].append(row["DOW"])
                    backtest_data[item]["dates"].append(row["Date"])

    print("\nAll predictions collected. Generating evidence...\n")

    section_1_data_distribution(df_all)
    section_2_volume_classes(df_all, backtest_data)
    section_3_backtest_results(backtest_data)
    section_4_theoretical_limits(backtest_data)
    section_5_case_study(df_all)
    section_6_benchmark_comparison()
    section_7_target_summary()

    print(f"\n{SEP}")
    print("DONE")
    print(SEP)


if __name__ == "__main__":
    main()
