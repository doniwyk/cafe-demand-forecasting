"""Feature Selection Analysis (Evidence-Driven) — Multi-Item Statistical Edition
================================================================================
Builds ALL candidate features from raw data, then COMPUTES evidence
for every inclusion/exclusion decision using model-free statistics
across ALL items (not just Husgendam Ice).

Sections 1-4: Statistical analysis (autocorrelation, sparsity, collinearity, target corr)
Sections 5:   Mutual Information across all items (replaces XGBoost importance)
Section 6:    Group-level information contribution (replaces model ablation)
Section 7:    Conditional dependency analysis — Lag_1 persistence vs DOW cycle
Section 8:    Exclusion summary

Figures are saved to figures/feature_discovery/.
Per-item MI table is saved to CSV for document reference.

Run: conda run -n cafe python exploration/features/feature_analysis.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from xgboost import XGBRegressor
from datetime import timedelta

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_regression

BASE_DIR = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(BASE_DIR))

from inference.forecast import (
    load_item_data,
    load_all_items,
    build_item_features,
    compute_dow_stats,
    FEATURE_COLS,
    QUANTILE,
    DEFAULT_XGB_PARAMS,
    FRI_SAT_UPWEIGHT,
    _should_skip,
)

TARGET_ITEM = "Kopi Susu Husgendam Ice"
SEP = "=" * 70
POST_REBRAND = "2025-05-01"
MIN_NONZERO = 60
TOP_N = 58  # all non-skip items with enough data

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

FEATURE_GROUPS = {
    "Lag": ["Lag_1", "Lag_7", "Lag_14", "Lag_28", "Lag_182"],
    "Diff": ["Diff_1"],
    "Rolling/EWMA": ["Roll_Mean_7", "Roll_Mean_28", "Roll_Std_7", "EWMA_7", "EWMA_28"],
    "Trend/Momentum": ["Trend_7", "Momentum"],
    "DOW Identity": ["DOW", "Is_Weekend"],
    "DOW Stats": ["DOW_Avg", "DOW_P75", "DOW_P90", "DOW_Std", "DOW_Median"],
    "Weekly Ratio": ["Weekly_Ratio", "Seasonal_Diff"],
}

FIG_DIR = BASE_DIR / "figures" / "feature_discovery"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR = BASE_DIR / "tables"
TABLE_DIR.mkdir(parents=True, exist_ok=True)


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


def _post_rebrand_nonzero(item_name: str, df_all: pd.DataFrame) -> pd.DataFrame | None:
    df = load_item_data(item_name, df_all)
    if df is None or len(df) == 0:
        return None
    df = df[df["Date"] >= POST_REBRAND].copy()
    non_zero = df[df["Quantity_Sold"] > 0].copy()
    if len(non_zero) < MIN_NONZERO:
        return None
    return non_zero


def _collect_items(df_all: pd.DataFrame) -> list[str]:
    all_items = sorted(df_all[~df_all["Item"].apply(_should_skip)]["Item"].unique())
    valid = []
    for item in all_items:
        df = _post_rebrand_nonzero(item, df_all)
        if df is not None:
            valid.append(item)
    print(f"  Collected {len(valid)}/{len(all_items)} valid items (>= {MIN_NONZERO} non-zero post-rebrand days)")
    return valid


# ─────────────────────────────────────────────
# Section 1 — Autocorrelation
# ─────────────────────────────────────────────
def section_1_autocorrelation(df_all: pd.DataFrame, items: list[str]):
    print(f"\n{SEP}")
    print("SECTION 1: AUTOCORRELATION (aggregated across items)")
    print(SEP)

    lags = [1, 2, 3, 5, 7, 14, 21, 28, 60, 90, 182]

    all_corrs: dict[int, list[float]] = {lag: [] for lag in lags}
    for item in items:
        df = _post_rebrand_nonzero(item, df_all)
        if df is None:
            continue
        qty = df.sort_values("Date")["Quantity_Sold"].values
        for lag in lags:
            if len(qty) > lag:
                r = np.corrcoef(qty[lag:], qty[:-lag])[0, 1]
                all_corrs[lag].append(r)

    print(f"\n  {'Lag':>5s}  {'Mean r':>8s}  {'Std r':>8s}  {'N items':>8s}")
    print(f"  {'-'*35}")
    for lag in lags:
        vals = all_corrs[lag]
        if vals:
            print(f"  {lag:>5d}  {np.mean(vals):>+8.4f}  {np.std(vals):>8.4f}  {len(vals):>8d}")

    # Generate figure
    means = [np.mean(all_corrs[lag]) if all_corrs[lag] else 0 for lag in lags]
    stds = [np.std(all_corrs[lag]) if all_corrs[lag] else 0 for lag in lags]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.errorbar(lags, means, yerr=stds, fmt="o-", capsize=4, color="#2c3e50")
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.5)
    ax.set_xlabel("Lag (days)")
    ax.set_ylabel("Pearson r")
    ax.set_title("Autocorrelation (mean ± std across items)")
    ax.set_xticks(lags)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "07_autocorrelation.png", dpi=150)
    plt.close(fig)
    print(f"\n  Saved: figures/feature_discovery/07_autocorrelation.png")


# ─────────────────────────────────────────────
# Section 2 — Sparsity
# ─────────────────────────────────────────────
def section_2_nan_sparsity(df_all: pd.DataFrame, items: list[str]):
    print(f"\n{SEP}")
    print("SECTION 2: SPARSITY (aggregated across items)")
    print(SEP)

    lags = [1, 7, 14, 28, 182]
    all_pcts: dict[int, list[float]] = {lag: [] for lag in lags}
    for item in items:
        df = _post_rebrand_nonzero(item, df_all)
        if df is None:
            continue
        g = df.sort_values("Date")["Quantity_Sold"]
        for lag in lags:
            nan_pct = g.shift(lag).isna().sum() / len(g) * 100
            all_pcts[lag].append(nan_pct)

    print(f"\n  {'Feature':>12s}  {'Mean NaN %':>10s}  {'Std NaN %':>10s}  {'Min':>8s}  {'Max':>8s}")
    print(f"  {'-'*55}")
    for lag in lags:
        vals = all_pcts[lag]
        if vals:
            print(f"  {'Lag_' + str(lag):>12s}  {np.mean(vals):>10.2f}  {np.std(vals):>10.2f}  {np.min(vals):>8.2f}  {np.max(vals):>8.2f}")


# ─────────────────────────────────────────────
# Section 3 — Collinearity
# ─────────────────────────────────────────────
def section_3_collinearity(df_all: pd.DataFrame, items: list[str]):
    print(f"\n{SEP}")
    print("SECTION 3: COLLINEARITY (aggregated across items)")
    print(SEP)

    all_corr_mats: list[pd.DataFrame] = []
    all_pairs: dict[tuple[str, str], list[float]] = {}
    for item in items:
        df = _post_rebrand_nonzero(item, df_all)
        if df is None:
            continue
        df_feat = _build_all_candidates(df.copy())
        available = [f for f in ALL_CANDIDATE_FEATURES if f in df_feat.columns]
        corr = df_feat[available].corr()
        all_corr_mats.append(corr)
        for i, f1 in enumerate(available):
            for f2 in available[i + 1:]:
                r = corr.loc[f1, f2]
                if abs(r) >= 0.70:
                    all_pairs.setdefault((f1, f2), []).append(r)

    # Generate mean correlation heatmap across all items
    mean_corr = sum(all_corr_mats) / len(all_corr_mats)
    fig, ax = plt.subplots(figsize=(14, 11))
    sns.heatmap(mean_corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                vmin=-1, vmax=1, linewidths=0.5, ax=ax,
                cbar_kws={"label": "Mean Pearson r"})
    ax.set_title(f"Mean Inter-Feature Correlation (average of {len(items)} items)", fontsize=13)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "09_feature_correlation.png", dpi=150)
    plt.close(fig)
    print(f"\n  Saved: figures/feature_discovery/09_feature_correlation.png (mean of {len(items)} items)")

    print(f"\n  Pairs with |r| >= 0.70 in at least one item:")
    print(f"  {'Feature A':>20s}  {'Feature B':>20s}  {'Mean r':>8s}  {'Std r':>8s}  {'Freq':>6s}")
    print(f"  {'-'*70}")
    sorted_pairs = sorted(all_pairs.items(), key=lambda x: abs(np.mean(x[1])), reverse=True)
    for (f1, f2), vals in sorted_pairs:
        print(f"  {f1:>20s}  {f2:>20s}  {np.mean(vals):>+8.4f}  {np.std(vals):>8.4f}  {len(vals):>6d}")

    return all_pairs


# ─────────────────────────────────────────────
# Section 4 — Target Correlation
# ─────────────────────────────────────────────
def section_4_target_correlation(df_all: pd.DataFrame, items: list[str]):
    print(f"\n{SEP}")
    print("SECTION 4: TARGET CORRELATION (aggregated across items)")
    print(SEP)

    all_corrs: dict[str, list[float]] = {f: [] for f in ALL_CANDIDATE_FEATURES}
    for item in items:
        df = _post_rebrand_nonzero(item, df_all)
        if df is None:
            continue
        df_feat = _build_all_candidates(df.copy())
        for feat in ALL_CANDIDATE_FEATURES:
            if feat in df_feat.columns:
                r = df_feat[feat].corr(df_feat["Quantity_Sold"])
                if not np.isnan(r):
                    all_corrs[feat].append(r)

    print(f"\n  {'Feature':>20s}  {'Mean r':>8s}  {'Std r':>8s}  {'|r|>=0.2':>9s}  {'N':>5s}")
    print(f"  {'-'*60}")
    results = []
    for feat, vals in sorted(all_corrs.items(), key=lambda x: abs(np.mean(x[1])), reverse=True):
        if vals:
            pct_strong = sum(1 for v in vals if abs(v) >= 0.2) / len(vals) * 100
            print(f"  {feat:>20s}  {np.mean(vals):>+8.4f}  {np.std(vals):>8.4f}  {pct_strong:>8.0f}%  {len(vals):>5d}")
            results.append({"feature": feat, "mean_r": np.mean(vals), "std_r": np.std(vals), "n": len(vals)})

    pd.DataFrame(results).to_csv(TABLE_DIR / "target_correlation.csv", index=False)
    return all_corrs


# ─────────────────────────────────────────────
# Section 5 — Mutual Information (multi-item)
# ─────────────────────────────────────────────
def section_5_mutual_information(df_all: pd.DataFrame, items: list[str]):
    print(f"\n{SEP}")
    print("SECTION 5: MUTUAL INFORMATION (model-free, across all items)")
    print(SEP)
    print("  MI measures any dependency (linear + non-linear) between feature and target.")
    print("  Unlike Pearson r, it captures non-linear patterns (e.g. DOW_Avg's effect via splits).")

    all_mi: dict[str, list[float]] = {f: [] for f in ALL_CANDIDATE_FEATURES}
    for idx, item in enumerate(items):
        if (idx + 1) % 10 == 0:
            print(f"  Processing item {idx+1}/{len(items)}: {item}")
        df = _post_rebrand_nonzero(item, df_all)
        if df is None:
            continue
        df_feat = _build_all_candidates(df.copy())

        available = [f for f in ALL_CANDIDATE_FEATURES if f in df_feat.columns]
        X = df_feat[available].values.astype(np.float64)
        y = df_feat["Quantity_Sold"].values.astype(np.float64).ravel()

        mi_vals = mutual_info_regression(X, y, random_state=42)
        for feat, mi in zip(available, mi_vals):
            if not np.isnan(mi):
                all_mi[feat].append(mi)

    print(f"\n  {'Feature':>20s}  {'Mean MI':>8s}  {'Std MI':>8s}  {'MI>0.05':>9s}  {'Pearson r':>10s}  {'N':>5s}")
    print(f"  {'-'*70}")

    # Also load Pearson r for comparison
    pearson = section_4_target_correlation(df_all, items) if False else None
    # We already computed pearson above, let's just reference the saved file
    try:
        pearson_df = pd.read_csv(TABLE_DIR / "target_correlation.csv")
        pearson_map = dict(zip(pearson_df["feature"], pearson_df["mean_r"]))
    except FileNotFoundError:
        pearson_map = {}

    results_rows = []
    for feat, vals in sorted(all_mi.items(), key=lambda x: np.mean(x[1]), reverse=True):
        if vals:
            pct_strong = sum(1 for v in vals if v >= 0.05) / len(vals) * 100
            p_r = pearson_map.get(feat, float("nan"))
            print(f"  {feat:>20s}  {np.mean(vals):>8.4f}  {np.std(vals):>8.4f}  {pct_strong:>8.0f}%  {p_r:>+10.4f}  {len(vals):>5d}")
            results_rows.append({
                "feature": feat,
                "mean_mi": np.mean(vals),
                "std_mi": np.std(vals),
                "pearson_r": p_r,
                "n_items": len(vals),
            })

    # Save table
    mi_df = pd.DataFrame(results_rows)
    mi_df.to_csv(TABLE_DIR / "mutual_information.csv", index=False)
    print(f"\n  Saved: tables/mutual_information.csv")

    # Generate figure: MI bar chart with error bars, sorted descending
    fig, ax = plt.subplots(figsize=(12, 6))
    names = [r["feature"] for r in results_rows]
    means = [r["mean_mi"] for r in results_rows]
    stds = [r["std_mi"] for r in results_rows]
    colors = ["#e74c3c" if "Lag_1" in n or "Diff_1" in n else "#3498db" for n in names]
    ax.barh(range(len(names)), means, xerr=stds, color=colors, capsize=3, edgecolor="black", linewidth=0.5)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.set_xlabel("Mutual Information (nats)")
    ax.set_title("Mutual Information with Target (mean ± std across items)")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "10_feature_importance.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: figures/feature_discovery/10_feature_importance.png")

    # Generate grouped comparison: MI by category (excluding Diff — excluded feature)
    EXCLUDED_GROUPS = {"Diff"}
    fig, ax = plt.subplots(figsize=(10, 5))
    group_labels = [k for k in FEATURE_GROUPS if k not in EXCLUDED_GROUPS]
    group_means = []
    group_stds = []
    for label in group_labels:
        feats = FEATURE_GROUPS[label]
        g_vals = [r["mean_mi"] for r in results_rows if r["feature"] in feats]
        group_means.append(np.mean(g_vals) if g_vals else 0)
        group_stds.append(np.std(g_vals) if g_vals else 0)
    colors2 = plt.cm.Set2(np.linspace(0, 1, len(group_labels)))
    ax.barh(range(len(group_labels)), group_means, color=colors2, edgecolor="black", linewidth=0.5)
    ax.set_yticks(range(len(group_labels)))
    ax.set_yticklabels(group_labels)
    ax.set_xlabel("Mean Mutual Information (nats)")
    ax.set_title("Information Contribution by Feature Group")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "12_model_comparison.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: figures/feature_discovery/12_model_comparison.png (group contribution, Diff excluded)")

    return all_mi


# ─────────────────────────────────────────────
# Section 6 — Group Information Contribution
# ─────────────────────────────────────────────
def section_6_group_contribution(df_all: pd.DataFrame, items: list[str]):
    print(f"\n{SEP}")
    print("SECTION 6: GROUP INFORMATION CONTRIBUTION")
    print(SEP)
    print("  Measures total information each feature group provides about the target.")
    print("  Groups with higher total MI contribute more predictive signal.")

    # Read MI from previous section
    try:
        mi_df = pd.read_csv(TABLE_DIR / "mutual_information.csv")
    except FileNotFoundError:
        print("  ERROR: Run section 5 first to generate mutual_information.csv")
        return

    mi_map = dict(zip(mi_df["feature"], mi_df["mean_mi"]))

    print(f"\n  {'Group':<25s}  {'Total MI':>9s}  {'% of Total':>11s}  {'Features':>30s}")
    print(f"  {'-'*80}")

    total_mi = sum(mi_df["mean_mi"])
    group_results = []
    for label, feats in FEATURE_GROUPS.items():
        group_mi = sum(mi_map.get(f, 0) for f in feats)
        pct = group_mi / total_mi * 100 if total_mi > 0 else 0
        feat_list = ", ".join(f for f in feats if f in mi_map)
        print(f"  {label:<25s}  {group_mi:>9.4f}  {pct:>10.1f}%  {feat_list:<30s}")
        group_results.append({"group": label, "total_mi": group_mi, "pct": pct})

    pd.DataFrame(group_results).to_csv(TABLE_DIR / "group_contribution.csv", index=False)
    print(f"\n  Saved: tables/group_contribution.csv")


# ─────────────────────────────────────────────
# Section 7 — Conditional Dependency Analysis (Lag_1)
# ─────────────────────────────────────────────
def section_7_conditional_dependency(df_all: pd.DataFrame, items: list[str]):
    print(f"\n{SEP}")
    print("SECTION 7: CONDITIONAL DEPENDENCY — Lag_1 Persistence vs DOW Cycle")
    print(SEP)
    print("  Statistical evidence that Lag_1 creates a 'follow yesterday' trap.")
    print("  Compares how much information Lag_1 vs Lag_7 provides about the target.")
    print("  Shows that DOW features capture the weekly cycle more stably.")

    all_lag1: list[float] = []
    all_lag7: list[float] = []
    all_diff1: list[float] = []
    partial_entries: list[dict] = []

    for item in items:
        df = _post_rebrand_nonzero(item, df_all)
        if df is None:
            continue
        df_feat = _build_all_candidates(df.copy())

        # 1. Raw correlations
        r_lag1 = df_feat["Lag_1"].corr(df_feat["Quantity_Sold"])
        r_lag7 = df_feat["Lag_7"].corr(df_feat["Quantity_Sold"])
        r_diff1 = df_feat["Diff_1"].corr(df_feat["Quantity_Sold"])
        if any(np.isnan(x) for x in [r_lag1, r_lag7, r_diff1]):
            continue

        all_lag1.append(r_lag1)
        all_lag7.append(r_lag7)
        all_diff1.append(r_diff1)

        # 2. Partial correlation: Lag_1 | Lag_7 (controlling for Lag_7)
        r_lag1_lag7 = df_feat["Lag_1"].corr(df_feat["Lag_7"])
        partial_lag1_given_lag7 = (r_lag1 - r_lag7 * r_lag1_lag7) / (
            np.sqrt(1 - r_lag7**2) * np.sqrt(1 - r_lag1_lag7**2)
        )
        partial_lag7_given_lag1 = (r_lag7 - r_lag1 * r_lag1_lag7) / (
            np.sqrt(1 - r_lag1**2) * np.sqrt(1 - r_lag1_lag7**2)
        )
        partial_entries.append({
            "item": item,
            "r_lag1": r_lag1,
            "r_lag7": r_lag7,
            "partial_lag1_lag7": partial_lag1_given_lag7,
            "partial_lag7_lag1": partial_lag7_given_lag1,
        })

    print(f"\n  Results from {len(all_lag1)} items:")
    print(f"\n  {'Metric':<35s}  {'Mean':>8s}  {'Std':>8s}  {'Min':>8s}  {'Max':>8s}")
    print(f"  {'-'*70}")
    for label, vals in [
        ("r(target, Lag_1) — raw correlation", all_lag1),
        ("r(target, Lag_7) — raw correlation", all_lag7),
        ("r(target, Diff_1) — raw correlation", all_diff1),
    ]:
        print(f"  {label:<35s}  {np.mean(vals):>+8.4f}  {np.std(vals):>8.4f}  {np.min(vals):>+8.4f}  {np.max(vals):>+8.4f}")

    # Partial correlation summary
    print(f"\n  Partial correlations (controlling for the other lag):")
    print(f"  {'Metric':<50s}  {'Mean':>8s}  {'Std':>8s}")
    print(f"  {'-'*70}")
    p_lag1 = [e["partial_lag1_lag7"] for e in partial_entries]
    p_lag7 = [e["partial_lag7_lag1"] for e in partial_entries]
    print(f"  Partial r(Lag_1, target | Lag_7):                    {np.mean(p_lag1):>+8.4f}  {np.std(p_lag1):>8.4f}")
    print(f"  Partial r(Lag_7, target | Lag_1):                    {np.mean(p_lag7):>+8.4f}  {np.std(p_lag7):>8.4f}")

    # Count items where Lag_7 has higher partial than Lag_1
    lag7_wins = sum(1 for e in partial_entries if abs(e["partial_lag7_lag1"]) > abs(e["partial_lag1_lag7"]))
    print(f"\n  Lag_7 has higher partial correlation than Lag_1: {lag7_wins}/{len(partial_entries)} items")
    print(f"  Lag_1 has higher partial correlation than Lag_7: {len(partial_entries) - lag7_wins}/{len(partial_entries)} items")

    # Generate figure: bar chart comparing Lag_1 vs Lag_7 raw and partial
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.bar(["Lag_1", "Lag_7", "Diff_1"],
           [np.mean(all_lag1), np.mean(all_lag7), np.mean(all_diff1)],
           yerr=[np.std(all_lag1), np.std(all_lag7), np.std(all_diff1)],
           capsize=5, color=["#e74c3c", "#3498db", "#95a5a6"], edgecolor="black")
    ax.set_ylabel("Pearson r")
    ax.set_title("Raw Correlation with Target")
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.5)

    ax = axes[1]
    ax.bar(["Lag_1 | Lag_7", "Lag_7 | Lag_1"],
           [np.mean(p_lag1), np.mean(p_lag7)],
           yerr=[np.std(p_lag1), np.std(p_lag7)],
           capsize=5, color=["#e74c3c", "#3498db"], edgecolor="black")
    ax.set_ylabel("Partial Pearson r")
    ax.set_title("Partial Correlation (controlling for other lag)")
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.5)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "14_lag1_dependency.png", dpi=150)
    plt.close(fig)
    print(f"\n  Saved: figures/feature_discovery/14_lag1_dependency.png")

    # Save table
    pd.DataFrame(partial_entries).to_csv(TABLE_DIR / "lag1_dependency.csv", index=False)
    print(f"  Saved: tables/lag1_dependency.csv")

    # Generate a transition matrix for the top item
    print(f"\n\n  --- Conditional Distribution: {TARGET_ITEM} ---")
    df = _post_rebrand_nonzero(TARGET_ITEM, df_all)
    if df is not None:
        df_feat = _build_all_candidates(df.copy())
        df_feat = df_feat.dropna(subset=["Lag_1", "Quantity_Sold"])
        # Bin the previous day's value
        bins = [0, 1, 2, 3, 4, 5, 10, 100]
        labels = ["1", "2", "3", "4", "5", "6-10", "11+"]
        df_feat["Lag_1_bin"] = pd.cut(df_feat["Lag_1"], bins=bins, labels=labels, right=True)

        print(f"\n  P(qty[t] | qty[t-1]) — Conditional distribution for {TARGET_ITEM}:")
        print(f"  {'qty[t-1]':>10s}  {'N':>5s}  {'Mean qty[t]':>12s}  {'Std':>6s}")
        print(f"  {'-'*40}")
        for label in labels:
            subset = df_feat[df_feat["Lag_1_bin"] == label]
            if len(subset) > 0:
                print(f"  {label:>10s}  {len(subset):>5d}  {subset['Quantity_Sold'].mean():>12.2f}  {subset['Quantity_Sold'].std():>6.2f}")

        print(f"\n  Compare with DOW averages for same item:")
        dow_avg = compute_dow_stats(df)
        for dow in range(7):
            row = dow_avg[dow_avg["DOW"] == dow]
            if len(row) > 0:
                day_names = ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"]
                print(f"  {day_names[dow]:>10s}  avg={row['DOW_Avg'].values[0]:.2f}  p75={row['DOW_P75'].values[0]:.2f}")


# ─────────────────────────────────────────────
# Section 8 — Exclusion Summary
# ─────────────────────────────────────────────
def section_8_exclusion_summary(df_all: pd.DataFrame, items: list[str]):
    print(f"\n{SEP}")
    print("SECTION 8: FINAL FEATURE SET — Evidence-based summary")
    print(SEP)

    # Read computed tables
    try:
        mi_df = pd.read_csv(TABLE_DIR / "mutual_information.csv")
        mi_map = dict(zip(mi_df["feature"], mi_df["mean_mi"]))
    except FileNotFoundError:
        mi_map = {}

    try:
        pearson_df = pd.read_csv(TABLE_DIR / "target_correlation.csv")
        pearson_map = dict(zip(pearson_df["feature"], pearson_df["mean_r"]))
    except FileNotFoundError:
        pearson_map = {}

    print(f"\nINCLUDED ({len(FEATURE_COLS)} features):")
    for feat in FEATURE_COLS:
        mi = mi_map.get(feat, float("nan"))
        r = pearson_map.get(feat, float("nan"))
        print(f"  {feat:>20s}  MI={mi:.4f}  r={r:+.4f}")

    excluded = [f for f in ALL_CANDIDATE_FEATURES if f not in FEATURE_COLS]
    print(f"\nEXCLUDED ({len(excluded)} features):")
    for feat in excluded:
        mi = mi_map.get(feat, float("nan"))
        r = pearson_map.get(feat, float("nan"))
        print(f"  {feat:>20s}  MI={mi:.4f}  r={r:+.4f}")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    print(SEP)
    print("FEATURE ANALYSIS (Evidence-Driven) — Multi-Item Statistical Edition")
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(SEP)

    print("\nLoading all items from database...")
    df_all = load_all_items()
    print(f"Loaded {len(df_all):,} rows")

    items = _collect_items(df_all)
    print(f"Using {len(items)} items for analysis")

    section_1_autocorrelation(df_all, items)
    section_2_nan_sparsity(df_all, items)
    section_3_collinearity(df_all, items)
    section_4_target_correlation(df_all, items)

    print(f"\n{'='*70}")
    print("SECTIONS 5-7: Multi-Item Statistical Analysis")
    print("=" * 70)

    mi_results = section_5_mutual_information(df_all, items)
    section_6_group_contribution(df_all, items)
    section_7_conditional_dependency(df_all, items)

    section_8_exclusion_summary(df_all, items)

    print(f"\n{SEP}")
    print(f"ANALYSIS COMPLETE — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Figures saved to: {FIG_DIR}")
    print(f"Tables saved to: {TABLE_DIR}")
    print(SEP)


if __name__ == "__main__":
    main()
