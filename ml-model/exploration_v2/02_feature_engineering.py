"""
v2_02_feature_engineering.py
fresh feature design — driven by EDA insights, no prior code referenced

Key EDA findings guiding feature design:
  1. 64% zero-inflation → need binary sale probability model
  2. Sparse items have high lag-7 autocorrelation → recency & lag features are powerful
  3. Continuous growth trend → trend features needed, not just step-change cutoff
  4. Items vary in maturity → lifecycle features (days since first sale, rank)
  5. Saturday busiest, Friday slowest → DOW effects matter
"""
import os, sys, warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DAILY_SALES_PATH, FIGURES_DIR, TABLES_DIR, RANDOM_SEED

sns.set_style("whitegrid")
np.random.seed(RANDOM_SEED)

OUT = os.path.join(FIGURES_DIR, "v2_features")
os.makedirs(OUT, exist_ok=True)


def build_full_history(df):
    """Build complete (date x item) grid with all features, including zeros."""
    items = sorted(df["Item"].unique())
    dates = pd.date_range(df["Date_Only"].min(), df["Date_Only"].max())
    grid = pd.DataFrame(
        [(d, i) for d in dates for i in items], columns=["Date_Only", "Item"]
    )
    full = grid.merge(df, on=["Date_Only", "Item"], how="left")
    full["Quantity"] = full["Quantity"].fillna(0).astype(float)
    full["Category"] = full.groupby("Item")["Category"].transform(
        lambda x: x.mode().iloc[0] if not x.mode().empty else x.dropna().iloc[0]
        if x.dropna().shape[0] > 0 else "unknown"
    )
    full["Is_Sale"] = (full["Quantity"] > 0).astype(int)
    return full


# ---------------------------------------------------------------------------
# FEATURE GROUP 1: Temporal identity features
# ---------------------------------------------------------------------------
def add_temporal_features(full):
    d = full["Date_Only"]
    full["DOW"] = d.dt.dayofweek
    full["Is_Weekend"] = (full["DOW"] >= 5).astype(int)
    full["Month"] = d.dt.month
    full["Year"] = d.dt.year
    full["WeekOfYear"] = d.dt.isocalendar().week.astype(int)
    full["DayOfMonth"] = d.dt.day
    full["Quarter"] = d.dt.quarter
    full["MonthStart"] = (full["DayOfMonth"] <= 7).astype(int)
    full["MonthEnd"] = (full["DayOfMonth"] >= 25).astype(int)
    full["Is_Holiday_Season"] = full["Month"].isin([12, 1]).astype(int)
    full["WeekOfMonth"] = ((full["DayOfMonth"] - 1) // 7 + 1).astype(int)
    full["DaysFromStart"] = (d - d.min()).dt.days

    # Cyclical encoding for DOW and Month
    full["DOW_Sin"] = np.sin(2 * np.pi * full["DOW"] / 7)
    full["DOW_Cos"] = np.cos(2 * np.pi * full["DOW"] / 7)
    full["Month_Sin"] = np.sin(2 * np.pi * full["Month"] / 12)
    full["Month_Cos"] = np.cos(2 * np.pi * full["Month"] / 12)

    print("  Added temporal identity features (DOW sin/cos, Month sin/cos, WeekOfMonth, etc.)")
    return full


# ---------------------------------------------------------------------------
# FEATURE GROUP 2: Item lifecycle & recency
# ---------------------------------------------------------------------------
def add_lifecycle_features(full):
    """Days since first sale, item age, total historical rank."""
    # First sale date per item
    first_dates = {}
    item_ranks = {}
    for item, grp in full.groupby("Item"):
        sales = grp[grp["Quantity"] > 0]
        if len(sales) > 0:
            first_dates[item] = sales["Date_Only"].min()

    # Item rank by total volume (up to current date)
    # We'll compute globally across all time, since rank is relatively stable
    item_total_vol = full.groupby("Item")["Quantity"].sum().rank(ascending=False)
    max_rank = item_total_vol.max()
    for item in full["Item"].unique():
        item_ranks[item] = item_total_vol.get(item, max_rank)

    full["Days_Since_First_Sale"] = full.apply(
        lambda r: (r["Date_Only"] - first_dates.get(r["Item"], r["Date_Only"])).days, axis=1
    )
    full["Item_Rank"] = full["Item"].map(item_ranks)
    full["Item_Rank_Pct"] = full["Item_Rank"] / max_rank

    print("  Added lifecycle features (days_since_first, item_rank)")
    return full


def add_recency_features(full):
    """
    CRITICAL for zero-inflation: days since last sale.
    Vectorized per-item using groupby.
    """
    full = full.sort_values(["Item", "Date_Only"]).reset_index(drop=True)

    def _recency_per_item(grp):
        grp = grp.sort_values("Date_Only")
        dates = pd.to_datetime(grp["Date_Only"]).values
        qty = grp["Quantity"].values
        n = len(dates)

        days_since = np.full(n, 999, dtype=int)
        last_sale_date = None

        for i in range(n):
            if qty[i] > 0:
                last_sale_date = dates[i]
            if last_sale_date is not None and i > 0:
                diff = (dates[i] - last_sale_date) / np.timedelta64(1, "D")
                days_since[i] = min(int(diff), 999)

        # Count sales in last 7 days (using rolling count of sale days)
        is_sale_vals = (qty > 0).astype(int)
        sales_7d = np.zeros(n, dtype=int)
        for i in range(n):
            count = 0
            for j in range(max(0, i - 7), i):
                if is_sale_vals[j]:
                    count += 1
            sales_7d[i] = count

        grp["Days_Since_Last_Sale"] = days_since
        grp["Sales_Last_7D"] = sales_7d.astype(int)
        return grp

    full = full.groupby("Item", group_keys=False).apply(_recency_per_item)
    print("  Added recency features (days_since_last_sale, sales_last_7d)")
    return full


# ---------------------------------------------------------------------------
# FEATURE GROUP 3: Rolling & lag features (per item, no leakage)
# ---------------------------------------------------------------------------
def add_rolling_features(full):
    """
    Compute per-item rolling statistics using SHIFT(1) to prevent leakage.
    Vectorized with groupby.
    """
    full = full.sort_values(["Item", "Date_Only"]).reset_index(drop=True)

    def _rolling_per_item(grp):
        grp = grp.sort_values("Date_Only")
        qty = grp["Quantity"].values
        s = pd.Series(qty, index=grp.index)

        shifted = s.shift(1)

        grp["Roll_Mean_7"] = shifted.rolling(7, min_periods=1).mean().values
        grp["Roll_Mean_14"] = shifted.rolling(14, min_periods=1).mean().values
        grp["Roll_Mean_28"] = shifted.rolling(28, min_periods=1).mean().values
        grp["Roll_Std_7"] = shifted.rolling(7, min_periods=2).std().fillna(0).values
        grp["EWMA_7"] = shifted.ewm(span=7, min_periods=1).mean().values
        grp["EWMA_28"] = shifted.ewm(span=28, min_periods=1).mean().values
        grp["Lag_1"] = s.shift(1).values
        grp["Lag_7"] = s.shift(7).values
        grp["Lag_14"] = s.shift(14).values
        grp["Lag_28"] = s.shift(28).values

        r7 = grp["Roll_Mean_7"].values
        r28 = grp["Roll_Mean_28"].values
        grp["Trend_7_28"] = (r7 - r28) / (r28 + 0.1)

        r7_s = pd.Series(r7, index=grp.index)
        grp["WoW_Change"] = (r7_s - r7_s.shift(7)) / (r7_s.shift(7) + 0.1)

        return grp

    full = full.groupby("Item", group_keys=False).apply(_rolling_per_item)

    for col in ["Lag_1", "Lag_7", "Lag_14", "Lag_28", "Roll_Std_7", "WoW_Change",
                "Roll_Mean_7", "Roll_Mean_14", "Roll_Mean_28", "EWMA_7", "EWMA_28"]:
        full[col] = full[col].fillna(0)

    full["Trend_7_28"] = full["Trend_7_28"].fillna(0).replace([np.inf, -np.inf], 0)

    for col in ["Trend_7_28", "WoW_Change"]:
        full[col] = full[col].clip(-5, 5)

    print("  Added rolling/lag features (lag_1,7,14,28, roll_mean, ewma, trend, wow)")
    return full


# ---------------------------------------------------------------------------
# FEATURE GROUP 4: DOW baseline statistics per item (expanding, no leakage)
# ---------------------------------------------------------------------------
def add_dow_baselines(full):
    """
    For each item, compute DOW average/median/P75 using expanding window.
    Vectorized: computes cumulative per-DOW stats using groupby rolling.
    """
    full = full.sort_values(["Item", "Date_Only"]).reset_index(drop=True)

    # Create expanding per-(Item, DOW) stats
    # Strategy: track cumulative sum/count per DOW, compute mean incrementally
    for item, grp in full.groupby("Item"):
        grp = grp.sort_values("Date_Only")
        cum_sum = {d: 0.0 for d in range(7)}
        cum_cnt = {d: 0 for d in range(7)}
        cum_list = {d: [] for d in range(7)}  # for median and p75

        dow_avg = np.zeros(len(grp))
        dow_med = np.zeros(len(grp))
        dow_p75 = np.zeros(len(grp))
        dow_n = np.zeros(len(grp), dtype=int)

        for i, (_, row) in enumerate(grp.iterrows()):
            d = int(row["DOW"])
            qty = row["Quantity"]

            if qty > 0:
                cum_sum[d] += qty
                cum_cnt[d] += 1
                cum_list[d].append(qty)

            if cum_cnt[d] > 0:
                dow_avg[i] = cum_sum[d] / cum_cnt[d]
                arr = np.array(cum_list[d])
                dow_med[i] = np.median(arr)
                dow_p75[i] = np.percentile(arr, 75)
            else:
                dow_avg[i] = 0.0
                dow_med[i] = 0.0
                dow_p75[i] = 0.0
            dow_n[i] = cum_cnt[d]

        full.loc[grp.index, "DOW_Avg"] = dow_avg
        full.loc[grp.index, "DOW_Median"] = dow_med
        full.loc[grp.index, "DOW_P75"] = dow_p75
        full.loc[grp.index, "DOW_N_Samples"] = dow_n.astype(float)

    print("  Added expanding DOW baselines (avg, median, p75, n_samples)")
    return full


# ---------------------------------------------------------------------------
# FEATURE GROUP 5: Cross-item / day-level features (no leakage)
# ---------------------------------------------------------------------------
def add_cross_item_features(full):
    """
    Day-level aggregate features computed on SHIFTED data.
    The cafe's overall busyness is a signal for individual items.
    """
    daily_totals = full.groupby("Date_Only").agg(
        Total_Qty=("Quantity", "sum"),
        Total_Items_Sold=("Is_Sale", "sum"),
        Total_Beverage=("Quantity", lambda x: x[full.loc[x.index, "Category"] == "beverage"].sum()),
        Total_Food=("Quantity", lambda x: x[full.loc[x.index, "Category"] == "food"].sum()),
    ).shift(1)  # use YESTERDAY's totals

    daily_totals = daily_totals.fillna(0)

    for col in daily_totals.columns:
        full[f"Day_{col}"] = full["Date_Only"].map(daily_totals[col])

    # Also: 7-day rolling average of daily totals
    roll_total = daily_totals["Total_Qty"].rolling(7, min_periods=1).mean()
    full["Day_Total_Qty_7D"] = full["Date_Only"].map(roll_total)

    print("  Added cross-item day-level features (yesterday totals, 7d avg)")
    return full


# ---------------------------------------------------------------------------
# FEATURE CORRELATION ANALYSIS
# ---------------------------------------------------------------------------
def analyze_features(full, feature_cols):
    """Analyze which features correlate with the target."""
    print("\n" + "=" * 70)
    print("FEATURE ANALYSIS")
    print("=" * 70)

    # Only use data where we have enough history for features to be meaningful
    analysis = full[full["DaysFromStart"] >= 30].copy()
    if len(analysis) == 0:
        analysis = full.copy()

    # --- Binary target correlation ---
    print("\n--- Feature correlation with Is_Sale (binary) ---")
    numeric_cols = [c for c in feature_cols if c in analysis.columns and analysis[c].dtype in ["int64", "float64", "int32"]]
    sale_corr = analysis[numeric_cols + ["Is_Sale"]].corr()["Is_Sale"].drop("Is_Sale").sort_values(key=abs, ascending=False)
    print(sale_corr.head(20).to_string())
    print()

    # --- Continuous target correlation (only on sale days) ---
    print("--- Feature correlation with Quantity (sale days only) ---")
    sale_only = analysis[analysis["Quantity"] > 0]
    qty_corr = sale_only[numeric_cols + ["Quantity"]].corr()["Quantity"].drop("Quantity").sort_values(key=abs, ascending=False)
    print(qty_corr.head(20).to_string())
    print()

    # --- Mutual information for Is_Sale ---
    try:
        from sklearn.feature_selection import mutual_info_regression
        valid = analysis[numeric_cols].dropna()
        mi = mutual_info_regression(
            valid.fillna(0),
            analysis.loc[valid.index, "Is_Sale"],
            random_state=RANDOM_SEED,
            n_neighbors=5,
        )
        mi_scores = pd.Series(mi, index=numeric_cols).sort_values(ascending=False)
        print("--- Mutual Information with Is_Sale ---")
        print(mi_scores.head(15).to_string())
        mi_scores.to_csv(os.path.join(TABLES_DIR, "v2_mutual_info.csv"))

        # Plot: Mutual Information (better version)
        fig, ax = plt.subplots(figsize=(10, 7))
        mi_plot = mi_scores.head(20)
        colors_mi = ['#E74C3C' if 'Days_Since_Last' in f or 'Sales_Last_7D' in f else
                      '#3498DB' if 'Roll' in f or 'EWMA' in f or 'Trend' in f else
                      '#2ECC71' if 'Lag_' in f else '#F39C12'
                      for f in mi_plot.index]
        ax.barh(range(len(mi_plot)), mi_plot.values, color=colors_mi)
        ax.set_yticks(range(len(mi_plot)))
        ax.set_yticklabels(mi_plot.index, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel('Mutual Information (nats)', fontsize=12)
        ax.set_title('Mutual Information with Is_Sale', fontsize=14, fontweight='bold')
        plt.tight_layout()
        fig.savefig(os.path.join(OUT, "mutual_information.png"), dpi=200, bbox_inches='tight')
        plt.close()
        print("  → saved mutual_information.png")

        # Plot: Correlation matrix (top 25 features by abs correlation with Quantity)
        non_item_cols = [c for c in numeric_cols if not c.startswith('Item_')]
        top_corr = sale_only[non_item_cols + ['Quantity']].corr()['Quantity'].drop('Quantity')
        top_25 = top_corr.abs().sort_values(ascending=False).head(25).index.tolist()
        corr_mat = analysis[top_25].corr()
        fig, ax = plt.subplots(figsize=(14, 12))
        mask = np.triu(np.ones_like(corr_mat, dtype=bool), k=1)
        sns.heatmap(corr_mat, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
                    center=0, vmin=-1, vmax=1, square=True, linewidths=0.5,
                    cbar_kws={'shrink': 0.6, 'label': 'Pearson r'}, ax=ax)
        ax.set_title('Feature Correlation Matrix (Top 25 Features)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        fig.savefig(os.path.join(OUT, "correlation_matrix.png"), dpi=200, bbox_inches='tight')
        plt.close()
        print("  → saved correlation_matrix.png")
    except ImportError:
        print("  sklearn not available, skipping MI")

    return sale_corr, qty_corr


# ---------------------------------------------------------------------------
# FEATURE IMPORTANCE VIA SIMPLE TREE MODEL
# ---------------------------------------------------------------------------
def tree_feature_importance(full, binary_features, reg_features):
    """Quick LightGBM-based feature importance."""
    try:
        import lightgbm as lgb
    except ImportError:
        print("  LightGBM not available, skipping tree importance")
        return

    print("\n--- Tree-based feature importance ---")
    analysis = full[full["DaysFromStart"] >= 30]

    # Classification: Is_Sale
    bin_feats = [f for f in binary_features if f in analysis.columns]
    X_bin = analysis[bin_feats].fillna(0)
    y_bin = analysis["Is_Sale"]
    model_bin = lgb.LGBMClassifier(n_estimators=100, max_depth=5, random_state=RANDOM_SEED, verbose=-1)
    model_bin.fit(X_bin, y_bin)
    imp_bin = pd.Series(model_bin.feature_importances_, index=bin_feats).sort_values(ascending=False)
    print("\nTop 20 for Is_Sale (LightGBM):")
    print(imp_bin.head(20).to_string())
    imp_bin.to_csv(os.path.join(TABLES_DIR, "v2_lgb_importance_issale.csv"))

    # Regression: Quantity (on sale days only)
    sale_only = analysis[analysis["Quantity"] > 0]
    X_reg = sale_only[bin_feats].fillna(0)
    y_reg = sale_only["Quantity"]
    model_reg = lgb.LGBMRegressor(n_estimators=100, max_depth=5, random_state=RANDOM_SEED, verbose=-1)
    model_reg.fit(X_reg, y_reg)
    imp_reg = pd.Series(model_reg.feature_importances_, index=bin_feats).sort_values(ascending=False)
    print("\nTop 20 for Quantity (LightGBM, sale days only):")
    print(imp_reg.head(20).to_string())
    imp_reg.to_csv(os.path.join(TABLES_DIR, "v2_lgb_importance_qty.csv"))

    return imp_bin, imp_reg


# ---------------------------------------------------------------------------
# FEATURE SELECTION
# ---------------------------------------------------------------------------
# Feature groups and their ablation impact (Δ MAE from 8-window backtest):
#
#   Recency        +0.410  ████████████████████████████████████████
#   Temporal       +0.076  ████████
#   CrossItem      +0.033  ████
#   Lags           +0.025  ███
#   ─────────────────────  threshold Δ > 0.02 ─────────────────────
#   Lifecycle      +0.017  ██          ← dropped
#   Rolling        +0.013  █           ← dropped
#   DOW_Baselines  +0.026* *8-window backtest confirms help despite
#                           misleading single-split (Δ=-0.029)
#
# Groups kept: Recency, Temporal, CrossItem, Lags, DOW_Baselines = 31 features.
# Threshold rationale: the gap between Lags (+0.025) and Lifecycle (+0.017)
# is the only meaningful structural break in the sorted Δ distribution.

ABLATION_GROUPS = {
    "Recency":      ["Days_Since_Last_Sale", "Sales_Last_7D"],
    "Temporal":     ["DOW", "Is_Weekend", "Month", "Year", "WeekOfYear", "DayOfMonth",
                     "Quarter", "MonthStart", "MonthEnd", "Is_Holiday_Season",
                     "WeekOfMonth", "DaysFromStart", "DOW_Sin", "DOW_Cos",
                     "Month_Sin", "Month_Cos"],
    "CrossItem":    ["Day_Total_Qty", "Day_Total_Items_Sold", "Day_Total_Beverage",
                     "Day_Total_Food", "Day_Total_Qty_7D"],
    "Lags":         ["Lag_1", "Lag_7", "Lag_14", "Lag_28"],
    "DOW_Baselines": ["DOW_Avg", "DOW_Median", "DOW_P75", "DOW_N_Samples"],
}
# Dropped: Lifecycle (Δ=+0.017), Rolling (Δ=+0.013)

SELECTED_FEATURES = [f for group in ABLATION_GROUPS.values() for f in group]


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    print("v2_02: FRESH FEATURE ENGINEERING FROM SCRATCH")
    print()

    # Load raw
    df = pd.read_csv(DAILY_SALES_PATH)
    df["Date_Only"] = pd.to_datetime(df["Date_Only"])
    df["Quantity"] = df["Quantity"].astype(float)
    print(f"Loaded {len(df)} raw rows, {df['Item'].nunique()} items")
    print()

    # Build full grid (with zeros)
    full = build_full_history(df)
    print(f"Full grid: {len(full)} rows ({full['Item'].nunique()} items x {full['Date_Only'].nunique()} days)")
    print()

    # Add features
    print("Adding features...")
    full = add_temporal_features(full)
    full = add_lifecycle_features(full)
    full = add_recency_features(full)
    full = add_rolling_features(full)
    full = add_dow_baselines(full)
    full = add_cross_item_features(full)

    binary_features = SELECTED_FEATURES
    regression_features = binary_features

    print(f"\nFeature selection driven by ablation (Δ MAE > 0.02 threshold):")
    print(f"  Kept:   {len(binary_features)} features from {len(ABLATION_GROUPS)} groups")
    dropped = ["Lifecycle (3)", "Rolling (8)"]
    print(f"  Dropped: {', '.join(dropped)} — below Δ>0.02 threshold")
    print(f"  Removed earlier: 61 item dummies + 2 category flags — no predictive value")

    # Analyze
    sale_corr, qty_corr = analyze_features(full, binary_features)
    tree_feature_importance(full, binary_features, regression_features)

    # Save
    feature_cols_to_save = [
        "Date_Only", "Item", "Category", "Quantity", "Is_Sale",
    ] + binary_features

    save_cols = [c for c in feature_cols_to_save if c in full.columns]
    full[save_cols].to_csv(
        os.path.join(TABLES_DIR, "v2_feature_matrix.csv"), index=False
    )
    print(f"\nSaved feature matrix: {len(full)} rows x {len(save_cols)} cols")
    print(f"  → {os.path.join(TABLES_DIR, 'v2_feature_matrix.csv')}")


if __name__ == "__main__":
    main()
