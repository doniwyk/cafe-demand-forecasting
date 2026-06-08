"""
v2_01_data_exploration.py
fresh EDA — look at the data with zero preconceptions
"""
import os, sys, warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from datetime import timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DAILY_SALES_PATH, FIGURES_DIR, TABLES_DIR, RANDOM_SEED

sns.set_style("whitegrid")
np.random.seed(RANDOM_SEED)

os.makedirs(os.path.join(FIGURES_DIR, "v2_eda"), exist_ok=True)
os.makedirs(TABLES_DIR, exist_ok=True)
OUT = os.path.join(FIGURES_DIR, "v2_eda")


def load_raw():
    df = pd.read_csv(DAILY_SALES_PATH)
    df["Date_Only"] = pd.to_datetime(df["Date_Only"])
    df["Quantity"] = df["Quantity"].astype(float)
    return df


def build_full_grid(df):
    """Create every (date, item) combination so we can measure zero-inflation"""
    dates = pd.date_range(df["Date_Only"].min(), df["Date_Only"].max())
    items = sorted(df["Item"].unique())
    grid = pd.DataFrame(
        [(d, i) for d in dates for i in items], columns=["Date_Only", "Item"]
    )
    full = grid.merge(df, on=["Date_Only", "Item"], how="left")
    full["Quantity"] = full["Quantity"].fillna(0)
    full["Is_Sale"] = (full["Quantity"] > 0).astype(int)
    return full


# ---------------------------------------------------------------------------
# 1. DATA OVERVIEW
# ---------------------------------------------------------------------------
def section1_overview(df):
    print("=" * 70)
    print("SECTION 1: DATA OVERVIEW")
    print("=" * 70)
    print(f"Rows: {len(df):,}")
    print(f"Columns: {list(df.columns)}")
    print(f"Date range: {df['Date_Only'].min().date()} → {df['Date_Only'].max().date()}")
    days_span = (df["Date_Only"].max() - df["Date_Only"].min()).days
    print(f"Calendar days: {days_span}")
    print(f"Unique items: {df['Item'].nunique()}")
    print(f"Categories: {df['Category'].nunique()} — {sorted(df['Category'].unique())}")
    print(f"Quantity range: {df['Quantity'].min():.0f} → {df['Quantity'].max():.0f}")
    print()

    # Quantity distribution
    qdist = df["Quantity"].value_counts().sort_index()
    pct_cum = 0
    print("Quantity | Count | Pct | CumPct")
    print("-" * 45)
    for q, c in qdist.items():
        p = c / len(df) * 100
        pct_cum += p
        print(f"  {int(q):>4}    | {c:>5} | {p:4.1f} | {pct_cum:5.1f}")
    print()

    # Per-item stats
    item_stats = df.groupby("Item").agg(
        n_days=("Date_Only", "nunique"),
        total_qty=("Quantity", "sum"),
        avg_qty=("Quantity", "mean"),
        max_qty=("Quantity", "max"),
        first_date=("Date_Only", "min"),
        last_date=("Date_Only", "max"),
    ).sort_values("total_qty", ascending=False)

    print("Top 15 items by volume:")
    print(item_stats.head(15).to_string())
    print()
    print("Bottom 10 items by volume:")
    print(item_stats.tail(10).to_string())
    print()

    # ABC breakdown
    total_vol = item_stats["total_qty"].sum()
    item_stats["cum_pct"] = item_stats["total_qty"].cumsum() / total_vol * 100
    a_items = (item_stats["cum_pct"] <= 70).sum()
    b_items = ((item_stats["cum_pct"] > 70) & (item_stats["cum_pct"] <= 90)).sum()
    c_items = (item_stats["cum_pct"] > 90).sum()
    print(f"ABC breakdown: A={a_items}, B={b_items}, C={c_items}")
    print(f"A volume: {item_stats[item_stats['cum_pct']<=70]['total_qty'].sum()/total_vol*100:.1f}%")
    print(f"B volume: {item_stats[(item_stats['cum_pct']>70)&(item_stats['cum_pct']<=90)]['total_qty'].sum()/total_vol*100:.1f}%")
    print(f"C volume: {item_stats[item_stats['cum_pct']>90]['total_qty'].sum()/total_vol*100:.1f}%")
    item_stats.to_csv(os.path.join(TABLES_DIR, "v2_item_stats.csv"))
    print()

    # Plot: top items
    fig, ax = plt.subplots(figsize=(14, 6))
    top15 = item_stats.head(15)
    ax.barh(range(len(top15)), top15["total_qty"].values)
    ax.set_yticks(range(len(top15)))
    ax.set_yticklabels(top15.index)
    ax.invert_yaxis()
    ax.set_xlabel("Total Quantity Sold")
    ax.set_title("Top 15 Items by Total Volume")
    plt.tight_layout()
    fig.savefig(os.path.join(OUT, "01_top_items.png"), dpi=150)
    plt.close()

    return item_stats


# ---------------------------------------------------------------------------
# 2. ZERO-INFLATION
# ---------------------------------------------------------------------------
def section2_zero_inflation(df):
    print("=" * 70)
    print("SECTION 2: ZERO-INFLATION ANALYSIS")
    print("=" * 70)

    full = build_full_grid(df)
    total_combos = len(full)
    zero_rows = (full["Quantity"] == 0).sum()
    nonzero_rows = (full["Quantity"] > 0).sum()
    zero_pct = zero_rows / total_combos * 100

    print(f"Total (date x item) combinations: {total_combos:,}")
    print(f"Zero-sales combos: {zero_rows:,}  ({zero_pct:.1f}%)")
    print(f"Non-zero combos: {nonzero_rows:,}  ({100-zero_pct:.1f}%)")
    print()

    # Per-item zero %
    item_zero = full.groupby("Item").agg(
        zero_pct=("Is_Sale", lambda x: (1 - x.mean()) * 100),
        nonzero_days=("Is_Sale", "sum"),
        total_days=("Is_Sale", "count"),
    ).sort_values("zero_pct")

    print("Items with highest and lowest zero-day %:")
    print(item_zero.head(10).to_string())
    print("...")
    print(item_zero.tail(10).to_string())
    print()

    # Daily zero %
    daily_zero = full.groupby("Date_Only").agg(
        items_sold=("Is_Sale", "sum"),
        total_items=("Is_Sale", "count"),
        zero_pct=("Is_Sale", lambda x: (1 - x.mean()) * 100),
        total_qty=("Quantity", "sum"),
    )

    print("Items sold per day:")
    print(daily_zero[["items_sold", "total_qty"]].describe())
    print()

    # Plot: daily items sold + total qty
    fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True)
    axes[0].plot(daily_zero.index, daily_zero["items_sold"], alpha=0.7, linewidth=0.5)
    axes[0].set_ylabel("Items Sold (count)")
    axes[0].set_title("Daily Number of Distinct Items Sold")
    axes[1].plot(daily_zero.index, daily_zero["total_qty"], alpha=0.7, linewidth=0.5, color="orange")
    axes[1].set_ylabel("Total Cups Sold")
    axes[1].set_title("Daily Total Quantity Sold")
    for ax in axes:
        ax.axvline(pd.Timestamp("2025-05-01"), color="red", linestyle="--", alpha=0.6, linewidth=1)
        ax.text(pd.Timestamp("2025-05-01"), ax.get_ylim()[1]*0.9, "May 2025", color="red", fontsize=8)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT, "02_daily_activity.png"), dpi=150)
    plt.close()

    item_zero.to_csv(os.path.join(TABLES_DIR, "v2_zero_inflation.csv"))
    return full


# ---------------------------------------------------------------------------
# 3. TIME PATTERNS
# ---------------------------------------------------------------------------
def section3_time_patterns(full):
    print("=" * 70)
    print("SECTION 3: TIME PATTERNS")
    print("=" * 70)

    daily = full.groupby("Date_Only").agg(
        total_qty=("Quantity", "sum"),
        n_items=("Is_Sale", "sum"),
        avg_qty_per_item=("Quantity", "mean"),
    )
    daily["DOW"] = daily.index.dayofweek
    daily["DOW_Name"] = daily.index.day_name()
    daily["Month"] = daily.index.month
    daily["Year"] = daily.index.year
    daily["WeekOfYear"] = daily.index.isocalendar().week.astype(int)
    daily["IsWeekend"] = (daily["DOW"] >= 5).astype(int)

    # --- 3a. DOW pattern ---
    print("--- Day of Week ---")
    dow = daily.groupby("DOW_Name").agg(
        avg_total_qty=("total_qty", "mean"),
        avg_n_items=("n_items", "mean"),
    ).reindex(["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"])
    print(dow.to_string())
    print()

    # DOW pattern per item (top items only)
    item_dow = full[full["Quantity"] > 0].copy()
    item_dow["DOW_Name"] = item_dow["Date_Only"].dt.day_name()
    top_items = item_dow.groupby("Item")["Quantity"].sum().nlargest(9).index
    fig, axes = plt.subplots(3, 3, figsize=(14, 12))
    for i, item in enumerate(top_items):
        ax = axes[i // 3][i % 3]
        sub = item_dow[item_dow["Item"] == item]
        dow_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        avg = sub.groupby("DOW_Name")["Quantity"].mean().reindex(dow_order)
        avg.plot(kind="bar", ax=ax, color="steelblue")
        ax.set_title(item, fontsize=9)
        ax.set_xticklabels(dow_order, rotation=45, fontsize=7)
        ax.set_ylabel("Avg Qty", fontsize=7)
    plt.suptitle("Per-Item Day-of-Week Average Quantity", fontsize=12)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT, "03_dow_per_item.png"), dpi=150)
    plt.close()
    print("→ saved 03_dow_per_item.png")
    print()

    # --- 3b. Monthly pattern ---
    print("--- Monthly ---")
    monthly = daily.groupby(["Year", "Month"]).agg(
        avg_qty=("total_qty", "mean"),
        total_qty=("total_qty", "sum"),
    ).reset_index()
    print(monthly.to_string())
    print()

    fig, ax = plt.subplots(figsize=(14, 5))
    pivot = monthly.pivot(index="Month", columns="Year", values="avg_qty")
    pivot.plot(ax=ax, marker="o")
    ax.set_xlabel("Month")
    ax.set_ylabel("Avg Daily Quantity")
    ax.set_title("Monthly Average Daily Sales by Year")
    ax.legend(title="Year")
    plt.tight_layout()
    fig.savefig(os.path.join(OUT, "04_monthly_pattern.png"), dpi=150)
    plt.close()

    # --- 3c. Trend (rolling) ---
    daily["Roll_Mean_28"] = daily["total_qty"].rolling(28).mean()
    daily["Roll_Std_28"] = daily["total_qty"].rolling(28).std()
    print(f"Latest 28-day avg: {daily['Roll_Mean_28'].iloc[-1]:.1f} ± {daily['Roll_Std_28'].iloc[-1]:.1f}")
    print(f"28-day avg min/max: {daily['Roll_Mean_28'].min():.1f} / {daily['Roll_Mean_28'].max():.1f}")
    print()

    fig, ax = plt.subplots(figsize=(16, 5))
    ax.fill_between(daily.index, daily["Roll_Mean_28"] - daily["Roll_Std_28"],
                    daily["Roll_Mean_28"] + daily["Roll_Std_28"], alpha=0.2)
    ax.plot(daily.index, daily["Roll_Mean_28"], linewidth=1.5, label="28-day MA")
    ax.plot(daily.index, daily["total_qty"], alpha=0.2, linewidth=0.5, label="Daily")
    ax.axvline(pd.Timestamp("2025-05-01"), color="red", linestyle="--", alpha=0.6)
    ax.set_ylabel("Total Quantity")
    ax.legend()
    ax.set_title("Daily Total Sales with 28-day Moving Average")
    plt.tight_layout()
    fig.savefig(os.path.join(OUT, "05_daily_trend.png"), dpi=150)
    plt.close()
    print("→ saved 05_daily_trend.png")
    print()

    # --- 3d. Week of year ---
    woy = daily.groupby("WeekOfYear")["total_qty"].mean()
    print(f"Weekly pattern: mean={woy.mean():.1f}, std={woy.std():.1f}, cv={woy.std()/woy.mean()*100:.1f}%")

    return daily


# ---------------------------------------------------------------------------
# 4. STRUCTURAL BREAK DETECTION
# ---------------------------------------------------------------------------
def section4_structural_break(full):
    print("=" * 70)
    print("SECTION 4: STRUCTURAL BREAK DETECTION")
    print("=" * 70)

    daily = full.groupby("Date_Only")["Quantity"].sum()

    # Split into 3-month windows and compare
    windows = pd.date_range(daily.index.min(), daily.index.max(), freq="3MS")
    for i in range(len(windows) - 1):
        start, end = windows[i], windows[i + 1] - timedelta(days=1)
        w = daily.loc[start:end]
        if len(w) > 30:
            print(f"  {start.date()} → {end.date()}: mean={w.mean():.1f}, std={w.std():.1f}, total={w.sum():.0f}, days={len(w)}")

    # Detect the biggest jump
    window_means = []
    for i in range(len(windows) - 1):
        start, end = windows[i], windows[i + 1] - timedelta(days=1)
        w = daily.loc[start:end]
        if len(w) > 30:
            window_means.append((start, w.mean()))

    changes = []
    for i in range(1, len(window_means)):
        prev_mean = window_means[i - 1][1]
        curr_mean = window_means[i][1]
        pct_change = (curr_mean - prev_mean) / (prev_mean + 0.001) * 100
        changes.append((window_means[i][0], pct_change))

    changes.sort(key=lambda x: abs(x[1]), reverse=True)
    print("\nLargest 3-month-window changes:")
    for date, pct in changes[:5]:
        print(f"  {date.date()}: {pct:+.0f}%")

    # Compute Cohen's d for pre vs post candidate cutoff
    candidates = [pd.Timestamp("2025-05-01"), pd.Timestamp("2025-06-01"), pd.Timestamp("2025-04-01"), pd.Timestamp("2024-01-01")]
    print("\nCohen's d for candidate break dates (pre vs post):")
    for cutoff in candidates:
        pre = daily[daily.index < cutoff]
        post = daily[daily.index >= cutoff]
        d = (post.mean() - pre.mean()) / np.sqrt((pre.std()**2 + post.std()**2) / 2)
        print(f"  {cutoff.date()}: d={d:.3f}  (pre_mean={pre.mean():.1f}, post_mean={post.mean():.1f}, pre_n={len(pre)}, post_n={len(post)})")

    return daily


# ---------------------------------------------------------------------------
# 5. ITEM LIFECYCLE
# ---------------------------------------------------------------------------
def section5_item_lifecycle(full):
    print("=" * 70)
    print("SECTION 5: ITEM LIFECYCLE")
    print("=" * 70)

    items = full[full["Quantity"] > 0].groupby("Item").agg(
        first_sale=("Date_Only", "min"),
        last_sale=("Date_Only", "max"),
        total_qty=("Quantity", "sum"),
        n_days=("Date_Only", "nunique"),
    )
    items["lifespan_days"] = (items["last_sale"] - items["first_sale"]).dt.days
    items["sales_rate"] = items["n_days"] / (items["lifespan_days"] + 1)
    items["sales_per_day"] = items["total_qty"] / (items["lifespan_days"] + 1)
    items["first_year"] = items["first_sale"].dt.year
    items["first_month"] = items["first_sale"].dt.to_period("M")

    print("Items by first sale year:")
    print(items["first_year"].value_counts().sort_index())
    print()

    print("Items by first sale month (2025-2026):")
    recent = items[items["first_sale"] >= "2025-01-01"]
    print(recent["first_month"].value_counts().sort_index().to_string())
    print()

    # Plot: lifespans
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    axes[0].hist(items["lifespan_days"].values, bins=30, edgecolor="white")
    axes[0].set_xlabel("Lifespan (days)")
    axes[0].set_title("Item Lifespan Distribution")
    axes[1].scatter(items["lifespan_days"], items["total_qty"], alpha=0.5)
    axes[1].set_xlabel("Lifespan (days)")
    axes[1].set_ylabel("Total Quantity")
    axes[1].set_title("Lifespan vs Total Volume")
    plt.tight_layout()
    fig.savefig(os.path.join(OUT, "06_item_lifecycle.png"), dpi=150)
    plt.close()
    print("→ saved 06_item_lifecycle.png")
    print()

    return items


# ---------------------------------------------------------------------------
# 6. SPARSITY & PREDICTABILITY
# ---------------------------------------------------------------------------
def section6_sparsity(full):
    print("=" * 70)
    print("SECTION 6: SPARSITY & PREDICTABILITY")
    print("=" * 70)

    # For each item, compute:
    # - mean sales per day (including zeros)
    # - CV of non-zero sales
    # - zero-day fraction
    # - max consecutive zero days
    # - autocorrelation at lag 7
    results = []
    for item, grp in full.groupby("Item"):
        grp = grp.sort_values("Date_Only")
        vals = grp["Quantity"].values
        nonzero_vals = vals[vals > 0]
        zero_frac = (vals == 0).mean()

        # consecutive zeros
        is_zero = (vals == 0).astype(int)
        consec = 0
        max_consec = 0
        for z in is_zero:
            if z:
                consec += 1
                max_consec = max(max_consec, consec)
            else:
                consec = 0

        # autocorrelation lag 7 on non-zero interpolated
        if len(nonzero_vals) >= 14:
            series = pd.Series(vals)
            acf7 = series.autocorr(lag=7) if series.std() > 0 else 0
        else:
            acf7 = np.nan

        results.append({
            "Item": item,
            "mean_sales": vals.mean(),
            "nonzero_mean": nonzero_vals.mean() if len(nonzero_vals) else 0,
            "cv_nonzero": nonzero_vals.std() / max(nonzero_vals.mean(), 0.1),
            "zero_pct": zero_frac * 100,
            "max_consec_zeros": max_consec,
            "acf_lag7": acf7,
            "n_days": len(vals),
            "n_nonzero": len(nonzero_vals),
        })

    sp = pd.DataFrame(results).set_index("Item").sort_values("mean_sales", ascending=False)
    print(sp.describe().to_string())
    print()

    # Items with strong autocorrelation (predictable)
    acf_items = sp.dropna(subset=["acf_lag7"]).sort_values("acf_lag7", ascending=False)
    print("Top 10 items by Lag-7 autocorrelation (most predictable):")
    print(acf_items[["mean_sales", "acf_lag7", "zero_pct"]].head(10).to_string())
    print()
    print("Bottom 10 items by Lag-7 autocorrelation (least predictable):")
    print(acf_items[["mean_sales", "acf_lag7", "zero_pct"]].tail(10).to_string())
    print()

    # Hard-to-predict items (high zero%, low acf, low mean)
    sp["hard_score"] = sp["zero_pct"] * (1 - sp["acf_lag7"].fillna(0))
    hardest = sp.sort_values("hard_score", ascending=False).head(15)
    print("Hardest-to-predict items (high zero% + low autocorrelation):")
    print(hardest[["mean_sales", "zero_pct", "acf_lag7", "hard_score"]].to_string())
    print()

    sp.to_csv(os.path.join(TABLES_DIR, "v2_sparsity.csv"))

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    axes[0].scatter(sp["mean_sales"], sp["zero_pct"], alpha=0.6)
    axes[0].set_xlabel("Mean Sales (incl zeros)")
    axes[0].set_ylabel("Zero-Day %")
    axes[0].set_title("Volume vs Sparsity")
    axes[1].scatter(sp["mean_sales"], sp["acf_lag7"].fillna(0), alpha=0.6)
    axes[1].set_xlabel("Mean Sales")
    axes[1].set_ylabel("Lag-7 Autocorrelation")
    axes[1].set_title("Volume vs Predictability")
    axes[2].scatter(sp["zero_pct"], sp["acf_lag7"].fillna(0), alpha=0.6, c=sp["mean_sales"], cmap="viridis")
    axes[2].set_xlabel("Zero-Day %")
    axes[2].set_ylabel("Lag-7 Autocorrelation")
    axes[2].set_title("Sparsity vs Predictability (color=volume)")
    plt.tight_layout()
    fig.savefig(os.path.join(OUT, "07_sparsity_predictability.png"), dpi=150)
    plt.close()
    print("→ saved 07_sparsity_predictability.png")

    return sp


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    print("v2_01: FRESH EDA FROM SCRATCH")
    print()
    df = load_raw()

    item_stats = section1_overview(df)
    full = section2_zero_inflation(df)
    daily = section3_time_patterns(full)
    daily_raw = section4_structural_break(full)
    items_lc = section5_item_lifecycle(full)
    sp = section6_sparsity(full)

    print("=" * 70)
    print("EDA COMPLETE — results in figures/v2_eda/ and tables/v2_*.csv")
    print("=" * 70)


if __name__ == "__main__":
    main()
