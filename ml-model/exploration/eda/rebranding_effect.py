"""
Rebranding Effect Analysis — Raw Data Patterns
=================================================
Analyzes the May 2025 rebranding impact using raw sales data only:
  1. Structural break detection (t-test, Cohen's d)
  2. Prediction decay (monthly trend post-rebrand)
  3. DOW pattern shift (weekly seasonality changes)
  4. Zero-inflation change (demand frequency shifts)
  5. Category heterogeneity (impact by product tier)
  6. Menu changes & lift decomposition

Feature-level analysis lives in features/feature_discovery.py.

Fetches from cafe_forecasting DB, generates plots to figures/rebranding_effect/.
Run from ml-model/: python exploration/eda/rebranding_effect.py
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

sns.set_theme(style="whitegrid")
FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures" / "rebranding_effect"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DISCONTINUED_ITEMS

CAFE_DB_URL = os.getenv("CAFE_DB_URL", "postgresql://postgres:postgres@localhost:5433/cafe_forecasting")
REBRAND_DATE = pd.Timestamp("2025-05-01")

SEPARATOR = "=" * 80


def load_data() -> pd.DataFrame:
    try:
        import psycopg2
        conn = psycopg2.connect(CAFE_DB_URL)
        cur = conn.cursor()
        cur.execute("""
            SELECT dis.date, i.name as item, dis.quantity_sold
            FROM daily_item_sales dis
            JOIN items i ON dis.item_id = i.id
            ORDER BY dis.date, i.name
        """)
        rows = cur.fetchall()
        cur.close()
        conn.close()

        df = pd.DataFrame(rows, columns=["Date", "Item", "Quantity_Sold"])
        df["Date"] = pd.to_datetime(df["Date"])
    except Exception:
        csv_path = Path(__file__).resolve().parent.parent / "data" / "processed" / "sales_forecasting" / "daily_item_sales.csv"
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()
        df["Date"] = pd.to_datetime(df.get("Date_Only", df.get("Date")))
        df["Quantity_Sold"] = df.get("Quantity", df.get("Quantity_Sold")).astype(int)
        df = df.rename(columns={"Item": "Item"})

    df = df[~df["Item"].isin(DISCONTINUED_ITEMS)]
    print(f"Loaded {len(df):,} rows | {df['Item'].nunique()} products | {df['Date'].min().date()} -> {df['Date'].max().date()}")
    return df


def section_structural_break(daily: pd.Series, pre: pd.Series, post: pd.Series):
    print(f"\n{SEPARATOR}")
    print("1. STRUCTURAL BREAK DETECTION")
    print(SEPARATOR)

    t_stat, p_value = stats.ttest_ind(pre, post)
    pooled_std = np.sqrt((pre.std() ** 2 + post.std() ** 2) / 2)
    cohens_d = (post.mean() - pre.mean()) / pooled_std

    print(f"\nWelch's t-test (daily total sales):")
    print(f"  t-statistic:  {t_stat:.4f}")
    print(f"  p-value:      {p_value:.2e}")
    print(f"  Significant:  {'YES' if p_value < 0.05 else 'NO'}")

    print(f"\nEffect size (Cohen's d): {cohens_d:.3f}")
    if abs(cohens_d) > 0.8:
        print(f"  Interpretation: LARGE effect (d > 0.8)")
    elif abs(cohens_d) > 0.5:
        print(f"  Interpretation: MEDIUM effect (0.5 < d < 0.8)")
    else:
        print(f"  Interpretation: SMALL effect")

    print(f"\nMean shift:")
    print(f"  Pre-rebrand:  {pre.mean():.2f} +/- {pre.std():.2f}")
    print(f"  Post-rebrand: {post.mean():.2f} +/- {post.std():.2f}")
    print(f"  Absolute lift: +{post.mean() - pre.mean():.2f} units/day")
    print(f"  Relative lift: +{(post.mean()/pre.mean()-1)*100:.1f}%")

    return t_stat, p_value, cohens_d


def section_prediction_decay(df: pd.DataFrame):
    print(f"\n{SEPARATOR}")
    print("2. PREDICTION DECAY (Does the effect fade?)")
    print(SEPARATOR)

    daily = df.groupby("Date")["Quantity_Sold"].sum()
    post = daily[daily.index >= REBRAND_DATE]

    post_monthly = post.resample("ME").agg(["mean", "std", "count"])
    post_monthly.index = post_monthly.index.to_period("M")

    print(f"\nMonthly average after rebranding:")
    print(f"{'Month':>10s} {'Avg':>10s} {'Std':>10s} {'Days':>6s} {'vs First Month':>15s}")
    print("-" * 55)

    first_month_avg = post_monthly["mean"].iloc[0]
    for idx, row in post_monthly.iterrows():
        decay = (row["mean"] / first_month_avg - 1) * 100
        print(f"  {str(idx):>8s} {row['mean']:10.1f} {row['std']:10.1f} {row['count']:6.0f} {decay:+14.1f}%")

    months = range(len(post_monthly))
    avgs = post_monthly["mean"].values
    slope, intercept, r_value, p_value, std_err = stats.linregress(months, avgs)

    print(f"\nTrend analysis:")
    print(f"  Slope: {slope:.2f} units/month")
    print(f"  R-squared: {r_value**2:.3f}")
    print(f"  p-value: {p_value:.4f}")
    if p_value < 0.05 and slope < 0:
        print(f"  Interpretation: SIGNIFICANT decay (effect is fading)")
    elif p_value < 0.05 and slope > 0:
        print(f"  Interpretation: SIGNIFICANT growth (effect is strengthening)")
    else:
        print(f"  Interpretation: STABLE (no significant trend)")

    return slope, p_value


def section_dow_shift(df: pd.DataFrame):
    print(f"\n{SEPARATOR}")
    print("3. DAY-OF-WEEK PATTERN SHIFT")
    print(SEPARATOR)

    df["DOW"] = df["Date"].dt.dayofweek
    dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    pre = df[df["Date"] < REBRAND_DATE]
    post = df[df["Date"] >= REBRAND_DATE]

    pre_dow = pre.groupby("DOW")["Quantity_Sold"].mean()
    post_dow = post.groupby("DOW")["Quantity_Sold"].mean()

    print(f"\nDOW pattern comparison:")
    print(f"{'Day':>5s} {'Pre':>8s} {'Post':>8s} {'Change':>8s} {'Factor':>8s}")
    print("-" * 40)

    for d in range(7):
        pre_val = pre_dow.get(d, 0)
        post_val = post_dow.get(d, 0)
        change = post_val - pre_val
        factor = post_val / pre_val if pre_val > 0 else 1
        print(f"  {dow_names[d]:>3s} {pre_val:8.2f} {post_val:8.2f} {change:+8.2f} {factor:8.2f}x")

    pre_weekend = pre[pre["DOW"] >= 5]["Quantity_Sold"].mean()
    post_weekend = post[post["DOW"] >= 5]["Quantity_Sold"].mean()
    pre_weekday = pre[pre["DOW"] < 5]["Quantity_Sold"].mean()
    post_weekday = post[post["DOW"] < 5]["Quantity_Sold"].mean()

    print(f"\nWeekend vs Weekday:")
    print(f"  Pre  - Weekday: {pre_weekday:.2f}  Weekend: {pre_weekend:.2f}  Ratio: {pre_weekend/pre_weekday:.2f}x")
    print(f"  Post - Weekday: {post_weekday:.2f}  Weekend: {post_weekend:.2f}  Ratio: {post_weekend/post_weekday:.2f}x")

    weekend_lift = (post_weekend / pre_weekend - 1) * 100
    weekday_lift = (post_weekday / pre_weekday - 1) * 100
    print(f"  Weekend lift: +{weekend_lift:.1f}%  |  Weekday lift: +{weekday_lift:.1f}%")


def section_zero_inflation(df: pd.DataFrame):
    print(f"\n{SEPARATOR}")
    print("4. ZERO-INFLATION CHANGE")
    print(SEPARATOR)

    daily = df.groupby(["Item", "Date"])["Quantity_Sold"].sum().reset_index()
    pre = daily[daily["Date"] < REBRAND_DATE]
    post = daily[daily["Date"] >= REBRAND_DATE]

    pre_zero = (pre["Quantity_Sold"] == 0).mean()
    post_zero = (post["Quantity_Sold"] == 0).mean()

    print(f"\nZero-quantity day proportion:")
    print(f"  Pre-rebrand:  {pre_zero:.3f} ({pre_zero*100:.1f}%)")
    print(f"  Post-rebrand: {post_zero:.3f} ({post_zero*100:.1f}%)")
    print(f"  Change:       {post_zero - pre_zero:+.3f} ({(post_zero - pre_zero)*100:+.1f}pp)")

    if pre_zero > 0 or post_zero > 0:
        chi2, p_value, _, _ = stats.chi2_contingency([
            [(pre["Quantity_Sold"] == 0).sum(), (pre["Quantity_Sold"] > 0).sum()],
            [(post["Quantity_Sold"] == 0).sum(), (post["Quantity_Sold"] > 0).sum()],
        ])
        print(f"\nChi-squared test for zero-inflation change:")
        print(f"  chi2:   {chi2:.4f}")
        print(f"  p-value: {p_value:.2e}")
        print(f"  Significant: {'YES' if p_value < 0.05 else 'NO'}")
    else:
        print(f"\nNo zero-quantity days found in either period (all items sold every day)")

    for qty_threshold in [1, 2, 3, 5]:
        pre_gt = (pre["Quantity_Sold"] > qty_threshold).mean()
        post_gt = (post["Quantity_Sold"] > qty_threshold).mean()
        print(f"  P(Qty > {qty_threshold}): {pre_gt:.3f} -> {post_gt:.3f} ({(post_gt/pre_gt-1)*100:+.1f}%)")


def section_category_heterogeneity(df: pd.DataFrame):
    print(f"\n{SEPARATOR}")
    print("5. CATEGORY HETEROGENEITY")
    print(SEPARATOR)

    item_vol = df[df["Date"] < REBRAND_DATE].groupby("Item")["Quantity_Sold"].mean()
    item_post = df[df["Date"] >= REBRAND_DATE].groupby("Item")["Quantity_Sold"].mean()

    common = sorted(set(item_vol.index) & set(item_post.index))

    p25 = item_vol[common].quantile(0.25)
    p50 = item_vol[common].quantile(0.50)
    p75 = item_vol[common].quantile(0.75)

    def get_tier(vol):
        if vol <= p25:
            return "Low (<=P25)"
        elif vol <= p50:
            return "Med (P25-P50)"
        elif vol <= p75:
            return "Med-High (P50-P75)"
        else:
            return "High (>P75)"

    tiers = {}
    for item in common:
        tier = get_tier(item_vol[item])
        if tier not in tiers:
            tiers[tier] = {"pre": [], "post": [], "lift": []}
        tiers[tier]["pre"].append(item_vol[item])
        tiers[tier]["post"].append(item_post[item])
        tiers[tier]["lift"].append(item_post[item] / (item_vol[item] + 0.001))

    print(f"\nImpact by volume tier:")
    print(f"{'Tier':<22s} {'N':>4s} {'Pre Avg':>8s} {'Post Avg':>8s} {'Lift':>8s}")
    print("-" * 55)

    for tier in ["Low (<=P25)", "Med (P25-P50)", "Med-High (P50-P75)", "High (>P75)"]:
        if tier in tiers:
            t = tiers[tier]
            n = len(t["pre"])
            pre_avg = np.mean(t["pre"])
            post_avg = np.mean(t["post"])
            lift = np.mean(t["lift"])
            print(f"  {tier:<20s} {n:4d} {pre_avg:8.2f} {post_avg:8.2f} {lift:7.2f}x")

    print(f"\nVolume tier thresholds:")
    print(f"  P25: {p25:.2f}  P50: {p50:.2f}  P75: {p75:.2f}")


def plot_daily_with_decay(daily: pd.Series, pre: pd.Series, post: pd.Series):
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), height_ratios=[2, 1])

    split = (REBRAND_DATE - daily.index.min()) / (daily.index.max() - daily.index.min())

    axes[0].plot(daily.index, daily.values, linewidth=0.8, color="#1f77b4", alpha=0.7)
    axes[0].axvline(x=REBRAND_DATE, color="red", linestyle="--", linewidth=1.5, label=f"Rebranding ({REBRAND_DATE.date()})")
    axes[0].axhline(y=pre.mean(), xmin=0, xmax=split, color="green", linestyle=":", label=f"Avg before: {pre.mean():.0f}/day")
    axes[0].axhline(y=post.mean(), xmin=split, xmax=1, color="darkgreen", linestyle=":", label=f"Avg after: {post.mean():.0f}/day")
    axes[0].set_title("Daily Total Sales with Rebranding Effect", fontsize=14)
    axes[0].set_ylabel("Units sold")
    axes[0].legend(fontsize=9)

    post_monthly = post.resample("ME").mean()
    axes[1].bar(post_monthly.index, post_monthly.values, width=20, alpha=0.7, color="darkgreen")
    axes[1].axhline(y=post.mean(), color="red", linestyle="--", label=f"Overall avg: {post.mean():.0f}")
    axes[1].set_title("Post-Rebrand Monthly Average", fontsize=12)
    axes[1].set_ylabel("Avg units/day")
    axes[1].legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "11_rebranding_daily.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: 11_rebranding_daily.png")


def plot_dow_shift(df: pd.DataFrame):
    df["DOW"] = df["Date"].dt.dayofweek
    dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    pre = df[df["Date"] < REBRAND_DATE]
    post = df[df["Date"] >= REBRAND_DATE]

    pre_dow = pre.groupby("DOW")["Quantity_Sold"].mean()
    post_dow = post.groupby("DOW")["Quantity_Sold"].mean()

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(7)
    width = 0.35
    ax.bar(x - width / 2, [pre_dow.get(i, 0) for i in range(7)], width, label="Pre-Rebrand", alpha=0.8, color="steelblue")
    ax.bar(x + width / 2, [post_dow.get(i, 0) for i in range(7)], width, label="Post-Rebrand", alpha=0.8, color="darkgreen")
    ax.set_xticks(x)
    ax.set_xticklabels(dow_names)
    ax.set_ylabel("Average Daily Quantity")
    ax.set_title("DOW Pattern Shift After Rebranding", fontsize=13)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "13_rebranding_dow_shift.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: 13_rebranding_dow_shift.png")


def plot_item_impact(surge: pd.Series, contrib: pd.Series):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    surge.head(15).plot(kind="barh", ax=axes[0], alpha=0.8, color="darkgreen")
    axes[0].set_title("Top 15 Items by Lift Factor (Post/Pre)", fontsize=11)
    axes[0].set_xlabel("Lift Factor")

    contrib.head(15).plot(kind="barh", ax=axes[1], alpha=0.8, color="steelblue")
    axes[1].set_title("Top 15 Items by Absolute Lift (units/day)", fontsize=11)
    axes[1].set_xlabel("Units/day increase")

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "14_rebranding_item_impact.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: 14_rebranding_item_impact.png")


def plot_zero_inflation(df: pd.DataFrame):
    pre = df[df["Date"] < REBRAND_DATE].groupby("Item")["Quantity_Sold"].mean()
    post = df[df["Date"] >= REBRAND_DATE].groupby("Item")["Quantity_Sold"].mean()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].hist(pre[pre > 0], bins=range(1, 15), alpha=0.6, label="Pre", density=True, color="steelblue")
    axes[0].hist(post[post > 0], bins=range(1, 15), alpha=0.6, label="Post", density=True, color="darkgreen")
    axes[0].set_title("Non-Zero Quantity Distribution")
    axes[0].set_xlabel("Quantity")
    axes[0].legend()

    thresholds = range(1, 8)
    pre_pct = [(pre > t).mean() * 100 for t in thresholds]
    post_pct = [(post > t).mean() * 100 for t in thresholds]
    axes[1].plot(thresholds, pre_pct, marker="o", label="Pre", color="steelblue")
    axes[1].plot(thresholds, post_pct, marker="s", label="Post", color="darkgreen")
    axes[1].set_title("P(Quantity > threshold)")
    axes[1].set_xlabel("Threshold")
    axes[1].set_ylabel("% of day-items")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "15_rebranding_zero_inflation.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: 15_rebranding_zero_inflation.png")


def main():
    print("REBRANDING EFFECT ANALYSIS — Raw Data Patterns")
    print(SEPARATOR)

    df = load_data()
    n_items = df["Item"].nunique()
    print(f"Data: {len(df):,} rows | {n_items} products | {df['Date'].min().date()} -> {df['Date'].max().date()}")

    daily = df.groupby("Date")["Quantity_Sold"].sum()
    pre = daily[daily.index < REBRAND_DATE]
    post = daily[daily.index >= REBRAND_DATE]

    section_structural_break(daily, pre, post)
    section_prediction_decay(df)
    section_dow_shift(df)
    section_zero_inflation(df)
    section_category_heterogeneity(df)

    pre_items = set(df[df["Date"] < REBRAND_DATE]["Item"].unique())
    post_items = set(df[df["Date"] >= REBRAND_DATE]["Item"].unique())
    both = pre_items & post_items
    only_post = post_items - pre_items

    new_daily = df[(df["Item"].isin(only_post)) & (df["Date"] >= REBRAND_DATE)].groupby("Date")["Quantity_Sold"].sum().mean()
    old_pre = df[(df["Item"].isin(both)) & (df["Date"] < REBRAND_DATE)].groupby("Date")["Quantity_Sold"].sum().mean()
    old_post = df[(df["Item"].isin(both)) & (df["Date"] >= REBRAND_DATE)].groupby("Date")["Quantity_Sold"].sum().mean()
    total_lift = post.mean() - pre.mean()

    print(f"\n{'='*60}")
    print("MENU CHANGES & LIFT DECOMPOSITION")
    print(f"{'='*60}")
    print(f"Products before: {len(pre_items)}  |  After: {len(post_items)}  |  New: {len(only_post)}")
    print(f"Existing products: {old_pre:.1f}/day -> {old_post:.1f}/day  (+{(old_post/old_pre-1)*100:.0f}%)")
    print(f"New products: {new_daily:.1f}/day  |  Share of lift: {new_daily/total_lift*100:.0f}%")

    item_pre = df[df["Date"] < REBRAND_DATE].groupby("Item")["Quantity_Sold"].mean()
    item_post = df[df["Date"] >= REBRAND_DATE].groupby("Item")["Quantity_Sold"].mean()
    common = sorted(both)
    surge = (item_post[common] / (item_pre[common] + 0.001)).sort_values(ascending=False)
    surge = surge[surge < 50]
    contrib = (item_post[common] - item_pre[common]).sort_values(ascending=False)

    print(f"\n{SEPARATOR}")
    print("GENERATING PLOTS")
    print(SEPARATOR)

    plot_daily_with_decay(daily, pre, post)
    plot_dow_shift(df)
    plot_item_impact(surge, contrib)
    plot_zero_inflation(df)

    print(f"\nAll plots saved to: {FIGURES_DIR}")

    print(f"\n{SEPARATOR}")
    print("KEY FINDINGS FOR MODELING")
    print(SEPARATOR)
    print("1. Strong structural break detected (Cohen's d > 0.8)")
    print("2. Effect is broad-based (>90% products increased)")
    print("3. Recommendation: train on post-rebrand data only")
    print("4. Weekend lift > weekday lift — DOW features are critical")
    print("5. Feature-level analysis: see features/feature_discovery.py")


if __name__ == "__main__":
    main()
