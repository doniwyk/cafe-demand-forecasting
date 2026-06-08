"""
Pipeline Dataset Statistics
===========================
Compute actual data statistics for each stage of the forecasting pipeline.
These numbers are used in bab_v_5.1.md to replace inline code blocks.

Run: python exploration/eda/pipeline_dataset_stats.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from inference.forecast import load_all_items, _should_skip, MIN_NONZERO_DAYS

REBRAND_DATE = pd.Timestamp("2025-05-01")
SEP = "=" * 72


def dataset_overview(df: pd.DataFrame):
    """5.1.2.1 — Load & Filter: raw dataset stats."""
    print(f"\n{SEP}")
    print("TABLE 5.1 — LOAD & FILTER: DATASET STATISTICS")
    print(SEP)

    # --- Raw dataset ---
    print(f"\nA. RAW DATASET (from DB)")
    print(f"  Total rows:              {len(df):>8,}")
    print(f"  Unique items:            {df['Item'].nunique():>8}")
    print(f"  Date range:              {df['Date'].min().date()}  to  {df['Date'].max().date()}")
    print(f"  Calendar days spanned:   {(df['Date'].max() - df['Date'].min()).days:>8,}")
    print(f"  Days with data:          {df['Date'].dt.date.nunique():>8,}")
    print(f"  Total quantity sold:     {df['Quantity_Sold'].sum():>8,}")
    print(f"  Mean per row:            {df['Quantity_Sold'].mean():>8.2f}")
    print(f"  Median:                  {df['Quantity_Sold'].median():>8.0f}")
    print(f"  Std:                     {df['Quantity_Sold'].std():>8.2f}")

    # Quantity distribution
    qty = df["Quantity_Sold"]
    print(f"\nB. QUANTITY DISTRIBUTION (all rows)")
    print(f"  {'Qty':>4s}  {'Count':>7s}  {'Pct':>6s}  {'Cumul':>6s}")
    print(f"  {'-'*28}")
    cumul = 0
    for val in range(0, 12):
        count = (qty == val).sum()
        if count == 0:
            continue
        pct = count / len(qty) * 100
        cumul += pct
        print(f"  {val:>4d}  {count:>7,d}  {pct:>5.1f}%  {cumul:>5.1f}%")
    count_11plus = (qty >= 11).sum()
    pct_11plus = count_11plus / len(qty) * 100
    cumul += pct_11plus
    print(f"  {'11+':>4s}  {count_11plus:>7,d}  {pct_11plus:>5.1f}%  {cumul:>5.1f}%")

    # Items
    items_with_zeros = df[df["Quantity_Sold"] == 0]["Item"].nunique()
    print(f"\nC. ITEMS")
    print(f"  Items with zero rows:    {items_with_zeros}")
    # Check _should_skip
    skipped_items = [i for i in df["Item"].unique() if _should_skip(i)]
    print(f"  Items flagged _should_skip: {len(skipped_items)} ({', '.join(skipped_items) if skipped_items else 'none'})")


def post_rebrand_stats(df: pd.DataFrame):
    """Stats after filtering to post-rebrand only."""
    print(f"\n{SEP}")
    print("POST-REBRAND DATA (>= 2025-05-01)")
    print(SEP)

    post = df[df["Date"] >= REBRAND_DATE].copy()

    print(f"\nA. OVERVIEW")
    print(f"  Rows:                    {len(post):>8,}")
    print(f"  Items:                   {post['Item'].nunique():>8}")
    print(f"  Date range:              {post['Date'].min().date()}  to  {post['Date'].max().date()}")
    print(f"  Days:                    {post['Date'].dt.date.nunique():>8,}")
    print(f"  Mean daily qty:          {post['Quantity_Sold'].mean():>8.2f}")
    print(f"  Median:                  {post['Quantity_Sold'].median():>8.0f}")

    # Non-zero days per item
    nz = post[post["Quantity_Sold"] > 0].groupby("Item").size()
    print(f"\nB. NON-ZERO DAYS PER ITEM (post-rebrand)")
    print(f"  Min:          {nz.min():>4d}  ({nz.idxmin()})")
    print(f"  Max:          {nz.max():>4d}  ({nz.idxmax()})")
    print(f"  Median:       {nz.median():>4.0f}")
    print(f"  Mean:         {nz.mean():>4.0f}")
    print(f"  Items >= {MIN_NONZERO_DAYS}:  {(nz >= MIN_NONZERO_DAYS).sum():>4d} / {len(nz)}")

    below = nz[nz < MIN_NONZERO_DAYS]
    if len(below) > 0:
        print(f"  Items below threshold:")
        for item, days in below.sort_values().items():
            print(f"    {item:<35s}  {days} non-zero days")

    # Final filtered
    valid = nz[nz >= MIN_NONZERO_DAYS].index
    final = post[post["Item"].isin(valid)]
    print(f"\nC. FINAL PIPELINE INPUT (post-rebrand + MIN_NONZERO_DAYS)")
    print(f"  Rows:                    {len(final):>8,}")
    print(f"  Items:                   {final['Item'].nunique():>8}")
    print(f"  Mean qty:                {final['Quantity_Sold'].mean():>8.2f}")
    print(f"  Median qty:              {final['Quantity_Sold'].median():>8.0f}")

    # Top 10 items
    item_totals = final.groupby("Item")["Quantity_Sold"].agg(["sum", "mean", "median", "std", "count"])
    item_totals["mean"] = item_totals["mean"].round(2)
    item_totals["std"] = item_totals["std"].round(2)
    item_totals = item_totals.sort_values("sum", ascending=False)

    print(f"\nD. TOP 10 ITEMS BY TOTAL SALES (post-rebrand, final set)")
    print(f"  {'Item':<33s}  {'Sum':>6s}  {'Mean':>6s}  {'Med':>4s}  {'Std':>5s}  {'Days':>5s}")
    print(f"  {'-'*65}")
    for i, (name, row) in enumerate(item_totals.head(10).iterrows(), 1):
        print(f"  {i:2d}. {name:<30s}  {row['sum']:>6.0f}  {row['mean']:>6.2f}  {row['median']:>4.0f}  {row['std']:>5.2f}  {row['count']:>5.0f}")

    print(f"\nE. BOTTOM 10 ITEMS")
    for i, (name, row) in enumerate(item_totals.tail(10).iterrows(), 1):
        print(f"  {i:2d}. {name:<30s}  {row['sum']:>6.0f}  {row['mean']:>6.2f}  {row['median']:>4.0f}  {row['std']:>5.2f}  {row['count']:>5.0f}")


def dow_statistics(df: pd.DataFrame):
    """5.1.2.2 — DOW statistics across all items (post-rebrand)."""
    print(f"\n{SEP}")
    print("TABLE 5.2 — DAY-OF-WEEK STATISTICS (post-rebrand, final items)")
    print(SEP)

    post = df[df["Date"] >= REBRAND_DATE].copy()
    nz = post[post["Quantity_Sold"] > 0].groupby("Item").size()
    valid = nz[nz >= MIN_NONZERO_DAYS].index
    final = post[post["Item"].isin(valid)].copy()
    final["DOW"] = final["Date"].dt.dayofweek

    dow_names = {0: "Senin", 1: "Selasa", 2: "Rabu", 3: "Kamis",
                 4: "Jumat", 5: "Sabtu", 6: "Minggu"}

    print(f"\nA. FULL DATA (including zero-qty days)")
    print(f"  {'Hari':<10s}  {'N':>6s}  {'Mean':>8s}  {'Median':>6s}  {'P75':>6s}  {'P90':>6s}  {'Std':>6s}  {'Zero%':>7s}")
    print(f"  {'-'*62}")
    for d in range(7):
        sub = final[final["DOW"] == d]
        mean = sub["Quantity_Sold"].mean()
        zero_pct = (sub["Quantity_Sold"] == 0).mean() * 100
        print(f"  {dow_names[d]:<10s}  {len(sub):>6,}  {mean:>8.2f}  {sub['Quantity_Sold'].median():>6.0f}  "
              f"{sub['Quantity_Sold'].quantile(0.75):>6.0f}  {sub['Quantity_Sold'].quantile(0.90):>6.0f}  "
              f"{sub['Quantity_Sold'].std():>6.2f}  {zero_pct:>6.2f}%")

    print(f"\nB. NON-ZERO ONLY")
    print(f"  {'Hari':<10s}  {'N':>6s}  {'Mean':>8s}  {'Median':>6s}  {'P75':>6s}  {'P90':>6s}  {'P95':>6s}  {'Std':>6s}")
    print(f"  {'-'*62}")
    for d in range(7):
        sub = final[(final["DOW"] == d) & (final["Quantity_Sold"] > 0)]
        if len(sub) == 0:
            continue
        print(f"  {dow_names[d]:<10s}  {len(sub):>6,}  {sub['Quantity_Sold'].mean():>8.2f}  "
              f"{sub['Quantity_Sold'].median():>6.0f}  {sub['Quantity_Sold'].quantile(0.75):>6.0f}  "
              f"{sub['Quantity_Sold'].quantile(0.90):>6.0f}  {sub['Quantity_Sold'].quantile(0.95):>6.0f}  "
              f"{sub['Quantity_Sold'].std():>6.2f}")

    # Representative item DOW stats
    top_item = "Kopi Susu Husgendam Ice"
    item_df = final[final["Item"] == top_item]
    print(f"\nC. DOW STATS FOR REPRESENTATIVE ITEM: {top_item}")
    print(f"  {'Hari':<10s}  {'N':>5s}  {'Mean':>8s}  {'Med':>6s}  {'P75':>6s}  {'P90':>6s}  {'P95':>6s}  {'Std':>6s}")
    print(f"  {'-'*62}")
    for d in range(7):
        sub = item_df[(item_df["DOW"] == d) & (item_df["Quantity_Sold"] > 0)]
        if len(sub) == 0:
            continue
        print(f"  {dow_names[d]:<10s}  {len(sub):>5d}  {sub['Quantity_Sold'].mean():>8.2f}  "
              f"{sub['Quantity_Sold'].median():>6.0f}  {sub['Quantity_Sold'].quantile(0.75):>6.0f}  "
              f"{sub['Quantity_Sold'].quantile(0.90):>6.0f}  {sub['Quantity_Sold'].quantile(0.95):>6.0f}  "
              f"{sub['Quantity_Sold'].std():>6.2f}")

    # Summary sentence
    dow_avg = final.groupby("DOW")["Quantity_Sold"].mean()
    min_dow = dow_avg.idxmin()
    max_dow = dow_avg.idxmax()
    print(f"\nD. SUMMARY")
    print(f"  Lowest avg DOW:  {dow_names[min_dow]} ({dow_avg[min_dow]:.2f})")
    print(f"  Highest avg DOW: {dow_names[max_dow]} ({dow_avg[max_dow]:.2f})")
    print(f"  Weekend ratio:   {dow_avg[[5,6]].mean() / dow_avg[[0,1,2,3,4]].mean():.2f}x")


def training_data_stats(df: pd.DataFrame):
    """5.1.3 — Training data per representative item."""
    print(f"\n{SEP}")
    print("TABLE 5.4 — TRAINING DATA: REPRESENTATIVE ITEM STATS")
    print(SEP)

    post = df[df["Date"] >= REBRAND_DATE].copy()
    nz = post[post["Quantity_Sold"] > 0].groupby("Item").size()
    valid = nz[nz >= MIN_NONZERO_DAYS].index
    final = post[post["Item"].isin(valid)].copy()

    top_item = "Kopi Susu Husgendam Ice"
    item_df = final[final["Item"] == top_item].copy()
    item_df["DOW"] = item_df["Date"].dt.dayofweek

    print(f"\nA. REPRESENTATIVE ITEM: {top_item}")
    print(f"  Total rows:          {len(item_df):>6}")
    print(f"  Non-zero rows:       {(item_df['Quantity_Sold'] > 0).sum():>6}")
    print(f"  Date range:          {item_df['Date'].min().date()}  to  {item_df['Date'].max().date()}")
    print(f"  Mean daily qty:      {item_df['Quantity_Sold'].mean():>6.2f}")
    print(f"  Median:              {item_df['Quantity_Sold'].median():>6.0f}")
    print(f"  Std:                 {item_df['Quantity_Sold'].std():>6.2f}")
    print(f"  Min:                 {item_df['Quantity_Sold'].min():>6}")
    print(f"  Max:                 {item_df['Quantity_Sold'].max():>6}")
    print(f"  Fri/Sat rows:        {item_df[item_df['DOW'].isin([4,5])].shape[0]:>6} "
          f"({item_df[item_df['DOW'].isin([4,5])].shape[0]/len(item_df)*100:.0f}%)")
    print(f"  Fri/Sat upweight:    3.0x")

    # For all items — training size summary
    print(f"\nB. TRAINING SIZE ACROSS ALL {final['Item'].nunique()} ITEMS")
    item_sizes = final.groupby("Item").size()
    print(f"  Min training rows:   {item_sizes.min():>6}  ({item_sizes.idxmin()})")
    print(f"  Max training rows:   {item_sizes.max():>6}  ({item_sizes.idxmax()})")
    print(f"  Median training rows:{item_sizes.median():>6.0f}")
    item_nonzero = final[final["Quantity_Sold"] > 0].groupby("Item").size()
    print(f"  Median non-zero rows:{item_nonzero.median():>6.0f}")


def main():
    print("=" * 72)
    print("  PIPELINE DATASET STATISTICS")
    print("  Generates the numbers used for bab_v_5.1.md tables")
    print("=" * 72)

    df = load_all_items()

    dataset_overview(df)
    post_rebrand_stats(df)
    dow_statistics(df)
    training_data_stats(df)

    print(f"\n{SEP}")
    print("DONE")
    print(SEP)


if __name__ == "__main__":
    main()
