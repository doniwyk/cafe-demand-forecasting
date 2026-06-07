"""
Feature Discovery (Updated with Rebranding Awareness)
=====================================================
Day-1 feature exploration: discover what patterns exist in the data,
then propose and evaluate candidate features for forecasting.
Includes rebranding-aware analysis based on EDA findings.

Key additions from EDA:
  - Structural break at May 2025 (Cohen's d = 1.675)
  - Feature distributions shifted significantly post-rebranding
  - Effect is strengthening (not fading)
  - Weekend lift > weekday lift

Fetches directly from hus_db (POS) when available, falls back to CSV.
Generates plots to figures/feature_discovery/.

Run from ml-model/: python exploration/features/feature_discovery.py
"""

import os
import sys
from pathlib import Path
from datetime import timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor
from scipy import stats

sns.set_theme(style="whitegrid")
FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures" / "feature_discovery"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import SALES_FORECASTING_DIR
from features import _split_train_val

SEPARATOR = "=" * 80

HUS_DB_URL = os.getenv("HUS_DB_URL", "postgresql://user:password@localhost:5432/hus_db")
REBRAND_DATE = pd.Timestamp("2025-05-01")

SKIP_PREFIXES = [
    "Add ", "Filter", "FIlter", "V60", "Harum Jasmine Tea",
    "Cookies Redvelvet", "Lotus Cheesecake", "Strawberry Cheesecake",
    "Kopi Susu Bersemi",
]
FALLBACK_VARIANT_MAP = {
    "Kopi Susu Husgendam": "Kopi Susu Husgendam Ice",
    "Cappucino": "Cappucino Ice",
}


def _should_skip(name: str) -> bool:
    return any(name.startswith(p) for p in SKIP_PREFIXES)


def _match_item(product_name: str, variant_name: str | None, cafe_items: set) -> str | None:
    product_name = (product_name or "").strip()
    variant_name = (variant_name or "").strip() if variant_name else None

    combined = f"{product_name} {variant_name}".strip() if variant_name else product_name
    if combined in cafe_items:
        return combined

    if product_name in cafe_items:
        return product_name

    if product_name in FALLBACK_VARIANT_MAP:
        fallback = FALLBACK_VARIANT_MAP[product_name]
        if fallback in cafe_items:
            return fallback

    return None


def load_from_hus_db() -> pd.DataFrame | None:
    try:
        import psycopg2
    except ImportError:
        print("psycopg2 not installed, cannot connect to hus_db")
        return None

    try:
        conn = psycopg2.connect(HUS_DB_URL)
    except Exception as e:
        print(f"Cannot connect to hus_db: {e}")
        return None

    print(f"Connected to hus_db, fetching sales data...")

    csv_path = SALES_FORECASTING_DIR / "daily_item_sales.csv"
    if csv_path.exists():
        csv_df = pd.read_csv(csv_path)
        csv_df.columns = csv_df.columns.str.strip()
        date_col = "Date_Only" if "Date_Only" in csv_df.columns else "Date"
        last_date = pd.to_datetime(csv_df[date_col]).max()
        since = (last_date + timedelta(days=1)).strftime("%Y-%m-%d")
        cafe_items = set(csv_df["Item"].str.strip().unique())
    else:
        since = "2022-01-01"
        cafe_items = set()

    cur = conn.cursor()
    cur.execute("""
        SELECT
            DATE(o.created_at) as sale_date,
            oi.product_name_snapshot,
            oi.variant_name_snapshot,
            SUM(oi.quantity) as total_qty
        FROM order_items oi
        JOIN orders o ON oi.order_id = o.id
        WHERE o.status = 'PAID'
          AND o.created_at >= %s
        GROUP BY DATE(o.created_at), oi.product_name_snapshot, oi.variant_name_snapshot
        ORDER BY sale_date
    """, (since,))
    rows = cur.fetchall()
    cur.close()
    conn.close()

    print(f"Fetched {len(rows)} rows from hus_db (since {since})")

    if not rows:
        if csv_path.exists():
            print("No new data, using CSV")
            return None
        print("No data in hus_db and no CSV found")
        return None

    matched_rows = []
    skipped = {}
    for sale_date, product_name, variant_name, qty in rows:
        item = _match_item(product_name, variant_name, cafe_items)
        if item is None:
            key = f"{product_name} {variant_name or ''}".strip()
            skipped[key] = skipped.get(key, 0) + int(qty)
            continue
        matched_rows.append({"Date": sale_date, "Item": item, "Quantity_Sold": int(qty)})

    print(f"Matched: {len(matched_rows)} rows")
    if skipped:
        print(f"Skipped: {sum(skipped.values())} units ({len(skipped)} products)")

    hus_df = pd.DataFrame(matched_rows)
    hus_df["Date"] = pd.to_datetime(hus_df["Date"])

    if csv_path.exists():
        csv_df = pd.read_csv(csv_path)
        csv_df.columns = csv_df.columns.str.strip()
        date_col = "Date_Only" if "Date_Only" in csv_df.columns else "Date"
        csv_df["Date"] = pd.to_datetime(csv_df[date_col])
        csv_df = csv_df.rename(columns={"Quantity": "Quantity_Sold"})
        csv_df = csv_df[~csv_df["Item"].str.strip().str.lower().str.startswith("add")]

        combined = pd.concat([csv_df, hus_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["Date", "Item"], keep="last")
        combined = combined.sort_values(["Date", "Item"]).reset_index(drop=True)
        print(f"Combined: {len(combined)} rows ({csv_df.shape[0]} CSV + {len(hus_df)} new)")
        return combined
    else:
        return hus_df


def load_and_prep_data(filepath: str | Path) -> pd.DataFrame:
    hus_df = load_from_hus_db()
    if hus_df is not None:
        df = hus_df.copy()
    else:
        print(f"Loading data from: {filepath}")
        df = pd.read_csv(filepath)
        df.columns = df.columns.str.strip()

        date_col = "Date_Only" if "Date_Only" in df.columns else "Date"
        qty_col = "Quantity" if "Quantity" in df.columns else "Quantity_Sold"
        df["Date"] = pd.to_datetime(df[date_col])
        df["Quantity_Sold"] = df[qty_col]
        df = df[~df["Item"].str.strip().str.lower().str.startswith("add")]

    df_freq = (
        df.set_index("Date")
        .groupby("Item")
        .resample("D")["Quantity_Sold"]
        .sum()
        .reset_index()
    )

    print(f"Aggregated to daily: {len(df_freq)} observations")
    print(f"Date range: {df_freq['Date'].min().date()} to {df_freq['Date'].max().date()}")
    return df_freq


def section_target_analysis(df: pd.DataFrame):
    print(f"\n{SEPARATOR}")
    print("TARGET ANALYSIS: What are we predicting?")
    print(SEPARATOR)

    qty = df["Quantity_Sold"]

    print(f"\nDistribution:")
    print(f"  Count:  {len(qty):,}")
    print(f"  Mean:   {qty.mean():.2f}")
    print(f"  Median: {qty.median():.2f}")
    print(f"  Std:    {qty.std():.2f}")
    print(f"  Min:    {qty.min():.0f}")
    print(f"  Max:    {qty.max():.0f}")

    quantiles = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    print(f"\nQuantiles:")
    for q in quantiles:
        print(f"  {q*100:5.1f}%: {qty.quantile(q):.0f}")

    max_val = int(qty.max())
    if max_val <= 20:
        edges = [0, 1, 2, 3, 5, 10, max_val + 1]
        labels = ["0", "1", "2", "3", "4-5", "6-10"]
    elif max_val <= 50:
        edges = [0, 1, 2, 3, 5, 10, 20, max_val + 1]
        labels = ["0", "1", "2", "3", "4-5", "6-10", "11-20"]
    else:
        edges = [0, 1, 2, 3, 5, 10, 20, 50, max_val + 1]
        labels = ["0", "1", "2", "3", "4-5", "6-10", "11-20", "21-50"]
    qty_binned = pd.cut(qty, bins=edges, labels=labels, right=False)
    dist = qty_binned.value_counts().sort_index()
    print(f"\nValue distribution:")
    for label, count in dist.items():
        pct = count / len(qty) * 100
        bar = "#" * int(pct / 2)
        print(f"  {label:>8s}: {count:6,} ({pct:5.1f}%)  {bar}")

    print(f"\nKey observation: {((qty == 0).sum() / len(qty) * 100):.1f}% of day-item combos are zero.")
    print(f"This is a sparse demand problem — most items sell 0-3 units per day.")


def section_autocorrelation(df: pd.DataFrame, max_lag: int = 30):
    print(f"\n{SEPARATOR}")
    print("AUTOCORRELATION: How does past predict future?")
    print(SEPARATOR)

    item_vols = df.groupby("Item")["Quantity_Sold"].sum().sort_values(ascending=False)
    top_items = item_vols.head(5).index.tolist()

    print(f"\nAnalyzing top 5 items by volume: {top_items}")
    print(f"\nLag autocorrelation (Pearson r between qty[t] and qty[t-lag]):")
    print(f"{'Lag':>5s}", end="")
    for item in top_items:
        print(f"  {item[:12]:>12s}", end="")
    print()

    for lag in [1, 2, 3, 4, 5, 7, 14, 21, 28]:
        print(f"  {lag:3d}", end="")
        for item in top_items:
            item_df = df[df["Item"] == item].sort_values("Date")
            q = item_df["Quantity_Sold"].values
            if len(q) > lag:
                r = np.corrcoef(q[lag:], q[:-lag])[0, 1]
                print(f"  {r:12.4f}", end="")
            else:
                print(f"  {'N/A':>12s}", end="")
        print()

    print(f"\nKey observations:")
    print(f"  - Lag-1 is typically the strongest predictor (yesterday predicts today)")
    print(f"  - Lag-7 captures weekly seasonality (same day last week)")
    print(f"  - Decay pattern shows how quickly demand reverts to mean")


def section_day_of_week_patterns(df: pd.DataFrame):
    print(f"\n{SEPARATOR}")
    print("DAY-OF-WEEK PATTERNS: Is there weekly seasonality?")
    print(SEPARATOR)

    item_vols = df.groupby("Item")["Quantity_Sold"].sum().sort_values(ascending=False)
    top_items = item_vols.head(5).index.tolist()

    df["DOW"] = df["Date"].dt.dayofweek
    dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    print(f"\nDOW factor by item (ratio of day avg to overall avg):")
    print(f"{'Item':<25s}", end="")
    for d in range(7):
        print(f"  {dow_names[d]:>5s}", end="")
    print()

    for item in top_items:
        item_df = df[df["Item"] == item]
        overall_avg = item_df["Quantity_Sold"].mean()
        dow_avg = item_df.groupby("DOW")["Quantity_Sold"].mean()
        print(f"  {item[:23]:<23s}", end="")
        for d in range(7):
            factor = dow_avg.get(d, overall_avg) / overall_avg if overall_avg > 0 else 1
            print(f"  {factor:5.2f}", end="")
        print()

    print(f"\nKey observations:")
    print(f"  - Factors > 1.0 = above-average day, < 1.0 = below-average day")
    print(f"  - DOW factors can be used as post-processing adjustments")


def section_rolling_window_analysis(df: pd.DataFrame):
    print(f"\n{SEPARATOR}")
    print("ROLLING WINDOW ANALYSIS: What windows capture demand patterns?")
    print(SEPARATOR)

    item_vols = df.groupby("Item")["Quantity_Sold"].sum().sort_values(ascending=False)
    top_items = item_vols.head(3).index.tolist()

    windows = [3, 5, 7, 14, 21, 28, 60]

    print(f"\nRolling mean correlation with target (qty[t]) at different windows:")
    print(f"{'Window':>8s}", end="")
    for item in top_items:
        print(f"  {item[:12]:>12s}", end="")
    print(f"  {'Average':>12s}")
    print("-" * (8 + 14 * (len(top_items) + 1)))

    for w in windows:
        corrs = []
        print(f"  {w:6d}", end="")
        for item in top_items:
            item_df = df[df["Item"] == item].sort_values("Date")
            q = item_df["Quantity_Sold"]
            rolled = q.shift(1).rolling(w, min_periods=1).mean()
            valid = ~(rolled.isna() | q.isna())
            if valid.sum() > 10:
                r = q[valid].corr(rolled[valid])
                corrs.append(r)
                print(f"  {r:12.4f}", end="")
            else:
                print(f"  {'N/A':>12s}", end="")
        if corrs:
            print(f"  {np.mean(corrs):12.4f}")
        else:
            print()

    print(f"\nRolling std correlation with target:")
    print(f"{'Window':>8s}", end="")
    for item in top_items:
        print(f"  {item[:12]:>12s}", end="")
    print(f"  {'Average':>12s}")
    print("-" * (8 + 14 * (len(top_items) + 1)))

    for w in windows:
        corrs = []
        print(f"  {w:6d}", end="")
        for item in top_items:
            item_df = df[df["Item"] == item].sort_values("Date")
            q = item_df["Quantity_Sold"]
            rolled_std = q.shift(1).rolling(w, min_periods=1).std()
            valid = ~(rolled_std.isna() | q.isna())
            if valid.sum() > 10:
                r = q[valid].corr(rolled_std[valid])
                corrs.append(r)
                print(f"  {r:12.4f}", end="")
            else:
                print(f"  {'N/A':>12s}", end="")
        if corrs:
            print(f"  {np.mean(corrs):12.4f}")
        else:
            print()

    print(f"\nKey observations:")
    print(f"  - Shorter windows (3-7) capture recent demand level")
    print(f"  - Longer windows (28-60) capture baseline/trend")
    print(f"  - Rolling std captures demand volatility (uncertainty)")


def section_trend_features(df: pd.DataFrame):
    print(f"\n{SEPARATOR}")
    print("TREND FEATURES: Differences and acceleration")
    print(SEPARATOR)

    item_vols = df.groupby("Item")["Quantity_Sold"].sum().sort_values(ascending=False)
    top_items = item_vols.head(5).index.tolist()

    print(f"\nFirst-order difference (qty[t] - qty[t-1]) correlation with target:")
    for item in top_items:
        item_df = df[df["Item"] == item].sort_values("Date")
        q = item_df["Quantity_Sold"]
        diff1 = q.diff(1)
        valid = ~(diff1.isna() | q.isna())
        if valid.sum() > 10:
            r = q[valid].corr(diff1[valid])
            print(f"  {item:<35s}  r={r:+.4f}")

    print(f"\nSecond-order difference (acceleration) correlation with target:")
    for item in top_items:
        item_df = df[df["Item"] == item].sort_values("Date")
        q = item_df["Quantity_Sold"]
        accel = q.diff(1).diff(1)
        valid = ~(accel.isna() | q.isna())
        if valid.sum() > 10:
            r = q[valid].corr(accel[valid])
            print(f"  {item:<35s}  r={r:+.4f}")

    print(f"\nKey observations:")
    print(f"  - Diff_1 captures short-term trend direction")
    print(f"  - Acceleration captures change in momentum")
    print(f"  - These are raw signals that XGBoost can split on non-linearly")


def section_proposed_features(df: pd.DataFrame):
    print(f"\n{SEPARATOR}")
    print("PROPOSED FEATURE SET (Updated with Rebranding Awareness)")
    print(SEPARATOR)

    print(f"""
Based on the analysis above and EDA findings, here are candidate features:

TREND FEATURES (capture direction and momentum):
  1. Lag_1          qty[t-1] — strongest single predictor
  2. Diff_1         qty[t] - qty[t-1] — short-term trend direction
  3. Accel_2        diff(t) - diff(t-1) — change in momentum

SMOOTHING FEATURES (capture baseline level):
  4. Roll_Mean_7    7-day rolling mean of lagged values — weekly baseline
  5. Roll_Mean_28   28-day rolling mean of lagged values — monthly baseline
  6. EWMA_7         Exponential weighted MA (span=7) — recent-weighted baseline
  7. EWMA_28        Exponential weighted MA (span=28) — longer-term baseline

VOLATILITY FEATURES (capture demand uncertainty):
  8. Roll_Std_7     7-day rolling std — short-term demand volatility
  9. Roll_Q95_7     7-day rolling 95th percentile — demand spike ceiling

SEASONAL FEATURES (capture weekly pattern):
  10. Seasonal_Strength  qty[t-1] / (qty[t-4] + 1) - 1 — weekly seasonality ratio

REBRANDING FEATURES (from EDA structural break analysis):
  11. IsPostRebrand      Binary: 1 if date >= 2025-05-01, else 0
  12. MonthsSinceRebrand Months since rebranding (0 before, 0.33/0.67/1.0/...)

TOTAL: 12 features

Key findings from EDA:
  - Structural break detected (Cohen's d = 1.675, p < 1e-188)
  - Effect is STRENGTHENING (slope: +2.96 units/month)
  - Weekend lift (+27%) > Weekday lift (+21%)
  - All feature distributions shifted significantly post-rebranding
""")


def section_feature_correlations(df: pd.DataFrame):
    print(f"\n{SEPARATOR}")
    print("PROPOSED FEATURE CORRELATIONS (with Rebranding Features)")
    print(SEPARATOR)

    print(f"\nBuilding feature matrix...")
    data = df[["Item", "Date", "Quantity_Sold"]].copy().sort_values(["Item", "Date"]).reset_index(drop=True)

    for item in data["Item"].unique():
        mask = data["Item"] == item
        g = data.loc[mask, "Quantity_Sold"]

        data.loc[mask, "Lag_1"] = g.shift(1).values
        data.loc[mask, "Diff_1"] = g.diff(1).values
        data.loc[mask, "Accel_2"] = g.diff(1).diff(1).values

        shifted = g.shift(1)
        data.loc[mask, "Roll_Mean_7"] = shifted.rolling(7, min_periods=1).mean().values
        data.loc[mask, "Roll_Mean_28"] = shifted.rolling(28, min_periods=1).mean().values
        data.loc[mask, "Roll_Std_7"] = shifted.rolling(7, min_periods=1).std().values
        data.loc[mask, "Roll_Q95_7"] = shifted.rolling(7, min_periods=1).quantile(0.95).values
        data.loc[mask, "EWMA_7"] = shifted.ewm(span=7, adjust=False).mean().values
        data.loc[mask, "EWMA_28"] = shifted.ewm(span=28, adjust=False).mean().values

        g_lag1 = g.shift(1)
        g_lag4 = g.shift(4)
        data.loc[mask, "Seasonal_Strength"] = (g_lag1 / (g_lag4 + 1) - 1).values

    data["IsPostRebrand"] = (data["Date"] >= REBRAND_DATE).astype(int)
    data["MonthsSinceRebrand"] = ((data["Date"] - REBRAND_DATE).dt.days / 30.44).clip(lower=0)

    data = data.fillna(0).replace([np.inf, -np.inf], 0)

    features = [
        "Lag_1", "Diff_1", "Accel_2",
        "Roll_Mean_7", "Roll_Mean_28", "EWMA_7", "EWMA_28",
        "Roll_Std_7", "Roll_Q95_7",
        "Seasonal_Strength",
        "IsPostRebrand", "MonthsSinceRebrand",
    ]

    print(f"\nCorrelation with target (Quantity_Sold):")
    target_corr = data[features + ["Quantity_Sold"]].corr()["Quantity_Sold"].drop("Quantity_Sold").sort_values(key=abs, ascending=False)
    for feat, val in target_corr.items():
        bar = "+" * int(abs(val) * 30) if val > 0 else "-" * int(abs(val) * 30)
        print(f"  {feat:<22s} {val:+.4f}  {bar}")

    print(f"\nInter-feature correlations (|r| > 0.7):")
    corr = data[features].corr()
    high_corr = []
    for i, f1 in enumerate(features):
        for f2 in features[i + 1:]:
            r = corr.loc[f1, f2]
            if abs(r) > 0.7:
                high_corr.append((f1, f2, r))
    high_corr.sort(key=lambda x: abs(x[2]), reverse=True)

    if high_corr:
        for f1, f2, r in high_corr:
            print(f"  {f1:<22s} <-> {f2:<22s}  r={r:+.4f}")
    else:
        print(f"  None found.")

    return data, features


def section_feature_importance(data: pd.DataFrame, features: list):
    print(f"\n{SEPARATOR}")
    print("FEATURE IMPORTANCE (XGBoost global model)")
    print(SEPARATOR)

    target = "Quantity_Sold"
    train_data, val_data = _split_train_val(data)

    print(f"Training global model on {len(train_data):,} rows...")
    model = XGBRegressor(
        objective="count:poisson",
        n_estimators=300,
        learning_rate=0.03,
        max_depth=5,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.7,
        reg_alpha=0.5,
        reg_lambda=1.0,
        random_state=42,
        early_stopping_rounds=30,
    )
    model.fit(
        train_data[features],
        train_data[target],
        eval_set=[(val_data[features], val_data[target])],
        verbose=False,
    )

    importance = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)

    print(f"\nFeature importance (gain):")
    for feat, imp in importance.items():
        bar = "#" * int(imp / importance.max() * 40)
        print(f"  {feat:<22s} {imp:.4f}  {bar}")

    print(f"\nTop 3 features: {list(importance.head(3).index)}")
    print(f"Bottom 3 features: {list(importance.tail(3).index)}")

    print(f"\nKey observations:")
    print(f"  - Diff_1 and Lag_1 dominate — autoregressive signal is strongest")
    print(f"  - Rolling/EWMA features provide baseline context")
    print(f"  - Seasonal_Strength captures weekly pattern")
    print(f"  - XGBoost uses features non-linearly, so even low-importance features")
    print(f"    may contribute via interaction effects")

    return importance


def train_global_model(data: pd.DataFrame, features: list) -> pd.Series:
    """Train global model and return feature importance (for plot generation)."""
    target = "Quantity_Sold"
    train_data, val_data = _split_train_val(data)

    model = XGBRegressor(
        objective="count:poisson",
        n_estimators=300,
        learning_rate=0.03,
        max_depth=5,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.7,
        reg_alpha=0.5,
        reg_lambda=1.0,
        random_state=42,
        early_stopping_rounds=30,
    )
    model.fit(
        train_data[features],
        train_data[target],
        eval_set=[(val_data[features], val_data[target])],
        verbose=False,
    )

    return pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)


def section_per_item_importance(data: pd.DataFrame, features: list, top_n: int = 5):
    print(f"\n{SEPARATOR}")
    print(f"PER-ITEM FEATURE IMPORTANCE (top {top_n} items)")
    print(SEPARATOR)

    target = "Quantity_Sold"
    item_vols = data.groupby("Item")[target].sum().sort_values(ascending=False)
    top_items = item_vols.head(top_n).index.tolist()

    all_importances = {}
    for item in top_items:
        item_df = data[data["Item"] == item]
        if len(item_df) < 60:
            print(f"  {item}: insufficient data ({len(item_df)} rows), skipping")
            continue

        train_data, val_data = _split_train_val(item_df)
        model = XGBRegressor(
            objective="count:poisson",
            n_estimators=200,
            learning_rate=0.03,
            max_depth=4,
            random_state=42,
            early_stopping_rounds=20,
        )
        model.fit(
            train_data[features],
            train_data[target],
            eval_set=[(val_data[features], val_data[target])] if len(val_data) > 0 else None,
            verbose=False,
        )
        imp = pd.Series(model.feature_importances_, index=features)
        all_importances[item] = imp

    if not all_importances:
        print("No items had sufficient data.")
        return

    imp_df = pd.DataFrame(all_importances)
    avg_imp = imp_df.mean(axis=1).sort_values(ascending=False)

    print(f"\nAverage feature importance across {len(all_importances)} items:")
    for feat, imp in avg_imp.items():
        bar = "#" * int(imp / avg_imp.max() * 40)
        print(f"  {feat:<22s} {imp:.4f}  {bar}")

    print(f"\nPer-item breakdown:")
    header = f"{'Feature':<22s}" + "".join(f"{item[:12]:>14s}" for item in all_importances.keys())
    print(header)
    for feat in features:
        row = f"  {feat:<20s}"
        for item in all_importances.keys():
            row += f"  {imp_df.loc[feat, item]:12.4f}"
        print(row)


def section_rebranding_feature_analysis(data: pd.DataFrame, features: list):
    print(f"\n{SEPARATOR}")
    print("REBRANDING FEATURE ANALYSIS")
    print(SEPARATOR)

    pre = data[data["Date"] < REBRAND_DATE]
    post = data[data["Date"] >= REBRAND_DATE]

    print(f"\n1. Feature mean comparison (pre vs post rebranding):")
    print(f"{'Feature':<22s} {'Pre':>10s} {'Post':>10s} {'p-value':>10s}")
    print("-" * 55)

    for feat in features:
        if feat in ["IsPostRebrand", "MonthsSinceRebrand"]:
            continue
        pre_vals = pre[feat].dropna()
        post_vals = post[feat].dropna()
        if len(pre_vals) > 10 and len(post_vals) > 10:
            _, p = stats.ttest_ind(pre_vals, post_vals)
            print(f"  {feat:<20s} {pre_vals.mean():10.3f} {post_vals.mean():10.3f} {p:10.2e}")

    print(f"\n2. Model comparison: WITH vs WITHOUT rebranding features")
    target = "Quantity_Sold"
    base_features = [f for f in features if f not in ["IsPostRebrand", "MonthsSinceRebrand"]]

    train_data, val_data = _split_train_val(data)

    model_base = XGBRegressor(
        objective="count:poisson", n_estimators=200, learning_rate=0.03,
        max_depth=4, random_state=42, early_stopping_rounds=20,
    )
    model_base.fit(
        train_data[base_features], train_data[target],
        eval_set=[(val_data[base_features], val_data[target])],
        verbose=False,
    )
    base_rmse = np.sqrt(((val_data[target] - model_base.predict(val_data[base_features]))**2).mean())

    model_full = XGBRegressor(
        objective="count:poisson", n_estimators=200, learning_rate=0.03,
        max_depth=4, random_state=42, early_stopping_rounds=20,
    )
    model_full.fit(
        train_data[features], train_data[target],
        eval_set=[(val_data[features], val_data[target])],
        verbose=False,
    )
    full_rmse = np.sqrt(((val_data[target] - model_full.predict(val_data[features]))**2).mean())

    print(f"  Base model ({len(base_features)} features): RMSE = {base_rmse:.4f}")
    print(f"  Full model ({len(features)} features): RMSE = {full_rmse:.4f}")
    print(f"  Improvement: {(base_rmse - full_rmse) / base_rmse * 100:+.2f}%")

    print(f"\n3. Feature importance comparison (pre vs post rebranding)")
    imp_pre = {}
    imp_post = {}
    for feat in features:
        pre_corr = np.abs(pre[feat].corr(pre["Quantity_Sold"]))
        post_corr = np.abs(post[feat].corr(post["Quantity_Sold"]))
        imp_pre[feat] = pre_corr
        imp_post[feat] = post_corr

    print(f"{'Feature':<22s} {'Pre r':>10s} {'Post r':>10s} {'Delta':>10s}")
    print("-" * 55)
    for feat in features:
        delta = imp_post[feat] - imp_pre[feat]
        print(f"  {feat:<20s} {imp_pre[feat]:10.4f} {imp_post[feat]:10.4f} {delta:+10.4f}")


def section_dow_interaction(data: pd.DataFrame, features: list):
    print(f"\n{SEPARATOR}")
    print("DOW x REBRANDING INTERACTION")
    print(SEPARATOR)

    data["DOW"] = data["Date"].dt.dayofweek
    dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    pre = data[data["Date"] < REBRAND_DATE]
    post = data[data["Date"] >= REBRAND_DATE]

    pre_dow = pre.groupby("DOW")["Quantity_Sold"].mean()
    post_dow = post.groupby("DOW")["Quantity_Sold"].mean()

    print(f"\nDOW lift after rebranding:")
    print(f"{'Day':>5s} {'Pre':>8s} {'Post':>8s} {'Lift':>8s}")
    print("-" * 35)

    for d in range(7):
        lift = (post_dow.get(d, 1) / pre_dow.get(d, 1) - 1) * 100
        print(f"  {dow_names[d]:>3s} {pre_dow.get(d, 0):8.2f} {post_dow.get(d, 0):8.2f} {lift:+7.1f}%")

    pre_weekend = pre[pre["DOW"] >= 5]["Quantity_Sold"].mean()
    post_weekend = post[post["DOW"] >= 5]["Quantity_Sold"].mean()
    pre_weekday = pre[pre["DOW"] < 5]["Quantity_Sold"].mean()
    post_weekday = post[post["DOW"] < 5]["Quantity_Sold"].mean()

    weekend_lift = (post_weekend / pre_weekend - 1) * 100
    weekday_lift = (post_weekday / pre_weekday - 1) * 100

    print(f"\nWeekend lift: +{weekend_lift:.1f}%  |  Weekday lift: +{weekday_lift:.1f}%")
    print(f"Interaction: Weekend gets {weekend_lift - weekday_lift:+.1f}pp more lift")


def plot_target_distribution(df: pd.DataFrame):
    qty = df["Quantity_Sold"]
    qty_nonzero = qty[qty > 0]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    qty.value_counts().head(10).sort_index().plot(kind="bar", ax=axes[0], alpha=0.8)
    axes[0].set_title("Target Distribution (Quantity_Sold)")
    axes[0].set_xlabel("Quantity")
    axes[0].set_ylabel("Count")

    if len(qty_nonzero) > 0:
        qty_nonzero.hist(bins=30, ax=axes[1], alpha=0.8, edgecolor="black")
        axes[1].set_title("Non-Zero Quantity Distribution")
        axes[1].set_xlabel("Quantity")
        axes[1].set_ylabel("Count")

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "06_target_distribution.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: 06_target_distribution.png")


def plot_autocorrelation(df: pd.DataFrame, max_lag: int = 28):
    item_vols = df.groupby("Item")["Quantity_Sold"].sum().sort_values(ascending=False)
    top_items = item_vols.head(5).index.tolist()

    fig, ax = plt.subplots(figsize=(12, 6))
    lags = range(1, max_lag + 1)

    for item in top_items:
        item_df = df[df["Item"] == item].sort_values("Date")
        q = item_df["Quantity_Sold"].values
        corrs = []
        for lag in lags:
            if len(q) > lag:
                r = np.corrcoef(q[lag:], q[:-lag])[0, 1]
                corrs.append(r)
            else:
                corrs.append(np.nan)
        ax.plot(lags, corrs, marker="o", markersize=3, label=item[:20], linewidth=1.5)

    ax.set_title("Autocorrelation by Lag (Top 5 Items)", fontsize=14)
    ax.set_xlabel("Lag (days)")
    ax.set_ylabel("Correlation (r)")
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.legend(fontsize=8, loc="upper right")
    ax.set_xticks(range(0, max_lag + 1, 7))
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "07_autocorrelation.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: 07_autocorrelation.png")


def plot_rolling_correlation(df: pd.DataFrame):
    item_vols = df.groupby("Item")["Quantity_Sold"].sum().sort_values(ascending=False)
    top_items = item_vols.head(3).index.tolist()
    windows = [3, 5, 7, 14, 21, 28, 60]

    fig, ax = plt.subplots(figsize=(10, 6))

    for item in top_items:
        item_df = df[df["Item"] == item].sort_values("Date")
        q = item_df["Quantity_Sold"]
        corrs = []
        for w in windows:
            rolled = q.shift(1).rolling(w, min_periods=1).mean()
            valid = ~(rolled.isna() | q.isna())
            if valid.sum() > 10:
                r = q[valid].corr(rolled[valid])
                corrs.append(r)
            else:
                corrs.append(np.nan)
        ax.plot(windows, corrs, marker="o", label=item[:20], linewidth=1.5)

    ax.set_title("Rolling Mean Correlation with Target", fontsize=14)
    ax.set_xlabel("Window Size (days)")
    ax.set_ylabel("Correlation (r)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "08_rolling_correlation.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: 08_rolling_correlation.png")


def plot_feature_correlation_heatmap(data: pd.DataFrame, features: list):
    corr = data[features].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        ax=ax,
        square=True,
        linewidths=0.5,
        annot_kws={"size": 8},
    )
    ax.set_title("Feature Correlation Heatmap", fontsize=14)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "09_feature_correlation.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: 09_feature_correlation.png")


def plot_feature_importance(importance: pd.Series):
    fig, ax = plt.subplots(figsize=(10, 6))
    importance.sort_values().plot(kind="barh", ax=ax, alpha=0.8)
    ax.set_title("Global Feature Importance (XGBoost Gain)", fontsize=14)
    ax.set_xlabel("Importance")
    for i, v in enumerate(importance.sort_values()):
        ax.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "10_feature_importance.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: 10_feature_importance.png")


def plot_rebranding_feature_shift(data: pd.DataFrame):
    features = ["Lag_1", "Diff_1", "Roll_Mean_7", "EWMA_7", "Seasonal_Strength"]
    pre = data[data["Date"] < REBRAND_DATE]
    post = data[data["Date"] >= REBRAND_DATE]

    fig, axes = plt.subplots(1, 5, figsize=(18, 4))
    for i, feat in enumerate(features):
        pre_vals = pre[feat].clip(-10, 20)
        post_vals = post[feat].clip(-10, 20)
        axes[i].hist(pre_vals, bins=30, alpha=0.5, label="Pre", density=True, color="blue")
        axes[i].hist(post_vals, bins=30, alpha=0.5, label="Post", density=True, color="red")
        axes[i].set_title(feat, fontsize=10)
        axes[i].legend(fontsize=8)
    fig.suptitle("Feature Distribution Shift (Pre vs Post Rebrand)", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "11_rebranding_feature_shift.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: 11_rebranding_feature_shift.png")


def plot_model_comparison(data: pd.DataFrame, features: list):
    target = "Quantity_Sold"
    base_features = [f for f in features if f not in ["IsPostRebrand", "MonthsSinceRebrand"]]

    train_data, val_data = _split_train_val(data)

    model_base = XGBRegressor(
        objective="count:poisson", n_estimators=200, learning_rate=0.03,
        max_depth=4, random_state=42, early_stopping_rounds=20,
    )
    model_base.fit(train_data[base_features], train_data[target],
                   eval_set=[(val_data[base_features], val_data[target])], verbose=False)
    pred_base = model_base.predict(val_data[base_features])

    model_full = XGBRegressor(
        objective="count:poisson", n_estimators=200, learning_rate=0.03,
        max_depth=4, random_state=42, early_stopping_rounds=20,
    )
    model_full.fit(train_data[features], train_data[target],
                   eval_set=[(val_data[features], val_data[target])], verbose=False)
    pred_full = model_full.predict(val_data[features])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].scatter(pred_base, val_data[target], alpha=0.3, s=10)
    axes[0].plot([0, 10], [0, 10], "r--", linewidth=1)
    rmse_base = np.sqrt(((val_data[target] - pred_base)**2).mean())
    axes[0].set_title(f"Base Model (RMSE={rmse_base:.3f})")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Actual")

    axes[1].scatter(pred_full, val_data[target], alpha=0.3, s=10)
    axes[1].plot([0, 10], [0, 10], "r--", linewidth=1)
    rmse_full = np.sqrt(((val_data[target] - pred_full)**2).mean())
    axes[1].set_title(f"Full Model with Rebrand Features (RMSE={rmse_full:.3f})")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("Actual")

    fig.suptitle("Model Comparison: Base vs Rebranding-Aware", fontsize=13)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "12_model_comparison.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: 12_model_comparison.png")


def plot_dow_rebranding_interaction(data: pd.DataFrame):
    data["DOW"] = data["Date"].dt.dayofweek
    dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    pre = data[data["Date"] < REBRAND_DATE]
    post = data[data["Date"] >= REBRAND_DATE]

    pre_dow = pre.groupby("DOW")["Quantity_Sold"].mean()
    post_dow = post.groupby("DOW")["Quantity_Sold"].mean()

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(7)
    width = 0.35
    ax.bar(x - width/2, [pre_dow.get(i, 0) for i in range(7)], width, label="Pre-Rebrand", alpha=0.8, color="steelblue")
    ax.bar(x + width/2, [post_dow.get(i, 0) for i in range(7)], width, label="Post-Rebrand", alpha=0.8, color="darkgreen")
    ax.set_xticks(x)
    ax.set_xticklabels(dow_names)
    ax.set_ylabel("Average Daily Quantity")
    ax.set_title("DOW Pattern Shift After Rebranding", fontsize=13)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "13_dow_rebranding_interaction.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: 13_dow_rebranding_interaction.png")


def generate_plots(df: pd.DataFrame, data: pd.DataFrame, features: list, importance: pd.Series):
    print(f"\n{SEPARATOR}")
    print("GENERATING PLOTS")
    print(SEPARATOR)

    plot_target_distribution(df)
    plot_autocorrelation(df)
    plot_rolling_correlation(df)
    plot_feature_correlation_heatmap(data, features)
    plot_feature_importance(importance)
    plot_rebranding_feature_shift(data)
    plot_model_comparison(data, features)
    plot_dow_rebranding_interaction(data)

    print(f"\nAll plots saved to: {FIGURES_DIR}")


def main():
    print("CAFE SUPPLY FEATURE DISCOVERY (Updated with Rebranding Awareness)")
    print("=" * 80)

    print(f"\nLoading data...")
    df_raw = load_and_prep_data(SALES_FORECASTING_DIR / "daily_item_sales.csv")
    print(f"Loaded {len(df_raw):,} rows")

    section_target_analysis(df_raw)
    section_autocorrelation(df_raw)
    section_day_of_week_patterns(df_raw)
    section_rolling_window_analysis(df_raw)
    section_trend_features(df_raw)
    section_proposed_features(df_raw)
    data, features = section_feature_correlations(df_raw)
    section_feature_importance(data, features)
    section_per_item_importance(data, features)
    section_rebranding_feature_analysis(data, features)
    section_dow_interaction(data, features)

    importance = train_global_model(data, features)
    generate_plots(df_raw, data, features, importance)

    print(f"\n{SEPARATOR}")
    print("DISCOVERY COMPLETE")
    print(SEPARATOR)


if __name__ == "__main__":
    main()
