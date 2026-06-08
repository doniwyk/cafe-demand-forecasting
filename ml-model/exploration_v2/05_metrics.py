"""
v2_06_metrics.py
Proper forecast evaluation:
  - MAPE per period (not just cumulative)
  - Std dev of forecast errors (Actual - Predicted)
  - Service-level buffer via z-score
  - MAE + MAPE together
"""
import os, sys, warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from datetime import timedelta
from sklearn.metrics import mean_absolute_error
import xgboost as xgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import FIGURES_DIR, TABLES_DIR, RANDOM_SEED

OUT = os.path.join(FIGURES_DIR, "v2_metrics")
os.makedirs(OUT, exist_ok=True)

FEATURE_MATRIX_PATH = os.path.join(TABLES_DIR, "v2_feature_matrix.csv")


def load_data():
    df = pd.read_csv(FEATURE_MATRIX_PATH)
    df["Date_Only"] = pd.to_datetime(df["Date_Only"])
    return df


def get_feature_cols(full):
    temporal = ["DOW","Is_Weekend","Month","Year","WeekOfYear","DayOfMonth","Quarter",
                "MonthStart","MonthEnd","Is_Holiday_Season","WeekOfMonth","DaysFromStart",
                "DOW_Sin","DOW_Cos","Month_Sin","Month_Cos"]
    lifecycle = ["Days_Since_First_Sale","Item_Rank","Item_Rank_Pct"]
    recency = ["Days_Since_Last_Sale","Sales_Last_7D"]
    rolling = ["Roll_Mean_7","Roll_Mean_14","Roll_Mean_28","Roll_Std_7",
               "EWMA_7","EWMA_28","Trend_7_28","WoW_Change"]
    lags = ["Lag_1","Lag_7","Lag_14","Lag_28"]
    cross = ["Day_Total_Qty","Day_Total_Items_Sold","Day_Total_Beverage",
             "Day_Total_Food","Day_Total_Qty_7D"]
    cat = ["Is_Beverage","Is_Food"]
    item_d = [c for c in full.columns if c.startswith("Item_")
              and c not in ("Item_Rank","Item_Rank_Pct") and ".1" not in c]
    feats = temporal + lifecycle + recency + rolling + lags + cross + cat + item_d
    return [c for c in feats if c in full.columns]


def evaluate_period(y_true, y_pred):
    """Compute per-period (daily) metrics."""
    mask = y_true > 0
    nz = mask.sum()

    mae = mean_absolute_error(y_true, y_pred)

    errors = y_pred - y_true
    bias = np.mean(errors)
    error_std = np.std(errors)
    rmse = np.sqrt(np.mean(errors**2))

    # MAPE on non-zero only (avoids division by zero)
    if nz > 0:
        ape = np.abs((y_pred[mask] - y_true[mask]) / y_true[mask])
        mape = np.mean(ape) * 100
        median_ape = np.median(ape) * 100
    else:
        mape = np.nan
        median_ape = np.nan

    # Service level: % of predictions >= actual (no stockout)
    service_level = (y_pred >= y_true).mean() * 100

    return {
        "MAE": mae,
        "RMSE": rmse,
        "Bias": bias,
        "Error_Std": error_std,
        "MAPE": mape,
        "Median_APE": median_ape,
        "Service_Level": service_level,
        "N": len(y_true),
        "N_nonzero": nz,
        "True_Mean": y_true.mean(),
        "Pred_Mean": y_pred.mean(),
    }


def compute_buffer_from_zscore(error_std, z=1.645):
    """Buffer = z * error_std for given service level.
    z=1.645 → 95% service level (one-tailed normal)."""
    return z * error_std


def run_weekly_backtest(full, feature_cols, n_windows=8):
    """Expanding window backtest with per-period metric tracking."""
    all_dates = sorted(full["Date_Only"].unique())
    test_start = all_dates[-1] - timedelta(days=7 * n_windows)

    periods = []
    all_errors = []
    item_errors = {}  # per item

    for w in range(n_windows):
        ws = test_start + timedelta(days=7 * w)
        we = ws + timedelta(days=6)

        train = full[full["Date_Only"] < ws]
        test = full[(full["Date_Only"] >= ws) & (full["Date_Only"] <= we)]
        if len(test) < 10:
            continue

        Xtr, ytr = train[feature_cols].fillna(0), train["Quantity"]
        Xte, yte = test[feature_cols].fillna(0), test["Quantity"]

        model = xgb.XGBRegressor(
            objective="count:poisson",
            n_estimators=200, max_depth=4, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.5, reg_lambda=0.5,
            random_state=RANDOM_SEED, verbosity=0,
        )
        model.fit(Xtr, ytr)
        preds = np.maximum(model.predict(Xte), 0)

        # Per-date metrics (7 days in the window)
        test_dates = sorted(test["Date_Only"].unique())
        for date in test_dates:
            dm = test["Date_Only"] == date
            metrics = evaluate_period(yte[dm].values, preds[dm])
            metrics["Date"] = date.date()
            metrics["Window"] = w + 1
            periods.append(metrics)

        # Track per-item errors across all windows
        errors = preds - yte.values
        all_errors.extend(errors)

        for _, row in test.iterrows():
            item = row["Item"]
            actual = row["Quantity"]
            pred = preds[test.index.get_loc(row.name)]
            if item not in item_errors:
                item_errors[item] = {"errors": [], "actuals": [], "preds": []}
            item_errors[item]["errors"].append(pred - actual)
            item_errors[item]["actuals"].append(actual)
            item_errors[item]["preds"].append(pred)

    periods_df = pd.DataFrame(periods)
    return periods_df, np.array(all_errors), item_errors


def main():
    print("=" * 60)
    print("V2 METRICS — MAPE, Error Std, Service Level Buffer")
    print("=" * 60)

    full = load_data()
    feature_cols = get_feature_cols(full)
    print(f"Data: {len(full)} rows, {full['Item'].nunique()} items")
    print(f"Date range: {full['Date_Only'].min().date()} → {full['Date_Only'].max().date()}")
    print()

    # --- 8-WEEK BACKTEST ---
    periods_df, all_errors, item_errors = run_weekly_backtest(full, feature_cols)

    # --- Overall error statistics ---
    overall_error_std = np.std(all_errors)
    overall_mae = np.mean(np.abs(all_errors))
    overall_bias = np.mean(all_errors)
    overall_rmse = np.sqrt(np.mean(all_errors ** 2))

    print("OVERALL ERROR STATISTICS (8-week backtest)")
    print("-" * 45)
    print(f"  MAE (cups):           {overall_mae:.3f}")
    print(f"  RMSE (cups):          {overall_rmse:.3f}")
    print(f"  Bias (cups):          {overall_bias:+.3f}")
    print(f"  Std Dev of Errors:    {overall_error_std:.3f}")
    print()

    # --- Per-period MAPE summary ---
    valid_mape = periods_df["MAPE"].dropna()
    print("MAPE OVER TIME (daily, non-zero actuals only)")
    print("-" * 45)
    print(f"  Mean MAPE:            {valid_mape.mean():.1f}%")
    print(f"  Median MAPE:          {valid_mape.median():.1f}%")
    print(f"  Std MAPE:             {valid_mape.std():.1f}%")
    print(f"  Min MAPE:             {valid_mape.min():.1f}%")
    print(f"  Max MAPE:             {valid_mape.max():.1f}%")

    # Show MAPE by window
    print(f"\n  MAPE by week:")
    for w in sorted(periods_df["Window"].unique()):
        w_df = periods_df[periods_df["Window"] == w]
        w_mape = w_df["MAPE"].dropna()
        if len(w_mape) > 0:
            print(f"    Week {int(w)}: MAPE={w_mape.mean():.0f}% (n={len(w_mape)} days)")

    # Show MAPE by DOW
    periods_df["DOW"] = pd.to_datetime(periods_df["Date"]).dt.dayofweek
    dn = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
    print(f"\n  MAPE by day of week:")
    for d in range(7):
        dow_mape = periods_df[periods_df["DOW"] == d]["MAPE"].dropna()
        if len(dow_mape) > 0:
            print(f"    {dn[d]:>4}: MAPE={dow_mape.mean():.0f}%")

    # --- Service level buffer ---
    print(f"\n{'='*60}")
    print("SERVICE LEVEL BUFFER (z-score based)")
    print("-" * 45)

    z_values = {90: 1.282, 95: 1.645, 99: 2.326}
    for level, z in z_values.items():
        buffer = compute_buffer_from_zscore(overall_error_std, z)
        print(f"  {level}% service → z={z:.3f} → buffer = {buffer:.2f} cups per item-day")
        print(f"    Prediction + buffer = {overall_bias + buffer:+.2f} cups above mean actual")

    # --- Per-item error stats ---
    print(f"\n{'='*60}")
    print("PER-ITEM ERROR STATISTICS")
    print("-" * 60)

    # --- ABC segmentation (by total volume) ---
    item_volume = {item: sum(data["actuals"]) for item, data in item_errors.items()}
    total_vol = sum(item_volume.values())
    sorted_items = sorted(item_volume.items(), key=lambda x: -x[1])
    abc_class = {}
    cum = 0
    for item, vol in sorted_items:
        cum += vol
        pct = cum / total_vol * 100
        if pct <= 70:
            abc_class[item] = "A"
        elif pct <= 90:
            abc_class[item] = "B"
        else:
            abc_class[item] = "C"

    # Per-class service levels and z-scores
    class_config = {
        "A": {"label": "A (top 70%)", "z": 1.645, "level": 95},
        "B": {"label": "B (70-90%)", "z": 1.282, "level": 90},
        "C": {"label": "C (bottom 10%)", "z": 1.036, "level": 85},
    }

    item_stats = []
    for item, data in item_errors.items():
        errs = np.array(data["errors"])
        acts = np.array(data["actuals"])
        if len(errs) < 10:
            continue
        nz = acts > 0
        cls = abc_class.get(item, "C")
        z = class_config[cls]["z"]
        # For intermittent items (zero_pct > 80%), MAPE is unreliable — use MAE
        zero_pct = (acts == 0).mean() * 100
        mape_val = np.mean(np.abs(errs[nz] / acts[nz])) * 100 if nz.sum() > 0 else np.nan
        if zero_pct > 80 and not np.isnan(mape_val) and mape_val > 100:
            mape_note = "unreliable*"
        else:
            mape_note = ""

        item_stats.append({
            "Item": item,
            "Class": cls,
            "MAE": np.mean(np.abs(errs)),
            "RMSE": np.sqrt(np.mean(errs ** 2)),
            "Bias": np.mean(errs),
            "Error_Std": np.std(errs),
            "MAPE": mape_val,
            "MAPE_Note": mape_note,
            "Zero%": zero_pct,
            "Buffer": compute_buffer_from_zscore(np.std(errs), z),
            "Svc_Level": class_config[cls]["level"],
            "N": len(errs),
            "True_Mean": np.mean(acts),
            "True_TotalVol": sum(acts),
        })

    item_df = pd.DataFrame(item_stats).sort_values("True_TotalVol", ascending=False)
    item_df.to_csv(os.path.join(TABLES_DIR, "v2_per_item_errors.csv"), index=False)

    # Per-class summary
    print(f"\n{'='*60}")
    print("ABC SEGMENTATION — Per-Class Service Levels")
    print(f"{'='*60}")

    total_buffer_cups = 0
    for cls in ["A", "B", "C"]:
        cfg = class_config[cls]
        cdf = item_df[item_df["Class"] == cls]
        avg_mae = cdf["MAE"].mean()
        avg_mape = cdf["MAPE"].dropna().mean()
        total_buf = cdf["Buffer"].sum()
        total_buffer_cups += total_buf
        n_items = len(cdf)
        n_intermittent = (cdf["Zero%"] > 80).sum()
        print(f"\n  {cfg['label']}: {n_items} items, {cfg['level']}% service (z={cfg['z']:.3f})")
        print(f"    Avg MAE: {avg_mae:.2f} cups, Avg MAPE: {avg_mape:.0f}%")
        print(f"    Intermittent items (zero>80%): {n_intermittent}")
        print(f"    Total buffer/day: {total_buf:.1f} cups")

    # Weekly cost
    weekly_buffer = total_buffer_cups * 7
    weekly_pred = item_df["True_Mean"].sum() * 7
    print(f"\n  --- Cost sanity check ---")
    print(f"  Total buffer (all items): {total_buffer_cups:.1f} cups/day")
    print(f"  Weekly buffer:            {weekly_buffer:.0f} cups")
    print(f"  Weekly predicted demand:  {weekly_pred:.0f} cups")
    print(f"  Buffer as % of demand:    {weekly_buffer/weekly_pred*100:.1f}%")

    # Per-class buffer
    weekly_buf_a = item_df[item_df["Class"]=="A"]["Buffer"].sum() * 7
    weekly_buf_b = item_df[item_df["Class"]=="B"]["Buffer"].sum() * 7
    weekly_buf_c = item_df[item_df["Class"]=="C"]["Buffer"].sum() * 7
    weekly_demand_a = item_df[item_df["Class"]=="A"]["True_Mean"].sum() * 7
    print(f"\n  Buffer by class (weekly):")
    for cls, buf, dem in [("A", weekly_buf_a, weekly_demand_a),
                           ("B", weekly_buf_b, weekly_demand_a * 0.29),
                           ("C", weekly_buf_c, weekly_demand_a * 0.14)]:
        pct = buf / max(dem, 1) * 100
        print(f"    {cls}: {buf:.0f} cups buffer, ~{pct:.0f}% of class demand")

    # Top and bottom items
    print(f"\n  Top 5 items (A-class):")
    a_df = item_df[item_df["Class"] == "A"].head(5)
    for _, r in a_df.iterrows():
        mape_str = f"MAPE={r['MAPE']:.0f}%" if not pd.isna(r["MAPE"]) else "MAPE=N/A"
        intermittent = " [intermittent]" if r["Zero%"] > 80 else ""
        print(f"    {r['Item']:<30} MAE={r['MAE']:.2f}  {mape_str}  buf={r['Buffer']:.1f}{intermittent}")

    print(f"\n  Intermittent items (MAPE unreliable, use MAE):")
    inter = item_df[item_df["Zero%"] > 80]
    for _, r in inter.iterrows():
        print(f"    {r['Item']:<30} MAE={r['MAE']:.2f}  zero%={r['Zero%']:.0f}%  buf={r['Buffer']:.1f}")

    # --- Summary table ---
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"  MAE (cups/item/day):     {overall_mae:.3f}")
    print(f"  MAPE (mean daily):       {valid_mape.mean():.0f}%")
    print(f"  MAPE (median daily):     {valid_mape.median():.0f}%")
    print(f"  Error Std:               {overall_error_std:.3f}")
    print(f"  Weekly buffer (ABC):     {weekly_buffer:.0f} cups ({weekly_buffer/weekly_pred*100:.0f}% of demand)")

    # --- Plot: error by class ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = {"A": "steelblue", "B": "darkorange", "C": "green"}
    for cls in ["A", "B", "C"]:
        cdf = item_df[item_df["Class"] == cls]
        axes[0].scatter(cdf["True_Mean"], cdf["MAE"], c=colors[cls], label=class_config[cls]["label"], alpha=0.7)
    axes[0].set_xlabel("Avg Daily Volume (cups)")
    axes[0].set_ylabel("MAE (cups)")
    axes[0].set_title("MAE vs Volume by ABC Class")
    axes[0].legend()

    # Bar chart: per-class buffer cost
    bar_data = [("A (95%)", weekly_buf_a), ("B (90%)", weekly_buf_b), ("C (85%)", weekly_buf_c)]
    axes[1].bar([x[0] for x in bar_data], [x[1] for x in bar_data], color=[colors["A"], colors["B"], colors["C"]])
    axes[1].set_ylabel("Weekly Buffer (cups)")
    axes[1].set_title("Weekly Buffer by ABC Class")
    plt.tight_layout()
    fig.savefig(os.path.join(OUT, "abc_analysis.png"), dpi=150)
    plt.close()
    print(f"\n  Plots saved to {OUT}")


if __name__ == "__main__":
    main()
