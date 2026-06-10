"""
v2_04_evaluation.py
Final evaluation — comparing v2 approaches against each other
and against v1 benchmarks from thesis.

Key v1 benchmarks to beat (from exploration v1 README):
  - MAE: 1.29 cups/item/day (blended model)
  - RMSE < Std(actual): achieved (1.80 < 1.98)
  - wMAPE < 20%: not met
  - R² ≥ 0.6: not met
  - Fri+Sat MAE: 1.30
  - Accuracy ≤20%: ~35% (XGB)
  - Bias: +0.06

v2 targets (fresh approach):
  - MAE < 1.0 ✓ (0.699 from 03_model_experiments)
  - wMAPE: need to check
  - Per-item, per-DOW breakdowns
"""
import os, sys, warnings, json
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
)
import xgboost as xgb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    FIGURES_DIR, MODELS_DIR, TABLES_DIR,
    RANDOM_SEED, N_BACKTEST_WINDOWS, BACKTEST_WINDOW_DAYS,
)

sns.set_style("whitegrid")
np.random.seed(RANDOM_SEED)

OUT = os.path.join(FIGURES_DIR, "v2_evaluation")
os.makedirs(OUT, exist_ok=True)

FEATURE_MATRIX_PATH = os.path.join(TABLES_DIR, "v2_feature_matrix.csv")


def load_data():
    df = pd.read_csv(FEATURE_MATRIX_PATH)
    df["Date_Only"] = pd.to_datetime(df["Date_Only"])
    return df


def get_optimized_features(full):
    """
    Features selected by ablation: Δ MAE > 0.02 threshold.
    Groups kept: Temporal, Recency, Lags, DOW_Baselines, CrossItem = 31 features.
    Dropped: Item dummies, category flags, Lifecycle, Rolling.
    """
    temporal = ["DOW", "Is_Weekend", "Month", "Year", "WeekOfYear", "DayOfMonth",
                "Quarter", "MonthStart", "MonthEnd", "Is_Holiday_Season",
                "WeekOfMonth", "DaysFromStart", "DOW_Sin", "DOW_Cos",
                "Month_Sin", "Month_Cos"]
    recency = ["Days_Since_Last_Sale", "Sales_Last_7D"]
    lags = ["Lag_1", "Lag_7", "Lag_14", "Lag_28"]
    cross = ["Day_Total_Qty", "Day_Total_Items_Sold", "Day_Total_Beverage",
             "Day_Total_Food", "Day_Total_Qty_7D"]

    all_feats = temporal + recency + lags + cross
    return list(dict.fromkeys([c for c in all_feats if c in full.columns]))


def build_model():
    return xgb.XGBRegressor(
        objective="count:poisson",
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.5, reg_lambda=0.5,
        random_state=RANDOM_SEED, verbosity=0,
    )


def evaluate_predictions(y_true, y_pred, label=""):
    """Compute comprehensive metrics."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(np.maximum(y_pred, 0))

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    bias = np.mean(y_pred - y_true)
    std_actual = np.std(y_true)

    wmape = 100 * np.sum(np.abs(y_true - y_pred)) / max(np.sum(y_true), 0.01)

    # R² on non-zero
    nonzero = y_true > 0
    r2_nonzero = r2_score(y_true[nonzero], y_pred[nonzero]) if nonzero.sum() > 5 else np.nan
    r2_all = r2_score(y_true, y_pred) if len(y_true) > 5 else np.nan

    # Accuracy buckets on non-zero
    ape = np.abs((y_pred[nonzero] - y_true[nonzero]) / y_true[nonzero])
    within_20 = (ape <= 0.2).mean() * 100
    within_50 = (ape <= 0.5).mean() * 100

    # Exact match (for integer predictions)
    pred_rounded = np.round(y_pred).clip(0)
    exact_match = (pred_rounded[nonzero] == y_true[nonzero]).mean() * 100

    # Over/under prediction breakdown
    over_pct = (y_pred > y_true).mean() * 100
    under_pct = (y_pred < y_true).mean() * 100
    exact_pct = (y_pred == y_true).mean() * 100

    return {
        "MAE": mae,
        "RMSE": rmse,
        "Std_Actual": std_actual,
        "RMSE_vs_Std": "✓" if rmse < std_actual else "✗",
        "Bias": bias,
        "wMAPE": wmape,
        "R2": r2_all,
        "R2_nonzero": r2_nonzero,
        "Within_20pct": within_20,
        "Within_50pct": within_50,
        "Exact_Match_pct": exact_match,
        "Overpredict_pct": over_pct,
        "Underpredict_pct": under_pct,
        "Exact_pct": exact_pct,
        "N": len(y_true),
        "N_nonzero": nonzero.sum(),
        "True_Mean": y_true.mean(),
        "Pred_Mean": y_pred.mean(),
        "True_Mean_Nonzero": y_true[nonzero].mean(),
        "Pred_Mean_Nonzero": y_pred[nonzero].mean(),
    }


# ---------------------------------------------------------------------------
# FINAL BACKTEST
# ---------------------------------------------------------------------------
def run_final_backtest(full, feature_cols):
    print("=" * 70)
    print("FINAL EXPANDING-WINDOW BACKTEST")
    print("=" * 70)

    all_dates = sorted(full["Date_Only"].unique())
    test_end = all_dates[-1].date()
    test_start = test_end - timedelta(days=BACKTEST_WINDOW_DAYS * N_BACKTEST_WINDOWS)

    all_predictions = []
    all_actuals = []
    window_summaries = []

    # ABC classification based on full history (excluding test windows)
    pre_backtest = full[full["Date_Only"] < pd.Timestamp(test_start)]
    item_volume = pre_backtest.groupby("Item")["Quantity"].sum().sort_values(ascending=False)
    total_vol = item_volume.sum()
    item_abc = {}
    cum = 0
    for item, vol in item_volume.items():
        cum += vol
        pct = cum / total_vol * 100
        if pct <= 70:
            item_abc[item] = "A"
        elif pct <= 90:
            item_abc[item] = "B"
        else:
            item_abc[item] = "C"

    for window in range(N_BACKTEST_WINDOWS):
        w_start = pd.Timestamp(test_start) + timedelta(days=BACKTEST_WINDOW_DAYS * window)
        w_end = w_start + timedelta(days=BACKTEST_WINDOW_DAYS - 1)

        train = full[full["Date_Only"] < w_start].copy()
        test = full[(full["Date_Only"] >= w_start) & (full["Date_Only"] <= w_end)].copy()

        if len(train) < 100 or len(test) < 10:
            continue

        print(f"\nWindow {window+1}: {w_start.date()} → {w_end.date()}")
        print(f"  Train: {len(train)} rows, {train['Date_Only'].nunique()} days")
        print(f"  Test:  {len(test)} rows, {test['Date_Only'].nunique()} days, "
              f"{(test['Quantity'] > 0).sum()} non-zero")

        X_train = train[feature_cols].fillna(0)
        y_train = train["Quantity"]
        X_test = test[feature_cols].fillna(0)
        y_test = test["Quantity"]

        model = build_model()
        model.fit(X_train, y_train)
        y_pred = np.maximum(model.predict(X_test), 0)

        all_predictions.extend(y_pred)
        all_actuals.extend(y_test.values)

        # Per-window metrics
        metrics = evaluate_predictions(y_test.values, y_pred)
        metrics["window"] = window + 1
        metrics["start_date"] = str(w_start.date())
        metrics["end_date"] = str(w_end.date())
        window_summaries.append(metrics)

        # Per-DOW breakdown
        print(f"  --- DOW breakdown ---")
        for dow in range(7):
            dow_mask = test["DOW"] == dow
            if dow_mask.sum() < 5:
                continue
            dow_mae = mean_absolute_error(
                y_test.values[dow_mask], y_pred[dow_mask]
            )
            dow_names = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
            print(f"    {dow_names[dow]}: MAE={dow_mae:.4f} (n={dow_mask.sum()})")

        # Per-ABC breakdown
        print(f"  --- ABC breakdown ---")
        for abc_class in ["A", "B", "C"]:
            abc_items = [i for i, c in item_abc.items() if c == abc_class]
            abc_mask = test["Item"].isin(abc_items)
            if abc_mask.sum() == 0:
                continue
            abc_mae = mean_absolute_error(
                y_test.values[abc_mask], y_pred[abc_mask]
            )
            abc_nonzero = (y_test.values[abc_mask] > 0).sum()
            print(f"    {abc_class}: MAE={abc_mae:.4f} (n={abc_mask.sum()}, nonzero={abc_nonzero})")

    # Overall metrics
    print(f"\n{'='*70}")
    print("OVERALL RESULTS")
    print(f"{'='*70}")
    overall = evaluate_predictions(np.array(all_actuals), np.array(all_predictions))
    for k, v in overall.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    # Top 10 items
    all_pred_series = pd.Series(all_predictions)
    all_actual_series = pd.Series(all_actuals)

    # Reconstruct the item-date mapping for per-item analysis
    test_data_for_items = full[
        (full["Date_Only"] >= pd.Timestamp(test_start))
    ].copy()

    if len(test_data_for_items) == len(all_actuals):
        test_data_for_items["Predicted"] = all_pred_series.values
        test_data_for_items["Actual"] = all_actual_series.values
        test_data_for_items["Error"] = test_data_for_items["Predicted"] - test_data_for_items["Actual"]
        test_data_for_items["AbsError"] = np.abs(test_data_for_items["Error"])

        item_perf = test_data_for_items.groupby("Item").agg(
            MAE=("AbsError", "mean"),
            Bias=("Error", "mean"),
            True_Mean=("Actual", "mean"),
            Pred_Mean=("Predicted", "mean"),
            N=("AbsError", "count"),
            N_Nonzero=("Actual", lambda x: (x > 0).sum()),
            ABC=("Item", lambda x: item_abc.get(x.iloc[0], "?")),
        ).sort_values("True_Mean", ascending=False)

        print(f"\n{'='*70}")
        print("TOP 15 ITEMS BY VOLUME")
        print(f"{'='*70}")
        print(item_perf.head(15).to_string())
        item_perf.to_csv(os.path.join(TABLES_DIR, "v2_item_performance.csv"))

        # Plot: per-item MAE vs True Mean
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = {"A": "green", "B": "orange", "C": "red"}
        for cls in ["A", "B", "C"]:
            cls_data = item_perf[item_perf["ABC"] == cls]
            ax.scatter(cls_data["True_Mean"], cls_data["MAE"],
                      c=colors[cls], label=f"Class {cls}", alpha=0.7, s=50)
        ax.set_xlabel("True Mean Quantity")
        ax.set_ylabel("MAE")
        ax.set_title("Per-Item MAE vs Mean Volume")
        ax.legend()
        plt.tight_layout()
        fig.savefig(os.path.join(OUT, "item_mae_vs_volume.png"), dpi=150)
        plt.close()

    # Window summary table
    ws_df = pd.DataFrame(window_summaries)
    ws_df.to_csv(os.path.join(TABLES_DIR, "v2_window_summaries.csv"), index=False)

    # Save all predictions
    if len(test_data_for_items) == len(all_actuals):
        preds_out = test_data_for_items[["Date_Only", "Item", "Actual", "Predicted", "Error", "ABC"]]
        preds_out.to_csv(os.path.join(TABLES_DIR, "v2_predictions.csv"), index=False)

    return overall, item_perf if len(test_data_for_items) == len(all_actuals) else None


# ---------------------------------------------------------------------------
# COMPARISON WITH OLD BENCHMARKS
# ---------------------------------------------------------------------------
def compare_with_v1(overall):
    """Compare v2 results with v1 benchmarks from thesis."""
    print(f"\n{'='*70}")
    print("V2 vs V1 BENCHMARK COMPARISON")
    print(f"{'='*70}")

    benchmarks = {
        "MAE (cups)": {"v1": 1.29, "v2": overall["MAE"], "target": "< 1.0", "better": "lower"},
        "RMSE": {"v1": 1.80, "v2": overall["RMSE"], "target": f"< {overall['Std_Actual']:.2f}", "better": "lower"},
        "wMAPE (%)": {"v1": None, "v2": overall["wMAPE"], "target": "< 20%", "better": "lower"},
        "R²": {"v1": None, "v2": overall["R2"], "target": "≥ 0.6", "better": "higher"},
        "Within ±20%": {"v1": 35.0, "v2": overall["Within_20pct"], "target": None, "better": "higher"},
        "Bias": {"v1": 0.06, "v2": overall["Bias"], "target": None, "better": "lower"},
    }

    print(f"{'Metric':<25} {'v1':>8} {'v2':>8} {'Change':>12} {'Target':>12}")
    print("-" * 70)
    for metric, vals in benchmarks.items():
        v1 = vals["v1"]
        v2 = vals["v2"]
        target = vals["target"] or "-"
        if v1 is not None:
            if vals["better"] == "lower":
                change = ((v1 - v2) / v1 * 100) if v1 != 0 else 0
                arrow = "↓" if v2 < v1 else "↑"
            else:
                change = ((v2 - v1) / v1 * 100) if v1 != 0 else 0
                arrow = "↑" if v2 > v1 else "↓"
            change_str = f"{change:+.1f}% {arrow}"
        else:
            change_str = "-"
        v1_str = f"{v1:>8.3f}" if v1 is not None else "    None"
        target_str = f"{target:>12}" if target is not None else "         -"
        print(f"{metric:<25} {v1_str} {v2:>8.3f} {change_str:>12} {target_str}")


# ---------------------------------------------------------------------------
# HYPERPARAMETER SENSITIVITY
# ---------------------------------------------------------------------------
def hyperparameter_sensitivity(full, feature_cols):
    """Test sensitivity to key hyperparameters on a single train/test split."""
    print(f"\n{'='*70}")
    print("HYPERPARAMETER SENSITIVITY")
    print(f"{'='*70}")

    cutoff = full["Date_Only"].max() - timedelta(days=BACKTEST_WINDOW_DAYS)
    train = full[full["Date_Only"] < cutoff]
    test = full[full["Date_Only"] >= cutoff]

    X_train = train[feature_cols].fillna(0)
    y_train = train["Quantity"]
    X_test = test[feature_cols].fillna(0)
    y_test = test["Quantity"]

    print(f"{'Config':<40} {'MAE':>8} {'wMAPE':>8} {'R²_nz':>8}")
    print("-" * 70)

    param_grid = [
        {"max_depth": 3, "n_estimators": 100, "learning_rate": 0.03},
        {"max_depth": 4, "n_estimators": 200, "learning_rate": 0.05},
        {"max_depth": 5, "n_estimators": 200, "learning_rate": 0.05},
        {"max_depth": 6, "n_estimators": 300, "learning_rate": 0.03},
        {"max_depth": 4, "n_estimators": 300, "learning_rate": 0.03},
        {"max_depth": 4, "n_estimators": 200, "learning_rate": 0.1},
    ]

    best = None
    best_mae = float("inf")

    for params in param_grid:
        model = xgb.XGBRegressor(
            objective="count:poisson",
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.5, reg_lambda=0.5,
            random_state=RANDOM_SEED, verbosity=0,
            **params,
        )
        model.fit(X_train, y_train)
        y_pred = np.maximum(model.predict(X_test), 0)

        mae = mean_absolute_error(y_test, y_pred)
        wmape = 100 * np.sum(np.abs(y_test.values - y_pred)) / max(y_test.sum(), 0.01)

        nonzero = y_test > 0
        r2_nz = r2_score(y_test[nonzero], y_pred[nonzero]) if nonzero.sum() > 5 else 0

        label = f"depth={params['max_depth']}, n={params['n_estimators']}, lr={params['learning_rate']}"
        print(f"{label:<40} {mae:>8.4f} {wmape:>8.1f} {r2_nz:>8.4f}")

        if mae < best_mae:
            best_mae = mae
            best = {**params, "mae": mae, "wmape": wmape, "r2_nz": r2_nz}

    print(f"\nBest: max_depth={best['max_depth']}, n_estimators={best['n_estimators']}, "
          f"lr={best['learning_rate']} → MAE={best['mae']:.4f}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    print("v2_04: FINAL EVALUATION")
    print()

    full = load_data()
    print(f"Data: {len(full)} rows, {full['Item'].nunique()} items "
          f"({full['Date_Only'].min().date()} → {full['Date_Only'].max().date()})")

    feature_cols = get_optimized_features(full)
    print(f"Features: {len(feature_cols)}")
    print()

    # Hyperparameter sensitivity first (quick)
    hyperparameter_sensitivity(full, feature_cols)

    # Final backtest
    overall, item_perf = run_final_backtest(full, feature_cols)

    # Compare with v1
    compare_with_v1(overall)

    print(f"\n{'='*70}")
    print("EVALUATION COMPLETE")
    print(f"Results: tables/v2_*.csv, figures/v2_evaluation/")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
