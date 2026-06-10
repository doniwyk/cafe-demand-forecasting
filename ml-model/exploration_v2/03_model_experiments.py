"""
v2_03_model_experiments.py
fresh model experiments — XGBoost + RandomForest only (matching thesis scope)

New approaches compared to exploration v1:
  1. TWO-STAGE HURDLE MODEL: classify Is_Sale first, then regress Quantity
  2. SINGLE-STAGE: standard regression on all data with zeros
  3. POISSON objective for count data (XGBoost count:poisson)
  4. Expanding window backtest with growing training data
  5. Feature selection informed by fresh EDA insights
"""
import os, sys, warnings, json
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from datetime import timedelta
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, roc_auc_score, classification_report,
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    FIGURES_DIR, MODELS_DIR, TABLES_DIR,
    RANDOM_SEED, N_BACKTEST_WINDOWS, BACKTEST_WINDOW_DAYS, MIN_NONZERO_DAYS,
)

sns.set_style("whitegrid")
np.random.seed(RANDOM_SEED)

OUT = os.path.join(FIGURES_DIR, "v2_features")
os.makedirs(OUT, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(TABLES_DIR, exist_ok=True)

FEATURE_MATRIX_PATH = os.path.join(TABLES_DIR, "v2_feature_matrix.csv")


# ---------------------------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------------------------
def load_feature_matrix():
    df = pd.read_csv(FEATURE_MATRIX_PATH)
    df["Date_Only"] = pd.to_datetime(df["Date_Only"])
    return df


def get_feature_columns(full):
    """
    Features are selected by ablation: Δ MAE > 0.02 threshold.
    Groups: Recency, Temporal, CrossItem, Lags, DOW_Baselines = 31 features.
    Dropped: Lifecycle (Δ=+0.017), Rolling (Δ=+0.013).
    """
    temporal = ["DOW", "Is_Weekend", "Month", "Year", "WeekOfYear", "DayOfMonth",
                "Quarter", "MonthStart", "MonthEnd", "Is_Holiday_Season",
                "WeekOfMonth", "DaysFromStart", "DOW_Sin", "DOW_Cos",
                "Month_Sin", "Month_Cos"]
    recency = ["Days_Since_Last_Sale", "Sales_Last_7D"]
    lags = ["Lag_1", "Lag_7", "Lag_14", "Lag_28"]
    dow_baselines = ["DOW_Avg", "DOW_Median", "DOW_P75", "DOW_N_Samples"]
    cross = ["Day_Total_Qty", "Day_Total_Items_Sold", "Day_Total_Beverage",
             "Day_Total_Food", "Day_Total_Qty_7D"]

    all_feats = temporal + recency + lags + dow_baselines + cross
    available = list(dict.fromkeys([c for c in all_feats if c in full.columns]))
    return available


# ---------------------------------------------------------------------------
# TWO-STAGE HURDLE MODEL
# ---------------------------------------------------------------------------
class TwoStageHurdleModel:
    """
    Stage 1: Binary classifier — will this item sell today?
    Stage 2: Regressor — if yes, how many cups?

    Final prediction: P(Is_Sale) * E[Quantity | Is_Sale]
    """
    def __init__(self, clf_type="xgb", reg_type="xgb", random_state=RANDOM_SEED):
        self.clf_type = clf_type
        self.reg_type = reg_type
        self.random_state = random_state
        self.clf = None
        self.reg = None
        self.features = None

    def _build_clf(self):
        if self.clf_type == "xgb":
            return xgb.XGBClassifier(
                n_estimators=200, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                reg_alpha=0.5, reg_lambda=0.5,
                random_state=self.random_state, eval_metric="logloss",
                verbosity=0,
            )
        elif self.clf_type == "rf":
            return RandomForestClassifier(
                n_estimators=200, max_depth=10,
                min_samples_leaf=5, class_weight="balanced",
                random_state=self.random_state, n_jobs=-1,
            )

    def _build_reg(self):
        if self.reg_type == "xgb":
            return xgb.XGBRegressor(
                n_estimators=200, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                reg_alpha=0.5, reg_lambda=0.5,
                random_state=self.random_state, verbosity=0,
            )
        elif self.reg_type == "rf":
            return RandomForestRegressor(
                n_estimators=200, max_depth=10,
                min_samples_leaf=5,
                random_state=self.random_state, n_jobs=-1,
            )

    def fit(self, X, y, features):
        """Unified interface: y is Quantity (both binary and continuous targets derived from it)."""
        self.features = features
        X_fit = X[features].fillna(0).values
        y_binary = (y > 0).astype(int).values

        self.clf = self._build_clf()
        self.clf.fit(X_fit, y_binary)

        nonzero_mask = y > 0
        if nonzero_mask.sum() > 0:
            self.reg = self._build_reg()
            self.reg.fit(X_fit[nonzero_mask], y.values[nonzero_mask])
        else:
            self.reg = None

        return self

    def predict(self, X):
        X_pred = X[self.features].fillna(0).values
        proba = self.clf.predict_proba(X_pred)[:, 1]

        if self.reg is not None:
            qty_given_sale = self.reg.predict(X_pred)
        else:
            qty_given_sale = np.ones(len(X_pred))

        qty_given_sale = np.maximum(qty_given_sale, 0)
        return proba * qty_given_sale


# ---------------------------------------------------------------------------
# SINGLE-STAGE MODEL
# ---------------------------------------------------------------------------
class SingleStageModel:
    """
    Standard single-stage regression on all data (including zeros).
    Supports XGBoost Poisson objective for count data.
    """
    def __init__(self, model_type="xgb", use_poisson=True, random_state=RANDOM_SEED):
        self.model_type = model_type
        self.use_poisson = use_poisson
        self.random_state = random_state
        self.model = None
        self.features = None

    def _build(self):
        if self.model_type == "xgb":
            if self.use_poisson:
                return xgb.XGBRegressor(
                    objective="count:poisson",
                    n_estimators=200, max_depth=4, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8,
                    reg_alpha=0.5, reg_lambda=0.5,
                    random_state=self.random_state, verbosity=0,
                )
            else:
                return xgb.XGBRegressor(
                    n_estimators=200, max_depth=4, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8,
                    reg_alpha=0.5, reg_lambda=0.5,
                    random_state=self.random_state, verbosity=0,
                )
        elif self.model_type == "rf":
            return RandomForestRegressor(
                n_estimators=200, max_depth=10,
                min_samples_leaf=5,
                random_state=self.random_state, n_jobs=-1,
            )

    def fit(self, X, y, features):
        self.features = features
        X_fit = X[features].fillna(0).values
        self.model = self._build()
        self.model.fit(X_fit, y)
        return self

    def predict(self, X):
        X_pred = X[self.features].fillna(0).values
        return np.maximum(self.model.predict(X_pred), 0)


# ---------------------------------------------------------------------------
# DOW BASELINE (naive)
# ---------------------------------------------------------------------------
class DOWBaseline:
    """Simple DOW historical average baseline."""
    def __init__(self):
        self.dow_stats = {}

    def fit(self, train_df):
        for dow in range(7):
            mask = (train_df["DOW"] == dow) & (train_df["Quantity"] > 0)
            if mask.sum() > 0:
                vals = train_df.loc[mask, "Quantity"]
                self.dow_stats[dow] = {
                    "mean": vals.mean(),
                    "median": vals.median(),
                    "p75": vals.quantile(0.75),
                }
            else:
                self.dow_stats[dow] = {"mean": 0, "median": 0, "p75": 0}
        return self

    def predict(self, X, stat="mean"):
        return np.array([self.dow_stats.get(int(d), {"mean": 0})[stat]
                         for d in X["DOW"]])


# ---------------------------------------------------------------------------
# EVALUATION
# ---------------------------------------------------------------------------
def evaluate_global_model(model, X_test, y_test, name=""):
    """Evaluate a global model on test data."""
    preds = model.predict(X_test)
    actuals = y_test.values

    metrics = {
        "MAE": mean_absolute_error(actuals, preds),
        "RMSE": np.sqrt(mean_squared_error(actuals, preds)),
        "Bias": np.mean(preds - actuals),
    }

    if actuals.sum() > 0:
        metrics["wMAPE"] = 100 * np.sum(np.abs(actuals - preds)) / actuals.sum()
    else:
        metrics["wMAPE"] = np.nan

    non_zero = actuals > 0
    if non_zero.sum() > 5:
        metrics["R2_nonzero"] = r2_score(actuals[non_zero], preds[non_zero])

    # Accuracy buckets on non-zero
    if non_zero.sum() > 5:
        ape = np.abs((preds[non_zero] - actuals[non_zero]) / actuals[non_zero])
        metrics["pct_within_20pct"] = (ape <= 0.2).mean() * 100
        metrics["pct_within_50pct"] = (ape <= 0.5).mean() * 100

    return metrics


# ---------------------------------------------------------------------------
# EXPANDING WINDOW BACKTEST
# ---------------------------------------------------------------------------
def run_expanding_backtest(full, feature_cols, model_configs, min_train_date=None):
    """
    Expanding window backtest.
    For each of N windows, train on all data before the window, test on the window.
    """
    if min_train_date is None:
        min_train_date = full["Date_Only"].min()

    all_dates = sorted(full["Date_Only"].unique())
    test_end = all_dates[-1]
    test_start = test_end - timedelta(days=BACKTEST_WINDOW_DAYS * N_BACKTEST_WINDOWS)

    results = []

    for window in range(N_BACKTEST_WINDOWS):
        w_start = test_start + timedelta(days=BACKTEST_WINDOW_DAYS * window)
        w_end = w_start + timedelta(days=BACKTEST_WINDOW_DAYS - 1)

        train_mask = (full["Date_Only"] >= min_train_date) & (full["Date_Only"] < w_start)
        test_mask = (full["Date_Only"] >= w_start) & (full["Date_Only"] <= w_end)

        train = full[train_mask].copy()
        test = full[test_mask].copy()

        if len(train) < 100 or len(test) < 10:
            print(f"  Window {window+1}: {w_start.date()}→{w_end.date()} — SKIP (train={len(train)}, test={len(test)})")
            continue

        print(f"  Window {window+1}: {w_start.date()}→{w_end.date()} | train={len(train)}, test={len(test)}")

        # DOW Baseline
        dow_baseline = DOWBaseline()
        dow_baseline.fit(train)

        window_results = {
            "window": window + 1,
            "start": str(w_start.date()),
            "end": str(w_end.date()),
            "train_days": train["Date_Only"].nunique(),
            "train_rows": len(train),
            "test_items": test["Item"].nunique(),
            "test_rows": len(test),
            "test_nonzero": (test["Quantity"] > 0).sum(),
            "true_mean": test["Quantity"].mean(),
        }

        for name, config in model_configs.items():
            model = config["factory"](**config.get("kwargs", {}))
            model.fit(train, train["Quantity"], features=feature_cols)

            metrics = evaluate_global_model(model, test, test["Quantity"], name=name)
            for k, v in metrics.items():
                window_results[f"{name}_{k}"] = v

            # Also compute per-item metrics
            item_metrics = {}
            for item in test["Item"].unique():
                item_test = test[test["Item"] == item]
                if len(item_test) < 2 or item_test["Quantity"].sum() < 1:
                    continue
                item_pred = model.predict(item_test)
                item_mae = mean_absolute_error(item_test["Quantity"], item_pred)
                item_metrics[item] = item_mae

            window_results[f"{name}_item_MAE_median"] = np.median(list(item_metrics.values())) if item_metrics else np.nan

        # DOW baseline (compute once per window)
        for stat in ["mean", "median", "p75"]:
            dow_pred = dow_baseline.predict(test, stat=stat)
            dow_mae = mean_absolute_error(test["Quantity"], dow_pred)
            window_results[f"DowBaseline_{stat}_MAE"] = dow_mae

        results.append(window_results)

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# PER-ITEM MODEL EVALUATION
# ---------------------------------------------------------------------------
def evaluate_per_item_models(full, feature_cols):
    """
    For each item with sufficient data, train a dedicated model.
    Compare to global model.
    """
    print("\n=== PER-ITEM MODEL ANALYSIS ===")

    items = sorted(full["Item"].unique())
    per_item_results = []

    for item in items:
        item_data = full[full["Item"] == item].sort_values("Date_Only")
        n_nonzero = (item_data["Quantity"] > 0).sum()

        if n_nonzero < MIN_NONZERO_DAYS:
            continue

        # Last 14 days as test
        test_size = min(14, max(7, n_nonzero // 10))
        train = item_data.iloc[:-test_size]
        test = item_data.iloc[-test_size:]

        # Per-item XGBoost
        try:
            model = SingleStageModel(model_type="xgb", use_poisson=False)
            model.fit(train, train["Quantity"], features=feature_cols)
            preds = model.predict(test)
            mae = mean_absolute_error(test["Quantity"], preds)
        except Exception as e:
            mae = np.nan

        # Moving average baseline
        if len(train) >= 7:
            ma_pred = np.full(len(test), train["Quantity"].tail(7).mean())
            ma_mae = mean_absolute_error(test["Quantity"], ma_pred)
        else:
            ma_mae = np.nan

        per_item_results.append({
            "Item": item,
            "n_train": len(train),
            "n_test": len(test),
            "n_nonzero": n_nonzero,
            "test_mean_qty": test["Quantity"].mean(),
            "xgb_item_mae": mae,
            "ma7_mae": ma_mae,
        })

    per_df = pd.DataFrame(per_item_results)
    per_df.to_csv(os.path.join(TABLES_DIR, "v2_per_item_results.csv"), index=False)
    print(f"Evaluated {len(per_df)} items individually")

    avg_mae = per_df["xgb_item_mae"].mean()
    avg_ma7 = per_df["ma7_mae"].mean()
    print(f"  Avg per-item XGB MAE: {avg_mae:.3f}")
    print(f"  Avg MA7 baseline MAE: {avg_ma7:.3f}")
    if not np.isnan(avg_ma7):
        print(f"  Improvement vs MA7: {(1 - avg_mae/avg_ma7)*100:.1f}%")

    return per_df


# ---------------------------------------------------------------------------
# MODEL COMPARISON
# ---------------------------------------------------------------------------
def run_model_comparison(full, feature_cols):
    """
    Compare multiple model configurations:
    1. XGBoost Single-Stage (baseline)
    2. XGBoost Two-Stage Hurdle
    3. RF Single-Stage
    4. DOW baselines
    """
    print("=" * 70)
    print("EXPANDING WINDOW BACKTEST")
    print("=" * 70)

    # Use data from 2024 onward for efficiency (but include trend)
    train_cutoff = full["Date_Only"].max() - timedelta(days=365*2 - BACKTEST_WINDOW_DAYS * N_BACKTEST_WINDOWS)

    model_configs = {
        "XGB_Single": {
            "factory": lambda **kw: SingleStageModel(model_type="xgb", use_poisson=False, **kw),
            "kwargs": {},
        },
        "XGB_Poisson": {
            "factory": lambda **kw: SingleStageModel(model_type="xgb", use_poisson=True, **kw),
            "kwargs": {},
        },
        "XGB_Hurdle": {
            "factory": lambda **kw: TwoStageHurdleModel(clf_type="xgb", reg_type="xgb", **kw),
            "kwargs": {},
        },
        "RF_Single": {
            "factory": lambda **kw: SingleStageModel(model_type="rf", use_poisson=False, **kw),
            "kwargs": {},
        },
    }

    results = run_expanding_backtest(full, feature_cols, model_configs, train_cutoff)

    print("\n=== SUMMARY ===")
    for col in results.columns:
        if "MAE" in col and "item" not in col:
            valid = results[col].dropna()
            if len(valid) > 0:
                print(f"  {col}: {valid.mean():.3f} ± {valid.std():.3f}")

    results.to_csv(os.path.join(TABLES_DIR, "v2_backtest_results.csv"), index=False)
    return results


# ---------------------------------------------------------------------------
# FEATURE ABLATION
# ---------------------------------------------------------------------------
def run_feature_ablation(full, feature_cols):
    """Run feature ablation for both XGBoost and Random Forest."""
    print("\n=== FEATURE GROUP ABLATION (XGBoost + RF) ===")

    groups = {
        "Temporal": ["DOW", "Is_Weekend", "Month", "Year", "WeekOfYear", "DayOfMonth",
                     "Quarter", "MonthStart", "MonthEnd", "Is_Holiday_Season",
                     "WeekOfMonth", "DaysFromStart", "DOW_Sin", "DOW_Cos",
                     "Month_Sin", "Month_Cos"],
        "Recency": ["Days_Since_Last_Sale", "Sales_Last_7D"],
        "Lags": ["Lag_1", "Lag_7", "Lag_14", "Lag_28"],
        "DOW_Baselines": ["DOW_Avg", "DOW_Median", "DOW_P75", "DOW_N_Samples"],
        "CrossItem": ["Day_Total_Qty", "Day_Total_Items_Sold", "Day_Total_Beverage",
                      "Day_Total_Food", "Day_Total_Qty_7D"],
    }

    available_groups = {k: [f for f in v if f in feature_cols] for k, v in groups.items()}

    train_mask = full["Date_Only"] >= full["Date_Only"].max() - timedelta(days=365)
    train = full[train_mask].copy()
    test = full[full["Date_Only"] > train["Date_Only"].max()].copy()

    if len(test) < 10:
        test = train.tail(max(7, len(train) // 10))
        train = train.iloc[:len(test)]

    base_features = [f for f in feature_cols if f in full.columns]
    X_tr = train[base_features].fillna(0)
    y_tr = train["Quantity"]
    X_te = test[base_features].fillna(0)
    y_te = test["Quantity"]

    ablation = []

    for model_label, build_fn in [
        ("XGB_Single", lambda: SingleStageModel(model_type="xgb", use_poisson=False)),
        ("RF",         lambda: SingleStageModel(model_type="rf" if RandomForestRegressor else "rf", use_poisson=False)),
    ]:
        base_model = build_fn()
        base_model.fit(train, train["Quantity"], features=base_features)
        base_mae = mean_absolute_error(y_te, base_model.predict(test))
        print(f"\n  [{model_label}] All features ({len(base_features)}): MAE = {base_mae:.3f}")

        for group_name, group_feats in available_groups.items():
            if not group_feats:
                continue
            reduced = [f for f in base_features if f not in group_feats]
            if not reduced:
                continue
            model = build_fn()
            model.fit(train, train["Quantity"], features=reduced)
            mae = mean_absolute_error(test["Quantity"], model.predict(test))
            delta = mae - base_mae
            ablation.append({
                "model": model_label,
                "removed_group": group_name,
                "n_removed": len(group_feats),
                "MAE": mae,
                "MAE_delta": delta,
            })
            sign = "+" if delta > 0 else ""
            print(f"    Without {group_name} ({len(group_feats)} feats): MAE = {mae:.3f}  (Δ={sign}{delta:.3f})")

    abl_df = pd.DataFrame(ablation)
    abl_df.to_csv(os.path.join(TABLES_DIR, "v2_ablation.csv"), index=False)

    # Plot: side-by-side comparison
    pivot = abl_df.pivot(index="removed_group", columns="model", values="MAE_delta")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    for ax, model_col in zip(axes, ["XGB_Single", "RF"]):
        deltas = pivot[model_col].reindex(pivot.index[::-1])
        colors_abl = ['#E74C3C' if v > 0 else '#27AE60' for v in deltas.values]
        bars = ax.barh(range(len(deltas)), deltas.values, color=colors_abl)
        ax.axvline(0, color='black', linewidth=0.8)
        ax.set_title(f'{model_col}', fontsize=13, fontweight='bold')
        if ax == axes[0]:
            ax.set_ylabel('Removed Group', fontsize=11)
        ax.set_yticklabels([])
        for bar, val in zip(bars, deltas.values):
            ax.text(bar.get_width() + (0.01 if val > 0 else -0.03),
                    bar.get_y() + bar.get_height()/2,
                    f'{val:+.3f}', va='center', fontsize=9, fontweight='bold')

    axes[0].set_yticklabels(deltas.index, fontsize=9)
    fig.supxlabel('Δ MAE (cups)', fontsize=11, y=0.02)
    fig.suptitle('Feature Group Ablation — XGBoost vs Random Forest', fontsize=14, fontweight='bold')
    plt.tight_layout()

    # Add group labels between the two plots
    handles = [plt.Rectangle((0,0),1,1,facecolor='#E74C3C'), plt.Rectangle((0,0),1,1,facecolor='#27AE60')]
    labels = ['Harmful when removed (kept feature is useful)', 'Helpful when removed (feature is harmful)']
    fig.legend(handles, labels, loc='lower center', ncol=2, fontsize=9, frameon=False)

    fig.savefig(os.path.join(OUT, "ablation_results.png"), dpi=200, bbox_inches='tight')
    plt.close()
    print("  → saved ablation_results.png (XGBoost vs RF comparison)")

    return abl_df, pivot


def plot_xgb_feature_importance(full, feature_cols):
    """Generate XGBoost feature importance plot."""
    from datetime import timedelta
    cutoff = full["Date_Only"].max() - timedelta(days=60)
    train = full[full["Date_Only"] <= cutoff]

    model = xgb.XGBRegressor(
        objective="count:poisson", n_estimators=200, max_depth=4,
        learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.5, reg_lambda=0.5,
        random_state=RANDOM_SEED, verbosity=0,
    )
    model.fit(train[feature_cols].fillna(0), train["Quantity"])
    imp = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)

    top25 = imp.head(25)
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = []
    for feat in top25.index:
        if feat in ['Days_Since_Last_Sale', 'Sales_Last_7D']: colors.append('#E74C3C')
        elif 'Roll' in feat or 'EWMA' in feat or 'Trend' in feat or 'WoW' in feat: colors.append('#3498DB')
        elif feat.startswith('Lag_'): colors.append('#2ECC71')
        elif feat in ['DOW','Is_Weekend','Month','Year','WeekOfYear','DayOfMonth','Quarter',
                       'MonthStart','MonthEnd','Is_Holiday_Season','WeekOfMonth','DaysFromStart',
                       'DOW_Sin','DOW_Cos','Month_Sin','Month_Cos']: colors.append('#F39C12')
        elif feat.startswith('Day_'): colors.append('#9B59B6')
        elif feat in ['Days_Since_First_Sale','Item_Rank','Item_Rank_Pct']: colors.append('#1ABC9C')
        elif feat.startswith('Item_'): colors.append('#95A5A6')
        else: colors.append('#7F8C8D')

    ax.barh(range(len(top25)), top25.values, color=colors)
    ax.set_yticks(range(len(top25)))
    ax.set_yticklabels(top25.index, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel('Feature Importance (gain)', fontsize=12)
    ax.set_title('XGBoost Feature Importance — Top 25 Features', fontsize=14, fontweight='bold')

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#E74C3C', label='Recency'),
        Patch(facecolor='#3498DB', label='Rolling/EWMA'),
        Patch(facecolor='#2ECC71', label='Lags'),
        Patch(facecolor='#F39C12', label='Temporal'),
        Patch(facecolor='#9B59B6', label='Cross-item'),
        Patch(facecolor='#1ABC9C', label='Lifecycle'),
        Patch(facecolor='#95A5A6', label='Item Dummies'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9, framealpha=0.9)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT, "feature_importance_xgb.png"), dpi=200, bbox_inches='tight')
    plt.close()
    print("  → saved feature_importance_xgb.png")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    print("v2_03: FRESH MODEL EXPERIMENTS")
    print()

    full = load_feature_matrix()
    print(f"Loaded feature matrix: {len(full)} rows, {full['Item'].nunique()} items")
    print(f"Date range: {full['Date_Only'].min().date()} → {full['Date_Only'].max().date()}")
    print()

    feature_cols = get_feature_columns(full)
    print(f"Total features: {len(feature_cols)}")
    print()

    # Feature ablation
    abl_df = run_feature_ablation(full, feature_cols)
    print()

    # XGBoost feature importance
    plot_xgb_feature_importance(full, feature_cols)
    print()

    # Per-item analysis
    per_item = evaluate_per_item_models(full, feature_cols)
    print()

    # Full backtest comparison
    backtest_results = run_model_comparison(full, feature_cols)
    print()

    print("=" * 70)
    print("MODEL EXPERIMENTS COMPLETE")
    print("Results in tables/v2_backtest_results.csv, v2_ablation.csv, v2_per_item_results.csv")
    print("=" * 70)


if __name__ == "__main__":
    main()
