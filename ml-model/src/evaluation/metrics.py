import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from typing import Dict, Any


def weighted_mape(y_true: pd.Series, y_pred: pd.Series) -> float:
    return 100 * abs(y_true - y_pred).sum() / y_true.sum()


def per_period_median_accuracy(y_true: pd.Series, y_pred: pd.Series, item_col: pd.Series, min_actual: float = 2) -> float:
    df = pd.DataFrame({"item": item_col, "true": y_true, "pred": y_pred})
    df = df[df["true"] >= min_actual]
    if len(df) == 0:
        return 0.0
    accs = 100 * (1 - (df["pred"] - df["true"]).abs() / df["true"])
    return round(float(accs.median()), 1)


def per_period_within_threshold(y_true: pd.Series, y_pred: pd.Series, item_col: pd.Series, threshold: float = 0.2) -> float:
    df = pd.DataFrame({"item": item_col, "true": y_true, "pred": y_pred})
    mask = df["true"] >= 2
    if mask.sum() == 0:
        return 0.0
    within = (df.loc[mask].eval("abs(pred - true) / true") <= threshold).sum()
    return round(100 * within / mask.sum(), 1)


def compute_metrics(y_true, y_pred) -> Dict[str, float]:
    return {
        "r2": round(r2_score(y_true, y_pred), 4),
        "wmape": round(weighted_mape(y_true, y_pred), 2),
        "mae": round(mean_absolute_error(y_true, y_pred), 2),
    }


def compute_item_metrics(y_true, y_pred, item_col) -> Dict[str, float]:
    return {
        "r2": round(r2_score(y_true, y_pred), 4),
        "wmape": round(weighted_mape(y_true, y_pred), 2),
        "mae": round(mean_absolute_error(y_true, y_pred), 2),
        "median_period_accuracy": per_period_median_accuracy(y_true, y_pred, item_col),
        "periods_within_20pct": per_period_within_threshold(y_true, y_pred, item_col, 0.2),
        "periods_within_50pct": per_period_within_threshold(y_true, y_pred, item_col, 0.5),
    }


def classify_abc(df: pd.DataFrame, volume_col: str = "Quantity_Sold") -> pd.DataFrame:
    item_vol = df.groupby("Item")[volume_col].sum().sort_values(ascending=False)
    total = item_vol.sum()
    item_vol = pd.DataFrame(
        {
            "Vol": item_vol,
            "Cum": item_vol.cumsum(),
            "Pct": item_vol.cumsum() / total,
        }
    )
    item_vol["Class"] = item_vol["Pct"].apply(
        lambda x: "A" if x <= 0.70 else ("B" if x <= 0.90 else "C")
    )
    return item_vol


def generate_abc_analysis(df: pd.DataFrame, frequency: str = "weekly") -> Dict[str, Any]:
    y_true = df["Quantity_Sold"]
    y_pred = df["Predicted"]
    item_col = df["Item"]

    global_metrics = compute_item_metrics(y_true, y_pred, item_col)

    abc_df = classify_abc(df)
    df = df.copy()
    df["Class"] = df["Item"].map(abc_df["Class"])

    period_label = "day" if frequency == "daily" else "week"

    class_metrics = {}
    for c in ["A", "B", "C"]:
        sub = df[df["Class"] == c]
        if len(sub) == 0:
            continue
        class_metrics[c] = {
            "n_items": sub["Item"].nunique(),
            "wmape": round(weighted_mape(sub["Quantity_Sold"], sub["Predicted"]), 1),
            "median_period_acc": per_period_median_accuracy(
                sub["Quantity_Sold"], sub["Predicted"], sub["Item"]
            ),
        }

    top_items = (
        df[df["Class"] == "A"]
        .groupby("Item")[["Quantity_Sold", "Predicted"]]
        .sum()
        .sort_values("Quantity_Sold", ascending=False)
        .head(10)
    )
    top_items["accuracy_pct"] = (
        100
        * (
            1
            - abs(top_items["Predicted"] - top_items["Quantity_Sold"])
            / top_items["Quantity_Sold"]
        )
    ).round(1)

    return {
        "global_metrics": global_metrics,
        "class_metrics": class_metrics,
        "top_items": top_items.reset_index().to_dict("records"),
        "abc_classifications": abc_df.reset_index().to_dict("records"),
        "frequency": frequency,
    }


def print_abc_report(analysis: Dict[str, Any]):
    gm = analysis["global_metrics"]
    freq = analysis.get("frequency", "weekly")
    p = "day" if freq == "daily" else "week"
    P = p.capitalize()
    Ps = p.capitalize() + "s"

    print("\n" + "=" * 90)
    print("MODEL PERFORMANCE")
    print("=" * 90)
    print(f"Global R2              : {gm['r2']:.4f}")
    print(f"Global wMAPE           : {gm['wmape']:.2f}%")
    print(f"Global MAE             : {gm['mae']:.2f}")
    print(f"Median {P} Accuracy    : {gm['median_period_accuracy']:.1f}%")
    print(f"{Ps} within ±20%       : {gm['periods_within_20pct']:.1f}%")
    print(f"{Ps} within ±50%       : {gm['periods_within_50pct']:.1f}%")

    print("\nABC BY CLASS")
    print("-" * 60)
    for c in ["A", "B", "C"]:
        if c not in analysis["class_metrics"]:
            continue
        cm = analysis["class_metrics"][c]
        print(
            f"{c}-Class | Items: {cm['n_items']:2d} | "
            f"wMAPE: {cm['wmape']:5.1f}% | Med.{P[:1]}: {cm['median_period_acc']:5.1f}%"
        )

    print("\nTOP 10 CLASS A ITEMS")
    print("-" * 60)
    for item in analysis["top_items"]:
        print(
            f"  {item['Item']:<25} Actual: {item['Quantity_Sold']:6.0f}  "
            f"Pred: {item['Predicted']:6.0f}  Acc: {item['accuracy_pct']:5.1f}%"
        )
