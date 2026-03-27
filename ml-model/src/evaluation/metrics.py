import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from typing import Dict, Any


def weighted_mape(y_true: pd.Series, y_pred: pd.Series) -> float:
    return 100 * abs(y_true - y_pred).sum() / y_true.sum()


def volume_accuracy(y_true: pd.Series, y_pred: pd.Series) -> float:
    return 100 * (1 - abs(y_true.sum() - y_pred.sum()) / y_true.sum())


def compute_metrics(y_true, y_pred) -> Dict[str, float]:
    return {
        "r2": round(r2_score(y_true, y_pred), 4),
        "wmape": round(weighted_mape(y_true, y_pred), 2),
        "mae": round(mean_absolute_error(y_true, y_pred), 2),
        "volume_accuracy": round(volume_accuracy(y_true, y_pred), 2),
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


def generate_abc_analysis(df: pd.DataFrame) -> Dict[str, Any]:
    y_true = df["Quantity_Sold"]
    y_pred = df["Predicted"]

    global_metrics = compute_metrics(y_true, y_pred)

    abc_df = classify_abc(df)
    df = df.copy()
    df["Class"] = df["Item"].map(abc_df["Class"])

    class_metrics = {}
    for c in ["A", "B", "C"]:
        sub = df[df["Class"] == c]
        if len(sub) == 0:
            continue
        class_metrics[c] = {
            "n_items": sub["Item"].nunique(),
            "wmape": round(weighted_mape(sub["Quantity_Sold"], sub["Predicted"]), 1),
            "volume_accuracy": round(
                volume_accuracy(sub["Quantity_Sold"], sub["Predicted"]), 1
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
    }


def print_abc_report(analysis: Dict[str, Any]):
    gm = analysis["global_metrics"]
    print("\n" + "=" * 90)
    print("MODEL PERFORMANCE")
    print("=" * 90)
    print(f"Global R2        : {gm['r2']:.4f}")
    print(f"Global wMAPE     : {gm['wmape']:.2f}%")
    print(f"Global MAE       : {gm['mae']:.2f}")
    print(f"Total Vol Acc    : {gm['volume_accuracy']:.2f}%")

    print("\nABC BY CLASS")
    print("-" * 60)
    for c in ["A", "B", "C"]:
        if c not in analysis["class_metrics"]:
            continue
        cm = analysis["class_metrics"][c]
        print(
            f"{c}-Class | Items: {cm['n_items']:2d} | "
            f"wMAPE: {cm['wmape']:5.1f}% | Vol.Acc: {cm['volume_accuracy']:5.1f}%"
        )

    print("\nTOP 10 CLASS A ITEMS")
    print("-" * 60)
    for item in analysis["top_items"]:
        print(
            f"  {item['Item']:<25} Actual: {item['Quantity_Sold']:6.0f}  "
            f"Pred: {item['Predicted']:6.0f}  Acc: {item['accuracy_pct']:5.1f}%"
        )
