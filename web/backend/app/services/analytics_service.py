import pandas as pd

from app.config import DAILY_ITEM_SALES_PATH, ASSOCIATION_RULES_PATH
from app.models.analytics import ABCItem, ABCAnalysisResponse, AssociationRule
from src.evaluation.metrics import classify_abc


def get_abc_analysis() -> ABCAnalysisResponse:
    df = pd.read_csv(DAILY_ITEM_SALES_PATH)
    abc_df = classify_abc(df, volume_col="Quantity_Sold")
    class_metrics = {}
    for cls in ["A", "B", "C"]:
        sub = abc_df[abc_df["Class"] == cls]
        if len(sub) == 0:
            continue
        class_metrics[cls] = {
            "n_items": len(sub),
            "total_volume": float(sub["Vol"].sum()),
            "pct_volume": float(sub["Pct"].iloc[-1] * 100) if len(sub) > 0 else 0,
        }

    classifications = [
        ABCItem(
            item=row["Item"] if "Item" in row.index else row.index[0],
            vol=float(row["Vol"]),
            cum=float(row["Cum"]),
            pct=float(row["Pct"]),
            class_label=str(row["Class"]),
        )
        for _, row in abc_df.iterrows()
    ]

    return ABCAnalysisResponse(
        class_metrics=class_metrics,
        classifications=classifications,
    )


def get_metrics():
    df = pd.read_csv(DAILY_ITEM_SALES_PATH)
    from app.ml.engine import run_evaluate

    analysis = run_evaluate(df)
    return analysis["global_metrics"]


def get_top_items(n: int = 20) -> list[dict]:
    df = pd.read_csv(DAILY_ITEM_SALES_PATH)
    top = df.groupby("Item")["Quantity_Sold"].sum().sort_values(ascending=False).head(n)
    return [{"item": item, "total_quantity": float(qty)} for item, qty in top.items()]


def get_association_rules(
    min_confidence: float = 0.3,
    min_lift: float = 1.0,
) -> list[AssociationRule]:
    try:
        df = pd.read_csv(ASSOCIATION_RULES_PATH)
    except FileNotFoundError:
        return []

    df = df[df["confidence"] >= min_confidence]
    df = df[df["lift"] >= min_lift]
    df = df.sort_values("lift", ascending=False).head(100)

    rules = []
    for _, row in df.iterrows():
        rules.append(
            AssociationRule(
                antecedents=str(row.get("antecedents", "")),
                consequents=str(row.get("consequents", "")),
                support=float(row.get("support", 0)),
                confidence=float(row.get("confidence", 0)),
                lift=float(row.get("lift", 0)),
            )
        )
    return rules
