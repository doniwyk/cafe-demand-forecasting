from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from app.config import DAILY_ITEM_SALES_PATH, FORECAST_SUMMARY_PATH
from app.ml.engine import run_evaluate


def main():
    print("Reading daily item sales...")
    df_daily = pd.read_csv(DAILY_ITEM_SALES_PATH)

    print("Running evaluation pipeline (this may take a while)...")
    analysis = run_evaluate(df_daily)

    summary = {
        "global_metrics": analysis["global_metrics"],
        "class_metrics": {
            cls: metrics for cls, metrics in analysis["class_metrics"].items()
        },
        "top_items": analysis["top_items"],
    }

    FORECAST_SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(FORECAST_SUMMARY_PATH, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Forecast summary saved to {FORECAST_SUMMARY_PATH}")


if __name__ == "__main__":
    main()
