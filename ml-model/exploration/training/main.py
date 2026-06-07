"""Main training orchestrator - runs both XGBoost and Random Forest."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import train_xgboost
import train_rf


def main():
    print("=" * 70)
    print("TRAINING PIPELINE (Exploration)")
    print("=" * 70)

    print("\n" + "=" * 70)
    print("1/2 TRAINING XGBOOST")
    print("=" * 70)
    xgb_metrics = train_xgboost.main()

    print("\n" + "=" * 70)
    print("2/2 TRAINING RANDOM FOREST")
    print("=" * 70)
    rf_metrics = train_rf.main()

    print("\n" + "=" * 70)
    print("CROSS-VALIDATION COMPARISON")
    print("=" * 70)
    print(f"\n{'Model':<25s} {'RMSE':>10s} {'MAE':>10s} {'R²':>10s} {'wMAPE':>10s} {'±20%':>10s} {'±50%':>10s}")
    print("-" * 87)
    print(f"{'XGBoost':<25s} {xgb_metrics['rmse']:10.2f} {xgb_metrics['mae']:10.2f} {xgb_metrics['r2']:10.4f} {xgb_metrics['wmape']:9.2f}% {xgb_metrics['periods_within_20pct']:9.1f}% {xgb_metrics['periods_within_50pct']:9.1f}%")
    print(f"{'Random Forest':<25s} {rf_metrics['rmse']:10.2f} {rf_metrics['mae']:10.2f} {rf_metrics['r2']:10.4f} {rf_metrics['wmape']:9.2f}% {rf_metrics['periods_within_20pct']:9.1f}% {rf_metrics['periods_within_50pct']:9.1f}%")

    if xgb_metrics["rmse"] < rf_metrics["rmse"]:
        print(f"\n  -> WINNER: XGBoost (RMSE {xgb_metrics['rmse']:.2f} vs {rf_metrics['rmse']:.2f})")
    else:
        print(f"\n  -> WINNER: Random Forest (RMSE {rf_metrics['rmse']:.2f} vs {xgb_metrics['rmse']:.2f})")


if __name__ == "__main__":
    main()
