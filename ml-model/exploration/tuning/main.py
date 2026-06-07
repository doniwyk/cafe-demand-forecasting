"""Run both XGBoost and RF tuning, then compare."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import xgboost_tuning
import rf_tuning


def main():
    print("=" * 70)
    print("HYPERPARAMETER TUNING PIPELINE")
    print("=" * 70)

    print("\n" + "=" * 70)
    print("1/2 TUNING XGBOOST")
    print("=" * 70)
    xgb_params = xgboost_tuning.main()

    print("\n" + "=" * 70)
    print("2/2 TUNING RANDOM FOREST")
    print("=" * 70)
    rf_params = rf_tuning.main()

    print("\n" + "=" * 70)
    print("TUNING COMPLETE")
    print("=" * 70)
    print(f"\nXGBoost params saved to: models/exploration/tuning/xgboost_best_params.json")
    print(f"RF params saved to:      models/exploration/tuning/rf_best_params.json")
    print(f"\nNext step: run training/main.py to train final models with these params.")


if __name__ == "__main__":
    main()
