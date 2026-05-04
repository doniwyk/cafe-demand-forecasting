"""
Cafe Supply Forecasting CLI

Unified entry point for model evaluation and training.

Usage:
    python scripts/04_forecast.py -f weekly train           # Weekly train + 3-month forecast
    python scripts/04_forecast.py -f daily train            # Daily train + 3-month forecast
    python scripts/04_forecast.py train --all               # Train both daily + weekly
    python scripts/04_forecast.py -f weekly evaluate        # Weekly evaluation only
    python scripts/04_forecast.py -f daily evaluate         # Daily evaluation only
    python scripts/04_forecast.py evaluate --all            # Evaluate both daily + weekly
    python scripts/04_forecast.py train --no-forecast       # Train + save models only
    python scripts/04_forecast.py --help
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import argparse

from src.models.features import create_features
from src.models.forecaster import (
    load_and_prep_data,
    train_and_predict,
    train_models,
    generate_future_features,
    predict,
)
from src.evaluation.metrics import generate_abc_analysis, print_abc_report
from src.utils.config import PROCESSED_DIR, SALES_FORECASTING_DIR, PREDICTIONS_DIR, MODELS_DIR, get_feature_columns


def cmd_evaluate(args):
    print("=" * 80)
    print(f"MODEL EVALUATION ({args.frequency.upper()})")
    print("=" * 80)

    df_raw = load_and_prep_data(SALES_FORECASTING_DIR / "daily_item_sales.csv", frequency=args.frequency)

    print("\nCreating features...")
    df_feat = create_features(df_raw, frequency=args.frequency)
    print(f"Features created: {df_feat.shape[1]} columns")

    print(f"\nTraining & evaluating on last 12 {args.frequency} periods...")
    test_pred = train_and_predict(df_feat, frequency=args.frequency)

    analysis = generate_abc_analysis(test_pred, frequency=args.frequency)
    print_abc_report(analysis)


def cmd_train(args):
    print("=" * 80)
    print(f"MODEL TRAINING ({args.frequency.upper()})")
    print("=" * 80)

    freq = args.frequency
    model_dir = MODELS_DIR / freq
    pred_file = PREDICTIONS_DIR / freq / "3_month_forecasts.csv"

    df_raw = load_and_prep_data(SALES_FORECASTING_DIR / "daily_item_sales.csv", frequency=freq)

    print("\nCreating features...")
    df_feat = create_features(df_raw, frequency=freq)
    print(f"Features created: {df_feat.shape[1]} columns")

    print(f"\nSaving models to: {model_dir}")
    item_models, global_model, dow_factors = train_models(df_feat, model_dir, frequency=freq)

    if not args.no_forecast:
        print("\n" + "=" * 80)
        print("GENERATING 3-MONTH FORECAST")
        print("=" * 80)

        future_features = generate_future_features(df_feat, future_weeks=12, frequency=freq)
        print(f"Future feature rows: {len(future_features)}")

        future_predictions = predict(
            future_features,
            item_models=item_models,
            global_model=global_model,
            dow_factor_dict=dow_factors,
            frequency=freq,
        )
        print(f"Predictions generated: {len(future_predictions)} rows")

        pred_file.parent.mkdir(parents=True, exist_ok=True)
        future_predictions[["Date", "Item", "Predicted"]].rename(
            columns={"Predicted": "Quantity_Sold"}
        ).to_csv(pred_file, index=False)
        print(f"Forecasts saved to: {pred_file}")
    else:
        print("\nSkipping forecast generation (--no-forecast)")

    print("\n" + "=" * 80)
    print("MODEL EVALUATION")
    print("=" * 80)

    test_pred = train_and_predict(df_feat, frequency=freq)
    analysis = generate_abc_analysis(test_pred, frequency=freq)
    print_abc_report(analysis)

    print("\nTraining complete!")


def main():
    parser = argparse.ArgumentParser(
        prog="forecast.py",
        description="Cafe Supply Forecasting — evaluate or train XGBoost models",
    )
    parser.add_argument(
        "-f", "--frequency",
        choices=["daily", "weekly"],
        default="weekly",
        help="Prediction frequency (default: weekly)",
    )
    subparsers = parser.add_subparsers(dest="command")

    eval_parser = subparsers.add_parser(
        "evaluate",
        help="Evaluate model metrics on last 12 periods (no model saving)",
    )
    eval_parser.add_argument(
        "--all",
        action="store_true",
        dest="run_all",
        help="Evaluate both daily and weekly frequencies",
    )

    train_parser = subparsers.add_parser(
        "train",
        help="Train models, save .pkl files, and generate 3-month forecast",
    )
    train_parser.add_argument(
        "--no-forecast",
        action="store_true",
        help="Skip 3-month forecast generation, only save models",
    )
    train_parser.add_argument(
        "--all",
        action="store_true",
        dest="run_all",
        help="Train both daily and weekly frequencies",
    )

    args = parser.parse_args()

    if args.command == "evaluate":
        if args.run_all:
            for freq in ["daily", "weekly"]:
                args.frequency = freq
                cmd_evaluate(args)
        else:
            cmd_evaluate(args)
    elif args.command == "train":
        if args.run_all:
            for freq in ["daily", "weekly"]:
                args.frequency = freq
                cmd_train(args)
        else:
            cmd_train(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
