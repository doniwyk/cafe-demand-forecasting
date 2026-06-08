"""
exploration_v2 config
fresh start — no imports from exploration/
"""
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "..", "data", "processed", "sales_forecasting")
FIGURES_DIR = os.path.join(ROOT, "figures")
MODELS_DIR = os.path.join(ROOT, "models")
TABLES_DIR = os.path.join(ROOT, "tables")

DAILY_SALES_PATH = os.path.join(DATA_DIR, "daily_item_sales.csv")

RANDOM_SEED = 42
MIN_NONZERO_DAYS = 30
N_BACKTEST_WINDOWS = 8
BACKTEST_WINDOW_DAYS = 7
