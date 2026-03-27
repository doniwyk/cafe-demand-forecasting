# Cafe Supply Forecasting — ML Model

XGBoost-based sales forecasting system for a cafe. Predicts weekly item-level sales, derives raw material procurement needs, and evaluates model performance via ABC classification.

## Quick Start

```bash
conda activate cafe
cd ml-model
pip install -e .   # one-time install for CLI command
```

### Full Pipeline (from raw data to trained models)

```bash
# 1. Merge Indonesian + English sales CSVs
python scripts/01_merge_sales_data.py

# 2. Clean & standardize item names (prompts for discontinued item removal)
python scripts/02_clean_sales_data.py --remove

# 3. Aggregate to daily item sales
python scripts/001_data_transformation.py

# 4. Train models & generate forecasts (saves .pkl files)
forecast train
```

### Forecast CLI

```bash
forecast <command> [options]
```

| Command | Description |
|---------|-------------|
| `forecast evaluate` | Evaluate model metrics on last 12 weeks (no model saving) |
| `forecast train` | Train models, save `.pkl` files, generate 3-month forecast |
| `forecast train --no-forecast` | Train and save models only, skip forecast generation |

**Examples:**

```bash
# Quick evaluation — see R², wMAPE, ABC analysis, top item accuracy
forecast evaluate

# Full train + save + forecast
forecast train

# Train + save models only (no 3-month forecast CSV)
forecast train --no-forecast

# Show help
forecast --help
```

**`evaluate` output includes:**
- **Global R²** — overall goodness of fit
- **Global wMAPE** — weighted mean absolute percentage error (lower = better)
- **Global MAE** — average absolute error in units
- **Volume Accuracy** — how close total predicted volume is to actual
- **ABC class breakdown** — accuracy per A/B/C item class
- **Top 10 Class A items** — per-item accuracy for highest-volume items

**`train` output files:**
- `models/global_model.pkl` — global fallback XGBoost model
- `models/item_models.pkl` — per-item XGBoost models (dict)
- `models/dow_factors.json` — day-of-week adjustment factors
- `models/model_metadata.json` — training date, feature list, item list
- `data/predictions/3_month_forecasts.csv` — 12-week future forecast

**Console output** includes the same evaluation metrics as `002_data_forecast.py` plus model saving confirmation.

### Load Models in Python

```python
from src.models.forecaster import load_models, predict

item_models, global_model, dow_factors = load_models()
# or from custom path:
item_models, global_model, dow_factors = load_models("path/to/models")

predictions = predict(future_feature_df)
```

## Project Structure

```
ml-model/
├── data/
│   ├── raw/
│   │   ├── bom/                     # Bill of materials
│   │   │   ├── menu_bom.csv         # Menu item -> ingredient mapping
│   │   │   └── condiment_bom.csv    # Condiment -> sub-ingredient mapping
│   │   └── sales/                   # Raw POS sales exports (ID + EN)
│   ├── processed/
│   │   ├── sales_data.csv           # Merged raw sales
│   │   ├── sales_data_cleaned.csv   # Standardized + filtered sales
│   │   ├── daily_item_sales.csv     # Daily quantity per item
│   │   ├── daily_category_sales.csv # Daily quantity per category
│   │   └── daily_total_sales.csv    # Daily total across all items
│   └── predictions/
│       └── 3_month_forecasts.csv    # Generated 12-week forecast
├── models/
│   ├── global_model.pkl             # Global fallback XGBRegressor
│   ├── item_models.pkl              # Per-item XGBRegressor dict
│   ├── dow_factors.json             # Day-of-week scaling factors
│   └── model_metadata.json          # Training metadata
├── notebooks/
│   ├── 001_data_transformation.ipynb
│   ├── 002_eda.ipynb
│   └── 003_data_forecast.ipynb
├── scripts/
│   ├── 01_merge_sales_data.py       # Merge ID + EN CSVs
│   ├── 02_clean_sales_data.py       # Clean & standardize
│   ├── 03_preprocess_raw_materials.py  # Compute raw material needs
│   ├── 001_data_transformation.py   # Aggregate to daily sales
│   ├── 002_data_forecast.py         # Quick forecast + ABC eval (no save)
│   └── train_models.py              # Full training: save .pkl + forecast
├── src/
│   ├── data/
│   │   ├── loader.py                # CSV loading utilities
│   │   ├── merger.py                # Indonesian/English file merger
│   │   ├── cleaner.py               # SalesDataCleaner class
│   │   └── transformer.py           # SalesDataTransformer class
│   ├── models/
│   │   ├── features.py              # Calendar, lag, rolling, EWMA features
│   │   ├── forecaster.py            # train_models, load_models, predict
│   │   └── raw_materials.py         # RawMaterialProcessor (BOM expansion)
│   ├── evaluation/
│   │   └── metrics.py               # wMAPE, R², MAE, ABC classification
│   └── utils/
│       └── config.py                # Paths, holidays, feature list
├── tests/
├── pyproject.toml
├── environment.yml
└── requirements.txt
```

## Scripts Reference

| Script | Input | Output | Description |
|--------|-------|--------|-------------|
| `forecast.py evaluate` | `daily_item_sales.csv` | stdout | Evaluate model metrics, no saving |
| `forecast.py train` | `daily_item_sales.csv` | `models/*.pkl` + forecast CSV | Train, save models, generate forecast |
| `forecast.py train --no-forecast` | `daily_item_sales.csv` | `models/*.pkl` | Train and save models only |
| `01_merge_sales_data.py` | `data/raw/sales/*.csv` | `sales_data.csv` | Merges Indonesian & English POS exports |
| `02_clean_sales_data.py` | `sales_data.csv` + `menu_bom.csv` | `sales_data_cleaned.csv` | Renames items, expands packages, removes discontinued |
| `001_data_transformation.py` | `sales_data_cleaned.csv` + `menu_bom.csv` | `daily_item_sales.csv` etc. | Aggregates to daily, adds temporal features |
| `03_preprocess_raw_materials.py` | `sales_data_cleaned.csv` + BOMs | `daily_raw_material_requirements.csv` | Computes daily raw material needs |

## src/ Package API

### Data Loading

```python
from src.data.loader import (
    load_daily_item_sales,      # -> pd.DataFrame
    load_daily_category_sales,  # -> pd.DataFrame
    load_daily_total_sales,     # -> pd.DataFrame
    load_forecasts,             # -> pd.DataFrame
    load_menu_bom,              # -> pd.DataFrame
    load_condiment_bom,         # -> pd.DataFrame
)
```

### Feature Engineering

```python
from src.models.features import create_features

# Expects columns: Date, Item, Quantity_Sold
# Adds: Month, Week, DOY, DOW, Sin_Week, Cos_Week, Lag_*, Roll_*, EWMA_*, etc.
df_features = create_features(df)
```

### Training & Prediction

```python
from src.models.forecaster import train_models, load_models, predict, generate_future_features

# Train and save
item_models, global_model, dow_factors = train_models(df_features, output_dir="models/")

# Load saved models
item_models, global_model, dow_factors = load_models(model_dir="models/")

# Predict on existing feature DataFrame
predictions = predict(df_features, item_models=item_models, global_model=global_model, dow_factor_dict=dow_factors)

# Generate future feature vectors (for 12 weeks ahead)
future_features = generate_future_features(df_daily, future_weeks=12)
future_preds = predict(future_features)
```

### Evaluation

```python
from src.evaluation.metrics import compute_metrics, generate_abc_analysis, print_abc_report

# Requires columns: Quantity_Sold, Predicted
metrics = compute_metrics(df["Quantity_Sold"], df["Predicted"])
# -> {"r2": 0.85, "wmape": 32.1, "mae": 2.3, "volume_accuracy": 97.5}

analysis = generate_abc_analysis(df)
# -> {"global_metrics": {...}, "class_metrics": {...}, "top_items": [...], "abc_classifications": [...]}

print_abc_report(analysis)  # Pretty-print to stdout
```

### Raw Material Requirements

```python
from src.models.raw_materials import RawMaterialProcessor

processor = RawMaterialProcessor(
    sales_path="data/processed/sales_data_cleaned.csv",
    menu_bom_path="data/raw/bom/menu_bom.csv",
    condiment_bom_path="data/raw/bom/condiment_bom.csv",
)
requirements = processor.process_sales_data()
# -> DataFrame with columns: Date, Raw_Material, Quantity_Required

# Or compute from any sales DataFrame directly
requirements = processor.compute_material_requirements(sales_df)
```

## Model Details

- **Algorithm**: XGBoost with Tweedie loss (`reg:tweedie`, variance_power=1.5)
- **Granularity**: Weekly per-item forecasts
- **Features** (28): Calendar (month, week, day-of-year), seasonality (sin/cos week), trend, payday/holiday/Ramadan flags, rebranding indicators, lags (1,2,4), rolling stats (mean 4/12, std 4, q95 4), EWMA (4/12), momentum, post-rebrand surge ratio
- **Strategy**: Per-item models for items with >=40 training weeks; global fallback otherwise
- **Post-processing**: Day-of-week factor adjustment applied to raw predictions
- **ABC Classification**: A (top 70% volume), B (70-90%), C (bottom 10%)

## Environment Setup

```bash
# Create from environment.yml
conda env create -f environment.yml
conda activate cafe

# Or manually
conda create -n cafe python=3.13
conda activate cafe
pip install pandas numpy scikit-learn xgboost
```

To update after installing new packages:

```bash
./export-env.sh
```
