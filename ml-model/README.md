# Cafe Supply Forecasting — ML Model

XGBoost-based sales forecasting system for a cafe. Predicts daily/weekly item-level sales, derives raw material procurement needs, and evaluates model performance via ABC classification.

## Quick Start

```bash
conda activate cafe
cd ml-model
pip install -e .   # one-time install for CLI command
```

### Full Pipeline

```bash
# 0. Pull latest BOM from database
python scripts/00_pull_from_hus_db.py

# 1. Merge Indonesian + English sales CSVs
python scripts/01_merge_sales_data.py

# 2. Clean & standardize item names, remove discontinued
python scripts/02_clean_sales_data.py --remove

# 3. Aggregate to daily item sales
python scripts/03_transform_sales.py

# 4. Forecast item sales (train + evaluate + save)
python scripts/04_forecast.py -f weekly train

# 5. Map forecast to raw material procurement needs
python scripts/05_forecast_to_materials.py -f weekly
```

### Forecast CLI (04_forecast.py)

```bash
python scripts/04_forecast.py -f weekly train          # Train + save + forecast
python scripts/04_forecast.py -f daily evaluate        # Evaluate only (no save)
python scripts/04_forecast.py -f weekly train --no-forecast  # Train + save only
```

**`evaluate` output includes:**
- **Global R²** — overall goodness of fit
- **Global wMAPE** — weighted mean absolute percentage error (lower = better)
- **Global MAE** — average absolute error in units
- **Volume Accuracy** — how close total predicted volume is to actual
- **ABC class breakdown** — accuracy per A/B/C item class
- **Top 10 Class A items** — per-item accuracy for highest-volume items

**`train` output files:**
- `models/{daily,weekly}/global_model.pkl` — global fallback XGBoost model
- `models/{daily,weekly}/item_models.pkl` — per-item XGBoost models (dict)
- `models/{daily,weekly}/dow_factors.json` — day-of-week adjustment factors
- `models/{daily,weekly}/model_metadata.json` — training date, feature list, item list
- `data/predictions/{daily,weekly}/3_month_forecasts.csv` — 12-week future forecast

### Raw Material Forecast (05_forecast_to_materials.py)

```bash
python scripts/05_forecast_to_materials.py -f weekly
```

Takes `3_month_forecasts.csv` × BOM → `raw_material_forecast.csv` with columns: `Date, Raw_Material, Quantity_Required`.

### Load Models in Python

```python
from src.models.forecaster import load_models, predict

item_models, global_model, dow_factors = load_models()
predictions = predict(future_feature_df)
```

## Project Structure

```
ml-model/
├── data/
│   ├── raw/
│   │   ├── bom/
│   │   │   ├── menu_bom.csv         # Menu item → ingredient mapping
│   │   │   └── condiment_bom.csv    # Condiment → sub-ingredient mapping
│   │   └── sales/                   # Raw POS sales exports (ID + EN)
│   ├── processed/
│   │   ├── sales_data.csv           # Merged raw sales
│   │   ├── sales_data_cleaned.csv   # Standardized + filtered sales
│   │   └── sales_forecasting/       # Daily aggregated files
│   │       ├── daily_item_sales.csv
│   │       ├── daily_category_sales.csv
│   │       └── daily_total_sales.csv
│   └── predictions/
│       └── {daily,weekly}/
│           ├── 3_month_forecasts.csv
│           └── raw_material_forecast.csv
├── models/{daily,weekly}/
│   ├── global_model.pkl
│   ├── item_models.pkl
│   ├── dow_factors.json
│   └── model_metadata.json
├── scripts/
│   ├── 00_pull_from_hus_db.py       # Pull BOM from database
│   ├── 01_merge_sales_data.py       # Merge ID + EN CSVs
│   ├── 02_clean_sales_data.py       # Clean & standardize
│   ├── 03_transform_sales.py        # Aggregate to daily sales
│   ├── 04_forecast.py               # Train/evaluate XGBoost models
│   └── 05_forecast_to_materials.py  # Forecast → raw material needs
├── src/
│   ├── data/
│   │   ├── loader.py
│   │   ├── merger.py
│   │   ├── cleaner.py
│   │   └── transformer.py
│   ├── models/
│   │   ├── features.py
│   │   ├── forecaster.py
│   │   └── raw_materials.py
│   ├── evaluation/
│   │   └── metrics.py
│   └── utils/
│       └── config.py
├── tests/
├── pyproject.toml
├── environment.yml
└── requirements.txt
```

## Scripts Reference

| Script | Input | Output | Description |
|--------|-------|--------|-------------|
| `00_pull_from_hus_db.py` | hus_db (PostgreSQL) | `menu_bom.csv`, `condiment_bom.csv` | Pull BOM data from database |
| `01_merge_sales_data.py` | `data/raw/sales/*.csv` | `sales_data.csv` | Merges Indonesian & English POS exports |
| `02_clean_sales_data.py` | `sales_data.csv` + BOM | `sales_data_cleaned.csv` | Renames items, removes discontinued |
| `03_transform_sales.py` | `sales_data_cleaned.csv` + BOM | `sales_forecasting/*.csv` | Aggregates to daily sales |
| `04_forecast.py train` | `daily_item_sales.csv` | `models/*.pkl` + forecast CSV | Train, save models, generate forecast |
| `04_forecast.py evaluate` | `daily_item_sales.csv` | stdout | Evaluate model metrics |
| `05_forecast_to_materials.py` | `3_month_forecasts.csv` + BOMs | `raw_material_forecast.csv` | Map forecast to raw material needs |

## Model Details

- **Algorithm**: XGBoost with Tweedie loss (`reg:tweedie`, variance_power=1.5)
- **Granularity**: Daily or weekly per-item forecasts
- **Features** (28): Calendar (month, week, day-of-year), seasonality (sin/cos week), trend, payday/holiday/Ramadan flags, rebranding indicators, lags (1,2,4), rolling stats (mean 4/12, std 4, q95 4), EWMA (4/12), momentum, post-rebrand surge ratio
- **Strategy**: Per-item models for items with >=40 training weeks; global fallback otherwise
- **Post-processing**: Day-of-week factor adjustment applied to raw predictions
- **ABC Classification**: A (top 70% volume), B (70-90%), C (bottom 10%)

## Environment Setup

```bash
conda env create -f environment.yml
conda activate cafe

# Or manually
conda create -n cafe python=3.13
conda activate cafe
pip install -r requirements.txt
```
