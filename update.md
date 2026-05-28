# Session Update — May 28, 2026

## Overview

This session covered: weekly→daily forecasting, model accuracy improvements (hyperparameters, features, zero-inflated data), forecast caching, date range filtering UI, and HUS POS database sync integration.

---

## 1. Model Accuracy Improvements

### 1a. Hyperparameter Tuning (XGBoost & Random Forest)

**Root issue**: Per-item model threshold was 100 weekly records (~2 years). No items qualified, so 100% of predictions came from a generic global model. Blend ratio was 15% per-item / 85% global, giving near-zero weight to item-specific patterns. Trees were shallow (max_depth=2-3) with heavy regularization.

**Changes** (`forecaster.py`, `forecaster_rf.py`):
- `MIN_TRAIN_RECORDS`: 100→24 weekly, 180→60 daily → 41 of 61 items get per-item models
- `_BLEND_ALPHA`: 0.15→0.5 (equal weight per-item + global)
- Global params: n_estimators 300→600, max_depth 3→5, learning_rate 0.015→0.03, reg reduced (alpha 5→0.5, lambda 10→1)
- Item params: n_estimators 150→400, max_depth 2→4, learning_rate 0.015→0.03, reg reduced
- `_EARLY_STOPPING_ROUNDS`: 15→30

**Result**: XGBoost R² from ~0.83→0.93, wMAPE 21%→6%.

### 1b. Daily Feature Expansion

**New features** (`features.py` + `config.py`):
- `DayOfMonth`, `IsWeekend`, `Sin_DOW`, `Cos_DOW` — day-level temporal signals
- `Lag_8`, `Lag_13` (weekly), `Lag_3`, `Lag_28` (daily) — longer-range lags
- `Momentum`, `Seasonal_Strength` — trend and periodicity signals
- `Days_Post_Ramadan` — days since last Ramadan ended (0 during Ramadan, capped at 60), enables model to learn post-Ramadan demand recovery curve

### 1c. Zero-Sale Day Filling

**Root issue**: `daily_item_sales` only stores rows where `quantity_sold > 0`. Zero-sale days were missing, so the model learned every item sells every day. "Anindita Honey Butter" (1 sale/week) appeared to sell daily.

**Fix** (`_clean_data` in engine.py, `get_forecasts` in forecast_service.py):
- `groupby("Item").resample("D").sum().fillna(0)` — fills missing dates with 0 for each item
- Training now includes both sale and zero-sale days → model learns realistic sparsity

**Result**: Anindita Honey Butter now shows 0 on 11/15 days, 1 on 4/15 days (matching real sparse pattern). Popular items like Kopi Susu Husgendam Ice predict 4-6/day instead of inflated 10/day.

### 1d. Objective Function: Tweedie → Poisson

- XGBoost objective changed from `reg:tweedie` to `count:poisson` — better for count data

### 1e. Smarter Rounding for Slow Items

- Predictions with `raw_adj < 1.5` use `floor()` instead of `round()` — values like 0.8 become 0 instead of 1
- Applied to XGBoost, RF, SARIMAX, Prophet predict functions

---

## 2. Weekly → Daily Forecasting

### 2a. Pipeline Switch

**Files**: `engine.py`

- Removed `_to_weekly()` aggregation → replaced with `_clean_data()` that keeps daily granularity + fills zero days
- `_FREQUENCY = "daily"` passed to all training and inference calls
- Training minimum records: 60 for daily
- Forecast produces 84 daily predictions (was 12 weekly) per 12-week window

### 2b. Batch-Recursive Prediction

**Root issue**: Original code filled future dates with `Quantity_Sold=0`, causing lag features to cascade to zero → predictions decayed to zero by week 8-12.

**Fix** (`generate_forecast` in engine.py):
- XGBoost/RF: predict in 7-day batches (4 batches for 28 days instead of 28 daily iterations)
- Each batch's predictions feed back as pseudo-observed values for next batch's feature computation
- 5.5x speedup: 21s→3.8s for 4-week forecast
- SARIMAX/Prophet: also use recursive day-by-day approach

### 2c. Forecast Caching

**Files**: `forecast_service.py`
- Module-level `_forecast_cache` dict per model type
- First request generates 8+ weeks of forecast → caches result
- Subsequent requests filter from cache (~0.1s)
- Auto-invalidated on retrain via `invalidate_forecast_cache(model_type)`

### 2d. Smart Date Range Generation

- Backend calculates `forecast_weeks` dynamically based on requested end_date
- Defaults to 4 weeks when no date range specified
- Capped to cover exactly the requested range + buffer

---

## 3. HUS POS Database Sync

### 3a. Sync Script

**New file**: `web/backend/scripts/sync_hus_sales.py`

- Queries hus_db (PostgreSQL 15 on port 5432) for completed orders
- Matches POS `product_name_snapshot + variant_name_snapshot` to `cafe_forecasting.items`
- Skips add-ons (Add Espresso, Add Telur, etc.), V60 filter coffees, and new products by default
- Fallback map for products missing variants (e.g. "Kopi Susu Husgendam" → "Kopi Susu Husgendam Ice")
- `--include-new` flag adds unmatched products to `items` table automatically
- Uses `ON CONFLICT DO UPDATE` for idempotent inserts into `daily_item_sales`
- Fetched 2,181 rows (5,662 units) of April-May 2026 data from hus_db

### 3b. Retrain Integration

**Files**: `routers/forecasts.py`, `models/forecast.py`, `settings.tsx`

- `RetrainRequest` gains `sync_hus: bool` and `include_new_products: bool` fields
- When `sync_hus=true`, the sync script runs before training — logs appear in training progress
- Settings page has two checkboxes: "Sync from hus_db before training" and "Include new products"

---

## 4. UI Improvements

### 4a. Forecasts Page Date Range Picker

**File**: `frontend/src/routes/forecasts.tsx`

- Calendar component with range mode (2-month view) for filtering
- Item selector now sends `item` param to API (was present in UI but not passed to API)
- `page_size` removed from frontend params — backend defaults to 10,000 (all records)
- Default shows 4 weeks of data without requiring date selection
- Data table below chart shows per-date predictions when an item is selected
- **Past date handling**: when user picks dates before forecast window, returns actual historical sales from `daily_item_sales` instead of empty results

### 4b. i18n

**Files**: `en.json`, `id.json`
- Added: `forecasts.item`, `forecasts.date`, `forecasts.dateRange`, `forecasts.selectDateRange`

### 4c. New Components

- `checkbox.tsx`, `label.tsx` — shadcn/ui components for settings page

---

## 5. Files Changed

| File | Changes |
|------|---------|
| `ml-model/src/models/forecaster.py` | Hyperparams, Poisson objective, zero-threshold rounding, batch feature gen |
| `ml-model/src/models/forecaster_rf.py` | Hyperparams, zero-threshold rounding |
| `ml-model/src/models/forecaster_sarimax.py` | MIN_TRAIN lowered, FIT_KWARGS maxiter 5→20, threshold rounding |
| `ml-model/src/models/forecaster_prophet.py` | MIN_TRAIN lowered, threshold rounding |
| `ml-model/src/models/features.py` | New features: DayOfMonth, IsWeekend, Sin/Cos_DOW, Lag_8/13/3/28, Momentum, Seasonal_Strength, Days_Post_Ramadan |
| `ml-model/src/utils/config.py` | Updated FEATURE_COLUMNS and FEATURE_COLUMNS_DAILY |
| `web/backend/app/ml/engine.py` | Daily frequency, `_clean_data` with zero-fill, batch-recursive forecast, feature import for Days_Post_Ramadan |
| `web/backend/app/services/forecast_service.py` | Forecast caching, daily resampling + zero-fill, date range → actual/forecast routing |
| `web/backend/app/routers/forecasts.py` | Sync integration in retrain, cache invalidation |
| `web/backend/app/routers/materials.py` | model_type param support |
| `web/backend/app/services/material_service.py` | model_type param |
| `web/backend/app/services/recipe_material_service.py` | model_type param, NaN fix in recipe matching |
| `web/backend/app/services/analytics_service.py` | Live inference for ABC analysis (replaced Forecast table dependency) |
| `web/backend/app/models/forecast.py` | sync_hus, include_new_products fields |
| `web/frontend/src/routes/forecasts.tsx` | Date range picker, data table, per-item API filtering, snapshot for past dates |
| `web/frontend/src/routes/settings.tsx` | Sync/new-products checkboxes |
| `web/frontend/src/hooks/use-forecasts.ts` | sync params in retrain mutation |
| `web/frontend/src/hooks/use-materials.ts` | model_type from context |
| `web/frontend/src/lib/api.ts` | model_type in material endpoints |
| `web/frontend/src/lib/locales/en.json` | New translation keys |
| `web/frontend/src/lib/locales/id.json` | New translation keys |
| `web/backend/scripts/sync_hus_sales.py` | **New** — HUS POS sync script |
| `web/frontend/src/components/ui/checkbox.tsx` | **New** |
| `web/frontend/src/components/ui/label.tsx` | **New** |
| `documentation.md` | **New** — full system documentation |
| `.gitignore` | Added `docs/` |
