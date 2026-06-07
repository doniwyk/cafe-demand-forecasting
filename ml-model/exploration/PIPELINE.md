# Exploration Pipeline

Step-by-step flow from raw data to production forecasts.

**Prerequisites:** Conda env `cafe` with `cafe_db` running on `localhost:5433`.

```
eda/ → features/ → tuning/ → inference/
```

---

## Step 1: EDA — Understand the raw data

### 1a. Data Exploration
```bash
python exploration/eda/data_exploration.py
```
**What it does:** Explores raw sales data structure, temporal patterns (daily/monthly/DOW), top/bottom items by volume, sparsity, missing dates, zero-quantity days, data quality checks.

**Output:** Console report + 5 plots in `figures/data_exploration/`

**Key findings used later:**
- 66 items, Jan 2022 – May 2026, ~36k rows
- 9 missing dates (closures)
- Top item: Kopi Susu Husgendam Ice (avg 4.6/day)

### 1b. Rebranding Effect
```bash
python exploration/eda/rebranding_effect.py
```
**What it does:** Analyzes the May 2025 rebranding impact on raw sales — structural break detection (t-test, Cohen's d), monthly trend post-rebrand, DOW pattern shifts, zero-inflation changes, impact by product tier, menu changes & lift decomposition.

**Output:** Console report + 4 plots in `figures/rebranding_effect/`

**Key findings used later:**
- Cohen's d = 1.675 (LARGE structural break)
- +125% sales lift, effect is strengthening (not fading)
- Weekend lift (+89%) > weekday lift (+77%)
- Recommendation: train on post-rebrand data only

---

## Step 2: Features — Evidence-driven feature analysis

### 2a. Feature Analysis
```bash
python exploration/features/feature_analysis.py
```
**What it does:** Builds all candidate features from raw data, then **computes evidence** for every inclusion/exclusion decision. No hardcoded opinions — every number is derived from the actual data. 8 sections: autocorrelation at each lag, NaN sparsity analysis, inter-feature collinearity (computed r values), target correlation, production model feature importance, ablation study (remove feature groups, measure Fri/Sat impact), Lag_1 vs no-Lag_1 comparison on real Fri/Sat rows, and final exclusion summary with evidence references.

**Output:** Console report (8 sections, all data-backed)

**Key findings:**
- 15 features selected, 12 excluded with computed evidence
- Lag_7 has strongest weekly signal (r=+0.33 at lag-7)
- Lag_182 excluded: 12.5% NaN, weak autocorrelation r=+0.13
- Lag_1 excluded: model with Lag_1 drags Fri predictions toward yesterday's value instead of DOW pattern (Section 7 shows concrete comparison)
- Roll_Std_7 excluded: collinear with Roll_Mean_7 (r=+0.76)
- Top 3 model importance: EWMA_28, Roll_Mean_28, DOW_Avg

---

## Step 3: Tuning — Find optimal hyperparameters

### 3a. Quantile XGBoost Tuning (current/active)
```bash
python exploration/tuning/tune_quantile.py
```
**What it does:** Sequential grid search over 8 XGBoost hyperparameters for quantile regression (q=0.75). Evaluates with pinball loss on a held-out validation set. Includes Fri/Sat sample upweighting (3x) and 3-fold expanding-window cross-validation.

**Output:** Console report + `models/exploration/tuning/quantile_best_params.json`

**Results:**
| Parameter | Before | After Tuning |
|-----------|--------|-------------|
| n_estimators | 600 | **200** |
| max_depth | 5 | **3** |
| colsample_bytree | 0.8 | **1.0** |
| learning_rate | 0.04 | 0.04 (same) |
| Pinball loss | 1.647 | **1.559** (-5.3%) |

### 3b. Legacy: XGBoost + RF Tuning (older pipeline)
```bash
python exploration/tuning/main.py
```
**What it does:** Tunes XGBoost (huber loss) and Random Forest with RMSE. Uses old feature set (includes Lag_1/Diff_1). Used by the `training/` pipeline, not by the current `inference/` pipeline.

**Output:** `models/exploration/tuning/xgboost_best_params.json`, `rf_best_params.json`

---

## Step 4: Inference — Production forecasts

### 4a. Forecast
```bash
python exploration/inference/forecast.py
```
**What it does:** Production inference for all items. Uses a blended approach:
- **Fri/Sat:** 70% DOW_P75 baseline + 30% quantile XGBoost
- **Weekdays:** 60% DOW_Median baseline + 40% quantile XGBoost

Loads tuned params from Step 3a. Trains per-item quantile XGBoost on post-rebrand data with 15 features. Forecasts 7 days ahead.

**Output:** Console table + `models/exploration/inference/forecasts.csv` (all items) + `forecast_metadata.json`

### 4b. Backtest
```bash
python exploration/inference/backtest.py
```
**What it does:** Validates the forecast approach on 5 historical test periods (Mar–May 2026). For each period, trains on data up to the test period, forecasts 7 days, compares predictions vs actuals. Reports overall MAE/RMSE/MAPE, metrics by DOW, by period, by item, accuracy buckets, and bias direction.

**Output:** Console report + `models/exploration/inference/backtest_results.csv`

**Results:**
| Metric | Value |
|--------|-------|
| Overall MAE | 1.35 |
| Overall RMSE | 1.89 |
| Fri+Sat MAE | 1.45 |
| Bias | +0.04 (near zero) |
| Within 50% | 54.0% |

---

## Legacy Pipeline (training/ + evaluation/)

These are from the initial exploration and use a different modeling approach (global + per-item blend with Lag_1/Diff_1 features, huber loss). The current inference pipeline supersedes them.

### Training
```bash
python exploration/training/main.py
```
Trains XGBoost + Random Forest with global/per-item blend, 3-fold CV, ABC evaluation.

### Evaluation
```bash
python exploration/evaluation/evaluate.py
```
Evaluates saved models with expanding-window CV and true holdout.

---

## Directory Structure

```
exploration/
├── config.py                  # Paths, feature list, constants
├── features.py                # Shared feature engineering (used by training/)
│
├── eda/                       # Step 1: Raw data exploration
│   ├── data_exploration.py    #   1a. Data overview, patterns, quality
│   └── rebranding_effect.py   #   1b. Rebranding impact on raw sales
│
├── features/                  # Step 2: Feature analysis
│   └── feature_analysis.py    #   2a. Evidence-driven feature selection
│
├── tuning/                    # Step 3: Hyperparameter optimization
│   ├── tune_quantile.py       #   3a. Quantile XGBoost tuning (active)
│   ├── xgboost_tuning.py      #   3b. Legacy XGBoost tuning
│   ├── rf_tuning.py           #   3b. Legacy RF tuning
│   └── main.py                #   3b. Legacy orchestrator
│
├── inference/                 # Step 4: Production forecasting
│   ├── forecast.py            #   4a. Generate 7-day forecasts
│   └── backtest.py            #   4b. Validate on historical data
│
├── training/                  # Legacy: model training
├── evaluation/                # Legacy: model evaluation
├── figures/                   # Generated plots
└── models/                    # Saved models, params, outputs
```
