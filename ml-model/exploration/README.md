# Cafe Supply Forecasting — Exploration Pipeline

Daily demand forecasting for 58 cafe items using quantile XGBoost + DOW baseline blending.

**Prerequisites:** Conda env `cafe` with PostgreSQL `cafe_db` running on `localhost:5433`.

```
eda/ → features/ → tuning/ → training/ → inference/
```

## Pipeline

### Step 1: EDA — Understand raw data

**1a. Data Exploration**
```bash
python exploration/eda/data_exploration.py
```
Explores raw sales data: structure, temporal patterns (daily/monthly/DOW), top/bottom items, sparsity, missing dates, zero-quantity days.

Output: Console report + 5 plots in `figures/data_exploration/`

**1b. Rebranding Effect**
```bash
python exploration/eda/rebranding_effect.py
```
Analyzes the May 2025 rebranding impact: structural break detection (t-test, Cohen's d=1.675), monthly trend, DOW pattern shifts, zero-inflation changes, impact by product tier.

Output: Console report + 4 plots in `figures/rebranding_effect/`

Key finding: **Train on post-rebrand data only** (+125% sales lift, effect still strengthening).

### Step 2: Features — Evidence-driven selection

```bash
python exploration/features/feature_analysis.py
```

Builds all 23 candidate features from raw data, then computes evidence for every inclusion/exclusion decision. 8 sections of data-backed analysis:

1. Autocorrelation at every lag
2. NaN sparsity per feature
3. Inter-feature collinearity (computed r)
4. Target correlation ranking
5. Production model feature importance
6. Ablation study (remove groups, measure Fri/Sat impact)
7. Lag_1 vs no-Lag_1 comparison on real rows
8. Final exclusion summary with evidence references

**16 features selected, 12 excluded:**

| Feature | Target r | Importance | Why included |
|---------|----------|------------|-------------|
| EWMA_28 | +0.46 | 0.103 | Long-term level (1st importance) |
| Roll_Mean_28 | +0.44 | 0.094 | Long-term smoothed average |
| DOW_Avg | +0.07 | 0.085 | Per-DOW historical average |
| EWMA_7 | +0.43 | 0.082 | Short-term level |
| Roll_Mean_7 | +0.45 | 0.079 | Recent weekly average |
| DOW | +0.06 | 0.072 | Day-of-week index |
| **Momentum** | +0.41 | 0.064 | (Roll_Mean_7 - DOW_Avg) / DOW_Avg — captures recent strength vs usual |
| Lag_28 | +0.23 | 0.061 | 4-week back same-day |
| Lag_7 | +0.33 | 0.061 | Weekly same-day reference |
| DOW_P75 | +0.05 | 0.058 | 75th percentile for this DOW |
| Lag_14 | +0.21 | 0.057 | Biweekly same-day reference |
| Trend_7 | +0.12 | 0.056 | Short vs long trend direction |
| DOW_Std | -0.04 | 0.053 | Per-DOW volatility |
| DOW_Median | +0.02 | 0.049 | Per-DOW median |
| DOW_P90 | +0.01 | 0.026 | 90th percentile for this DOW |
| Is_Weekend | +0.12 | 0.000 | Weekend flag (captured by DOW) |

Excluded with evidence: Lag_1 (drags Fri predictions to yesterday), Diff_1 (same), Lag_182 (12.5% NaN), Roll_Std_7 (collinear r=0.76), Weekly_Ratio/Seasonal_Diff (derivable from Lag_7/28), Accel_2 (noise), Roll_Q95_7, Seasonal_Strength, Monthly_Ratio, IsPostRebrand, MonthsSinceRebrand.

### Step 3: Tuning — Hyperparameter optimization

**3a. Quantile XGBoost**
```bash
python exploration/tuning/tune_quantile.py
```
Sequential grid search over 8 XGBoost hyperparameters for quantile regression (q=0.75). Pinball loss on held-out validation. 3-fold expanding-window CV.

Output: Console report + `models/exploration/tuning/quantile_best_params.json`

Tuned params:
| Parameter | Value |
|-----------|-------|
| n_estimators | 200 |
| max_depth | 3 |
| learning_rate | 0.02 |
| min_child_weight | 1 |
| subsample | 0.8 |
| colsample_bytree | 0.8 |
| reg_alpha | 1.0 |
| reg_lambda | 1.0 |

**3b. Blend Weights**
```bash
python exploration/tuning/tune_blend.py
```
Pre-computes raw XGB+RF predictions for 5 backtest periods (1238 predictions), then sequential grid search over baseline percentile, model weights, and RF weight. Evaluates with pinball loss.

Output: Console report + `models/exploration/tuning/blend_best_params.json`

Tuned blend:
| Parameter | Value |
|-----------|-------|
| weekend_baseline | P75 |
| weekend_model_w | 0.8 (80% model, 20% baseline) |
| weekday_model_w | 0.6 (60% model, 40% DOW_Median) |
| rf_weight | 0.0 (XGB-only wins for pinball loss) |

### Step 4: Model Comparison — XGBoost vs Random Forest

```bash
python exploration/training/model_comparison.py
```
Trains both XGBoost (quantile) and Random Forest on same data with 16 features. 3-fold expanding window CV with side-by-side metrics (RMSE, MAE, R2, wMAPE, accuracy buckets). Includes ABC analysis.

Output: Console report + `models/exploration/training/`

### Step 5: Inference — Production forecasts

**5a. Forecast**
```bash
python exploration/inference/forecast.py
```
Production inference for all 58 items. Per-item XGB trained on post-rebrand data with 16 features + 3x Fri/Sat upweight. Recursive 7-day forecasting with blend:

- **Fri/Sat:** 80% XGB prediction + 20% DOW_P75 baseline
- **Weekdays:** 60% XGB prediction + 40% DOW_Median baseline
- **Final output:** rounded to whole cups (>= 0.2 fractional rounds up)

Output: Console table + `models/exploration/inference/forecasts.csv`

**5b. Backtest**
```bash
python exploration/inference/backtest.py
```
5 historical test periods (Mar–May 2026) x 58 items = 1238 predictions.

Output: Console report + `models/exploration/inference/backtest_results.csv`

Results:
| Metric | Value |
|--------|-------|
| Overall MAE | 1.34 |
| Overall RMSE | 1.82 |
| Fri+Sat MAE | 1.38 |
| Bias | +0.20 (slight overpredict) |
| Within 50% | 54.0% |
| Within 100% | 75.1% |

---

## Architecture

```
forecast_item() recursive loop:
  for each forecast day:
    1. build_item_features() — compute all 16 features including Momentum
    2. xgb.predict() — raw quantile prediction
    3. blend with DOW baseline — 80/20 Fri+Sat, 60/40 weekdays
    4. feed blended value back as "yesterday" for next day's features
  after loop: round to whole cups (forecast_single only)
```

```
exploration/
├── config.py                  # Paths, legacy feature list (inference ignores)
├── features.py                # Legacy feature builder (inference ignores)
│
├── eda/                       # Step 1: Raw data exploration
│   ├── data_exploration.py
│   └── rebranding_effect.py
│
├── features/                  # Step 2: Feature analysis
│   └── feature_analysis.py    #   8-section evidence-driven analysis
│
├── tuning/                    # Step 3: Hyperparameter optimization
│   ├── tune_quantile.py       #   XGB quantile tuning
│   └── tune_blend.py          #   Blend weight search
│
├── training/                  # Step 4: Model comparison
│   ├── model_comparison.py    #   XGB vs RF side-by-side
│   └── metrics.py             #   Shared metrics (ABC, wMAPE)
│
├── inference/                 # Step 5: Production forecasting
│   ├── forecast.py            #   7-day forecasts (all items)
│   └── backtest.py            #   Historical validation
│
├── evaluation/                # Legacy: initial model evaluation
├── figures/                   # Generated plots
└── models/                    # Saved params, outputs (gitignored)
    └── exploration/
        └── tuning/
            ├── quantile_best_params.json
            ├── rf_best_params.json
            └── blend_best_params.json
```

## Key Design Decisions

| Decision | Evidence |
|----------|----------|
| Post-rebrand data only | Cohen's d=1.675 (large break), +125% lift |
| No Lag_1/Diff_1 | Section 7: model with Lag_1 drags Fri predictions to yesterday |
| Quantile regression (q=0.75) | MSE regresses to mean; quantile pushes toward upper range for supply |
| DOW_P75 baseline for Fri/Sat | P90/P95 increase bias (+0.21/+0.32) without improving pinball |
| XGB-only (rf_weight=0.0) | RF predicts mean, diluting quantile signal; pinball optimized for q=0.75 |
| 3x Fri/Sat sample upweight | Emphasizes weekend patterns in training |
| Momentum feature | 7th importance; high Fri/Sat (>=10) have Momentum +3.4 vs low (<5) -0.4 |
| 12-week DOW lookback, non-zero only | Cafe closures (qty=0) are NOT low demand |
| Round to whole cups at output only | Internal recursion uses decimals to avoid cascading error |
