# Model Comparison: XGBoost vs Random Forest vs SARIMAX vs Prophet

## Evaluation Setup

- **Train/Test Split**: Last 12 weeks held out for evaluation (2025-07-07 to 2025-09-29)
- **Training Period**: 2022-01-03 to 2025-06-30 (~183 weeks)
- **Items**: 72 menu items across ABC classes (28 A, 18 B, 26 C)
- **Granularity**: Weekly aggregation (W-MON)

## Global Metrics

| Metric               | XGBoost   | Random Forest | SARIMAX | Prophet   |
|----------------------|-----------|---------------|---------|-----------|
| **R2**               | **0.8199**| 0.7311        | 0.4174  | -5.8943   |
| **wMAPE (%)**        | **21.95** | 31.80         | 47.41   | 74.96     |
| **MAE**              | **1.61**  | 2.33          | 3.48    | 5.49      |
| **Volume Accuracy (%)**| 83.30  | **97.53**     | 89.70   | 96.14     |

## ABC Class Breakdown

### wMAPE by Class (lower is better)

| Class | XGBoost | Random Forest | SARIMAX | Prophet |
|-------|---------|---------------|---------|---------|
| A     | **19.1**| 27.5          | 42.6    | 82.5    |
| B     | **23.1**| 38.2          | 52.9    | 52.1    |
| C     | **38.4**| 47.7          | 68.2    | 69.6    |

### Volume Accuracy by Class (higher is better)

| Class | XGBoost | Random Forest | SARIMAX | Prophet |
|-------|---------|---------------|---------|---------|
| A     | 85.6    | **94.7**      | 85.2    | 90.2    |
| B     | 83.9    | 96.4          | **96.9**| 99.4    |
| C     | 67.0    | **95.6**      | 93.8    | 73.8    |

## Model Details

### XGBoost (Primary Model)

- **Algorithm**: `XGBRegressor` with Tweedie loss (`reg:tweedie`, variance_power=1.5)
- **Strategy**: Per-item models (n_estimators=1500) + global fallback (n_estimators=2000)
- **Features**: 28 engineered features (lags, rolling stats, EWMA, calendar, trend indicators)
- **Post-processing**: Day-of-week (DOW) factor adjustment
- **Strengths**: Best per-item accuracy across all metrics and ABC classes. Tweedie loss is well-suited for count-like sales data with right-skewed distributions.
- **Weaknesses**: Lower volume accuracy (83.3%) — tends to underpredict total volume slightly.

### Random Forest

- **Algorithm**: `RandomForestRegressor` (sklearn)
- **Strategy**: Per-item models (300 trees, max_depth=12) + global fallback (500 trees, max_depth=15)
- **Features**: Same 28 engineered features as XGBoost
- **Post-processing**: Same DOW factor adjustment
- **Strengths**: Best total volume accuracy (97.5%). Robust across all ABC classes for volume.
- **Weaknesses**: Higher per-item wMAPE (31.8%) — individual predictions are less precise, especially for high-variance items.

### SARIMAX

- **Algorithm**: `SARIMAX` from statsmodels — classical time series model
- **Strategy**: Per-item models with ADF stationarity check. Falls back to simpler (1,1,0) config on convergence failure.
- **Config**: Auto-selects between `(1,1,1)(1,1,1,52)` (non-stationary) and `(2,0,2)` (stationary)
- **Fallback**: Global mean for items with < 52 weeks of data
- **Strengths**: Reasonable volume accuracy (89.7%) without feature engineering. Classical interpretability.
- **Weaknesses**: Poor per-item accuracy (wMAPE 47.4%). Many items have insufficient history for reliable seasonal estimation. Convergence failures on several items. Cannot leverage cross-item patterns or external features.

### Prophet

- **Algorithm**: `Prophet` from Meta — additive decomposition model (trend + seasonality + holidays)
- **Strategy**: Per-item models with yearly + weekly (period=7) seasonality
- **Config**: `changepoint_prior_scale=0.05`, `seasonality_prior_scale=10.0`
- **Fallback**: Global mean for items with < 26 weeks of data
- **Strengths**: Good volume accuracy (96.1%). Handles trend changes automatically via changepoint detection.
- **Weaknesses**: Worst per-item accuracy (wMAPE 74.96%, R2 = -5.89). Designed for daily granularity with strong trends — weekly cafe sales are too sparse and noisy. Extreme overpredictions on some items (e.g., Pisang Goreng Aren: 423 predicted vs 157 actual).

## Key Findings

1. **XGBoost is the best overall model** — it dominates on all point-accuracy metrics (R2, wMAPE, MAE) across all ABC classes. The combination of rich feature engineering and Tweedie loss makes it well-suited for this forecasting task.

2. **Random Forest excels at volume accuracy** — if the goal is to avoid over/under-stocking in aggregate, RF's 97.5% volume accuracy is notable. It trades per-item precision for better aggregate estimates.

3. **Classical baselines (SARIMAX, Prophet) underperform significantly** — they lack access to cross-item features, lag/rolling statistics, and domain-specific engineered features that the tree-based models leverage. Their univariate nature is a limitation for this multi-item forecasting problem.

4. **Feature engineering is the key differentiator** — XGBoost and RF share the same 28 features. Their advantage over SARIMAX/Prophet comes primarily from these features (lags, rolling means, rebranding indicators, trend ratios) rather than the algorithms themselves.

5. **C-class items are hardest to predict** for all models — these low-volume, intermittent items have high variance relative to their mean, making accurate prediction inherently difficult.

## Ranking

| Rank | Model          | Best At                        |
|------|----------------|--------------------------------|
| 1    | XGBoost        | Per-item accuracy (R2, wMAPE)  |
| 2    | Random Forest  | Total volume accuracy          |
| 3    | SARIMAX        | Classical baseline, interpretable |
| 4    | Prophet        | Trend-heavy daily forecasting (not well-suited here) |
