# Exploration — Latest Update (2026-06-07)

## What Changed

### Inference Pipeline (New)

Created `inference/forecast.py` — a production-grade forecasting script that:

- **Loads all items** from `cafe_db` and forecasts the next 7 days
- Uses a **blended approach**: DOW percentile baseline + quantile XGBoost (q=0.75)
  - Fri/Sat: 70% DOW_P75 baseline + 30% model
  - Weekdays: 60% DOW_Median baseline + 40% model
- The DOW baseline captures weekend spikes that tree models systematically underpredict
- Skips items with fewer than 60 non-zero days
- Outputs saved to `models/exploration/inference/forecasts.csv`

### Backtesting (New)

Created `inference/backtest.py` — validates the forecast approach on historical data:

- 5 test periods (Mar–May 2026) × 58 items = 1,238 predictions
- Results: **MAE=1.35, MAPE=70.3%, near-zero bias (+0.04)**
- Fri+Sat: **MAE=1.45, MAPE=83.5%**

### Key Decisions This Session

| Decision | Why |
|----------|-----|
| Removed Lag_1/Diff_1 from inference features | Over-reliance on yesterday's value pulled predictions down on high-demand days |
| Switched to quantile regression (`reg:quantileerror`, q=0.75) | MSE/Huber loss regresses to mean; quantile pushes predictions toward upper range for supply planning |
| Added DOW percentile baselines (P75/P90/Median) | Tree models can't distinguish high vs low Fri/Sat — DOW stats directly encode the historical spike level |
| Chose P75 over P90 for Fri/Sat | Backtested better: Fri+Sat MAE 1.45 (P75) vs 1.91 (P90), bias +0.04 vs +0.26 |
| 12-week lookback for DOW stats, non-zero days only | Cafe closures (qty=0) should not contaminate DOW averages |

### Output Files

```
models/exploration/inference/
├── forecasts.csv              # 7-day forecasts for 58 items (406 rows)
├── forecast_metadata.json     # config snapshot
└── backtest_results.csv       # backtest predictions + errors
```

### How to Run

```bash
# Forecast all items
python exploration/inference/forecast.py

# Backtest
python exploration/inference/backtest.py
```

Requires conda env `cafe` and `cafe_db` running on localhost:5433.
