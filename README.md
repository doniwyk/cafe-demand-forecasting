# Cafe Supply Forecasting

ML-powered cafe supply forecasting system with multi-model support.

## Architecture

```
web/
  backend/          FastAPI (Python) - REST API + ML engine
  frontend/         React + Vite + TanStack Router + shadcn/ui
ml-model/           Scikit-learn / XGBoost / Prophet / Statsmodels models
```

## Supported Models

| Model | Description | Persistence |
|---|---|---|
| **XGBoost** | Gradient boosting with per-item + global fallback | Full (train/load/predict) |
| **Random Forest** | Ensemble with per-item + global fallback | Full (train/load/predict) |
| **Prophet** | Facebook Prophet with weekly seasonality | Full (train/load/predict) |
| **SARIMAX** | Seasonal ARIMA with stationarity detection | Full (train/load/predict) |

## Getting Started

### Dev Mode (hot reload)

```bash
docker compose -f docker-compose.dev.yml up
```

- Frontend: http://localhost:5174 (Vite HMR)
- Backend: http://localhost:8000 (uvicorn --reload)
- DB: localhost:5433

Source changes to `web/frontend/src/`, `web/backend/`, and `ml-model/` are reflected immediately.

### Prod Mode (static builds)

```bash
docker compose -f docker-compose.prod.yml up --build
```

- Frontend: http://localhost:3000 (nginx)
- Backend: http://localhost:8000
- DB: localhost:5433

## Key Features

### Global Model Selector

A model type dropdown in the header bar lets you switch between XGBoost, Random Forest, SARIMAX, and Prophet. All pages (Dashboard, Forecasts, Analytics) react to the selection and fetch data filtered by the chosen model.

### Settings Page (`/settings`)

- **Train Individual Models** - Train any model type independently
- **Train All (Sequential)** - Trains all 4 models one at a time
- **Live Status** - Real-time status badges (Training / Trained / Error) with 5s polling
- **Data Cleanup** - Delete stale model runs and forecasts from previous training sessions

### How Model Selection Works

1. The `ModelContext` in React stores the active `model_type`
2. All React Query hooks pass `model_type` to the API as a query parameter
3. Backend filters DB queries by `ModelRun.model_type` and `is_active = True`
4. Each model trains independently and stores its own `ModelRun` + `Forecast` rows
5. Only the latest active run per model type is shown

### API Endpoints

| Endpoint | Model Type Support |
|---|---|
| `GET /api/forecasts` | `?model_type=xgboost` |
| `GET /api/forecasts/summary` | `?model_type=sarimax` |
| `GET /api/analytics/metrics` | `?model_type=prophet` |
| `POST /api/forecasts/predict` | `{ "model_type": "random_forest" }` |
| `POST /api/forecasts/retrain` | `{ "model_type": "sarimax" }` |
| `GET /api/forecasts/retrain/status` | Per-model status dict |
| `POST /api/forecasts/cleanup` | Delete inactive runs + forecasts |

### Retrain Behavior

- Each retrain creates a new `ModelRun` row and sets the old one to `is_active = False`
- New forecasts are linked to the new run via `model_run_id`
- Old data is not deleted automatically (use the Cleanup button in Settings)
- Only one model type can be trained at a time per model

## Project Structure

```
web/backend/app/
  ml/engine.py              Multi-model dispatcher (train/load/predict)
  models/forecast.py        Pydantic schemas (PredictRequest, RetrainRequest)
  routers/forecasts.py      Forecast endpoints with model_type support
  routers/analytics.py      Analytics endpoints with model_type support
  services/
    forecast_service.py     Forecast business logic
    analytics_service.py    Analytics business logic

web/frontend/src/
  contexts/model-context.tsx  Global model type state (ModelProvider)
  hooks/use-forecasts.ts     React Query hooks with model_type
  hooks/use-analytics.ts     React Query hooks with model_type
  routes/
    __root.tsx               Layout with model selector dropdown
    index.tsx                Dashboard
    forecasts.tsx            Sales Forecast page
    analytics.tsx            Analytics page
    settings.tsx             Model training & cleanup management

ml-model/src/models/
  forecaster.py             XGBoost models
  forecaster_rf.py          Random Forest models
  forecaster_sarimax.py     SARIMAX models
  forecaster_prophet.py     Prophet models
```
