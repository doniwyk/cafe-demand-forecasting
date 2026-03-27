# Cafe Supply Forecasting — Project Plan

## Overview

Build an end-to-end cafe supply forecasting system that predicts daily sales of menu items, derives raw material procurement needs, and presents everything on an interactive dashboard. The system uses **XGBoost** for forecasting, **FastAPI** as the backend API, and **React + Vite + TypeScript + shadcn/ui** for the frontend.

---

## Architecture

```
┌─────────────────────────────────────────────────┐
│                   FastAPI Server                │
│  ┌───────────────────────────────────────────┐  │
│  │         /api/*  (REST endpoints)          │  │
│  │  - Sales data (historical + forecast)     │  │
│  │  - Raw material requirements              │  │
│  │  - ABC analysis & model metrics           │  │
│  │  - Association rules                      │  │
│  │  - Trigger retrain / re-predict           │  │
│  └───────────────┬───────────────────────────┘  │
│                  │                               │
│  ┌───────────────▼───────────────────────────┐  │
│  │       ML Engine (XGBoost models)          │  │
│  │  - Pre-computed CSV forecasts (default)   │  │
│  │  - On-demand prediction via API           │  │
│  └───────────────────────────────────────────┘  │
│                                                  │
│  ┌───────────────────────────────────────────┐  │
│  │       Static Files (React build)           │  │
│  │       /*  → served from /dist             │  │
│  └───────────────────────────────────────────┘  │
└─────────────────────────────────────────────────┘
```

Single deployment: FastAPI serves both the API and the built React frontend.

---

## Phase 1 — ML Module Refactoring

**Goal**: Move logic from `scripts/` notebooks into a reusable Python package under `ml-model/src/`.

### 1.1 Refactor `ml-model/src/` package structure

```
ml-model/src/
├── data/
│   ├── __init__.py
│   ├── loader.py            # Load CSVs from data/raw, data/processed
│   ├── transformer.py       # SalesDataTransformer (from 001_data_transformation.py)
│   ├── cleaner.py           # SalesDataCleaner (from 02_clean_sales_data.py)
│   └── merger.py            # Merge sales files (from 01_merge_sales_data.py)
├── models/
│   ├── __init__.py
│   ├── forecaster.py        # XGBRegressor model, train & predict (from 002_data_forecast.py)
│   ├── features.py          # Feature engineering: calendar, lags, rolling, rebranding
│   └── raw_materials.py     # RawMaterialProcessor (from 03_preprocess_raw_materials.py)
├── evaluation/
│   ├── __init__.py
│   └── metrics.py           # wMAPE, MAE, R², ABC classification
└── utils/
    ├── __init__.py
    └── config.py            # Paths, constants, holidays list
```

### 1.2 Tasks

- [ ] Extract `SalesDataTransformer` class into `src/data/transformer.py`
- [ ] Extract feature engineering functions into `src/models/features.py`
- [ ] Extract XGBoost training + prediction into `src/models/forecaster.py`
- [ ] Extract `RawMaterialProcessor` into `src/models/raw_materials.py`
- [ ] Extract evaluation / ABC analysis into `src/evaluation/metrics.py`
- [ ] Create `src/utils/config.py` with centralized paths and constants
- [ ] Refactor `scripts/` to use `src/` package imports (keep scripts as CLI entry points)
- [ ] Add `ml-model/pyproject.toml` with package metadata
- [ ] Save trained models as `.pkl` / `.json` files in `ml-model/models/`

### 1.3 Model persistence

- After training, serialize per-item XGBoost models to `ml-model/models/`
- Store model metadata: feature list, training date, performance metrics
- Global fallback model saved separately

---

## Phase 2 — FastAPI Backend

**Goal**: Build REST API serving historical data, forecasts, raw material needs, and model metrics.

### 2.1 Backend structure

```
web/backend/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI app, CORS, static file mount
│   ├── config.py            # Settings (paths, model paths, etc.)
│   ├── routers/
│   │   ├── __init__.py
│   │   ├── sales.py         # Historical sales endpoints
│   │   ├── forecasts.py     # Forecast data + trigger re-predict
│   │   ├── materials.py     # Raw material requirements
│   │   ├── analytics.py     # ABC analysis, metrics, association rules
│   │   └── items.py         # Item list, categories, metadata
│   ├── services/
│   │   ├── __init__.py
│   │   ├── sales_service.py
│   │   ├── forecast_service.py
│   │   ├── material_service.py
│   │   └── analytics_service.py
│   ├── models/              # Pydantic schemas
│   │   ├── __init__.py
│   │   ├── sales.py
│   │   ├── forecast.py
│   │   └── material.py
│   └── ml/                  # ML engine integration
│       ├── __init__.py
│       └── engine.py        # Load models, run predictions
├── requirements.txt
└── run.py                   # uvicorn entry point
```

### 2.2 API endpoints

#### Sales
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/sales/daily` | Daily item sales (paginated, filterable by item/date range) |
| GET | `/api/sales/daily/total` | Total daily sales across all items |
| GET | `/api/sales/daily/category` | Category-level daily aggregates |
| GET | `/api/sales/items` | List all unique items |
| GET | `/api/sales/categories` | List all categories |

#### Forecasts
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/forecasts` | Pre-computed forecast data (filterable by item/date range) |
| GET | `/api/forecasts/summary` | Forecast summary with accuracy metrics |
| POST | `/api/forecasts/predict` | Trigger on-demand prediction for given item(s) |
| POST | `/api/forecasts/retrain` | Trigger full model retraining |

#### Raw Materials
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/materials/daily` | Daily raw material requirements |
| GET | `/api/materials/forecast` | Projected raw material needs based on sales forecast |

#### Analytics
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/analytics/abc` | ABC classification analysis |
| GET | `/api/analytics/metrics` | Model performance metrics (wMAPE, R², MAE) |
| GET | `/api/analytics/top-items` | Top selling items by volume/revenue |
| GET | `/api/analytics/association-rules` | Item association rules |

### 2.3 Tasks

- [ ] Initialize FastAPI project with `requirements.txt` (fastapi, uvicorn, pandas, numpy, xgboost, scikit-learn, pydantic)
- [ ] Create `app/main.py` with CORS, router inclusion, static file serving
- [ ] Implement sales router + service
- [ ] Implement forecasts router + service (pre-computed CSV loading)
- [ ] Implement ML engine (`app/ml/engine.py`) — load serialized models, run predict
- [ ] Implement on-demand prediction endpoint
- [ ] Implement retrain endpoint (long-running, consider background task)
- [ ] Implement raw materials router + service
- [ ] Implement analytics router + service (ABC, metrics, association rules)
- [ ] Add Pydantic response models for all endpoints
- [ ] Add query parameter filtering (item, date range, category, pagination)
- [ ] Configure static file serving for React build output

---

## Phase 3 — React Frontend

**Goal**: Build interactive dashboard with shadcn/ui components showing forecasts, analytics, and raw material projections.

### 3.1 Frontend structure

```
web/frontend/
├── index.html
├── vite.config.ts
├── tsconfig.json
├── package.json
├── tailwind.config.ts
├── components.json          # shadcn/ui config
├── src/
│   ├── main.tsx
│   ├── App.tsx
│   ├── index.css
│   ├── lib/
│   │   ├── api.ts           # Axios/fetch client for /api/*
│   │   └── utils.ts         # shadcn/ui utils
│   ├── components/
│   │   ├── ui/              # shadcn/ui components (button, card, table, etc.)
│   │   ├── layout/
│   │   │   ├── Sidebar.tsx
│   │   │   ├── Header.tsx
│   │   │   └── Layout.tsx
│   │   ├── charts/
│   │   │   ├── SalesForecastChart.tsx    # Actual vs Predicted line chart
│   │   │   ├── DailySalesTrend.tsx       # Historical sales trend
│   │   │   ├── CategoryPieChart.tsx      # Sales by category
│   │   │   ├── ABCAnalysisChart.tsx      # ABC classification bar chart
│   │   │   ├── TopItemsChart.tsx         # Top items horizontal bar
│   │   │   ├── MaterialRequirementChart.tsx
│   │   │   └── AccuracyMetrics.tsx       # wMAPE, R², MAE cards
│   │   └── tables/
│   │       ├── SalesTable.tsx            # Daily sales data table
│   │       ├── ForecastTable.tsx         # Forecast results table
│   │       ├── MaterialTable.tsx         # Raw material requirements table
│   │       └── ItemSelector.tsx          # Multi-select item filter
│   ├── pages/
│   │   ├── Dashboard.tsx                 # Overview: KPI cards + top charts
│   │   ├── SalesForecast.tsx             # Detailed forecast view per item
│   │   ├── RawMaterials.tsx              # Raw material projection view
│   │   ├── Analytics.tsx                 # ABC analysis, association rules
│   │   └── Settings.tsx                  # Retrain model, config
│   ├── hooks/
│   │   ├── useSales.ts
│   │   ├── useForecasts.ts
│   │   ├── useMaterials.ts
│   │   └── useAnalytics.ts
│   └── types/
│       └── index.ts                      # TypeScript interfaces
```

### 3.2 Pages

#### Dashboard (Home)
- KPI cards: Total sales (today/this week/this month), # active items, model accuracy
- Sales trend chart (last 30 days actual)
- Top 10 items by quantity
- Category breakdown pie chart

#### Sales Forecast
- Item selector (multi-select dropdown)
- Date range picker
- Line chart: Actual vs Predicted quantities over time
- Forecast accuracy metrics (wMAPE, MAE, R²) per item
- Table with daily actual/predicted values
- ABC class badge per item

#### Raw Materials
- Date range picker
- Table: Raw material, predicted daily requirement, unit
- Bar chart: top 20 materials by projected requirement
- Option to aggregate by week/month

#### Analytics
- ABC classification table with item counts and accuracy per class
- Association rules table (antecedent → consequent, support, confidence, lift)
- Top items table with volume and accuracy
- Model performance summary

#### Settings
- "Retrain Model" button (triggers POST /api/forecasts/retrain)
- Training status indicator
- Model metadata display (last trained date, feature count, etc.)

### 3.3 Tech stack

| Tool | Purpose |
|------|---------|
| Vite | Build tool & dev server |
| TypeScript | Type safety |
| React 18+ | UI framework |
| React Router | Client-side routing |
| Tailwind CSS | Styling |
| shadcn/ui | Component library |
| Recharts | Charts |
| TanStack Query | Data fetching & caching |
| TanStack Table | Data tables |
| Axios | HTTP client |

### 3.4 Tasks

- [ ] Initialize Vite + React + TypeScript project
- [ ] Install and configure Tailwind CSS + shadcn/ui
- [ ] Set up React Router with page structure
- [ ] Build layout: Sidebar navigation + Header + main content area
- [ ] Create TypeScript types/interfaces for API responses
- [ ] Create API client (`lib/api.ts`) with base URL config
- [ ] Build custom hooks (useSales, useForecasts, useMaterials, useAnalytics)
- [ ] Build Dashboard page (KPI cards, trend chart, top items, category pie)
- [ ] Build Sales Forecast page (item selector, actual vs predicted chart, table)
- [ ] Build Raw Materials page (table, bar chart, date range)
- [ ] Build Analytics page (ABC table, association rules, model metrics)
- [ ] Build Settings page (retrain button, status, metadata)
- [ ] Add loading states, error handling, empty states
- [ ] Responsive design (mobile-friendly sidebar collapse)
- [ ] Configure Vite build output to `web/backend/static/` for FastAPI serving

---

## Phase 4 — Integration & Deployment

### 4.1 Tasks

- [ ] Configure FastAPI to serve React build from `static/`
- [ ] Set up `vite build` output path to `../backend/static`
- [ ] Create `run.py` entry point: `uvicorn app.main:app --host 0.0.0.0 --port 8000`
- [ ] Add API proxy config for Vite dev server (proxy `/api` to FastAPI port)
- [ ] Create `Makefile` or `justfile` for common commands:
  - `make dev-backend` — start FastAPI
  - `make dev-frontend` — start Vite dev server
  - `make dev` — start both concurrently
  - `make build` — build React + prepare for production
  - `make train` — run ML training pipeline
- [ ] Write basic integration tests (FastAPI TestClient)
- [ ] Add `.env.example` with configurable paths

---

## Phase 5 — Documentation & Polish

### 5.1 Tasks

- [ ] Update root `README.md` with project overview and quick start
- [ ] Update `ml-model/README.md` with updated src/ package docs
- [ ] Add `web/backend/README.md` with API documentation
- [ ] Add `web/frontend/README.md` with setup instructions
- [ ] Add OpenAPI/Swagger description enhancements to FastAPI
- [ ] (Optional) Docker setup for single-command deployment

---

## Execution Order

```
Phase 1 (ML Refactoring)
    │
    ├── 1.1 Refactor src/ package          ~2-3 days
    ├── 1.2 Model persistence               ~1 day
    │
    ▼
Phase 2 (FastAPI Backend)                   ~3-4 days
    │
    ├── 2.1 Project setup + sales router
    ├── 2.2 Forecasts router + ML engine
    ├── 2.3 Materials + analytics routers
    ├── 2.4 Filtering, pagination, schemas
    │
    ▼
Phase 3 (React Frontend)                    ~4-5 days
    │
    ├── 3.1 Project setup + layout
    ├── 3.2 Dashboard page
    ├── 3.3 Sales Forecast page
    ├── 3.4 Raw Materials + Analytics pages
    ├── 3.5 Settings + polish
    │
    ▼
Phase 4 (Integration)                       ~1-2 days
    │
    ├── 4.1 Static file serving + dev proxy
    ├── 4.2 Makefile / scripts
    ├── 4.3 Tests
    │
    ▼
Phase 5 (Documentation)                     ~1 day
```

**Estimated total: ~12-16 days**

---

## Data Flow Summary

```
Raw Sales CSVs (ID + EN)
        │
        ▼
   [01_merge]  ──►  sales_data.csv
        │
        ▼
   [02_clean]  ──►  sales_data_cleaned.csv
        │
        ├──────────────────┐
        ▼                  ▼
   [001_transform]    [03_preprocess]
        │                  │
        ▼                  ▼
   daily_item_sales    daily_raw_material
   daily_category      _requirements.csv
   daily_total_sales
        │
        ▼
   [002_forecast]  ──►  3_month_forecasts.csv
                         trained_models/*.pkl
        │
        ▼
   ┌─────┴─────┐
   ▼           ▼
FastAPI     React Dashboard
(/api/*)    (charts, tables, KPIs)
```
