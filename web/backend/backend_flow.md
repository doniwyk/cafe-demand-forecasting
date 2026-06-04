# Backend Flow

## Project Structure

```
web/backend/
├── run.py                          # Entry point
├── requirements.txt                # Dependencies
├── alembic.ini                     # DB migration config
├── app/
│   ├── main.py                     # FastAPI app factory, CORS, SPA fallback
│   ├── config.py                   # Paths, DB URL, JWT settings
│   ├── core/
│   │   ├── constants.py            # Shared constants (MODEL_TYPES, SQL fragments)
│   │   └── deps.py                 # Shared FastAPI deps (get_session, require_auth)
│   ├── db/
│   │   ├── base.py                 # SQLAlchemy DeclarativeBase
│   │   ├── engine.py               # Async + sync engines/sessions
│   │   └── models.py               # All ORM models
│   ├── models/                     # Pydantic request/response schemas
│   ├── repositories/               # Data access layer
│   │   ├── __init__.py             # BaseRepository (count, paginate)
│   │   ├── sales_repository.py     # Sales DB queries
│   │   ├── forecast_repository.py  # Forecast/model DB queries
│   │   └── material_repository.py  # Material DB queries
│   ├── services/                   # Business logic
│   │   ├── auth.py                 # JWT + bcrypt (AuthService)
│   │   ├── sales_service.py        # Sales CRUD
│   │   ├── forecast_service.py     # Forecast generation + cache
│   │   ├── material_service.py     # Material queries + BOM-based forecast
│   │   ├── recipe_material_service.py  # BOM → material mapping (sync session)
│   │   ├── analytics_service.py    # ABC analysis, metrics
│   │   └── retrain_service.py      # Background model training orchestration
│   ├── ml/
│   │   └── engine.py               # ML model registry, train/predict/forecast
│   └── routers/                    # HTTP route handlers
│       ├── auth.py                 # POST /login, GET /me
│       ├── sales.py                # GET /daily, /items, /categories
│       ├── forecasts.py            # GET /, /summary, POST /predict, /retrain
│       ├── materials.py            # GET /daily, /forecast, /daily-forecast
│       └── analytics.py            # GET /abc, /metrics, /top-items, /rules
└── scripts/
    ├── seed_data.py                # DB seeding from CSV
    ├── seed_user.py                # Create admin user
    └── sync_hus_sales.py           # POS sync
```

## Request Flow

```
HTTP Request
    │
    ▼
┌────────────────────────────────────────────────────┐
│  uvicorn (run.py → app.main:app)                   │
│  • CORS middleware                                  │
│  • Auth middleware (except /api/auth and /api/health)│
└─────────────────────┬──────────────────────────────┘
                      │
                      ▼
┌────────────────────────────────────────────────────┐
│  Router (app/routers/)                              │
│  • Parse request params / body                      │
│  • Inject dependencies (get_session, require_auth)  │
│  • Delegate to service                              │
│  • Return HTTP response                             │
└─────────────────────┬──────────────────────────────┘
                      │
                      ▼
┌────────────────────────────────────────────────────┐
│  Service (app/services/)                            │
│  • Business logic, validation                       │
│  • Orchestrates repositories + ML engine           │
│  • Transforms data for response                    │
└─────────┬──────────────────────────┬───────────────┘
          │                          │
          ▼                          ▼
┌──────────────────┐    ┌──────────────────────────┐
│  Repository       │    │  ML Engine               │
│  (app/repositories)│    │  (app/ml/engine.py)      │
│  • SQLAlchemy ORM │    │  • Model registry         │
│  • Single-respons.│    │  • train_and_evaluate     │
│    DB queries     │    │  • generate_forecast      │
└────────┬─────────┘    └───────────┬──────────────┘
         │                          │
         ▼                          ▼
┌──────────────────┐    ┌──────────────────────────┐
│  PostgreSQL      │    │  ml-model/src/models/    │
│  (Docker :5433)  │    │  • forecaster.py (XGB)   │
│                  │    │  • forecaster_rf.py      │
│                  │    │  • forecaster_sarimax.py │
│                  │    │  • forecaster_prophet.py │
│                  │    │  Saved models:            │
│                  │    │  ml-model/models/daily/  │
└──────────────────┘    └──────────────────────────┘
```

## Detailed Module Flows

### 1. Authentication

```
POST /api/auth/login
  │ body: { email, password }
  ▼
auth_router → auth_service.authenticate(session, email, password)
  │
  ├─ SELECT * FROM users WHERE email = ?
  ├─ bcrypt.verify(password, user.hashed_password)
  └─ return JWT.encode({ sub: user.id, exp: ... })

All other routes:
  │ Header: Authorization: Bearer <token>
  ▼
core/deps.py: require_auth()
  ├─ HTTPBearer extracts token
  ├─ JWT.decode(token) → user_id
  └─ raises 401 if invalid
```

### 2. Sales Data (Read path)

```
GET /api/sales/daily?item=&start_date=&end_date=&page=&page_size=
  │
  ▼
sales_router → sales_service.get_daily_sales(session, ...)
  │
  ▼
SalesRepository.get_daily_sales(item, start_date, end_date, page, page_size)
  │
  ├─ query = SELECT daily_item_sales.*, items.name
  │           FROM daily_item_sales JOIN items
  │           WHERE filters ORDER BY date, name
  ├─ _paginate(query, page, page_size)
  │   ├─ _count: SELECT COUNT(*) FROM (subquery)
  │   └─ query.offset().limit()
  │
  ▼
Return DailySalePage { data: DailySale[], total, page, page_size }
```

### 3. Forecast Generation (Read + ML)

```
GET /api/forecasts?item=&start_date=&end_date=&model_type=&page=&page_size=
  │
  ▼
forecasts_router → forecast_service.get_forecasts(session, ...)
  │
  ├─ ForecastRepository.get_sales_dataframe()
  │   └─ Raw SQL: SELECT dis.date, i.name, dis.quantity_sold
  │               FROM daily_item_sales dis JOIN items i
  │
  ├─ _resample_daily(df) — group by item + date, sum qty
  │
  ├─ _get_or_generate_forecast(df, weeks, model_type)
  │   ├─ Check _forecast_cache[model_type]
  │   ├─ Miss → asyncio.to_thread(generate_forecast)
  │   │   └─ app/ml/engine.py: generate_forecast()
  │   │       ├─ _clean_data() — strip "Add" items, resample daily
  │   │       ├─ _ensure_models_loaded() — load from ml-model/models/
  │   │       ├─ Recursive batch prediction (7-day windows)
  │   │       └─ Returns DataFrame [Date, Item, Predicted]
  │   └─ Update cache
  │
  ├─ _filter_forecast() by date range + item
  └─ Paginate + return ForecastPage
```

### 4. Model Retraining (Write + ML + Heavy)

```
POST /api/forecasts/retrain
  │ body: { model_type, max_items, sync_hus }
  ▼
forecasts_router → retrain_service.start_retrain(...)
  │
  ├─ (Optional) sync_hus_sales.sync_sales()
  │
  ├─ Set status = "training"
  │
  ├─ asyncio loop.run_in_executor(ThreadPoolExecutor)
  │   └─ _run_training_sync(model_type, max_items)
  │       │
  │       ├─ Load data from DB via sync_session
  │       │   SELECT dis.date, i.name, dis.quantity_sold
  │       │   FROM daily_item_sales dis JOIN items i
  │       │
  │       ├─ run_train_and_evaluate(df, model_type)
  │       │   │  (app/ml/engine.py)
  │       │   ├─ _clean_data(df)
  │       │   ├─ _import_model_fns(model_type) — lazy import
  │       │   │   ├─ xgboost       → src.models.forecaster
  │       │   │   ├─ random_forest → src.models.forecaster_rf
  │       │   │   ├─ sarimax       → src.models.forecaster_sarimax
  │       │   │   └─ prophet       → src.models.forecaster_prophet
  │       │   ├─ create_features() (for tree models)
  │       │   ├─ fns["train"](data, ML_MODELS_DIR)
  │       │   ├─ fns["train_and_predict"](data)
  │       │   └─ generate_abc_analysis(test_pred)
  │       │
  │       ├─ Deactivate old ModelRun
  │       ├─ Create new ModelRun + ModelRunClassMetric + ModelRunTopItem
  │       ├─ invalidate_forecast_cache(model_type)
  │       └─ Return { status, global_metrics, class_metrics }
  │
  └─ Update status = "success" | "error"
```

### 5. Material Requirements (Read + ML + BOM)

```
GET /api/materials/daily-forecast
  │
  ▼
recipe_material_service.get_daily_material_forecast(...)
  │
  ├─ Load sales data from DB → DataFrame
  ├─ generate_forecast(df, weeks=12, model_type) — via ML engine
  │
  ├─ _map_forecast_to_materials(forecast_df)
  │   ├─ Query product_recipe_ingredients JOIN products, materials, condiments
  │   ├─ For each forecast (item, date, qty):
  │   │   ├─ Match item to product/variant
  │   │   ├─ Multiply recipe_qty × forecast_qty → material requirement
  │   │   └─ Expand condiments via _expand_condiment()
  │   └─ GROUP BY date, raw_material
  │
  └─ Filter by date range, paginate
```

### 6. ABC Analytics

```
GET /api/analytics/abc
  │
  ▼
analytics_service.get_abc_analysis(session, model_type)
  │
  ├─ If model_type provided:
  │   ├─ Get sales data → generate_forecast(df) → predicted volumes
  │   └─ Rank items by predicted volume
  │
  ├─ If no model_type (historical):
  │   └─ SELECT item, SUM(quantity_sold) FROM daily_item_sales GROUP BY item
  │
  └─ _compute_abc_classification(rows)
      ├─ Sort by volume descending
      ├─ Cumulative % → A (≤70%), B (≤90%), C (>90%)
      └─ Return ABCAnalysisResponse
```

## Key Design Patterns

### Layer Separation
```
Router  →  Service  →  Repository  →  DB
   │           │
   │           └──→  ML Engine  →  ml-model/src/
   │
   └── deps: get_session, require_auth
```

### Dependency Injection
- `get_session()` — yields `AsyncSession` per request
- `require_auth()` — validates JWT, returns `user_id`
- All routers inject these via `Depends()`

### Repository Pattern
- `BaseRepository` provides `_count()` and `_paginate()`
- Each repository encapsulates all queries for its domain
- Services create repo instances: `repo = SalesRepository(session)`

### ML Model Registry
- `_model_fns_cache` maps model_type → { train, load, predict, train_and_predict, needs_features }
- Lazy imports happen once, dispatch uses dict lookup instead of if/elif
- `_models_cache` holds loaded models in memory

### Forecast Caching
- `_forecast_cache: dict[str, pd.DataFrame]` in forecast_service
- Invalidated on retrain, shared across forecast/material services

### Background Training
- Retrain runs in `ThreadPoolExecutor` (max 4 workers)
- Status tracked per model_type in `RetrainState`
- Cancellable via `_cancelled` flag
- Logs captured via `LogCapture` (stdout redirect)

## Data Sources

| Source | Location | Used By |
|---|---|---|
| PostgreSQL | `cafe-dev-db-1:5432` | All repositories |
| Daily item sales CSV | `ml-model/data/processed/daily_item_sales.csv` | Seeding |
| Menu BOM CSV | `ml-model/data/raw/bom/menu_bom.csv` | Material forecasting |
| Condiment BOM CSV | `ml-model/data/raw/bom/condiment_bom.csv` | Material forecasting |
| Trained models | `ml-model/models/daily/` | ML engine |
| External POS DB | `hus_db` (configurable) | sync_hus_sales.py |
