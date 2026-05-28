# Cafe Supply Forecasting — Complete System Documentation

> Covers everything from zero setup through daily operation: running the app, managing the database, training ML models, and using every feature.

---

## Table of Contents

- [1. Prerequisites](#1-prerequisites)
- [2. Quick Start (Local, No Docker)](#2-quick-start-local-no-docker)
- [3. Quick Start (Docker)](#3-quick-start-docker)
- [4. Environment Variables](#4-environment-variables)
- [5. Database Setup & Migrations](#5-database-setup--migrations)
- [6. Seeding the Database](#6-seeding-the-database)
- [7. Running the Application](#7-running-the-application)
- [8. User Authentication](#8-user-authentication)
- [9. System Features — Daily Operations](#9-system-features--daily-operations)
- [10. ML Pipeline & Model Training](#10-ml-pipeline--model-training)
- [11. Retraining Models via UI](#11-retraining-models-via-ui)
- [12. Production Deployment](#12-production-deployment)
- [13. Architecture Diagram](#13-architecture-diagram)
- [14. Troubleshooting](#14-troubleshooting)

---

## 1. Prerequisites

| Dependency | Version | Purpose |
|---|---|---|
| **Python** | 3.10+ (3.10 recommended) | FastAPI backend + ML models |
| **PostgreSQL** | 16 | Primary database |
| **Node.js** | 20+ | Frontend build |
| **Bun** | 1.x | Frontend package manager + dev runner |
| **Docker** (optional) | 24+ | Containerized dev/prod |
| **Conda** (optional) | — | ML model environment |

---

## 2. Quick Start (Local, No Docker)

### 2.1. Start PostgreSQL

```bash
# Via Homebrew (macOS)
brew install postgresql@16
brew services start postgresql@16

# Create the database
createdb cafe_forecasting
```

Or using Docker for just the database:

```bash
docker run -d \
  --name cafe-db \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_DB=cafe_forecasting \
  -p 5432:5432 \
  postgres:16-alpine
```

### 2.2. Set Up Python Environment

```bash
cd web

# Create virtual environment
make venv

# Install Python dependencies
make install
```

### 2.3. Install Frontend Dependencies

```bash
cd web
bun install
```

### 2.4. Configure Environment

```bash
cp web/backend/.env.example web/backend/.env
```

Edit `web/backend/.env` if needed. Defaults work with local PostgreSQL:

```
DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/cafe_forecasting
SECRET_KEY=dev-secret-key-change-in-production
ACCESS_TOKEN_EXPIRE_MINUTES=480
```

### 2.5. Run Database Migrations

```bash
cd web/backend
.venv/bin/alembic upgrade head
```

### 2.6. Seed the Database

```bash
# Create default admin user
cd web/backend
../.venv/bin/python scripts/seed_user.py

# Seed ML data from CSV files
cd ml-model
python scripts/seed_database.py
```

### 2.7. Start the App

```bash
cd web
bun run dev
```

- **Frontend**: http://localhost:5173
- **Backend**: http://localhost:8000
- **Health check**: http://localhost:8000/api/health

### 2.8. Login

| Email | Password |
|---|---|
| `manager@cafe.com` | `password` |

---

## 3. Quick Start (Docker)

### 3.1. Development

```bash
docker compose up
```

| Service | URL | Notes |
|---|---|---|
| Frontend | http://localhost:5174 | Vite HMR, proxies `/api` → backend |
| Backend | http://localhost:8001 | uvicorn `--reload` |
| Database | `localhost:5433` | PostgreSQL 16 |

Source changes to `web/frontend/src/`, `web/backend/`, and `ml-model/` are reflected immediately via volume mounts.

### 3.2. Run Migrations & Seed Inside Docker

```bash
# Run migrations
docker compose exec backend alembic upgrade head

# Seed admin user
docker compose exec backend python scripts/seed_user.py

# Seed database from CSVs
docker compose exec backend python -c "
import sys, os
sys.path.insert(0, '/app/ml-model')
os.chdir('/app/ml-model')
from scripts.seed_database import main
main()
"
```

### 3.3. Production

```bash
docker compose -f docker-compose.prod.yml up --build
```

| Service | URL | Notes |
|---|---|---|
| Frontend | http://localhost:3000 | Nginx serving built SPA |
| Backend | http://localhost:8001 | No reload |
| Database | `localhost:5433` | PostgreSQL 16 |

---

## 4. Environment Variables

### Backend (`web/backend/.env`)

| Variable | Default | Description |
|---|---|---|
| `DATABASE_URL` | `postgresql+asyncpg://postgres:postgres@localhost:5432/cafe_forecasting` | Async PostgreSQL connection string |
| `SECRET_KEY` | `dev-secret-key-change-in-production` | JWT signing secret |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | `480` | Token lifetime in minutes (8 hours) |

### Frontend (`web/frontend/.env`)

| Variable | Default | Description |
|---|---|---|
| `VITE_API_URL` | `http://localhost:8001` | Backend URL (Vite proxy target) |

### Docker Compose Overrides

In `docker-compose.yml`, `DATABASE_URL` points to `postgresql+asyncpg://postgres:postgres@db:5432/cafe_forecasting` and `PYTHONPATH=/app/ml-model` is set automatically.

---

## 5. Database Setup & Migrations

### 5.1. Schema Overview

The database contains 22 tables across 5 logical groups:

| Group | Tables | Purpose |
|---|---|---|
| **Auth** | `users` | Login, JWT |
| **Menu** | `categories`, `items`, `bom_recipes`, `condiment_recipes` | Menu items + Bill of Materials |
| **Sales** | `sales_cleaned`, `daily_item_sales`, `daily_category_sales`, `daily_total_sales` | Historical POS data |
| **ML** | `model_runs`, `model_run_class_metrics`, `model_run_top_items`, `forecasts`, `item_abc`, `association_rules`, `raw_material_requirements` | Training runs, forecasts, analytics |
| **Inventory** | `products`, `product_variants`, `materials`, `condiments`, `product_recipe_ingredients`, `condiment_ingredients` | Product BOM catalog |

### 5.2. Migration Commands

All commands run from `web/backend/`:

```bash
# Apply all pending migrations
alembic upgrade head

# Rollback one step
alembic downgrade -1

# Rollback to a specific revision
alembic downgrade 0001

# Show current revision
alembic current

# Show migration history
alembic history

# Create a new auto-generated migration
alembic revision --autogenerate -m "description"

# Create an empty migration (for manual SQL)
alembic revision -m "description"
```

### 5.3. Migration Structure

```
web/backend/app/db/migrations/
├── env.py                 # Async Alembic env (reads config.py DATABASE_URL)
├── script.py.mako         # Migration template
└── versions/
    └── 0001_initial_schema.py   # Creates all 22 tables
```

The `env.py` dynamically loads `DATABASE_URL` from `app.config` (which reads `.env` via `python-dotenv`), overriding the hardcoded URL in `alembic.ini`.

### 5.4. Creating Tables from Scratch

If you skip Alembic and want SQLAlchemy to create all tables directly:

```python
from app.db.base import Base
from app.db.engine import sync_engine
Base.metadata.create_all(sync_engine)
```

The `seed_database.py` script uses this approach when the database is empty.

### 5.5. Reset Database (Destroy & Recreate)

```bash
# Drop and recreate (PostgreSQL)
dropdb cafe_forecasting
createdb cafe_forecasting

# Then re-run migrations
cd web/backend
alembic upgrade head

# Re-seed
cd web/backend
../.venv/bin/python scripts/seed_user.py
cd ../ml-model
python scripts/seed_database.py
```

---

## 6. Seeding the Database

### 6.1. Admin User

Creates the default login account. Idempotent (skips if exists):

```bash
cd web/backend
../.venv/bin/python scripts/seed_user.py
```

### 6.2. CSV Data Seeding

Seeds all business data from CSV files in `ml-model/data/`. Run from `ml-model/`:

```bash
cd ml-model

# Full seed (includes ~55K sales rows)
python scripts/seed_database.py

# Skip the large sales_cleaned table (fast mode)
python scripts/seed_database.py --skip-sales-cleaned

# Clear all data first, then reseed
python scripts/seed_database.py --truncate
```

**What gets seeded (in order):**

| Step | Table(s) | Source CSV |
|---|---|---|
| 1 | `categories` | `data/raw/bom/menu_bom.csv` |
| 2 | `items` | `data/raw/bom/menu_bom.csv` |
| 3 | `bom_recipes` | `data/raw/bom/menu_bom.csv` |
| 4 | `condiment_recipes` | `data/raw/bom/condiment_bom.csv` |
| 5 | `sales_cleaned` | `data/processed/sales_data_cleaned.csv` |
| 6 | `daily_item_sales` | `data/processed/daily_item_sales.csv` |
| 7 | `daily_category_sales` | `data/processed/daily_category_sales.csv` |
| 8 | `daily_total_sales` | `data/processed/daily_total_sales.csv` |
| 9 | `model_runs` + `model_run_class_metrics` + `model_run_top_items` | `models/model_metadata.json` + `data/predictions/forecast_summary.json` |
| 10 | `forecasts` | `data/predictions/3_month_forecasts.csv` |
| 11 | `item_abc` | `data/processed/daily_item_sales.csv` (computed) |
| 12 | `association_rules` | `data/processed/association_rules_fpgrowth.csv` |

### 6.3. Required CSV Files

The seed script expects these files under `ml-model/`:

```
data/
├── raw/bom/
│   ├── menu_bom.csv
│   └── condiment_bom.csv
├── processed/
│   ├── sales_data_cleaned.csv
│   ├── daily_item_sales.csv
│   ├── daily_category_sales.csv
│   ├── daily_total_sales.csv
│   └── association_rules_fpgrowth.csv
├── predictions/
│   ├── 3_month_forecasts.csv
│   └── forecast_summary.json
└── models/
    └── model_metadata.json
```

---

## 7. Running the Application

### 7.1. Commands Reference

All commands from `web/` unless noted:

| Command | What It Does |
|---|---|
| `bun run dev` | Start frontend (Vite :5173) + backend (uvicorn :8000) concurrently |
| `bun run dev:frontend` | Start frontend only |
| `bun run dev:backend` | Start backend only |
| `bun run build` | Build frontend for production → `frontend/dist/` |
| `bun run start` | Start backend in production (serves API + static frontend) |
| `bun run lint` | Lint frontend code |
| `make venv` | Create Python virtual env at `web/.venv/` |
| `make install` | Create venv + install pip dependencies |
| `make backend` | Start uvicorn on :8000 with reload |
| `make start` | Start uvicorn on :8000 without reload |

### 7.2. Development Mode Details

When you run `bun run dev`, two processes start in parallel:

**Backend** (`make backend`):
- `uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload --app-dir backend`
- Auto-restarts on Python file changes
- Serves API at `http://localhost:8000/api/*`

**Frontend** (`bun --cwd frontend dev`):
- Vite dev server on `http://localhost:5173`
- Hot Module Replacement for instant UI updates
- Proxies `/api/*` → `http://localhost:8000` (no CORS issues)

### 7.3. Production Mode Details

```bash
cd web
bun run build   # builds frontend to frontend/dist/
make start      # runs uvicorn without reload
```

FastAPI serves everything on a single port (8000):
- `/assets/*` — static JS/CSS from the built frontend
- `/api/*` — API routers
- All other routes — `index.html` (SPA fallback)

No Nginx or separate web server is required.

### 7.4. Starting Specific Backend Without Make

```bash
# Direct via uvicorn
cd web/backend
../.venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# Or via run.py helper
cd web/backend
../.venv/bin/python run.py
```

---

## 8. User Authentication

### 8.1. How Auth Works

1. User submits email + password → `POST /api/auth/login`
2. Backend validates against bcrypt hash in `users` table
3. Returns JWT (`access_token`) with 8-hour expiry (configurable)
4. Frontend stores token in `localStorage` key `auth_token`
5. All subsequent API calls include `Authorization: Bearer <token>` header
6. On 401 response, frontend clears token and redirects to `/login`
7. On page load, frontend calls `GET /api/auth/me` to validate stored token

### 8.2. Default Credentials

| Email | Password | Role |
|---|---|---|
| `manager@cafe.com` | `password` | Cafe Manager (admin) |

### 8.3. Creating Additional Users

```python
# Example: create another user via Python
from app.db.engine import async_session
from app.models.user import User
from app.services.auth import auth_service
from sqlalchemy import select

async with async_session() as session:
    user = User(
        email="staff@cafe.com",
        name="Staff Member",
        hashed_password=auth_service.hash_password("staff123"),
    )
    session.add(user)
    await session.commit()
```

### 8.4. Password Change (via Python shell)

```python
from app.db.engine import async_session
from app.models.user import User
from app.services.auth import auth_service
from sqlalchemy import select

async with async_session() as session:
    result = await session.execute(select(User).where(User.email == "manager@cafe.com"))
    user = result.scalar_one_or_none()
    user.hashed_password = auth_service.hash_password("newpassword")
    await session.commit()
```

---

## 9. System Features — Daily Operations

### 9.1. Login

1. Open the app → redirected to `/login` if unauthenticated
2. Enter `manager@cafe.com` / `password`
3. Click **Login** → redirected to Dashboard

### 9.2. Dashboard (`/`)

Shows a high-level overview:
- **KPI cards**: Active Items count, Best Model Accuracy (%), Forecasted Items count
- **Bar chart**: Top 10 best-selling items by volume
- **Table**: Forecast summary per ABC class (A/B/C) with wMAPE and Volume Accuracy

### 9.3. Switching ML Models

The **Model Switcher** dropdown in the header (visible on all pages) lets you switch between:
- XGBoost
- Random Forest
- SARIMAX
- Prophet

Changing the selection reloads data on the current page for the chosen model.

### 9.4. Viewing Forecasts (`/forecasts`)

1. Select an item from the combobox
2. View a line chart comparing **actual sales** vs **predicted sales**
3. Below the chart, a **Top Items Accuracy** table shows forecast precision for top items

### 9.5. Daily Material Needs (`/materials/daily-need`)

Shows raw material quantities needed per day, calculated by multiplying forecasted sales with BOM recipes.

**Filters available:**
- By material type (dropdown)
- By date range (date picker)

**Export:**
- Click **Export CSV** to download filtered data as a `.csv` file

### 9.6. Analytics (`/analytics`)

Two main sections:

**ABC Analysis:**
- Bar chart of item distribution across A (high-value), B (medium), C (low-value) classes
- Tabbed tables for each class showing: item name, total volume, cumulative percentage

**Model Metrics:**
- Four metric cards: R², wMAPE, MAE, Volume Accuracy

### 9.7. Settings (`/settings`)

Controls for model training:
- **Train Individual Models** — select a model type and click Train
- **Train All (Sequential)** — trains all 4 models one at a time
- **Real-time Status** — logs stream every 5 seconds; shows Training/Trained/Error badges
- **Cancel** — abort an active training session
- **Cleanup** — delete stale model runs and forecasts from previous training sessions

### 9.8. Logout

Click the user info in the header → **Logout** → token cleared → redirected to `/login`

---

## 10. ML Pipeline & Model Training

### 10.1. Pipeline Overview

The ML pipeline processes raw POS exports through 5 stages:

```
Raw CSVs → Merge → Clean → Transform → Forecast → Material Requirements
```

### 10.2. Running the Full Pipeline

All commands from `ml-model/`:

```bash
# 0. Pull latest BOM from database (optional)
python scripts/00_pull_from_hus_db.py

# 1. Merge Indonesian + English sales CSVs
python scripts/01_merge_sales_data.py

# 2. Clean & standardize item names, remove discontinued
python scripts/02_clean_sales_data.py --remove

# 3. Aggregate to daily item sales
python scripts/03_transform_sales.py

# 4. Train + forecast
python scripts/04_forecast.py -f weekly train

# 5. Map forecast to raw material needs
python scripts/05_forecast_to_materials.py -f weekly
```

### 10.3. Forecast CLI Reference

```bash
python scripts/04_forecast.py -f weekly train          # Train + save + forecast
python scripts/04_forecast.py -f daily train            # Daily granularity
python scripts/04_forecast.py -f weekly evaluate        # Evaluate only (no save)
python scripts/04_forecast.py -f weekly train --no-forecast  # Train + save only
python scripts/04_forecast.py -f daily train --max-items 10  # Train only top 10 items
```

**`evaluate` output includes:**
- Global R², wMAPE, MAE, Volume Accuracy
- ABC class breakdown (wMAPE per class)
- Top 10 Class A items with per-item accuracy

**`train` output files saved to:**
```
models/{daily,weekly}/
├── global_model.pkl           # Global fallback XGBoost model
├── item_models.pkl            # Per-item XGBoost models (dict)
├── dow_factors.json            # Day-of-week adjustment factors
├── model_metadata.json         # Training date, features, item list
└── data/predictions/{daily,weekly}/
    └── 3_month_forecasts.csv   # 12-week future forecast
```

### 10.4. Model Details

| Model | Algorithm | File | Strategy |
|---|---|---|---|
| **XGBoost** | `reg:tweedie`, variance_power=1.5 | `forecaster.py` | Per-item (≥40 weeks) + global fallback |
| **Random Forest** | Ensemble, 100 trees | `forecaster_rf.py` | Per-item + global fallback |
| **SARIMAX** | Seasonal ARIMA with exogenous regressors | `forecaster_sarimax.py` | Stationarity detection, auto order selection |
| **Prophet** | Facebook Prophet, weekly seasonality | `forecaster_prophet.py` | Changepoint detection, holiday effects |

**Features** (28): Calendar (month, week, day-of-year), seasonality (sin/cos), trend, payday/holiday/Ramadan flags, rebranding indicators, lag 1/2/4, rolling mean 4/12, rolling std 4, rolling q95 4, EWMA 4/12, momentum, post-rebrand surge ratio.

**Post-processing**: Day-of-week factor adjustment applied to raw predictions.

### 10.5. ABC Classification

- **Class A**: top 70% of cumulative volume
- **Class B**: 70–90% of cumulative volume
- **Class C**: bottom 10% of cumulative volume

---

## 11. Retraining Models via UI

### 11.1. Triggering a Retrain (Settings Page)

1. Go to **Settings** (`/settings`)
2. Select a model type from the dropdown (XGBoost, Random Forest, SARIMAX, Prophet)
3. Click **Train**
4. The backend spawns a background task using `ThreadPoolExecutor`
5. The frontend polls `GET /api/forecasts/retrain/status` every 5 seconds
6. Logs stream in real time in the UI log area

### 11.2. What Happens During Retraining

1. Backend loads historical sales data from `daily_item_sales` table
2. Invokes `ml/engine.py:run_train_and_evaluate()` for the selected model type
3. ML engine imports the model from `ml-model/src/models/`
4. Features are engineered, model is trained (per-item + global), evaluated
5. Forecasts are generated for the next 84 days (12 weeks)
6. Results are saved to:
   - New `ModelRun` row (with metrics: R², wMAPE, MAE, Volume Accuracy)
   - `ModelRunClassMetric` rows (per ABC class)
   - `ModelRunTopItem` rows (top items accuracy)
   - `Forecast` rows (per item × date × predicted quantity)
   - Previous active run has `is_active` set to `False`

### 11.3. Retrain API Endpoints

| Endpoint | Method | Purpose |
|---|---|---|
| `/api/forecasts/retrain` | POST | Start training `{ "model_type": "xgboost" }` |
| `/api/forecasts/retrain/status` | GET | Live status + logs for all models |
| `/api/forecasts/retrain/cancel` | POST | Cancel active training `{ "model_type": "xgboost" }` |
| `/api/forecasts/cleanup` | POST | Delete inactive runs + forecasts |

### 11.4. On-Demand Prediction

```bash
curl -X POST http://localhost:8000/api/forecasts/predict \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{ "items": ["Espresso", "Latte"], "weeks": 4, "model_type": "xgboost" }'
```

---

## 12. Production Deployment

### 12.1. Docker Production Stack

```bash
docker compose -f docker-compose.prod.yml up --build -d
```

Three services:
- **db**: PostgreSQL 16 with persistent volume `pgdata_prod`
- **backend**: Python 3.10, FastAPI on :8001, no reload, serves API
- **frontend**: Nginx on :3000, serves built SPA, proxies `/api` → backend

### 12.2. Entrypoint Script

The Docker backend uses `entrypoint.sh` which checks if the `models/` volume is empty. If so, it copies seed models from `models.dist/` (if present):

```bash
# web/backend/entrypoint.sh logic:
# 1. If /app/ml-model/models/ is empty AND models.dist/ exists → copy seed
# 2. If empty and no models.dist → create empty dir
# 3. Then exec the CMD (uvicorn)
```

### 12.3. Building for Production (Manual)

```bash
# Build frontend
cd web/frontend && bun install && bun run build

# Start backend (serves both API and frontend)
cd web && make start
# → http://localhost:8000
```

---

## 13. Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                        Browser                               │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  React 19 + Vite + TanStack Router + shadcn/ui       │   │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────┐  │   │
│  │  │Dashboard │ │Forecasts │ │Analytics │ │Settings│  │   │
│  │  └──────────┘ └──────────┘ └──────────┘ └────────┘  │   │
│  │           ┌──────────────────────────┐                │   │
│  │           │  ModelContext (global)    │                │   │
│  │           │  AuthContext (JWT state)  │                │   │
│  │           └──────────────────────────┘                │   │
│  └──────────────────────────────────────────────────────┘   │
└──────────────────────┬──────────────────────────────────────┘
                       │ HTTP (JSON) + JWT
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  FastAPI (Python 3.10)          Port 8000/8001              │
│  ┌─────────┐ ┌──────────┐ ┌───────────┐ ┌───────────────┐  │
│  │ Routers │ │ Services │ │ ML Engine │ │ Auth (JWT)    │  │
│  ├─────────┤ ├──────────┤ ├───────────┤ ├───────────────┤  │
│  │ /auth   │ │ sales    │ │ XGBoost   │ │ bcrypt verify │  │
│  │ /sales  │ │ forecast │ │ RF        │ │ HS256 encode  │  │
│  │ /forecast│ │ analytics│ │ SARIMAX   │ │               │  │
│  │ /materials│ │ material │ │ Prophet   │ │               │  │
│  │ /analytics│ │          │ │           │ │               │  │
│  └─────────┘ └──────────┘ └───────────┘ └───────────────┘  │
│                │                                            │
│  ┌─────────────┴─────────────────────────────────────────┐  │
│  │  /frontend/dist/ (SPA static files in production)     │  │
│  └───────────────────────────────────────────────────────┘  │
└──────────────────────┬──────────────────────────────────────┘
                       │ SQLAlchemy (asyncpg)
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  PostgreSQL 16            cafe_forecasting                  │
│  ┌──────┐ ┌──────┐ ┌──────────┐ ┌──────────┐               │
│  │ Menu │ │Sales │ │ ML Runs  │ │Inventory │               │
│  │ cats │ │daily │ │ metrics  │ │ products  │               │
│  │ items│ │cleaned│ │forecasts │ │ materials│               │
│  │ BOM  │ │      │ │ ABC/AR   │ │ recipes  │               │
│  └──────┘ └──────┘ └──────────┘ └──────────┘               │
└─────────────────────────────────────────────────────────────┘
```

---

## 14. Troubleshooting

### 14.1. Database Connection Failed

**Error**: `could not connect to server: Connection refused`

**Fix**: Ensure PostgreSQL is running:
```bash
pg_isready -h localhost -p 5432
# Or restart:
brew services restart postgresql@16
```

### 14.2. Alembic: Target database is not up to date

```bash
cd web/backend
alembic upgrade head
```

### 14.3. Alembic: No such table

If migrations have never run but tables were created manually (e.g., by `seed_database.py`), Alembic doesn't know the state. Stamp it:

```bash
alembic stamp head
```

### 14.4. ModuleNotFoundError: No module named 'src'

The backend inserts `ml-model/` into `sys.path` in `config.py`. Ensure `PYTHONPATH` is set in Docker:

```yaml
environment:
  PYTHONPATH: /app/ml-model
```

For local dev, `config.py` does this automatically.

### 14.5. XGBoost Import Error in Docker

The Docker build uses `--no-deps` for xgboost to avoid dependency conflicts. If issues arise:

```dockerfile
RUN pip install --no-cache-dir --no-deps xgboost==2.0.3
```

### 14.6. Database Tables Already Exist Error During Seed

If `seed_database.py` detects existing tables, it skips `create_all`. To force a clean reseed:

```bash
python scripts/seed_database.py --truncate
```

### 14.7. Frontend Shows Blank Page

1. Check the browser console for errors
2. Ensure the backend is running (`curl http://localhost:8000/api/health`)
3. For production, ensure `bun run build` was run before starting
4. For Docker dev, check that `VITE_API_URL` points to the backend service

### 14.8. Login Returns 401

1. Ensure `seed_user.py` was run
2. Check the `users` table has the manager account
3. Verify `.env` has the correct `SECRET_KEY`

### 14.9. Retrain Gets Stuck

1. Check the backend logs for errors
2. Cancel the training from the Settings page
3. Try training a different model type
4. Restart the backend if the `ThreadPoolExecutor` is stuck

### 14.10. Common Port Conflicts

| Port | Service | Change |
|---|---|---|
| 5173 | Vite (dev frontend) | Edit `vite.config.ts` `server.port` |
| 8000 | uvicorn (local backend) | Edit `Makefile` or `run.py` |
| 8001 | uvicorn (Docker backend) | Edit `docker-compose.yml` |
| 5174 | Vite (Docker frontend) | Edit `docker-compose.yml` |
| 5432 | PostgreSQL (local) | — |
| 5433 | PostgreSQL (Docker) | Edit `docker-compose.yml` |
