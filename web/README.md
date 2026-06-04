# Cafe Supply Forecasting — Web Stack

## Prerequisites

- Python 3.10+
- [Bun](https://bun.sh) (JS runtime)
- Docker Desktop (PostgreSQL)

## Development Setup

```bash
# 1. Start PostgreSQL
docker compose up -d db

# 2. Seed database (from web/)
cd web
python backend/scripts/seed_data.py
python backend/scripts/seed_user.py    # creates user: admin@example.com / admin123

# 3. Install frontend deps + start dev servers
bun install
bun run dev             # frontend :5173, backend :8000
```

Or using Docker for all three services:

```bash
docker compose up -d
# frontend :5174, backend :8001, db :5433
```

## Database

PostgreSQL 16 runs in Docker on port **5433** (`cafe-dev-db-1`). Credentials: `postgres` / `postgres` / `cafe_forecasting`.

### Migrations (Alembic)

```bash
# From web/backend/
alembic upgrade head       # apply pending migrations
alembic revision --autogenerate -m "description"  # create new migration
```

Migrations live in `app/db/migrations/versions/`. The initial migration (`0001`) creates all tables; `0002` adds the `rmse` column.

### Seeding

| Script | Purpose |
|---|---|
| `scripts/seed_data.py` | Loads CSV from `../ml-model/data/processed/` into DB (daily sales, items, categories, BOM). **Run after `docker compose up -d db`** and before using the app. |
| `scripts/seed_user.py` | Creates default admin user. |
| `scripts/sync_hus_sales.py` | Pulls live POS sales from an external HUS database. |

## Project Structure

```
web/
├── package.json          Root scripts: dev, build, lint
├── Makefile              Python venv + backend tasks
├── architecture.md       Architecture deep-dive
├── README.md             ← you are here
├── backend/              FastAPI (Python)
│   ├── app/
│   │   ├── core/         constants, shared deps
│   │   ├── db/           ORM models, engine, Alembic migrations
│   │   ├── models/       Pydantic schemas
│   │   ├── repositories/ Data access layer
│   │   ├── services/     Business logic
│   │   ├── ml/           ML engine (registry pattern)
│   │   └── routers/      HTTP route handlers
│   └── scripts/          DB seeding, POS sync
├── frontend/             React + Vite + Tailwind + shadcn/ui
│   └── src/
│       ├── features/     Domain modules (dashboard, forecasts, analytics…)
│       └── lib/          i18n, API client, utilities
```

See `backend/backend_flow.md` for detailed request flows and architecture diagrams.

## API Overview

All endpoints are under `/api/`. Auth via `Authorization: Bearer <token>` (obtained from `POST /api/auth/login`).

### Auth

| Method | Path | Description |
|---|---|---|
| POST | `/api/auth/login` | `{ email, password }` → JWT token |
| GET | `/api/auth/me` | Current user info |

### Sales

| Method | Path | Description |
|---|---|---|
| GET | `/api/sales/daily` | Paginated daily sales (filter by item, date range) |
| GET | `/api/sales/items` | Menu items list |
| GET | `/api/sales/categories` | Sales categories |

### Forecasts

| Method | Path | Description |
|---|---|---|
| GET | `/api/forecasts` | Paginated forecast data (filter by item, model, date) |
| GET | `/api/forecasts/summary` | Aggregate metrics (R², wMAPE, MAE, RMSE, accuracy) |
| POST | `/api/forecasts/predict` | Run ad-hoc prediction |
| POST | `/api/forecasts/retrain` | Trigger model retraining (runs in background) |
| GET | `/api/forecasts/retrain/status` | Poll retraining progress |

### Analytics

| Method | Path | Description |
|---|---|---|
| GET | `/api/analytics/metrics` | Model performance metrics |
| GET | `/api/analytics/abc` | ABC classification |
| GET | `/api/analytics/top-items` | Top N items by accuracy |

### Materials

| Method | Path | Description |
|---|---|---|
| GET | `/api/materials/daily` | Daily raw material usage |
| GET | `/api/materials/forecast` | Forecast material requirements |
| GET | `/api/materials/daily-forecast` | BOM × forecast → daily material needs |

## Retraining Models

```bash
# Via API (single model)
curl -X POST http://localhost:8001/api/forecasts/retrain \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"model_type": "xgboost"}'

# Train all models sequentially
curl -X POST ... -d '{"model_type": "all"}'

# Poll status
curl http://localhost:8001/api/forecasts/retrain/status \
  -H "Authorization: Bearer <token>"
```

Supported `model_type` values: `xgboost`, `random_forest`, `sarimax`, `prophet`, `all`.

Retraining runs in a background thread. Progress updates and logs are available via the status endpoint. The retrain service (`app/services/retrain_service.py`):
1. Loads all daily sales from DB
2. Splits into train/test (last 12 months for test)
3. Trains the model, evaluates on test set
4. Saves metrics (R², wMAPE, MAE, RMSE, period accuracy) to `model_runs` table
5. Deactivates the previous model run for that model type

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `DATABASE_URL` | `postgresql+asyncpg://postgres:postgres@localhost:5432/cafe_forecasting` | DB connection string |
| `SECRET_KEY` | `dev-secret-key-change-in-production` | JWT signing key |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | `480` | Token TTL |

## Troubleshooting

**"Module not found" for ml-model imports**: Add `ml-model/` to `python.analysis.extraPaths` in `.vscode/settings.json`:

```json
{
  "python.analysis.extraPaths": ["ml-model"]
}
```

A `.vscode/settings.json` with this setting is already committed at the repo root.

**DB connection refused**: Ensure the db container is running and healthy:
```bash
docker compose ps
docker compose logs db
```

**Alembic detects no changes**: Make sure you're running from `web/backend/`:
```bash
cd web/backend && alembic check
```

**Frontend can't reach API**: The Vite dev server proxies `/api/*` to the backend. In Docker, check that `VITE_API_URL` points to `http://backend:8001`. For local dev, the proxy defaults to `http://localhost:8000`.
