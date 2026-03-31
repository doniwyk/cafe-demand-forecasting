# Architecture

## Monorepo Structure

```
web/
  package.json          Root scripts (dev, build, lint, start)
  Makefile              Python venv + backend tasks
  .gitignore            Combined ignores for Node + Python
  backend/              FastAPI application
    app/
      config.py         App config, paths, STATIC_DIR
      main.py           FastAPI app entrypoint, static serving
      routers/          API route handlers
      services/         Business logic layer
      models/           Data models/schemas
      ml/               ML engine
    requirements.txt    Python dependencies
    run.py              Direct uvicorn runner (dev convenience)
  frontend/             React + Vite + Tailwind application
    src/                Source code
    dist/               Built output (gitignored, served by FastAPI in prod)
    vite.config.ts      Vite config with /api proxy
    package.json        Frontend dependencies
```

## How It Works

### Development

Running `bun run dev` (from `web/`) starts both servers concurrently:

- **Frontend**: Vite dev server on `http://localhost:5173`
- **Backend**: Uvicorn on `http://localhost:8000`

The Vite dev server proxies all `/api/*` requests to the backend, so the frontend can call the API without CORS issues during development. This means all API calls from the React app should use relative paths (e.g., `/api/health`, not `http://localhost:8000/api/health`).

### Production

1. `bun run build` runs `vite build` in `frontend/`, outputting static files to `frontend/dist/`
2. FastAPI serves the built frontend:
   - `/assets/*` — served as static files (JS, CSS, images)
   - `/api/*` — handled by API routers
   - Everything else — falls back to `index.html` (SPA routing)
3. `bun run start` (or `make start`) runs uvicorn without `--reload` for production

In production, a single FastAPI process handles both the API and the frontend on one port (8000). No separate web server needed.

### Commands

| Command | Description |
|---|---|
| `bun run dev` | Start both frontend + backend in dev mode |
| `bun run dev:frontend` | Start frontend only |
| `bun run dev:backend` | Start backend only (via Makefile) |
| `bun run build` | Build frontend for production |
| `bun run lint` | Lint frontend code |
| `bun run start` | Start backend in production mode |
| `make venv` | Create Python virtual environment at `web/.venv` |
| `make install` | Create venv + install Python dependencies |
| `make clean` | Remove venv and built frontend |

## Key Design Decisions

**Root `package.json` + Makefile** instead of a workspace manager (Turborepo, pnpm workspaces). Since the backend is Python, there's no shared Node code between frontend and backend — a full workspace manager adds complexity without benefit.

**`concurrently`** is the only new root dependency. It runs both dev servers in a single terminal with color-coded output.

**Vite proxy** in development eliminates CORS. The backend still has CORS middleware configured for flexibility, but it isn't needed when using the proxy.

**FastAPI serves static files in production** — no Nginx or separate static file server required. The `STATIC_DIR` in `config.py` points to `frontend/dist/`, and `main.py` conditionally mounts static serving only when the directory exists (so the backend works fine without a frontend build during development).

**Python venv at `web/.venv`** (not inside `backend/`) keeps it at the workspace root, consistent with the monorepo structure.
