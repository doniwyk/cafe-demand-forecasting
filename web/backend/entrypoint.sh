#!/usr/bin/env bash
set -euo pipefail

MODELS_DIR="/app/ml-model/models"
MODELS_DIST="/app/ml-model/models.dist"

if [ ! -d "$MODELS_DIR" ] || [ -z "$(ls -A "$MODELS_DIR" 2>/dev/null)" ]; then
    if [ -d "$MODELS_DIST" ] && [ -n "$(ls -A "$MODELS_DIST" 2>/dev/null)" ]; then
        echo "[entrypoint] models volume is empty — seeding from models.dist"
        mkdir -p "$MODELS_DIR"
        cp -rn "$MODELS_DIST"/. "$MODELS_DIR"/
    else
        echo "[entrypoint] models volume is empty and models.dist not found — skipping seed"
        mkdir -p "$MODELS_DIR"
    fi
fi

exec "$@"
