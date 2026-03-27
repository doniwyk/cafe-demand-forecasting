#!/bin/bash
# ================================
# 🧠 Conda + Pip Environment Exporter
# -------------------------------
# Usage: ./export-env.sh
# It will:
#   1. Export your conda environment to environment.yml
#   2. Export all pip-installed packages to requirements.txt
# ================================

# Automatically detect the current Conda environment name
ENV_NAME=$(basename "$CONDA_PREFIX")

echo "🔍 Detected Conda environment: $ENV_NAME"

# Export Conda environment
echo "📦 Exporting full Conda environment to environment.yml..."
conda env export > environment.yml

# Export pip packages (only pip-installed ones)
echo "🐍 Exporting pip packages to requirements.txt..."
pip freeze > requirements.txt

# Optional cleanup: remove system prefix path lines for cleaner file
sed -i '' '/prefix:/d' environment.yml 2>/dev/null

echo "✅ Export complete!"
echo "  - Conda: environment.yml"
echo "  - Pip:   requirements.txt"
