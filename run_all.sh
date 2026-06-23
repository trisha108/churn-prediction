#!/usr/bin/env bash
# run_all.sh — Churn Platform end-to-end pipeline (macOS)
# Usage: chmod +x run_all.sh && ./run_all.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$ROOT/.venv"

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'
BLUE='\033[0;34m'; BOLD='\033[1m'; NC='\033[0m'

log()  { echo -e "${GREEN}[✓]${NC} $1"; }
warn() { echo -e "${YELLOW}[!]${NC} $1"; }
err()  { echo -e "${RED}[✗]${NC} $1"; exit 1; }
head() { echo -e "\n${BOLD}${BLUE}── $1 ─────────────────────────────────────────────${NC}"; }

head "Step 0: Verify dataset"
DATASET="$ROOT/data/WA_Fn-UseC_-HR-Employee-Attrition.csv"
[[ ! -f "$DATASET" ]] && err "Dataset not found at data/WA_Fn-UseC_-HR-Employee-Attrition.csv\nDownload from Kaggle and place in data/ folder."
log "Dataset found"

head "Step 1: Python environment"
[[ ! -d "$VENV" ]] && python3 -m venv "$VENV" && log "Created .venv"
source "$VENV/bin/activate"
pip install --quiet --upgrade pip
pip install --quiet -r "$ROOT/requirements.txt"
log "Dependencies installed"

head "Step 2: EDA + Feature Engineering"
python "$ROOT/scripts/01_eda_preprocessing.py"

head "Step 3: MLflow + Model Training (with tuning + SHAP)"
python "$ROOT/scripts/02_mlflow_training.py"
warn "MLflow UI: mlflow ui --backend-store-uri $ROOT/mlflow_runs → http://localhost:5000"

head "Step 4: Time-Series Churn Forecasting"
python "$ROOT/scripts/03_timeseries_forecast.py"

head "Step 5: Unit Tests"
pytest "$ROOT/tests/" -v --tb=short || warn "Some tests failed — check output above"

head "Step 6: Docker Build + Deploy (FastAPI)"
if ! command -v docker &>/dev/null; then
  warn "Docker not found — install from https://www.docker.com/products/docker-desktop"
  warn "Then: cd docker && docker-compose up --build"
else
  cd "$ROOT/docker" && docker-compose up --build -d churn-api && cd "$ROOT"
  log "FastAPI running → http://localhost:8000 | Docs → http://localhost:8000/docs"
  sleep 3
fi

head "Step 7: Test Endpoint"
if curl -sf http://localhost:8000/ > /dev/null 2>&1; then
  python "$ROOT/scripts/04_test_endpoint.py"
else
  warn "API not running — skipping endpoint tests"
fi

head "Step 8: AI Retention Recommendations (optional)"
if [[ -f "$ROOT/.env" ]] && grep -q "ANTHROPIC_API_KEY=sk-ant" "$ROOT/.env"; then
  python "$ROOT/scripts/05_retention_recommendations.py"
  log "AI recommendations generated → outputs/retention_recommendations.csv"
else
  warn "Skipping AI recommendations — add ANTHROPIC_API_KEY to .env file first"
  warn "Get free key: https://console.anthropic.com"
  warn "Then: cp .env.example .env  and add your key"
fi

head "All steps complete"
echo ""
echo -e "  ${BOLD}Dashboard:${NC}  python dashboard/app.py  →  ${BLUE}http://localhost:8050${NC}"
echo -e "  ${BOLD}MLflow UI:${NC}  mlflow ui --backend-store-uri $ROOT/mlflow_runs  →  ${BLUE}http://localhost:5000${NC}"
echo -e "  ${BOLD}API Docs:${NC}   ${BLUE}http://localhost:8000/docs${NC}"
echo ""
read -rp "  Launch Dash dashboard now? [y/N] " yn
[[ "$yn" =~ [Yy] ]] && python "$ROOT/dashboard/app.py" || warn "Run 'python dashboard/app.py' when ready"
