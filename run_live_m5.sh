#!/usr/bin/env bash
# ==============================================================================
# Run Script for Goldman Sachs AI Trading Indicator (5-Minute / 1-Hour)
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

BAR_MINUTES="${BAR_MINUTES:-5}"
PORT="${PORT:-8000}"

echo "======================================================================"
if [ "${BAR_MINUTES}" -eq 5 ]; then
    echo "Starting Goldman Sachs AI Trading Indicator (5-Minute Horizon Pipeline)"
else
    echo "Starting Goldman Sachs AI Trading Indicator (1-Hour Horizon Pipeline)"
fi
echo "======================================================================"

if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

mkdir -p data/live results/m5/tables results/m5/figures results/m5/logs

# Free port 8000 if in use
EXISTING_PID=$(lsof -ti tcp:${PORT} -sTCP:LISTEN 2>/dev/null || true)
if [ -n "${EXISTING_PID}" ]; then
    echo "Stopping existing process on port ${PORT} (PID ${EXISTING_PID})..."
    kill -15 ${EXISTING_PID} 2>/dev/null || true
    sleep 1
    RECHECK_PID=$(lsof -ti tcp:${PORT} -sTCP:LISTEN 2>/dev/null || true)
    if [ -n "${RECHECK_PID}" ]; then
        kill -9 ${RECHECK_PID} 2>/dev/null || true
        sleep 1
    fi
fi

if [ "${BAR_MINUTES}" -eq 5 ]; then
    echo "Running M5 Doctor diagnostics..."
    python3 -m src.live_m5.doctor
    echo "Starting FastAPI M5 server on http://localhost:${PORT}..."
    exec uvicorn src.live_m5.server:app --host 0.0.0.0 --port "${PORT}" --log-level info
else
    echo "Verifying 1-hour frozen model lock..."
    python3 -c "from src.live.engine import verify_model_lock; verify_model_lock()"
    echo "Starting FastAPI 1-hour server on http://localhost:${PORT}..."
    exec uvicorn src.live.server:app --host 0.0.0.0 --port "${PORT}" --log-level info
fi
