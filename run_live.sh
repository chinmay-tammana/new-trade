#!/usr/bin/env bash
# ==============================================================================
# Run Script for Goldman Sachs AI Trading Indicator Live System
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

echo "======================================================================"
echo "Starting Goldman Sachs AI Trading Indicator (V3 Live System)"
echo "======================================================================"

# Check Python environment
if [ -d ".venv" ]; then
    echo "Activating virtual environment (.venv)..."
    source .venv/bin/activate
fi

# Ensure output directories exist
mkdir -p data/live results/tables results/figures results/logs

# Check Alpaca credentials
if [ -z "${APCA_API_KEY_ID:-}" ] && [ ! -f ".env" ]; then
    echo "[WARNING] APCA_API_KEY_ID not found in environment and no .env file present."
    echo "Live data fetching from Alpaca may fail if credentials are missing."
fi

# Verify model artifacts lock
echo "Verifying frozen model artifacts integrity..."
python3 -c "from src.live.engine import verify_model_lock; verify_model_lock()"

# Start server
PORT="${PORT:-8000}"

# Auto-free port if an old server instance is already running
EXISTING_PID=$(lsof -ti tcp:${PORT} -sTCP:LISTEN 2>/dev/null || true)
if [ -n "${EXISTING_PID}" ]; then
    echo "Port ${PORT} is currently in use by PID(s): ${EXISTING_PID}"
    echo "Stopping existing process on port ${PORT} to restart..."
    kill -15 ${EXISTING_PID} 2>/dev/null || true
    sleep 1
    # Force kill if still lingering
    RECHECK_PID=$(lsof -ti tcp:${PORT} -sTCP:LISTEN 2>/dev/null || true)
    if [ -n "${RECHECK_PID}" ]; then
        kill -9 ${RECHECK_PID} 2>/dev/null || true
        sleep 1
    fi
fi

echo "Launching Live Indicator Server on http://localhost:${PORT}"
exec python3 -m src.live.server --port "${PORT}" "$@"
