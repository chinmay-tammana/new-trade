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
echo "Launching Live Indicator Server on http://localhost:${PORT}"
exec python3 -m src.live.server --port "${PORT}" "$@"
