# V3 Live Operations Runbook

This operational runbook provides instructions for deploying, running, monitoring, and evaluating the Goldman Sachs (`GS`) AI Trading Indicator live system.

---

## 1. Prerequisites and Setup

### 1.1 Environment Variables
Ensure Alpaca API credentials are set in `.env` or system environment:
```bash
export APCA_API_KEY_ID="<your_key_id>"
export APCA_API_SECRET_KEY="<your_secret_key>"
export APCA_API_BASE_URL="https://paper-api.alpaca.markets"
```

### 1.2 System Clock and Timezone
- The engine uses `America/New_York` (ET) for all session calculations and bar cutoffs.
- Verify system clock synchronization:
  ```bash
  date
  ```
- **Keep Machine Awake**: During NYSE market hours (09:30 to 16:00 ET, Monday through Friday), prevent the host system from sleeping:
  - On macOS:
    ```bash
    caffeinate -dimsu &
    ```
  - On Linux: Configure `systemd` or disable sleep settings.

---

## 2. Starting the Service

### 2.1 Using the Helper Script
```bash
chmod +x run_live.sh
./run_live.sh
```

### 2.2 Manual Execution (Direct Python)
```bash
source .venv/bin/activate
python -m src.live.server --port 8000
```

### 2.3 Container Execution (Docker)
```bash
docker build -t gs-live-indicator:v3 .
docker run -d --name gs-live \
  -p 8000:8000 \
  -v $(pwd)/data/live:/app/data/live \
  --env-file .env \
  gs-live-indicator:v3
```

---

## 3. Monitoring & Verification

### 3.1 Web Dashboard
Open [http://localhost:8000](http://localhost:8000) in your browser:
- Header shows live market status, next bar close timer, and feed status.
- Chart displays 1-hour candles with real-time forming bar.
- Signal banner displays directional prediction, confidence, volatility class, and return bands.
- Banners alert to active parity fallbacks, half-day sessions, or scheduled macroeconomic events.

### 3.2 API Health Probes
```bash
# Liveness probe
curl -s http://localhost:8000/healthz

# Engine status and parity gates
curl -s http://localhost:8000/api/status | jq .

# Latest generated signal
curl -s http://localhost:8000/api/latest | jq .
```

### 3.3 Database Audit Verification
To verify that no database rows have been tampered with or modified:
```bash
python3 -c "from src.live.store import LiveDatabase; db = LiveDatabase(); print(db.verify_hash_chain())"
```
A valid database will print `(True, N, None)`.

---

## 4. Accelerated Replay & Testing

To replay any historical trading day on an accelerated clock and verify 100% equivalence with historical models:
```bash
# Replay normal trading day
python -m src.live.server --replay 2026-11-06

# Replay half-day (13:00 early close)
python -m src.live.server --replay 2024-11-29
```

---

## 5. Live Scoring & Daily Evaluation

At the close of each trading day (after 17:00 ET):
```bash
python -m src.live.score --since 2026-11-10
```

Outputs generated:
1. `results/tables/live_scoreboard.csv`: Cumulative summary table tracking total signals, hits, per-side accuracy with 95% Wilson CIs, base rates, and net returns.
2. `results/tables/live_{date}.csv`: Detailed per-bar signal log.

*Note*: If fewer than 5 trading days have elapsed, the scoreboard prints an interim warning reflecting the statistical power limitations of small sample sizes.

---

## 6. Stopping and Restarting

### 6.1 Graceful Shutdown
Send `SIGINT` (Ctrl+C) or `SIGTERM`. The server will release its process lock (`data/live/live.lock`) and close open SQLite connections.

### 6.2 Mid-Session Restart
If the service restarts during an active trading session:
1. The engine checks the rolling history buffer ($\ge 600$ closed bars) from cached files and SQLite.
2. It fetches any missed closed bars. Any bar processed $>3$ minutes late is recorded with `late = 1` and excluded from official live evaluation metrics.
