# V3 Live Deployment Guide

This guide describes containerized and host-level deployment procedures for the Goldman Sachs AI Trading Indicator live system.

---

## 1. Environment and Configuration

The live service requires Alpaca credentials to stream real-time market data from the IEX feed:
- `APCA_API_KEY_ID`: Alpaca API Key ID.
- `APCA_API_SECRET_KEY`: Alpaca Secret Key.
- `APCA_API_BASE_URL`: Defaults to `https://paper-api.alpaca.markets`.

Store these in a secure `.env` file or export them into the container environment. Never commit `.env` to version control.

---

## 2. Docker Deployment

### 2.1 Build the Image
```bash
docker build -t gs-live-indicator:v3 .
```

### 2.2 Run with Persistent Volume Mount
To ensure the SQLite audit trail (`data/live/live.db`) and scoring tables survive container restarts, mount a host directory to `/app/data/live`:

```bash
docker run -d \
  --name gs-live-indicator \
  --restart unless-stopped \
  -p 8000:8000 \
  -v $(pwd)/data/live:/app/data/live \
  --env-file .env \
  gs-live-indicator:v3
```

### 2.3 Verify Health
The container defines an automated `HEALTHCHECK` running every 30 seconds against `/healthz`:
```bash
docker inspect --format='{{json .State.Health}}' gs-live-indicator | jq .
```

---

## 3. Host-Level Deployment (Systemd on Linux)

For bare-metal or cloud VM deployments without Docker, configure a systemd service unit:

Create `/etc/systemd/system/gs-live.service`:
```ini
[Unit]
Description=Goldman Sachs Live Trading Indicator
After=network.target

[Service]
Type=simple
User=appuser
WorkingDirectory=/opt/trade_indicator
EnvironmentFile=/opt/trade_indicator/.env
ExecStart=/opt/trade_indicator/.venv/bin/python -m src.live.server --port 8000
Restart=always
RestartSec=5s

[Install]
WantedBy=multi-user.target
```

Enable and start the service:
```bash
sudo systemctl daemon-reload
sudo systemctl enable gs-live
sudo systemctl start gs-live
sudo systemctl status gs-live
```

---

## 4. Production Security & Zero-Trust Verification

- **Read-Only API**: The FastAPI server exposes zero mutating endpoints (`POST`, `PUT`, `DELETE`).
- **Zero Secrets**: Alpaca credentials, API secrets, and server environment variables are stripped and never exposed via any API endpoint.
- **Model Tamper Proofing**: On startup, `src.live.engine` validates the SHA256 hashes of `models/model_v2.joblib`, `calib_v2.joblib`, and `calib_live.json` against `models/MODEL_SHA256.json`. Any artifact discrepancy prevents startup unless explicitly overridden with `--allow-model-change`.
- **Append-Only Auditing**: Every emitted signal is cryptographically chained in SQLite using SHA-256.
