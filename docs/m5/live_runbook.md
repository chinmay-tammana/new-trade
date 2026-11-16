# Live Operational Runbook: M5 (5-Minute) System

## 1. Pre-Market Startup Procedure (Before 09:00 ET / 19:30 IST)
1. **Activate Virtual Environment & Check Repository**:
   ```bash
   cd "/Users/chinmay/Desktop/trade final"
   source .venv/bin/activate
   ```
2. **Run System Diagnostic Doctor**:
   ```bash
   python3 -m src.live_m5.doctor
   ```
   Verify that all 6 diagnostic health checks report `PASSED`:
   - Dual clocks (ET & IST) synchronized.
   - Market calendar and next open/close times correct.
   - IEX real-time feed connectivity verified across all 6 symbols (GS, SPY, XLF, JPM, MS, BAC).
   - Frozen model SHA-256 matches `models/m5/MODEL_SHA256_m5.json`.
   - SQLite signals database and cryptographic hash chain verified unbroken.
3. **Launch the Live Server**:
   ```bash
   ./run_live_m5.sh
   ```
   The script starts the live operational server on `http://localhost:8000` with `BAR_MINUTES=5`.
4. **Prevent Computer Sleep**: Ensure macOS system sleep is disabled during regular trading hours (09:30 - 16:00 ET) using `caffeinate` or system preferences.

## 2. Intraday Operational Monitoring
- Open `http://localhost:8000` in Google Chrome or any modern browser.
- Verify the header badges: `SIGNALS ACTIVE`, `SIGNALS: IEX REAL-TIME (5-MIN)`, `MODEL: BB018397`.
- Signals are generated automatically every 5 minutes at bar close (09:35, 09:40, ..., 15:55 ET).
- At 16:00 ET, the final bar will automatically output `No signal (end of day)`.

## 3. Signal Rule Application CLI (`src.m5.apply_rule`)
To manage signal generation rules without touching models or deleting existing rows:
```bash
# Apply Always-Call rule (Rule v3 - directional on every eligible interval):
PYTHONPATH=. .venv/bin/python3 -m src.m5.apply_rule --mode always

# Apply quantile coverage target (e.g. 30% coverage):
PYTHONPATH=. .venv/bin/python3 -m src.m5.apply_rule --mode quantile --coverage 0.30

# Re-derive across a custom number of sessions (default: 10):
PYTHONPATH=. .venv/bin/python3 -m src.m5.apply_rule --mode always --sessions 15
```
**What the tool does**:
1. Derives directional sides from stored $P_{\text{cal}}$: Up if $P > 0.5$, Down if $P < 0.5$.
2. Sets confidence tiers: WEAK ($\le 52.1\%$), NORMAL ($52.1\% - 58.1\%$), STRONG ($\ge 58.1\%$).
3. Preserves `No signal` strictly for end-of-day bars (15:55 ET), data gaps, or exact 0.50 ties.
4. Non-destructively inserts new rows with `rule_version = 3` (`late = 1`) into `data/live/live_m5.db`.
5. Verifies the unbroken cryptographic SHA-256 hash chain.
6. Prints a per-day before-and-after comparison table for the last 5 sessions.
