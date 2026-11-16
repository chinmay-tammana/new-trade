# V3 Live Operations Runbook & Monday Live Test Guide

This operational runbook provides step-by-step instructions for deploying, inspecting, running, and auditing the Goldman Sachs (`GS`) AI Trading Indicator live system.

---

## 1. Monday Morning Live Test Checklist (Pre-Market)

Before the NYSE open on Monday:

1. **Keep System Awake**:
   Ensure the host system will not sleep or throttle network during market hours (09:30 to 16:00 ET, which is 20:00 to 02:30 IST):
   ```bash
   # On macOS
   caffeinate -dimsu &
   ```
2. **Run System Doctor (Health Check)**:
   Run before 09:00 ET (19:30 IST) to verify all 7 sub-systems:
   ```bash
   python -m src.live.doctor
   ```
   Confirm that all 7 checks pass:
   - System Clocks (ET and IST verified)
   - NYSE Calendar, Next Session Open (Monday 09:30 ET / 20:00 IST), and Next Delivery (Monday 10:46 ET / 21:16 IST)
   - Delay arithmetic (16 min = 960 s)
   - Frozen Model SHA256 Hashes (`75d0b0bf`, `0584bcff`, `bb56a297`)
   - SQLite Database & Cryptographic Hash Chain (schema_v=1 and schema_v=2 valid)
   - Alpaca Delayed SIP Feeds for all 8 symbols (GS, SPY, XLF, JPM, MS, BAC, TLT, VIXY)
   - Local web server response on port 8000
3. **Start the Live Dashboard & Engine**:
   Start `./run_live.sh` before 09:00 ET (19:30 IST):
   ```bash
   ./run_live.sh
   ```
4. **Open the Dashboard**:
   Navigate to [http://localhost:8000](http://localhost:8000). Verify:
   - Header badge shows `SIGNALS: SIP (15-min delayed)` and `PRICE: IEX live (display only)`.
   - `MODEL: 75d0b0bf` (verified green badge).
   - Dual clocks show synchronized New York (ET) and Indian Standard Time (IST).
   - Countdown timer shows time remaining until next signal delivery: `Monday 10:46 ET (21:16 IST)`.
5. **Auditing with the Signal Log Day View**:
   - The Signal Log below the chart provides a dedicated one-day view (defaults to latest scored session or active session).
   - Use `◀ Prev` and `Next ▶` or the date picker to inspect specific trading days.
   - All numbers on the day summary, table rows, hour strip, and sidebar balance card derive from a single source of truth (`src/live/log_engine.py`).
   - Click any row to expand its cryptographic audit details (`P(Up)`, model hash `75d0b0bf`, delivery delay, record hash).
   - Click any hour chip to highlight that candle on the chart and scroll smoothly to its row.

---

## 2. Live Signal Delivery Schedule (Delayed SIP)

The live engine operates on **Option B: Delayed SIP for Signals (>=15 min delay), IEX for Display Only**:
- **09:30 ET (20:00 IST)**: NYSE Opening Bell. First hourly candle starts forming.
- **10:30 ET (21:00 IST)**: Bar #1 closes. Ingestion starts with 15-minute regulatory end-lag buffer.
- **10:46 ET (21:16 IST)**: **Signal #1 Delivered** (emitted at `bar_end + 16 min`). Headline says: *"Signal for the hour starting 09:30 ET. Bar closed 10:30, delivered 10:46 ET. This hour is already 16 min old."* Covers remaining 44 min of the 10:30–11:30 candle.
- **11:46 ET (22:16 IST)**: **Signal #2 Delivered** (Bar 10:30–11:30 closed at 11:30 ET). Outcome of Signal #1 evaluates.
- **12:46 ET (23:16 IST)**: **Signal #3 Delivered** (Bar 11:30–12:30 closed at 12:30 ET).
- **13:46 ET (00:16 IST)**: **Signal #4 Delivered** (Bar 12:30–13:30 closed at 13:30 ET).
- **14:46 ET (01:16 IST)**: **Signal #5 Delivered** (Bar 13:30–14:30 closed at 14:30 ET).
- **15:46 ET (02:16 IST)**: **Signal #6 Delivered** (Bar 14:30–15:30 closed at 15:30 ET). Covers remaining 14 min of regular trading session.
- **16:00 ET (02:30 IST)**: NYSE Closing Bell. Final bar (15:30–16:00 ET) closes session and is marked **No signal (end of day)**.
- **12:30 ET (23:00 IST)**: **Signal #3** emitted.
- **13:30 ET (00:00 IST)**: **Signal #4** emitted.
- **14:30 ET (01:00 IST)**: **Signal #5** emitted.
- **15:30 ET (02:00 IST)**: **Signal #6** emitted.
- **16:00 ET (02:30 IST)**: Session Close. The 15:30–16:00 bar is an end-of-day half-hour bar; the engine emits `No signal (end of day)` and censors forward evaluation because trades are not held overnight.

Total evaluation signals for a full trading day: **approximately 6 signals**.

---

## 3. Handling Feeds and Anomalies

- **Missing Context Bar / Feed Stall**: If any context symbol (e.g. TLT or VIXY) lags or drops from the feed, the engine will NOT guess or hallucinate features; it emits `No signal (data gap)` with confidence `0.50`, logs the missing symbol, and continues.
- **Mid-Session Restart**: If the process restarts mid-day, it recovers historical bars from SQLite and marks any bar computed $>3$ minutes late with `late = 1`.
- **Anomalous Spikes**: If an intraday candle exceeds a 3% high-low range, the chart flags the bar with an anomaly badge for visual review while preserving true price action.

---

## 4. End-of-Day Audit Verification

At market close (after 16:00 ET / 02:30 IST):

1. **Verify Hash Chain Ledger**:
   ```bash
   python -m src.live.doctor
   ```
   Or via direct audit script:
   ```bash
   python -c "from src.live.store import LiveDatabase; db = LiveDatabase(); print(db.verify_hash_chain())"
   ```
   Must output `(True, N, None)` confirming cryptographic immutability from bar 0 to bar N.
2. **Score Day's Out-of-Sample Performance**:
   ```bash
   python -m src.live.score --since 2026-11-16
   ```
   Review `results/tables/live_scoreboard.csv`.

3. **Audit Scored Out-of-Sample Signals & Outcomes Table**:
   ```bash
   python -m src.live.audit_scoreboard
   ```
   Generates `results/tables/live_scoring_audit.csv` with per-bar forward return, label $y$, outcome, and Wilson intervals. Verifies that `No signal` rows remain unscored abstentions and only directional signals (`Up`/`Down`) are evaluated.

---

## 5. Statistical Policy Reminder

A 1-day live test ($N \approx 6$) serves as a real-time **engineering demonstration** of data pipelines, model inference, audit logging, and UI responsiveness. It has a 95% Wilson confidence interval spanning $>50$ percentage points and **cannot** be used to prove or disprove statistical edge. Formal statistical proof was conducted on the 1,422 sealed holdout candles ($p = 0.0417$).
