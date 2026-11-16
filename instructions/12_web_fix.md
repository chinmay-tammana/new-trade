V3.1-FIX: WEBSITE FIX. Continue in this project folder. The rules in AGENTS.md and instructions/ still apply (no placeholders, seed 42, never edit instructions/ except progress.md) with ONE override: IGNORE every GitHub and git rule for this task. Do not run git commit, push, tag, remote or secret-check-before-commit commands, and do not touch .gitignore or any repository settings. Just keep all files on disk. This prompt overrides any other conflict; record every override in progress.md.

STEP 0
- Save this prompt verbatim as instructions/12_web_fix.md and update AGENTS.md to read it last.
- model-v2 stays FROZEN: no retraining, no changes to features, q_lo, q_hi, calibrator, hyperparameters or model files. Verify by computing the sha256 of models/model_v2.joblib, models/calib_v2.joblib and models/calib_live.json and comparing them with models/MODEL_SHA256.json. If that file is missing, create it now from the current files and write in progress.md that it cannot prove nothing changed earlier. Existing rows in the signals table are never updated or deleted.
- Never print API keys, and never put keys or environment values in any API response, log file or the static page.
- Test policy from the TA (record in progress.md and docs): a ONE-DAY test is sufficient. The indicator uses 1-hour candles exactly as in the rest of the project; there is NO 10-minute mode. Remove any 10-minute or 5-day wording from the UI and docs. Update docs/live_expectations.md: a full day gives about 6 signals (closes at 10:30, 11:30, 12:30, 13:30, 14:30 and 15:30 ET), which is a demonstration, not statistical proof; the UI must say so.

A. REPRODUCE AND FIND ROOT CAUSES (do not guess)
1. Print the system date/time in ET and IST, whether the market is open, and the next open. Start the server, curl every /api endpoint, and capture the browser console errors. Write each root cause into progress.md with its fix. Known symptoms from my screenshot (page at localhost:8000, Saturday 01:19 ET):
   - The signal panel stays on "INITIALIZING" with confidence "--%".
   - "Next bar close" shows 3430:21 (minutes to Monday open formatted as mm:ss).
   - Badge "MODEL: UNKNOWN" next to "SIGNALS ACTIVE".
   - Volatility class, regime, hourly hit rate and the signal log are empty.
   - The chart ends around Nov 9: the sessions after the data freeze are missing, and there are no signal markers.
   - One candle around Nov 5 spans about 855 to 930, which looks like a bad IEX print.
2. Investigate the Nov 5 candle: list every 1h bar of the last 120 trading days with (high-low)/close > 3%, and compare each with the SIP-built bar. If an IEX print is bad, report it and mark such bars in the UI as "data anomaly" (never silently alter the model's inputs, never modify stored rows) and tell me.
3. Verify the model hash: find out why MODEL is UNKNOWN (missing models/MODEL_SHA256.json, wrong path or an unread field). Fix it. If the hash is unknown or mismatched the engine must refuse to emit signals and the UI must show a red "MODEL NOT VERIFIED" banner instead of "SIGNALS ACTIVE".

B. REQUIRED BEHAVIOUR
1. Backfill the chart with 1h IEX bars through the LAST COMPLETED SESSION (not just the freeze date), contiguous, 600+ bars of history. Chart timestamps in ET (the first bar of a day reads 09:30; verify it). Show ET and IST clocks in the header and footer.
2. Backfilled signals: for every closed bar of the last 10 completed sessions, compute the frozen model's signal with the same engine code and store it ONCE with late=true and a field sample = "IN-SAMPLE" if the bar date is on or before freeze_date (read configs/data_freeze.json; the frozen model was trained on that period) or "OUT-OF-SAMPLE" if it is after freeze_date. The UI shows both kinds but labels them clearly: IN-SAMPLE signals are for display only and are excluded from every scoreboard; OUT-OF-SAMPLE backfilled signals are scored and labelled "computed after the fact".
3. Market-closed mode (Saturday, Sunday, holidays, after 16:00 ET): the main panel never shows "INITIALIZING". It shows the latest stored signal with its bar time, its outcome (correct, wrong or pending), and a banner "MARKET CLOSED. Next signal: <weekday> 10:30 ET (<IST time>)". The countdown shows days/hours/minutes to the next session open, then mm:ss only during the session. Fix the 3430:21 bug and add a unit test for the formatter.
4. Main panel readable in one second:
   - A huge word UP (green, up-arrow), DOWN (red, down-arrow) or NO SIGNAL (grey), with confidence ("Up 58%") and the hour it covers ("covers 10:30 to 11:30 ET").
   - A STRONG tag, volatility class, regime, the band as two lines on the chart, the hourly hit rate with n.
   - Down and Up must be equally prominent. If the engine produces no Down signals over a full session, show a diagnostic note for me instead of hiding it.
   - Below the word: one plain sentence, for example "Model leans UP for the next hour. Single signals are weak evidence." Colour is never the only cue (shape and text too).
   - Remove the hard-coded statistical notice. Read the verified V2 numbers from the V3.1 verified-results file (create results/logs/verified_v2.json from the saved holdout predictions if it does not exist) and display only those.
5. Markers: one marker per closed candle (green up-arrow below, red down-arrow above, small grey dot for No signal, text "Up 58%"), with STRONG larger. Past markers never change between refreshes. A day-summary card for the selected day shows counts of Up, Down and No signal, hits, misses and pending, and the hit rate with n and the Wilson 95% CI. It must say that a one-day sample is a demonstration, not proof.
6. Badges must tell the truth: LIVE (data age under 3 minutes during the session), STALE (red, older), MARKET CLOSED, REPLAY, FEED: IEX, MODEL: <first 8 chars of sha256> (green if verified), MODEL NOT VERIFIED (red), SIGNALS DISABLED (if live_signals_enabled=false because a parity gate failed; say which gate). Never show SIGNALS ACTIVE unless the model is verified, live_signals_enabled is true and the data are fresh or the market is closed.

C. LIVE DATA ROBUSTNESS (the market is closed today, so build it and test it with the replay)
- The engine fetches the last hour of 1-minute IEX bars at bar_end + 10 s for GS and all 7 context symbols (retry every 5 s for up to 90 s), then computes and stores the signal exactly once. The forming candle comes from the websocket with a REST polling fallback every 15 s. /api/status reports: last minute bar received, its age in seconds, websocket state, the last engine run, last error and rows in the database.
- Write python3 -m src.live.doctor which prints and exits non-zero on any failure: times in ET and IST, market open and next open, Alpaca REST test per symbol (latest bar time and age), websocket test (only if the market is open), database state and hash-chain verification, model hash check, server status. I will run it on Monday before the market opens.

D. REPLAY AND TEST DAY
- Add a Replay control in the UI: a date picker and a speed slider that replays any completed day through the SAME engine into a temporary database, with a REPLAY badge, so I can demonstrate a full day even when the market is closed. Replay never writes to the live database.
- Replay equivalence test: replayed signals equal src/predict_day.py for the same date with the frozen model (same direction, confidence within 1e-9) on at least 3 days including one half-day.
- Tests: formatter, market-closed panel state, the badge matrix, the model hash, no INITIALIZING state after load when any stored signal exists, a UI snapshot that previous markers are unchanged after a new bar, no secrets in any API response, plus all earlier integrity checks (truncation test, ledger audit, live-compat). If playwright can be installed, add a headless smoke test: the page loads, shows UP/DOWN/NO SIGNAL (not INITIALIZING) within 10 s, the countdown matches the expected format, and there are no console errors. Otherwise use curl-level checks.

E. DOCS
- Update docs/user_guide.md, docs/live_runbook.md, docs/live_expectations.md, README.md, report/report.md and slides/outline.md for the one-day test and 1h candles. Add a Monday checklist: start run_live.sh before 09:00 ET (19:30 IST); the first live signal appears at 10:30 ET (21:00 IST); run the doctor first; keep the computer awake and online all day.
- Record progress and every decision in instructions/progress.md.

Final printed summary: the root cause of each bug; the explanation of the Nov 5 candle; the model hash status; the last session shown on the chart; counts of IN-SAMPLE and OUT-OF-SAMPLE backfilled signals with the out-of-sample hits, misses and Wilson CI (descriptive only, small n); replay equivalence results; doctor output; test results; what I must do on Monday.
