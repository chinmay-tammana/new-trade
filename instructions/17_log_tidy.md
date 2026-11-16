V3.6-LOGTIDY: TIDY THE SIGNAL LOG. Continue in this project folder. All rules in AGENTS.md and instructions/ still apply (no placeholders, seed 42, never edit instructions/ except progress.md) with ONE override: IGNORE every GitHub and git rule; do not run any git commands. Save this prompt verbatim as instructions/17_log_tidy.md and update AGENTS.md to read it last.

SCOPE AND HARD CONSTRAINTS
- Scope: ONLY the signal log on the page, plus read-only API endpoints that feed it. Do not change the model files, thresholds, calibrator, the signal engine, stored rows (no UPDATE, no DELETE, no schema change), scoring rules, the chart, the headline panel or the diagnostics content. No download or export buttons anywhere. Never put keys or environment values in any API response or static file.
- Every number on the page must come from ONE function that builds the rows and their summaries. The log, day header, hour strip, signal balance card and accuracy card all derive from it, so they can never disagree.

PROBLEMS FROM MY SCREENSHOT (localhost:8000)
1. The day header for 2026-11-13 says "Up: 8, Down: 16, No-Sig: 0 | Hits: 8, Misses: 16", but the Signal Balance card says Last 1 Session: Up 1, Down 1, No-Sig 5. Hits plus misses equals 24, so the header seems to aggregate many rows under one date.
2. The same bar appears twice (14:30: once "SIP (delayed)", once "IEX (shifted)"), although IEX rows are excluded from scoring. The title says "112 rows" for a period that has 24 out-of-sample candles.
3. Each row takes three lines because of the long "P(Up) 38.0%: needs >= 56.3% for Up..." text, so only about 3 rows are visible and the TA has to scroll and search.
4. The first columns are confusing: "TIME (ET) 14:30 | TIME (IST) 01:00 | DELIVERED 15:46 ET" does not say which hour is being predicted.
5. The log panel is short and the diagnostics section below pushes everything down.

A. ROOT CAUSES FIRST
Find why there are 112 rows and why the day header counts are wrong (grouping bug, mixed feeds, mixed horizons or duplicate inserts). Write each root cause and fix into progress.md.

B. NEW DESIGN: A ONE-DAY VIEW (the TA tests one trading day)
1. Default: ONE trading day at a time. Controls: Prev day / Next day buttons, a date input, "Latest session", and "Today" (when the market is open). Default day = today during a session, otherwise the latest session that has scored signals. Support ?day=YYYY-MM-DD in the URL.
2. Horizon tabs: "1 hour" (default) | "3 hours" | "Rest of day". One horizon at a time, so rows never mix horizons.
3. Day summary strip above the table, for example: "Fri 13 Nov 2026 | GS 910.10 -> 916.16 (+0.67%) | Signals 2 (Up 1, Down 1) | Correct 1 | Wrong 1 | Pending 0 | No signal 5 | Accuracy 1/2 (Wilson 95% CI [..])". Also show "demonstration, not proof (n=2)" in small text.
4. Hour strip: a row of chips, one per hour of the day (for example 10:30, 11:30, ..., 15:30), each showing arrow + word + confidence ("▲ Up 63%", "▼ Down 62%", "– No signal") and a small result mark (✔, ✘, ○). Colour is never the only cue. Clicking a chip scrolls to its row and highlights the candle on the chart.
5. Table: ONE compact line per hour, row height at most 56 px, 14 px minimum font, zebra stripes, tabular numerals, sticky header. Columns, in this order:
   - Hour predicted (ET), for example "10:30-11:30" (for other horizons the window of that horizon).
   - Issued (ET | IST), for example "10:46 | 21:16". For rows that were computed afterwards (late=true) show the time in grey italics with "(replayed)"; for true live rows show the real stored delivered_at in normal text.
   - Call: pill with arrow + word + confidence ("▲ UP 63%", "▼ DOWN 62%", "– NO SIGNAL"). For No signal show a short reason (end of day, data gap, inside the no-signal band, market closed) and NEVER a confidence or STRONG tag.
   - Strength: "STRONG" or blank.
   - Vol and band: for example "Normal | -46 / +37 bps".
   - Price move: "$907.63 -> $909.34 (+0.21%)", or "pending" until the next bar closes.
   - Result: ✔ Correct / ✘ Wrong / ○ Pending.
   - Source: one small pill (OUT-OF-SAMPLE, LIVE, IN-SAMPLE or REPLAY) with a tooltip.
6. The long probability line moves into an expandable detail row (click or tap a row): "P(Up) 38.0%. Up needs >= 56.3%, Down needs <= 43.4%" (thresholds read from the calibration file, never hard-coded), plus regime, feed, model hash prefix, bar close time, delivery delay and the stored record hash. Only one detail row open at a time.
7. De-duplicate: EXACTLY ONE row per (bar, horizon). Preference: LIVE sip_delayed, then replayed or backfilled sip_delayed. IEX (shifted) rows are hidden by default; a toggle "Show IEX rows (not scored)" shows them greyed out and clearly labelled, never inside the counts.
8. Filters, minimal: View = "Scored (out-of-sample + live)" (default) | "All including in-sample"; "Show No-signal rows" defaults to ON in the day view so the whole hourly timeline is visible.
9. Layout: the log sits directly below the chart. Reduce the chart height so that at 1440x900 the day summary, the hour strip and ALL rows of a full day (up to 7 rows) are visible without scrolling the page. Collapse the "Model & Feed Diagnostics" section by default behind a "Show diagnostics" toggle; its content stays unchanged.
10. Responsive: no horizontal scroll at 1024 px and above. Below 900 px each row becomes a compact card (call, hour, result on the first line; the rest below). Never clip text; wrap instead.
11. Live behaviour: during a session new rows appear without a page reload; the newest row is highlighted for 10 s; a pending row switches to Correct or Wrong when its outcome is written. Past rows are never reordered or altered. Keep the row-to-candle highlight on hover.
12. Convert ET to IST with zoneinfo (America/New_York to Asia/Kolkata); do not hard-code an offset.
13. Loading skeleton, a clear empty state with the reason ("No scored signals on this day; 6 exist if you enable 'All including in-sample'"), an error state with a Retry button; never an endless "Loading".

C. API (read-only; keep old endpoints working)
- GET /api/days returns days that have signals with counts per category.
- GET /api/log?day=&horizon=&view=&include_iex=&include_nosignal= returns {rows, summary}, where the summary is computed on the server from exactly those rows.

D. TESTS AND PROOF
- Unit tests: de-duplication (SIP preferred over IEX, one row per bar and horizon); summary equals the rows (Up + Down + No signal = rows; hits + misses + pending = Up + Down; No signal never counts as a hit or miss); the header, the strip and the balance card agree; ET to IST on DST boundaries (2026-03-08 and 2026-11-01); the late versus live display rule; the empty state.
- Real headless-browser test (tests/ui_render_check.py) at 1280x720, 1440x900, 1920x1080 and 390x844: the full day's rows are visible without page scroll at 1440x900; no horizontal scroll; each row is one line at 1280 and above; the strip chip count equals the row count; no IEX rows by default; no stuck loading text; no console errors; the day picker works (prev, next, date input, ?day= URL). Save screenshots to results/figures/ui_log_<width>x<height>.png, open them yourself, describe what you see, and fix anything unclear.
- Re-run the replay equivalence test and the model hash check (sha256 of models/model_v2.joblib, models/calib_v2.joblib and models/calib_live.json against models/MODEL_SHA256.json).

E. DOCS
Update docs/user_guide.md (a "How to read the log" section with an annotated example row), docs/live_runbook.md and README.md. Record every decision in instructions/progress.md.

Final printed summary: the root cause of the 112 rows and of the wrong day counts; the new column list; before-and-after row counts for 2026-11-13; test results per viewport; screenshot paths; model hash status.
