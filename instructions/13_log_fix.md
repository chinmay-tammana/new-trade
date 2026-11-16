V3.2-FIX: SIGNALS LOG AND PANEL FIX. Continue in this project folder. All rules in AGENTS.md and instructions/ still apply (no placeholders, seed 42, never edit instructions/ except progress.md) with ONE override: IGNORE every GitHub and git rule. Do not run any git commands. Save this prompt verbatim as instructions/13_log_fix.md and update AGENTS.md to read it last.

HARD CONSTRAINTS
- model-v2 stays FROZEN (no retraining, no changes to features, q_lo, q_hi, calibrator, hyperparameters or model files). Never UPDATE or DELETE existing rows of the signals table. Do not change live_signals_enabled or the parity-gate verdict; this task is display and scoring-audit only.
- Never put keys or environment values in any API response or static file.

A. REPRODUCE
Start the server, open the page at viewport sizes 1280x720, 1366x768, 1440x900 and 1920x1080, and record what is clipped. Root cause (print it into progress.md): the right column has a fixed height with overflow hidden and no scroll region, so the signals log is clipped.

B. SIGNALS LOG (the main fix)
1. Layout: make the page a 100vh flex column (header, main, footer). The right column becomes a flex column with min-height:0; it is scrollable as a whole if needed, and the signals log is its own panel with flex:1, min-height 260px and overflow-y:auto, with a sticky header row. The last row must always be reachable by scrolling at every viewport above.
2. Table design, no clipped or truncated text (wrap instead of cutting off):
   - Columns: Time ET | Time IST | Horizon | Signal (arrow + word, e.g. "▲ Up", "▼ Down", "– No signal") | Conf | Tier (STRONG or blank) | Outcome (✔ Correct, ✘ Wrong, ○ Pending) | Source tag.
   - Source tag shows IN-SAMPLE, OUT-OF-SAMPLE, LIVE or REPLAY as a coloured pill with a tooltip explaining it. Remove the cryptic "[S]" label. Add a one-line legend above the table.
   - For No signal rows: Conf is blank and the reason is a short text ("end of day", "data gap", "market closed").
   - Newest first, grouped by trading day with a day header showing the date and a mini summary (counts of Up, Down, No signal, hits, misses, pending).
   - Filter toggles: All | Out-of-sample + Live | Live only, and "Show No-signal rows" (default off). Default view: Out-of-sample + Live.
   - Clicking a row scrolls the chart to that candle and highlights it. Hovering a candle highlights its row.
   - A "Download CSV" button (GET /api/signals.csv, read-only) exporting the filtered view.
   - Colour is never the only cue (shape and text too).
3. Add 'limit' and 'offset' paging to /api/signals so the log stays fast with hundreds of rows, with "Load more" at the bottom.

C. OTHER VISIBLE BUGS IN THE SAME FILES
1. Countdown still shows 3342:21: write the formatter as a pure function: more than 24 h -> "2d 7h 12m"; 1 to 24 h -> "7h 12m"; under 1 h -> "mm:ss". Add unit tests for each range and for DST days. The same formatter feeds the header.
2. Headline: never clip. Use a fitted font size (CSS clamp or JS fit); "NO SIGNAL" is the headline and "end of day" or the reason goes in a smaller subtitle. The STRONG tag never overlaps text (put it on its own line). No confidence is shown for No signal.
3. Chart markers: the text label on every marker creates an unreadable pile. Show arrows only (STRONG slightly larger), and show the "Up 57%" text only for the last 6 bars, on hover, or when the bar spacing is wide enough that labels cannot overlap. Add a "labels on/off" toggle.
4. When SIGNALS DISABLED is shown, add a "Why?" link showing the three parity gates, their thresholds and the measured values (read from the saved parity results), and a line "Signals shown on the chart were computed after the fact and are display-only until the gates pass".

D. SCOREBOARD AUDIT (must be done; the panel currently shows 24 resolved signals and a 20.8% hit rate)
1. Write results/tables/live_scoring_audit.csv with one row per scored signal: signal bar start and end, horizon, direction, confidence, target bar end, close at the signal bar, close at the target bar, forward return, label y (1 if forward return > 0), correct or wrong, source tag. Verify: only OUT-OF-SAMPLE and LIVE signals are scored; No signal rows are never counted; the outcome uses the close of the NEXT bar versus the close of the signal bar; Up is correct iff y = 1 and Down is correct iff y = 0; no off-by-one bar; no inverted comparison.
2. Print the number of scored Up and Down signals separately with hits, per-side accuracy, the Wilson 95% CI and the exact binomial p-value against 50%. Compare with the base Up-rate and Down-rate over all candles of the same sessions. State plainly that this is a tiny descriptive sample.
3. If a scoring bug is found, do NOT edit stored signals. Compute corrected outcomes into a new table outcomes_v2, keep the old one, record the bug and fix in progress.md, and point the API and UI at outcomes_v2. If there is no bug, say so and keep the numbers as they are; do not hide a bad result.
4. The scorecard label must read "OUT-OF-SAMPLE HIT RATE (n=24, descriptive only)" with the Wilson CI, never a bare percentage.

E. TESTS
- Add a headless browser test if playwright can be installed (otherwise do CSS and DOM inspection through unit tests): for each viewport above, the log panel has scrollHeight > clientHeight and the last row can be scrolled into view; no table cell has scrollWidth > clientWidth; no horizontal page overflow; the headline fits; the countdown matches the format; there are no console errors. Also test that a mobile width of 390 px shows a stacked layout with a scrollable log.
- Unit tests: formatter, row rendering for each signal type, filters, paging, the CSV export (no keys, correct columns), scoring audit (hand-made cases for Up/Down correct and wrong, and an off-by-one trap).
- Re-run earlier integrity checks (truncation test, ledger audit, live-compat), the replay equivalence test and the model hash check. Confirm the sha256 of models/model_v2.joblib, models/calib_v2.joblib and models/calib_live.json match models/MODEL_SHA256.json.

F. DOCS
Update docs/user_guide.md (how to read the log, the source tags, the filters), docs/live_runbook.md, README.md, report/report.md and slides/outline.md. Record every decision in instructions/progress.md.

Final printed summary: root cause of the clipping; screenshots or measurements per viewport after the fix; the scoring audit result (bug or no bug, the per-side counts, hits, CI and p-value); the formatter test results; the model hash status; any remaining issue.
