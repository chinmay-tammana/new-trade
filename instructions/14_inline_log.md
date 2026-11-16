V3.3-FIX: SHOW THE SIGNAL LOG ON THE PAGE. Continue in this project folder. All rules in AGENTS.md and instructions/ still apply (no placeholders, seed 42, never edit instructions/ except progress.md) with ONE override: IGNORE every GitHub and git rule; do not run any git commands. Save this prompt verbatim as instructions/14_inline_log.md and update AGENTS.md to read it last.

HARD CONSTRAINTS
- model-v2 stays FROZEN (no retraining, no changes to features, q_lo, q_hi, calibrator, hyperparameters or model files). Never UPDATE or DELETE existing rows of the signals table. Do not change live_signals_enabled or the parity-gate verdict. Never put keys or environment values in any API response or static file.
- The previous round reported tests as passing while the real page was broken. This time every claim must be proven with a REAL BROWSER RENDER (see section F). If a browser cannot be installed, say so plainly and stop; do not claim success.

PROBLEM, from my screenshot at localhost:8000 (Saturday, market closed)
1. The signal log shows "(0 rows)" and "Loading signals..." and never fills. The table is also squeezed into the narrow right column. I want all signals visible on the website itself, without downloading anything.
2. The header clocks show "ET: --:--:-- | IST: --:--:--", return bands show "--", volatility and regime show "--", and the countdown still shows 3321:14: the page script is probably crashing early. Find the first console error.
3. The new controls (Historical Replay, Speed, Labels) are unstyled raw HTML and overlap the chart title; the styling regressed.
4. The right column is scrolled down so the headline (UP / DOWN / NO SIGNAL) is not visible on load.
5. The "Labels: OFF" toggle does not hide the marker text labels, which still pile up on the chart.
6. The scorecard shows "All-Candle: 20.8%", which is hits divided by ALL candles including abstentions (5/24); that is the wrong metric.
7. The earlier screenshot showed a NO SIGNAL headline with a STRONG tag; a No-signal bar must never be STRONG.

A. ROOT CAUSES FIRST
Start the server, open the page in a headless browser, capture ALL console errors and failed network requests, and curl /api/signals, /api/status, /api/latest. Write each root cause and fix into progress.md. Common suspects: a JS exception from the new filter or paging code, a changed API response shape, a changed query parameter, an element id mismatch, a CSS file path or class rename, the default filter ("Out-of-sample + Live") returning 0 rows because of a wrong source value.

B. SIGNAL LOG INLINE (main fix)
1. Remove the "Download CSV" button and every download link from the page. No file download is needed anywhere in the UI.
2. Move the log out of the sidebar. Make it a full-width panel directly BELOW the chart, always visible without scrolling the page at 1440x900, about 10 rows high, with its own scrollbar and a sticky header. The right column keeps only the headline panel and the summary cards.
3. Columns: Time (ET) | Time (IST) | Signal (arrow + word + confidence, e.g. "▼ Down 61%") | Tier (STRONG or blank) | Price move (close at signal bar -> close at next bar, and % change) | Outcome (✔ Correct, ✘ Wrong, ○ Pending) | Source (OUT-OF-SAMPLE, LIVE, IN-SAMPLE or REPLAY pill with tooltip).
4. No-signal rows: show the REASON (end of day, data gap, market closed, inside the no-signal band). No confidence and never a STRONG tag. Default view hides No-signal rows but a toggle shows them.
5. Newest first, grouped by trading day with a day header showing counts of Up, Down, No signal, hits, misses, pending. Filters: Out-of-sample + Live (default) | Live only | All including in-sample. If a filter returns 0 rows, show a clear message with the reason (for example "No out-of-sample signals match; 24 exist under 'All'") instead of an endless "Loading".
6. Loading and error states: while loading show a skeleton; on failure show the error text and a Retry button. Never leave "Loading signals..." forever (timeout after 8 s).
7. Clicking a row scrolls the chart to that candle and highlights it; hovering a candle highlights its row. Paging: load the newest 100 rows and add "Load more" at the bottom.
8. Readable at 1280x720, 1440x900, 1920x1080 and 390 px wide (stacked layout). No clipped text; wrap instead.

C. OTHER FIXES
1. Styling: restore the stylesheet look for all controls. Put Replay controls (date picker, Replay Day, speed) into one tidy toolbar above the chart that does not overlap the title; style buttons and selects consistently with the dark theme.
2. Clocks: ET and IST tick every second from the server time offset. Return bands, volatility and regime show the latest stored values, or "n/a (market closed, last signal <time>)", not bare "--".
3. Countdown: ONE shared formatter used everywhere (header, panel, tests): more than 24 h -> "2d 7h 12m"; 1 to 24 h -> "7h 12m"; under 1 h -> "mm:ss". Remove any duplicate formatting code. Test it through the rendered DOM, not only as a unit function.
4. On load, scroll the right column to the top. The headline panel (UP / DOWN / NO SIGNAL, confidence, covers-hour, STRONG tag on its own line, never clipped) is always visible. No confidence on No signal.
5. Labels toggle: OFF truly hides all marker text (arrows only, STRONG slightly larger); ON shows text only for the last 6 bars or on hover. Default OFF.
6. Scorecard: show "Signal accuracy = hits / directional signals" with n and the Wilson 95% CI, per side (Up x/y, Down x/y), "Coverage = directional / all candles (8 of 24)", and the label "OUT-OF-SAMPLE (descriptive only, small n)". Remove "All-Candle: 20.8%" or rename it as "Hits per candle incl. abstentions (not accuracy)" in small grey text.
7. STRONG audit: STRONG must never appear on a No-signal row. Report what fraction of out-of-sample directional signals are STRONG. If nearly all are STRONG the tier is not discriminating: say so in docs/live_expectations.md; do NOT change the frozen threshold.
8. Abstention audit: in the out-of-sample set 16 of 24 candles show No signal, while the V2 holdout had signals on about 89% of candles. Write results/tables/abstention_breakdown.csv with a count per reason (end of day, data gap per symbol, inside the no-signal band, other) and put a short explanation in progress.md. Do not change any thresholds.

D. DATA SANITY (display only)
Keep the existing handling of the suspicious Nov 5 candle, but make sure the log and chart read from the same single API source so the numbers always agree.

E. DOCS
Update docs/user_guide.md (how to read the log columns and source tags), docs/live_runbook.md, README.md, report/report.md and slides/outline.md. Record every decision in instructions/progress.md.

F. PROOF BY REAL BROWSER RENDER
- Install playwright (or another headless browser) and write tests/ui_render_check.py. For viewports 1280x720, 1440x900, 1920x1080 and 390x844 it must: load localhost:8000; assert zero console errors and zero failed requests; assert the log table has at least one visible row (or the clear empty-state message with a reason) within 8 s; assert no "Loading signals..." text remains; assert the clocks match HH:MM:SS; assert the countdown matches one of the three formats; assert the headline is fully visible (scrollWidth <= clientWidth) and inside the viewport on load; assert there is no Download button or link; assert Labels OFF hides the marker text; assert no horizontal page overflow; assert the log panel is visible without scrolling the page at 1440x900.
- Save a screenshot per viewport to results/figures/ui_check_<width>x<height>.png. Open the saved PNGs yourself, describe what you see, and fix anything where the log rows are not visible.
- Re-run the earlier checks: replay equivalence, truncation test, ledger audit, live-compat, and the model hash check (sha256 of models/model_v2.joblib, models/calib_v2.joblib and models/calib_live.json against models/MODEL_SHA256.json).

Final printed summary: the root cause of the empty log and the crash; the console and network errors before and after; a table of the rendered checks per viewport (pass or fail); the number of log rows shown per filter; the scorecard numbers (per-side hits, CI, coverage); the STRONG fraction; the abstention breakdown; the model hash status; any remaining issue.
