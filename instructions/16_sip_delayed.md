V3.5-SIPLIVE: OPTION B (DELAYED SIP FOR SIGNALS). Continue in this project folder. All rules in AGENTS.md and instructions/ still apply (no placeholders, seed 42, never edit instructions/ except progress.md) with ONE override: IGNORE every GitHub and git rule; do not run any git commands. Save this prompt verbatim as instructions/16_sip_delayed.md and update AGENTS.md to read it last.

HARD CONSTRAINTS
- model-v2 stays FROZEN (no retraining, no changes to features, q_lo, q_hi, calibrator, hyperparameters or model files). Never UPDATE or DELETE existing rows of the signals table; schema changes must be additive only (new columns or new tables), and the hash-chain verification must keep passing for old rows in their original canonical form (add a schema_v field; new rows include the new fields). Never put keys or environment values in any API response or static file.
- Forbidden: threshold changes, recalibration, forcing any side.
- Decision record: write in progress.md "Owner chose Option B: signals computed from delayed SIP, IEX for display only", with the reason (IEX feature distribution shift: 31.08% missing GS trade minutes, up_minute_ratio about 3.46 base std below SIP).

A. ENGINE SWITCH
1. The live signal engine uses feed=sip (delayed) for GS and all 7 context symbols, with the SAME aggregation, make_features and src/insights.py. Fetch the minute bars for a closed hour at bar_end + 16 min (the query end must be at least alpaca_end_lag_minutes old; retry every 20 s for up to 5 min; then record a data gap and proceed with what exists).
2. IEX is used ONLY for the forming candle and the price ticker, labelled "IEX live (display only)". It never feeds the model.
3. New columns on the signals table (additive): feed ("sip_delayed"), delivered_at, delivery_delay_s, schema_v. Existing rows keep feed="iex" semantics via a view or a default of "iex" for old rows; do not update them.
4. Stored IEX-derived signals (including the Nov 10-13 backfill) stay in the database untouched but the UI shows them with a pill "IEX (feature-shifted, not scored)" and every scoreboard excludes them.
5. Recompute the frozen model's signals for the 4 recent sessions (Nov 10-13) and the last 10 completed sessions from SIP minutes, and INSERT them as new rows (feed="sip_delayed", late=true, sample IN-SAMPLE or OUT-OF-SAMPLE as before). Verify the Nov 10-13 result against the diagnostic (expected about 2 Up, 4 Down, 18 No signal) and print any difference.

B. EXPLAIN THE 7 VERSUS 8
Rebuild the IEX 1h bars for Nov 10-13 now and compare them with the stored IEX bars and with the diagnostic's IEX computation (bar by bar). Fetch SIP minutes twice, a few minutes apart, and compare. Report whether IEX or SIP bars change between fetches (late prints), and which bar caused the 7 versus 8 Down difference. Print the feature-group swap-test table (IEX features with one group replaced by SIP values: volume, intrabar, peers/VIXY, others) with the resulting p_cal and sides for the 24 bars.

C. GATES AND STATUS
- Do not edit the old IEX gate results. Write docs/live_parity_gates_sip.md BEFORE testing: (a) the live engine's features from SIP minutes equal offline make_features on the same minutes within 1e-6 for the last 20 sessions; (b) SIP access works for all 8 symbols with the query end at least 15 minutes old; (c) replay equivalence with src/predict_day.py (direction equal, confidence within 1e-9) on at least 3 days including one half-day.
- live_signals_enabled becomes true ONLY if all three pass; otherwise stay false and report which one failed. The SIGNALS DISABLED banner must explain the real reason.
- Update python3 -m src.live.doctor: SIP access test per symbol, the delay arithmetic, the DB hash chain with schema_v, the model hash, and the next signal delivery time in ET and IST.

D. UI
1. Badges: "SIGNALS: SIP (15-min delayed)" and "PRICE: IEX live (display only)", MODEL hash, LIVE/STALE/MARKET CLOSED, SIGNALS DISABLED if the gates fail.
2. The headline says: "Signal for the hour starting 10:30 ET. Bar closed 10:30, delivered 10:46 ET. This hour is already 16 min old." Every log row shows bar closed time and delivered time (ET and IST). The countdown is to the next signal DELIVERY (bar_end + 16 min). Market closed: "Next signal: Monday 10:46 ET (21:16 IST)". Keep the single shared countdown formatter.
3. Fix the statistical notice. Read numbers from files and say: "The 52.25% accuracy [49.65, 54.84] and permutation p = 0.0417 were measured on the pre-refit model with thresholds 0.490 / 0.508 (88.65% coverage). The deployed rule uses thresholds 0.434 / 0.563 (about 30% target coverage) and has no held-out accuracy measurement." Remove "High signal coverage (~89%)" for the deployed rule. Never present the 52.25% as the accuracy of the signals shown.
4. Diagnostics section: add the FEED BIAS verdict, the swap-test table, the 7-versus-8 explanation, and the line "IEX signals are shown for transparency only."
5. Scoreboard: SIP-delayed rows only, OUT-OF-SAMPLE and LIVE only, label "descriptive only, small n", with per-side hits, Wilson CI and coverage. State plainly that the 5 of 8 earlier result came from IEX signals and is not scored.

E. DOCS
Update docs/user_guide.md (the delivery delay and how to read "delivered at"), docs/live_runbook.md, docs/live_expectations.md (a one-day test gives about 6 signals delivered at 10:46, 11:46, 12:46, 13:46, 14:46 and 15:46 ET; the last one covers only 14 remaining minutes; the end-of-day bar has no signal; tell the TA about the 15-minute delay), README.md, report/report.md (the V3 IEX decision, the failed gates, the feed-bias finding, the switch to delayed SIP, and the fact that the deployed rule has no held-out accuracy) and slides/outline.md. Add a Monday checklist: run the doctor, start run_live.sh before 09:00 ET (19:30 IST), the first signal arrives at 10:46 ET (21:16 IST).

F. PROOF
- Real headless-browser test (tests/ui_render_check.py) at 1280x720, 1440x900, 1920x1080 and 390x844: zero console errors, log rows visible, the new badges and headline text correct, no clipped text, countdown format valid. Save screenshots, open them yourself and describe them.
- Re-run the replay equivalence test with the SIP-delayed engine, the truncation test, the ledger audit and the model hash check (sha256 of models/model_v2.joblib, models/calib_v2.joblib and models/calib_live.json against models/MODEL_SHA256.json).

Final printed summary: the engine switch result; the Nov 10-13 SIP signal counts; the explanation of the 7 versus 8; the swap-test table; the three SIP gate results and live_signals_enabled; the doctor output; UI check results per viewport; model hash status; what I must do on Monday.
