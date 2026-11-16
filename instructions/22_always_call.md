V4.2-ALWAYSCALL: A DIRECTIONAL CALL ON EVERY INTERVAL (5-MINUTE SYSTEM). Continue in this project folder. All rules in AGENTS.md and instructions/ still apply (no placeholders, seed 42, never edit instructions/ except progress.md) with ONE override: IGNORE every GitHub and git rule; do not run any git commands. Save this prompt verbatim as instructions/22_always_call.md and update AGENTS.md to read it last. Record overrides in progress.md.

OWNER DECISION (record in progress.md, dated 2026-11-16)
"There is still too much No signal. Keep No signal minimal, or at least reduce it a lot." Interpretation: every eligible 5-minute bar gets an Up or Down call with a confidence tier. No signal remains ONLY for: the end-of-day bar (no forward interval in the same session), a data gap (a required symbol has no minute data), market closed, or an exact p_cal tie at 0.5. This is a SIGNAL-RULE change, NOT a model change.
Observed before the change, record verbatim: OOF accuracy 62.2% at 30% coverage; holdout 59.82% over 1,827 signals with 19.3% coverage; and the inconsistent figures "No-Signal 63,041 bars (70.0%)" with 74,971 bars (that is 84.1%).

HARD CONSTRAINTS
- Do NOT retrain or edit models/m5/model_m5.joblib or its calibrator. Do not touch the 1-hour system or data/live/live.db. No quotas, no forcing a side. The research holdout has already been inspected, so every number derived from it is descriptive, not a clean test; say so wherever it is shown.
- Never UPDATE or DELETE stored signal rows in data/live/live_m5.db. Schema changes are additive only. Old rows get rule_version 1 or 2 via defaults.
- All thresholds and tier cutoffs live in models/m5/rule_m5_v3.json with the active pointer models/m5/active_rule.json. The engine, replay, src/predict_day.py, UI and docs read them from there only. Remove every hard-coded threshold and print the grep result. Add the rule file hashes to models/m5/MODEL_SHA256_m5.json under "rule"; keep the model entries byte-identical.
- Never print keys; never put keys or environment values in any API response or static file.

A. AUDIT STATUS FIRST
Read docs/m5/audit_findings.md and results/m5/ from the V4.1 audit. If it does not exist, or any of its five parts was not run, run all of them now with the gates pre-registered in docs/m5/audit_plan.md (shuffled-label test, time-shift test, single-feature AUC, IEX print artifact test against the cached SIP 1-minute data, stability by year and time of day). Print a table: gate | result | number. If shuffled-label, time-shift or single-feature FAIL, STOP and report the leak. If the IEX print artifact is flagged, continue but show a red warning card on the page and put it at the TOP of the final summary.

B. RECONCILE THE NUMBERS (results/m5/coverage_reconciliation.csv and docs/m5/coverage_reconciliation.md)
One table for pooled OOF and for the holdout, from one denominator: total bars, labelled bars, directional, and No signal split by reason (inside the band, margin guard, end of day, data gap), each as a percentage of the labelled bars. Explain the 74,971 versus 39,778 bar counts, why "63,041 = 70.0%" is wrong, and why the holdout coverage was 19.3% against a 30% target (fixed OOF quantile thresholds versus the trailing-quantile rule versus the margin guard). Correct every wrong figure in the page, docs, report and slides; log each correction.

C. ALWAYS-CALL RULE (docs/m5/always_call_protocol.md BEFORE computing; never change it)
1. Rule: Up if p_cal > 0.5; Down if p_cal < 0.5; an exact tie is No signal. Confidence = max(p_cal, 1 - p_cal).
2. Tiers from the pooled pre-holdout OOF confidence distribution, stored in the rule file: STRONG = confidence >= the 75th percentile; WEAK = confidence <= the 25th percentile ("close to a coin flip"); NORMAL = in between.
3. Descriptive tables (OOF and holdout, labelled): all-bar accuracy against the larger base rate; balanced accuracy; ROC-AUC; per side and per tier accuracy with n and Wilson 95% CI; the permutation p-value for all-bar accuracy; net return per call for ALL calls and for STRONG-only at 0, 1, 2 and 5 basis points with block-bootstrap CI; the number of calls per day; the frequency of one-sided runs of 8 consecutive calls.
4. Pre-registered checks, reported as pass or fail: (a) all-bar holdout accuracy above the larger base rate with permutation p < 0.05, otherwise state plainly that no skill is shown at full coverage; (b) tier accuracy increases from WEAK to NORMAL to STRONG on both OOF and the holdout (if not, say the confidence is not informative); (c) ECE <= 0.03; (d) both sides present.
5. Re-derive the sides of ALL stored 5-minute signals of the last 10 completed sessions from stored p_cal (recompute from raw_minutes if p_cal is missing) and INSERT new rows with rule_version = 3 (late = true, same sample tags). Hide older-rule rows behind a toggle "Show older rules"; never inside counts. One row per (bar, horizon). Extend python3 -m src.m5.apply_rule with --mode always.
6. Print before-and-after counts (Up, Down, No signal) per day for the last 5 sessions.

D. UI
- Headline shows UP or DOWN for the current interval with the confidence and the tier chip; WEAK shows "close to a coin flip". NO SIGNAL appears only for the reasons listed in the owner decision, with the reason.
- Day view: the timeline strip has one chip per bar (arrow shape, tier shading and result mark); the table shows every bar by default, with filter chips All | NORMAL + STRONG | STRONG only; one compact line per bar (at most 44 px); the day summary shows accuracy for ALL calls and for STRONG-only side by side with n and Wilson CI, plus "demonstration, not proof (n = ...)". Never show a single headline accuracy number.
- Add the tier table and the all-bar versus STRONG-only comparison cards, always with n, CI and the sample tag ("holdout already inspected; descriptive"). If the audit raised a warning, keep the red warning card visible.
- Remove text that says the model "abstains to filter noise" or that presents the old 62% or 69.56% as the accuracy of what is on screen.

E. DOCS AND PROOF
- Update docs/m5/ (user guide: how to read tiers, why WEAK calls are weak; live expectations: about 77 calls per full day and the CI width for that n; runbook with the apply_rule command), README.md, report/report.md (the audit, the reconciliation, the always-call decision, the holdout caveat) and slides/outline.md. Record every decision in instructions/progress.md.
- Unit tests: tie handling; the tier boundaries; No signal only for the allowed reasons; one row per (bar, horizon); the summary equals the rows; no hard-coded thresholds (grep); the rule hash check; apply_rule is non-destructive.
- Real headless-browser test (tests/ui_render_check_m5.py) at 1280x720, 1440x900, 1920x1080 and 390x844: zero console errors, a full day of rows visible with a call on almost every bar, tier chips visible, no clipped text. Save screenshots to results/m5/figures/, open them yourself and describe them.
- Re-run the replay equivalence test (replay equals src/predict_day.py, same direction, confidence within 1e-9) on at least 3 days including one half-day, the truncation test, the ledger audit and the model hash check.

Final printed summary: the audit gate table (artifact warning first); the reconciliation table; the always-call results (all-bar accuracy against base rate and the permutation p-value, per-tier and per-side accuracy with CI, the monotonic check); the number of calls per day; before-and-after per-day counts for the last 5 sessions; replay, UI and hash check results; what I must decide.
