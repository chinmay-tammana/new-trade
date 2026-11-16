V3.4-DIAG: ONE-SIDED SIGNALS (Nov 10-13). Continue in this project folder. All rules in AGENTS.md and instructions/ still apply (no placeholders, seed 42, never edit instructions/ except progress.md) with ONE override: IGNORE every GitHub and git rule; do not run any git commands. Save this prompt verbatim as instructions/15_balance_diag.md and update AGENTS.md to read it last.

HARD CONSTRAINTS
- model-v2 stays FROZEN: no retraining, no changes to features, q_lo, q_hi, the calibrator, hyperparameters or model files. Never UPDATE or DELETE existing rows of the signals table. Do not change live_signals_enabled or the parity-gate verdict. Never put keys or environment values in any API response or static file.
- FORBIDDEN: any attempt to make signals look balanced, such as an artificial Up quota, shifted thresholds, a recalibration to the recent window, or mirroring Down into Up. If a recalibration seems warranted, document it as an option for me and wait for my explicit instruction.
- Every number in the UI, docs and report must come from a computed table in results/; no free-text explanations without a measured number behind them.

PROBLEM
In the recent out-of-sample sessions (Nov 10-13, 24 labelled candles) the frozen model produced 8 Down, 0 Up and 16 No signal. The V2 holdout had both sides. The agent's earlier explanations ("post-election pullback", "coverage deliberately tuned to about 30%") were not backed by data, and the coverage figures contradict each other: about 30% pooled OOF, 88.65% on the V2 holdout, about 33% on the recent days. Find out WHY, and fix only what is truly broken.

STEP 0: PRE-REGISTER (write docs/balance_diag_protocol.md BEFORE computing anything below)
Decision rules, never changed afterwards:
1. If section A7 finds a code bug (the live engine's features differ from offline make_features on the same bars by more than 1e-6, an inverted sign, wrong class-weight use, or a calibration direction error): fix ONLY the live and inference code path, add a regression test, and leave all model files untouched.
2. Else if the SIP-feature signals for the same bars are mixed (both sides) while the IEX ones are one-sided, or swapping one feature group from IEX to SIP values restores both sides: classify as FEED BIAS. No retraining. The UI shows an "IEX feed bias" warning. I will decide later between retraining on IEX data (a separate model) and using delayed SIP for a demo.
3. Else if one-sided runs of 8 directional signals occur in at least 5% of windows of 8 consecutive directional signals in the historical OOF and holdout predictions, or |Spearman rho| >= 0.30 between p_cal and trailing ret_5d or ret_20d: classify as TREND-FOLLOWING BEHAVIOUR, NOT A BUG. No change.
4. Otherwise: "INCONCLUSIVE, small sample". No change.

A. MEASURE (save CSVs to results/tables/balance_diag_*.csv and figures to results/figures/, each figure with caption and interpretation)
1. For each of the 24 labelled bars of Nov 10-13 and for the last 60 sessions, with the frozen final model, compute the signal TWICE: with IEX-built features and with SIP-built features (SIP minutes for these days are older than 15 minutes and available). Columns: bar time, source tag (IN-SAMPLE or OUT-OF-SAMPLE), raw P(Up), p_cal, side, close, next close, return, outcome.
2. Read the actual frozen q_lo and q_hi from the calibration file. Print the min and max p_cal in the Nov 10-13 window, how far the highest p_cal is below q_hi, the number of bars within 0.02 of q_hi and within 0.02 of q_lo, and side counts and agreement for IEX versus SIP.
3. Market facts: per-day open-to-close % move of GS for Nov 10-13 and the number of up bars and down bars among the 24 labelled candles, so any word like "pullback" has numbers behind it.
4. p_cal distribution comparison (mean, std, 5/25/50/75/95th percentiles, and the coverage at (q_lo, q_hi)): pooled pre-holdout OOF; the V2 holdout predictions from the pre-refit model; the final model over the last 60 sessions with IEX and with SIP; the 24 recent bars. Explain the 30% versus 88.65% versus 33% coverage contradiction using these numbers (for example a calibrator or distribution shift between OOF and the holdout).
5. Drivers: for the 24 bars compute XGBoost native contributions (pred_contribs=True). List the 5 features pushing most toward Down on average, with their values as percentiles of the Base training distribution. Over the last 250 sessions (SIP features) compute Spearman correlation of p_cal with ret_5d, ret_20d and ma_dist_50h.
6. Feed mismatch: for the 24 bars, per-feature IEX minus SIP difference in units of Base training std (top 10). Recompute p_cal using IEX features but replacing ONE feature group at a time (volume, intrabar, peers/VIXY, others) with SIP values, and report which group moves p_cal most. Diagnostic only; nothing is saved into the model.
7. Code check: confirm that the live engine's features for the same bars equal offline make_features on SIP data within 1e-6 (live-compat), that the calibrated probability is monotonic increasing in raw P(Up), and that class-balanced weights were applied as documented.
8. One-sidedness evidence: the binomial probability of 8 of 8 on one side given the pooled OOF Up share, AND the empirical frequency of one-sided windows of 8 consecutive directional signals in the OOF and holdout predictions (the latter is the one used by decision rule 3).

B. DECIDE
Apply the pre-registered rules in order. Write the verdict (BUG FIXED, FEED BIAS, TREND-FOLLOWING NOT A BUG, or INCONCLUSIVE) with the supporting numbers into progress.md and docs/balance_diag_findings.md. List the options for me with pros and cons (retrain on IEX data, use delayed SIP for demos, recalibrate on recent data with its risks, do nothing) without doing any of them.

C. UI (explain, never force)
1. Every log row and the headline panel show the probability behind the call, for example "P(Up) 47.8%: needs >= 56.3% for Up, <= 43.4% for Down" (use the real frozen thresholds from the calibration file, not hard-coded numbers).
2. Add a "Signal balance" card: counts of Up, Down and No signal for the last 1, 4 and 20 sessions (out-of-sample and live only). If at least 6 consecutive directional signals are on the same side, show: "All recent signals are on one side. This can reflect a real trend, the data feed or the model's trend-following; see Diagnostics."
3. Add a Diagnostics section on the same page (no downloads) showing from the CSVs: a p_cal chart with horizontal lines at q_lo and q_hi, IEX versus SIP sides, the market facts, and the verdict text.
4. Remove unsupported narrative claims from the dashboard, docs and report ("post-election pullback", "coverage deliberately tuned to about 30%") unless section A supports them; every explanation must cite a measured number.
5. Fix overstated wording everywhere: the pre-registered criterion (b) required signal accuracy >= 53% together with permutation p < 0.10. If the verified V2 holdout accuracy is below 53%, state that criterion (b) is NOT met, and remove phrases like "confirming the edge is statistically detectable". Report the permutation p-value as a number only, with the sentence "below the pre-registered accuracy bar of 53%".

D. PROOF
- Re-run tests/ui_render_check.py in a real headless browser at 1280x720, 1440x900, 1920x1080 and 390x844: zero console errors, the log rows visible, the new cards visible, no clipped text; save screenshots to results/figures/. Open the saved PNGs, describe what you see, and fix anything broken.
- Re-run the replay equivalence test and the model hash check (sha256 of models/model_v2.joblib, models/calib_v2.joblib and models/calib_live.json against models/MODEL_SHA256.json).

Final printed summary: the verdict and its evidence; side counts and agreement for IEX versus SIP; the p_cal range in the window against q_lo and q_hi; the explanation of the 30% / 88.65% / 33% coverage figures; the top drivers; the historical frequency of one-sided runs; what changed (code or UI) and what is untouched; my options with pros and cons.
