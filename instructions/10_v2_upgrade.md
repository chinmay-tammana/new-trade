V2 UPGRADE. Continue in this repository (https://github.com/chinmay-tammana/new-trade.git). All rules in AGENTS.md and instructions/ still apply (GitHub rules, secret check, no placeholders, seed 42, never edit instructions/ except progress.md) unless this prompt overrides them. Where this prompt conflicts with instructions/, this prompt wins; record every override in progress.md.

STEP 0
- Save this prompt verbatim as instructions/10_v2_upgrade.md. Add V2 phases to progress.md. Update AGENTS.md: "read instructions/10_v2_upgrade.md last; it overrides earlier files".
- Tag the current commit v1-negative-result and push the tag. Copy (do not delete) the v1 outputs (results/, models/, report drafts, RESULTS.md, split.json) to archive/v1_four_year_run/.
- Write into progress.md: "v1 result: Val 119 Up / 0 Down, 61.34% (73/119); Test 188 Up / 0 Down / 708 No signal, 46.28% (87/188), Wilson 95% CI [39.30%, 53.41%], always-Up on all test candles 48.88%, net return -7.24 bps per signal at 5 bps. Causes: (a) the Val window (Nov 2024 to Mar 2025) was a rally, so the Val search switched the Down side off and v1 allowed it; (b) the rule was tuned on 602 candles in one regime (selection optimism); (c) 'always-Up on the same candles' is identical to the accuracy of an Up-only rule, so its +0.00% excess was meaningless; (d) data stopped at 2025-10-15."

V2.1 DATA
1. Print the system date/time. Find out why the v1 data ends on 2025-10-15 (hard-coded end, stale cache or provider refusal) and fix it. Re-run the Alpaca availability test on recent dates. If the provider refuses SIP data older than 15 minutes, STOP and report the exact error.
2. history_years = 8 (about Nov 2018 onward; Alpaca history starts 2016), history_buffer_days 30. If earlier years fail, use as many as available and log it. Extend the cached monthly 1-minute chunks incrementally (resumable; never overwrite).
3. Context symbols: SPY, XLF, JPM, MS, BAC, TLT, VIXY (VIX proxy ETF, which has roll decay; note it). Same session-anchored 1h aggregation. Inner-join; report rows lost per symbol; drop a context symbol that loses more than 2% of GS bars.
4. Drop any 1h bar not fully closed at download time. Freeze: save data_end (last complete GS bar) to configs/data_freeze.json. Every fold and holdout boundary derives from it. Later runs reuse it unless --refresh is passed.
5. freeze_date = the last trading day BEFORE the TA's test day. TA test day: NOT SPECIFIED (default: freeze_date = most recent completed trading day). Store "test_day" and "freeze_date" in configs/config.py.
6. Re-run the audit and update the dataset card and data dictionary for the new span.

V2.2 FEATURES (add to the existing set; stationary, causal, computable live from 1-minute bars; the truncation test must pass for every new feature)
- Peers, rates, vol: ret_1 and ret_3 for JPM, MS, BAC, TLT, VIXY; gs_minus_peers = ret_1 minus mean peer ret_1; xlf-relative return.
- Intrabar (from the 1-minute bars inside each hour): realized_vol_1m (std of 1-min returns), up_minute_ratio, clv = ((close-low)-(high-close))/(high-low), signed_volume_ratio = (volume of up-minutes minus volume of down-minutes)/total volume, last15_ret.
- Levels and trend: dist_prev_close, dist_prev_high, dist_prev_low (in ATR units), dist_session_vwap, ret_1d, ret_5d, ret_20d (trading-day returns up to the previous close), ma_dist_50h and ma_dist_200h, efficiency_ratio_20 = |net move| / sum of |moves| over 20 bars, bars_since_open, bars_to_close.
- Keep a FEATURE_GROUPS dict (returns, momentum, vol, volume, time, market, peers, intrabar, levels) so ablations can drop whole groups.
- Document in docs/ that the intrabar and volume features need live 1-minute bars and that IEX-only live volume may differ from SIP.

V2.3 TARGETS AND HEADS
Direction heads (same labelling rules as v1: last bar of the day and overnight moves excluded; train on decisive rows only; evaluate on all labelled rows):
- H1: next 1h bar (as v1).
- H3: close of bar t+3 versus close of bar t, only if t+3 is on the same date.
- HD: session close versus close of bar t (rest of day), only for bars with at least 2 bars left.
- Decisive thresholds scale with sqrt(bars ahead).
Insight heads:
- VOL: XGB regressor on log((high-low)/close) of the next bar. Output High / Normal / Low by OOF terciles. Evaluate Spearman, MAE and R2 against persistence (previous bar's range) and an EWMA baseline.
- BAND: quantile regression (q=0.10 and 0.90) for the next-bar return. Evaluate coverage (target about 80%) and width against a constant-width band from the training std.
- REGIME: descriptive tag Uptrend / Downtrend / Chop from the sign of ret_20d and efficiency_ratio_20, with thresholds fixed from OOF data. Report direction-signal accuracy inside each regime.
Pre-registered choice of the PRIMARY direction horizon (fixed now): the horizon among H1, H3, HD with the highest Wilson 95% lower bound of pooled OOF signal accuracy, subject to each side having >= 150 signals and coverage >= 20%. Report all three.

V2.4 EVALUATION PROTOCOL (replaces the single Val block; the old Base/FT/Val/Test blocks are retired)
- RESEARCH HOLDOUT = bars from 2025-10-16 (the first day after v1's data) through freeze_date, leaving an 8-bar gap after the last pre-holdout bar. It never overlaps v1's test period. It is excluded from hyperparameter search, calibration, rule choice, feature-group choice, model choice and horizon choice. It is read ONCE.
- Walk-forward OOF on everything before the holdout: initial training = the oldest 2 years; then refit every quarter (63 trading days) on all earlier data (8-bar gap) and predict the next quarter. Pool all OOF predictions. Folds used for selection stop before the holdout; the folds that cover the holdout are run only in the final evaluation.
- Hyperparameter search (n_iter 30) uses only OOF folds before the holdout. Recency weighting: half-life chosen from {90, 180, 365 days, none}. Also compare a warm-start fine-tune variant (continue boosting on the latest 12 months) by pooled OOF log-loss.
- Candidate models: shallow XGBoost, HistGradientBoosting, elastic-net logistic regression, and the mean of their calibrated probabilities (ensemble). Choose by pooled OOF log-loss; ties go to the simpler model.
- Group ablations: drop one FEATURE_GROUP at a time, selected on pooled OOF log-loss. Keep a group only if dropping it does not improve log-loss.
- Calibration: Platt scaling on pooled OOF predictions from before the holdout, fitted with CLASS-BALANCED sample weights so that 0.5 means no directional lean. Reliability diagram and ECE overall and per regime.
- State in the report that OOF data is used for both selection and calibration (a small optimistic bias) and that the holdout adjudicates.

V2.5 SIGNAL RULE (BOTH SIDES MANDATORY)
- Up if p_cal >= q_hi; Down if p_cal <= q_lo. q_lo and q_hi are the (c/2) and (1 - c/2) quantiles of pooled OOF p_cal, where c is the total coverage swept over {0.15, 0.20, 0.25, 0.30, 0.40}. Choose c by the Wilson 95% lower bound of pooled OOF signal accuracy, subject to >= 150 signals on EACH side. Store q_lo and q_hi as frozen numbers so live signals never change.
- A side may be switched off ONLY if its pooled OOF accuracy has a 95% upper bound below 50% over >= 300 signals. Then state it loudly in docs, user guide and report.
- Code assertions: n_up >= 150 and n_down >= 150 on OOF (unless that exception applies). On the holdout, assert both counts > 0 and fail the run with a clear message otherwise. v1's zero-Down failure must not recur.
- Displayed confidence = p_cal for Up and 1 - p_cal for Down (always >= 50%). Report per-side counts, per-side accuracy and the confusion matrix.

V2.6 HONEST STATISTICS
- Replace "always-Up on the same candles" with: (i) always-Up and always-Down accuracy on ALL candles of the same period; (ii) per-side drift-adjusted accuracy (Up accuracy minus base Up-rate; Down accuracy minus base Down-rate); (iii) balanced accuracy; (iv) a permutation test (shuffle signals keeping the same Up/Down counts, 10,000 times) with the p-value of the observed accuracy; (v) accuracy by regime (bull / bear / chop by the sign of the trailing 60-trading-day SPY return) and per calendar year; (vi) net return per signal at 2, 5 and 10 bps with block-bootstrap CI.
- Write docs/problem_definition_v2.md BEFORE reading the holdout, with these pre-registered criteria (never change them): (a) pooled OOF log-loss below the constant-prior baseline; (b) holdout: >= 100 signals per side, and signal accuracy >= 53% with permutation p < 0.10, OR a plain statement that no skill was shown; (c) each side above its own base rate on the holdout, or the limitation stated; (d) ECE <= 0.05; (e) all integrity checks pass. Report failures plainly.

V2.7 FINAL MODEL, DAY REPLAY, INSIGHT OUTPUTS
- After the holdout has been read once and the verdict written, REFIT the chosen configuration on ALL data up to freeze_date (including the holdout), with the frozen hyperparameters, recency half-life and signal-rule procedure. Recompute the calibrator and q_lo/q_hi from pooled OOF predictions including the holdout's OOF predictions. Save models/model_v2.joblib and models/calib_v2.joblib. State in the report that the holdout numbers come from the pre-refit model, and that the TA's test day is the first truly untouched data.
- src/predict_day.py --date YYYY-MM-DD: replay a trading day bar by bar using ONLY data available at each bar close and a model trained on data strictly before that date (for dates after freeze_date use the frozen final model). Output results/tables/day_{date}.csv (timestamp, direction, confidence, volatility class, band low/high, regime, outcome) and results/figures/day_{date}.png (price with signal markers and band). Assert that a signal on a closed bar is never changed afterwards.
- src/insights.py: a function returning per-bar {direction, confidence, horizon, vol_class, band_low, band_high, regime, historical_hit_rate_for_this_hour_and_side}, reusable by a future live backend.

V2.8 CHECKS, DOCS, GITHUB
- Rerun all integrity checks: truncation test for every new feature, shuffled-label test on OOF, time-shift test, ledger audit (no holdout row in any fit before the single read), live-compat window test, secret check. Add a unit test that the rule produces both sides on synthetic data.
- Update docs: dataset card, data dictionary, EDA findings for new features (training-period data only), evaluation_protocol_v2, algorithm.md, user_guide (the three insight outputs and when not to trust each), report (planned vs implemented vs evidence; the v1 negative result as its own section) and slides/outline.md (add insight slides).
- Commit per phase, run the secret check before every commit, push (never force-push), tag model-v2 and push the tag. If a push fails, keep committing locally and tell me to run git push.
- Final printed summary: data span and bar counts; OOF log-loss per candidate; chosen model, horizon and coverage c; q_lo and q_hi; holdout Up/Down/No-signal counts, per-side accuracy with Wilson CI, balanced accuracy, permutation p-value, base rates, net return at 5 bps; VOL and BAND results; verdict on each pre-registered criterion; push status.
