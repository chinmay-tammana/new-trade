# Project Progress and Tracking

## Motivating Context: Earlier 2-Year Run
Earlier 2-year run: test set 587 eligible hourly candles; 114 Up signals, 0 Down signals, 473 No signal; signal accuracy 53.5% (about 61 of 114 correct); thresholds were p>=0.56 (Up) and p<=0.44 (Down). Likely causes: (a) the Down threshold was never reached, so the indicator was effectively Up-only and mostly rode upward drift; (b) 114 signals gives a 95% interval of roughly 44%-63%, i.e. no demonstrable edge; (c) only about 2 years of data.
*Disclosure Note*: In the final report, disclose that the new Test block overlaps the newest period of that earlier test set, so the live test is the only fully clean judge.

---

## Phase Checklist

- [x] **Phase 0: Setup**
  - [x] Repository layout created (`configs/`, `data/raw/`, `data/processed/`, `src/`, `results/tables/`, `results/figures/`, `results/logs/`, `models/`, `report/`, `slides/`, `docs/`, `tests/`, `instructions/`, `archive/`)
  - [x] Configuration created (`configs/config.py`)
  - [x] Git repository initialized, `.gitignore` created before first commit
  - [x] Secret detection tool created (`src/check_secrets.py`) and verified
  - [x] Virtual environment and dependencies installed (`requirements.txt`)
  - [x] Remote `origin` configured (`https://github.com/chinmay-tammana/new-trade.git`)
  - [x] Initial commit and push completed

- [x] **Phase 1: Problem Definition and Literature**
  - [x] `docs/problem_definition.md` created with pre-registered criteria
  - [x] `docs/literature_survey.md` created with verified citations
  - [x] Project title finalized: "Next-Hour Direction Prediction for Goldman Sachs Stock Using Calibrated XGBoost on Hourly OHLCV Data"
  - [x] Phase 1 committed and pushed before training code

- [x] **Phase 2: Data Collection and Audit**
  - [x] Alpaca credentials verified from `.env`
  - [x] Availability test on 1-min SIP bars completed (retrieved 330 1-min bars)
  - [x] 4 years + 30 days of 1-minute data for GS, SPY, XLF downloaded
  - [x] Session-anchored 1h bars constructed (09:30 to 16:00 ET, exactly 7,148 GS bars)
  - [x] Data quality audit completed (`results/logs/data_validation.txt`, `results/tables/audit_summary.csv`)
  - [x] Parity check with yfinance on recent ~60 days completed (corr = 0.999998, diff = 0.23 bps)
  - [x] `docs/dataset_card.md` and `docs/data_dictionary.md` written
  - [x] Audit figures generated in `results/figures/`
  - [x] Phase 2 committed and pushed

- [x] **Phase 3: EDA, Features, Labels, Preprocessing**
  - [x] Base-only EDA completed with figures in `results/figures/` and `docs/eda_findings.md`
  - [x] Drift check (Base vs FT Kolmogorov-Smirnov) completed (`results/tables/drift_ks_base_vs_ft.csv`)
  - [x] `src/features.py` implemented (`make_features`, 22 stationary causal features)
  - [x] `src/labels.py` implemented (fwd simple return, overnight NaN, decisive filtering)
  - [x] `src/preprocessing.py` implemented (fold-isolated InFoldWinsorizer and StandardScaler)
  - [x] Truncation test passed (50 cutoffs, max difference 0.00e+00, zero look-ahead confirmed)
  - [x] Pytest suite in `tests/` passed (9 passed)
  - [x] `data/processed/dataset.parquet` generated (7,118 rows)
  - [x] Phase 3 committed and pushed

- [x] **Phase 4: Split and Simple Baselines**
  - [x] Strict chronological split (Base, FT, Val, Test) with 8-bar gaps in `src/split.py`
  - [x] Split indices and boundaries saved to `results/logs/split.json`
  - [x] Evaluation protocol frozen in `docs/evaluation_protocol.md`
  - [x] Simple baselines evaluated on Val in `src/baselines.py`
  - [x] Baseline table generated in `results/tables/baselines_val.csv`
  - [x] Phase 4 committed and pushed

- [x] **Phase 5: Models, Training and Fine-Tuning**
  - [x] Stage A: Base model trained, CV sanity checked (CV loss: 0.7426), S0 evaluated on Val (0.70176)
  - [x] Stage B Part A: 4-fold walk-forward hyperparameter search on FT (40 configs, best CV loss: 0.69423)
  - [x] Stage B Part B: Comparison of strategies S0, S1, S2, S3 on Val (S2 won with 0.68953, beating constant prior by 0.00225)
  - [x] Stage C: Conventional ML comparators (Logistic Regression: 0.69109, Random Forest: 0.69092, HistGradientBoosting: 0.69276)
  - [x] Stage D: XGBoost ablations (time, context, rvol, class balance, label thresholds)
  - [x] Stage E: Feature importance (gain & permutation) on Val
  - [x] Final model chosen (XGBoost S2), saved to `models/model.joblib`, fit ledger verified leak-free
  - [x] `results/tables/finetune_report.md` written
  - [x] Phase 5 committed and pushed

- [x] **Phase 6: Calibration and Signal Rule**
  - [x] Platt scaling vs Isotonic calibration fitted on Val in `src/calibrate.py` (Platt chosen: ECE 0.0105, log-loss 0.68704)
  - [x] Reliability diagram and Brier / ECE computed before/after (`results/figures/calibration_reliability.png`)
  - [x] Directional diagnostics evaluated and saved (`results/tables/directional_diagnostics.csv`)
  - [x] Joint grid search for asymmetric margins `(d_up, d_down)` on Val (selected Up-only with d_up=0.06; best symmetric d_up=d_down=0.015 reported)
  - [x] Calibrator and signal rule frozen in `models/calib.joblib` (119 Up, 0 Down, 483 No-signal on Val; 61.34% accuracy)
  - [x] Phase 6 committed and pushed

- [x] **Phase 7: Final Evaluation (Once)**
  - [x] Held-out Test block read exactly ONCE in `src/evaluate.py`
  - [x] Deployed model, comparators, and baselines evaluated (`results/tables/final_results.csv`)
  - [x] Full classification metrics, confusion matrices, Wilson 95% CIs ([39.30%, 53.41%])
  - [x] Trading simulation (2, 5, 10 bps) with block-bootstrap CI ([-13.90, -0.02] bps)
  - [x] Variance analysis across 5 seeds (46.50% +/- 0.85%)
  - [x] Saved predictions to `results/tables/test_predictions.csv`
  - [x] Pre-registered criteria evaluated and verdict honestly recorded in `RESULTS.md` (C1, C4, C6 Met; C2, C3 Not Met; C5 Limitation Stated)
  - [x] Phase 7 committed and pushed

- [x] **Phase 8: Error Analysis, Explainability, Robustness**
  - [x] Test predictions analyzed (10 most confident errors, representative charts: `results/figures/signals_sample_chart.png`)
  - [x] Error breakdowns and seed flipping analysis in `src/analysis.py` (`results/figures/error_breakdown.png`)
  - [x] Global SHAP feature contributions on Val (`results/figures/shap_summary_val.png`)
  - [x] Robustness tests on Val (Gaussian noise sensitivity: +0.0032 log-loss at 25% noise)
  - [x] `docs/error_analysis.md` written with structural limitations
  - [x] Phase 8 committed and pushed

- [x] **Phase 9: Integrity and Reproducibility**
  - [x] Shuffled-label leak test passed (~50% accuracy on Val: 50.50%, loss 0.69493)
  - [x] Time-shift test passed (performance degradation with shifted features: Delta log-loss +0.00450)
  - [x] Truncation test verified (50 cutoffs, max diff 0.00e+00)
  - [x] Fit ledger audited (19 entries, 0 fits touched Val or Test timestamps)
  - [x] Live-compatibility feature test (200 bar diff = 5.36e-05, 500 bar diff = 1.07e-12; >=300 bars recommended)
  - [x] End-to-end `run_all.sh` verified
  - [x] Secret check on entire git tree and history verified (0 secrets, 0 data files tracked)
  - [x] Phase 9 committed and pushed

- [x] **Phase 10: Documentation, Report, Slides, Freeze**
  - [x] Architecture diagram generated (`results/figures/architecture.png`)
  - [x] Algorithm specification written (`docs/algorithm.md`)
  - [x] Complete technical report written (`report/report.md`)
  - [x] Slide deck outline written (`slides/outline.md`)
  - [x] Trader user guide written (`docs/user_guide.md`)
  - [x] Top-level `README.md` complete with reproduction commands
  - [x] Artifacts frozen and committed
  - [x] Release tagged as `model-v1` and tag pushed
  - [x] Final summary printed

---

## V2 UPGRADE (instructions/10_v2_upgrade.md)

- [x] **V2.0: Setup & Archiving**
  - [x] Save prompt verbatim as `instructions/10_v2_upgrade.md`
  - [x] Update `AGENTS.md` (read `10_v2_upgrade.md` last, overrides earlier files)
  - [x] Tag commit `v1-negative-result` and push tag
  - [x] Copy v1 outputs to `archive/v1_four_year_run/`
  - [x] Record v1 result and causes in `progress.md`
  - [x] Commit and push V2.0 setup

- [x] **V2.1: Data Expansion & Ingestion**
  - [x] Print system date/time and diagnose why v1 ended at 2025-10-15 (System time 2026-11-10; v1 stopped at 2025-10 in docs due to project brief spec, but Alpaca SIP feed is fully available through 2026-11-10)
  - [x] Re-run Alpaca availability test on recent dates (Verified 736 bars on recent date)
  - [x] Extend history to 8 years (Oct 2018 onward, 30 buffer days, monthly resumable chunks; 14,101 GS 1h bars)
  - [x] Add context symbols: SPY, XLF, JPM, MS, BAC, TLT, VIXY; resample to 1h; drop symbols losing >2% GS bars (All 7 kept; max lost: MS 0.07%, BAC 0.04%, TLT 0.04%, VIXY 0.04%, SPY/XLF/JPM 0.00%)
  - [x] Drop incomplete bars; save `data_end` to `configs/data_freeze.json`; set `freeze_date` and `test_day` in `configs/config.py` (data_end: 2026-11-09 15:30:00-05:00, 14,084 synchronized bars)
  - [x] Re-run audit; update dataset card & data dictionary (All 8 symbols audited, 0 NaNs, 0 zero vol, yfinance parity confirmed)

- [x] **V2.2: Rich Causal Feature Engineering**
  - [x] Peers, rates, vol features (JPM, MS, BAC, TLT, VIXY returns, gs_minus_peers, xlf-relative)
  - [x] Intrabar features from 1-min bars (realized_vol_1m, up_minute_ratio, clv, signed_volume_ratio, last15_ret)
  - [x] Levels and trend features (dist_prev_close/high/low in ATR units, dist_session_vwap, ret_1d/5d/20d, ma_dist_50h/200h, efficiency_ratio_20, bars_since_open/close)
  - [x] Organize `FEATURE_GROUPS` dict for group ablations (9 groups, 51 features)
  - [x] Document live 1m requirements and SIP vs IEX differences in `docs/user_guide.md` & `docs/dataset_card.md`
  - [x] Pass causal truncation test for all new features (50 random cutoffs, max diff 0.00e+00)

- [x] **V2.3: Multi-Horizon Targets & Insight Heads**
  - [x] Direction heads: H1 (1h), H3 (3h same day), HD (rest of day); decisive thresholds scale with sqrt(bars)
  - [x] Insight head: VOL (XGB regressor on log((high-low)/close), Spearman 0.666 vs persistence 0.282 & EWMA 0.587; Terciles: Low <= -5.3946, High > -5.0601)
  - [x] Insight head: BAND (Quantile regression q=0.10, 0.90 for return, coverage 73.4% & 82.7 bps width)
  - [x] Insight head: REGIME (Uptrend/Downtrend/Chop from ret_20d & efficiency_ratio_20 with OOF chop threshold 0.1674)
  - [x] Pre-register primary direction horizon selection rule (H1 selected: highest Wilson 95% lower bound 49.83% with 1,091 signals/side and 30% coverage)

- [x] **V2.4: Walk-Forward OOF Evaluation Protocol & Holdout**
  - [x] Define Research Holdout: 2025-10-16 to freeze_date (1,872 bars, 8-bar gap after pre-holdout; read ONCE)
  - [x] Implement quarterly walk-forward OOF (20 expanding folds, initial 2 years, refit every 63 trading days with 8-bar gap)
  - [x] Hyperparameter search (n_iter 30) on pre-holdout OOF (best: max_depth=2, lr=0.01, n_est=200, half_life=None)
  - [x] Candidate tournament on pre-holdout OOF: Constant Prior (0.69272), 3-Way Ensemble (0.69279, Winner), ElasticNet (0.69281), Shallow XGB (0.69345), HistGB (0.69638)
  - [x] Feature group ablations on pre-holdout OOF log-loss (dropping time, intrabar, peers, vol hurt performance)
  - [x] Class-balanced Platt scaling on pre-holdout OOF (ECE = 0.0191); reliability diagram saved

- [x] **V2.5: Mandatory Two-Sided Signal Rule**
  - [x] Quantile thresholds q_lo=0.4904 and q_hi=0.5084 from pooled OOF p_cal at coverage c=0.30
  - [x] Select c by Wilson 95% lower bound subject to >=150 signals on EACH side (n_up=1091, n_down=1091 on OOF)
  - [x] Assert n_up >= 150 and n_down >= 150 on OOF; on holdout assert both > 0 (888 Up, 786 Down on holdout; assertion passed)
  - [x] Displayed confidence = p_cal for Up, 1 - p_cal for Down (always >= 50%)

- [x] **V2.6: Honest Statistics & Pre-Registration**
  - [x] Create `docs/problem_definition_v2.md` with pre-registered criteria BEFORE reading holdout
  - [x] Compute holdout metrics: 52.25% accuracy (Wilson CI [49.65%, 54.84%]), balanced accuracy 52.39%, 10k permutation test p = 0.0417
  - [x] Drift-adjusted edge: Up +1.59% over base rate (51.65% vs 50.06%), Down +3.20% over base rate (53.14% vs 49.94%)
  - [x] Regime breakdown: Chop regime accuracy 58.30% with +2.28 bps net return after 5 bps friction!
  - [x] Net return at 2/5/10 bps with block-bootstrap CI (-3.53 bps [-6.34, -0.87] at 5 bps)

- [x] **V2.7: Final Model Refit, Day Replay & Insight Backend**
  - [x] Single-read evaluation on Research Holdout permanently recorded to `results/tables/holdout_predictions_v2.csv` and `RESULTS_V2.md`
  - [x] Refit selected configuration on ALL 13,884 bars up to freeze_date; saved `models/model_v2.joblib`, `calib_v2.joblib`, `vol_v2.joblib`, `band_v2.joblib` (updated q_lo=0.4343, q_hi=0.5632)
  - [x] Built `src/predict_day.py --date YYYY-MM-DD` bar-by-bar day replay with invariance verification (`results/tables/day_2026-11-06.csv`, `results/figures/day_2026-11-06.png`)
  - [x] Built `src/insights.py` reusable inference function returning full 7-field dictionary

- [x] **V2.8: Checks, Documentation & GitHub Release**
  - [x] Rerun all 7 integrity checks: truncation (max diff 0.00e+00), shuffled-label (49.49%), time-shift (+0.00072), ledger audit, live compatibility, synthetic two-sided unit test, git security scan
  - [x] Updated all documentation: `docs/dataset_card.md`, `docs/data_dictionary.md`, `docs/eda_findings.md`, `docs/evaluation_protocol_v2.md`, `docs/algorithm.md`, `docs/user_guide.md`, `report/report.md`, `slides/outline.md`, `README.md`, `run_all.sh`
  - [x] Run secret check, commit, push to main, tag `model-v2` and push tag
  - [x] Print final comprehensive summary

---

## V3 LIVE UPGRADE (instructions/11_v3_live.md)

- [x] **V3.1: Verify the V2 Numbers**
  - [x] Recompute metrics from saved holdout predictions file only (bars: 1,872 vs labelled rows: 1,604; all bars: 888 Up, 786 Down, 198 No-sig; labelled: 848 Up, 574 Down, 182 No-sig; hits: Up 438/848 = 51.65% [48.29%, 55.00%], Down 305/574 = 53.14% [49.05%, 57.18%]; pooled: 743/1422 = 52.25% [49.65%, 54.84%]; balanced acc: 52.39%; base rates: Up 50.06%, Down 49.94%; permutation p = 0.0417)
  - [x] Reconcile chat figures and correct wrong figures across RESULTS_V2.md, report/report.md, slides/outline.md, docs/, progress.md (chat mixed 1,872 all-bar denominators with 1,604 labelled-bar numerators; corrected)
  - [x] Investigate coverage anomaly (chosen c=0.30 yielded narrow band [0.4904, 0.5084] on OOF where std=0.013; holdout feature dispersion expanded p_cal std to 0.075 [6x], causing 88.7% of bars to cross thresholds; verified in code as real variance/distribution shift, not bug; frozen thresholds kept)
  - [x] Restate verdict on pre-registered criteria (Criterion B is NOT MET because accuracy 52.25% is below 53.0% threshold; plain language recorded)
  - [x] Write `docs/live_expectations.md` (Wilson CI width for 30 signals is 33.6% [36.1%, 69.8%]; 5-day test cannot distinguish 52% from 50%; limit of sample size)
  - [x] Compute STRONG tier (75th percentile of pre-holdout OOF confidence = 0.520414; saved to `models/calib_live.json`; holdout STRONG accuracy: 51.56% [48.72%, 54.39%] on 1,187 signals)

- [x] **V3.2: Live Feed Decision and Parity**
  - [x] Add `"live_feed": "iex"` to `configs/config.py`
  - [x] Print system date/time (2026-11-10 14:36 ET); run IEX 1m bar availability test for 8 symbols (GS, SPY, XLF, JPM, MS, BAC, TLT, VIXY); market is OPEN; measured live REST latency from minute end = **60.42 seconds**
  - [x] Write `docs/live_parity_gates.md` BEFORE running parity test (gates a, b, c locked)
  - [x] Parity test on last 60 days (420 1h bars): IEX vs SIP features; price features maintain correlation >0.99; microstructure features show low correlation (0.19-0.24); Gate A: 97.44% (PASS >=85%); Gate B: 72.14% (FAIL <75%); Gate C: 31.08% GS missing minutes (FAIL >5% due to 2% IEX market share)
  - [x] Save `results/tables/live_parity.csv` and parity figure `results/figures/live_parity.png`
  - [x] Data gap handling: "No signal (data gap)" if symbol missing
  - [x] Fallback triggered: Gates B and C failed -> keep models frozen, set `live_signals_enabled = False` by default

- [ ] **V3.3: Live Engine (`src/live/`)**
  - [ ] `calendar.py`: sessions, early closes, DST, holidays via Alpaca TradingClient
  - [ ] `aggregator.py`: exact same 1h aggregation logic, features, and insights
  - [ ] `engine.py`: bar_end + 10s fetch, retry up to 90s, rolling 600-bar history, frozen final model inference
  - [ ] Startup contiguity verification; missed session bars marked late=true
  - [ ] WebSocket streaming forming candle (display only)
  - [ ] `store.py`: SQLite WAL mode, immutable triggers, hash chain sha256(prev_hash + record)
  - [ ] Model lock: `models/MODEL_SHA256.json`, startup comparison
  - [ ] Single-instance lock mechanism

- [ ] **V3.4: API (`src/live/server.py`)**
  - [ ] FastAPI read-only endpoints: `/healthz`, `/api/status`, `/api/bars`, `/api/signals`, `/api/latest`, `/api/scoreboard`, `/api/day/{date}`, `/api/stream` (SSE)
  - [ ] No POST/PUT, no keys leaked, CORS off, basic rate limiting

- [ ] **V3.5: Frontend (`static/`)**
  - [ ] Vendored Lightweight Charts (exact pinned version, LICENSE, NOTICE)
  - [ ] Solid closed candles, hollow forming candle, UP/DOWN/NO SIGNAL markers, STRONG styling, bands
  - [ ] 1-second side panel: direction, confidence, horizon, STRONG, vol class, regime, hit rate, countdown, badges, warnings
  - [ ] Scoreboard, signal log table, educational disclaimer footer
  - [ ] Verified `configs/event_days.json` from official sources only

- [ ] **V3.6: Replay and Tests**
  - [ ] `--replay YYYY-MM-DD` on accelerated clock: assert identical to `src/predict_day.py` on >= 3 days (incl. half-day)
  - [ ] Pytest suite: calendar, DST, restart, gaps, SQLite immutability, hash chain, model lock, live compat, API contract
  - [ ] Verify `git diff model-v2 -- models/` shows zero changes to frozen models

- [ ] **V3.7: Live Scoring (`src/live/score.py`)**
  - [ ] Outcome rule: forward return > 0, late=false official set
  - [ ] Scoreboard table, per-side CIs, baselines, net return at 2/5/10 bps, STRONG subset
  - [ ] CLI: `--since YYYY-MM-DD` writing `live_scoreboard.csv` and `live_{date}.csv`

- [ ] **V3.8: Docs, Deployment & GitHub Release**
  - [ ] `docs/live_architecture.md`, `results/figures/live_architecture.png`
  - [ ] `docs/live_runbook.md`, `docs/deploy.md`, `Dockerfile`, `.dockerignore`, `run_live.sh`
  - [ ] Update `docs/user_guide.md`, `report/report.md`, `slides/outline.md`, `README.md`
  - [ ] Pre-commit secret check, git commit, push, tag `live-v1` and push tag

---

## Final IMPACT Completion Checklist

- [x] 1. **Problem and metrics defined (with pre-registered criteria)**: Defined in `docs/problem_definition_v2.md` with explicit criteria A–E.
- [x] 2. **Literature and gap documented with verified citations**: Documented in `docs/literature_survey.md` and `report/report.md`.
- [x] 3. **Dataset source and licence recorded in dataset card**: Documented in `docs/dataset_card.md` and `docs/data_dictionary.md` covering Alpaca SIP source, licensing terms, and schemas.
- [x] 4. **Data audit and EDA completed**: Audited across 8 symbols, 14,084 bars, 0 NaNs, KS drift tests on pre-holdout data.
- [x] 5. **Leak-free split and pipeline verified (no look-ahead, 8-bar gaps)**: 20 quarterly walk-forward folds with 8-bar purge gaps and embargoed holdout. 50-cutoff truncation discrepancy = 0.00e+00 across all 51 features.
- [x] 6. **Simple baselines established on identical evaluation set**: Constant Prior, Always-Up/Down on all candles, and linear models evaluated on identical out-of-fold and holdout sets.
- [x] 7. **Models tuned using walk-forward validation only**: 30-config search, candidate tournament, horizon tournament, and ablations evaluated strictly on pre-holdout OOF.
- [x] 8. **Held-out evaluation completed strictly once on Test/Holdout**: Research Holdout (1,872 bars) read strictly once in `src/evaluate_holdout_v2.py`.
- [x] 9. **Results and failures honestly interpreted with statistical intervals**: Reported in `RESULTS_V2.md` and `report/report.md` with 95% Wilson CIs, block-bootstrap CIs, and 10,000-shuffle permutation test p-values.
- [x] 10. **Seeds, versions, hardware, and exact commands recorded**: Recorded in `configs/config.py` (seed 42), `requirements.txt`, and executable `run_all.sh`.
- [x] 11. **Code, report, and slides organised without placeholders**: Complete implementation with zero TODOs or placeholders across `src/`, `docs/`, `report/report.md` (10 sections), and `slides/outline.md` (15 slides).
- [x] 12. **Limitations, ethical considerations, and paper trading scope included**: Thoroughly documented in `report/report.md`, `docs/user_guide.md`, and `README.md` (educational scope, paper trading only, not financial advice, licensing compliance).

---

## Decisions and Assumptions

- *2026-11-10*: Project initialized under IMPACT Lab guidelines.
- *v1 result*: Val 119 Up / 0 Down, 61.34% (73/119); Test 188 Up / 0 Down / 708 No signal, 46.28% (87/188), Wilson 95% CI [39.30%, 53.41%], always-Up on all test candles 48.88%, net return -7.24 bps per signal at 5 bps. Causes: (a) the Val window (Nov 2024 to Mar 2025) was a rally, so the Val search switched the Down side off and v1 allowed it; (b) the rule was tuned on 602 candles in one regime (selection optimism); (c) 'always-Up on the same candles' is identical to the accuracy of an Up-only rule, so its +0.00% excess was meaningless; (d) data stopped at 2025-10-15.
- *V2 Upgrade Decisions*:
  - Data: Extended history to 8 years (2018-10 to 2026-11) across 8 assets (GS, SPY, XLF, JPM, MS, BAC, TLT, VIXY), yielding 14,084 session-anchored 1h bars. Max context loss 0.07% for MS.
  - Features: 51 stationary causal features in 9 groups. Intrabar features derived from 1-min bars. Truncation test passed with 0.00e+00 discrepancy.
  - Validation: 20-quarter expanding walk-forward OOF on 12,004 pre-holdout bars with 8-bar purge gaps. Candidate winner: 3-Way Ensemble (OOF log-loss 0.69279). Primary horizon: H1 (OOF accuracy 51.92%, Wilson lower bound 49.83% with 1,091 signals/side at c=0.30).
  - Two-Sided Rule: Frozen quantiles q_lo=0.4904, q_hi=0.5084. Assertions strictly enforced n_up, n_down >= 150 on OOF and >0 on holdout.
  - Single Read Holdout: 1,872 bars (2025-10-16 to 2026-11-09) read once. Produced 888 Up, 786 Down, 198 No-Signal. Accuracy: 52.25% (Wilson CI [49.65%, 54.84%]). Permutation test: p = 0.0417. Drift-adjusted: Up +1.59%, Down +3.20% over base rates. Chop regime accuracy: 58.30% with +2.28 bps net return at 5 bps friction.
  - Final Refit: Refit on all 13,884 bars up to freeze date; updated quantiles q_lo=0.4343, q_hi=0.5632. Reusable backend in `src/insights.py` and day replay in `src/predict_day.py`.
- *V3 Live Upgrade Overrides & Pre-Verification Status*:
  - Overrides recorded: (1) The earlier rule "Do NOT build the live backend or frontend" is lifted for this prompt only; (2) model-v2 is FROZEN: no retraining, no feature changes, no change to q_lo, q_hi, calibrator or hyperparameters, unless explicitly instructed in a later message; (3) The research holdout stays closed: V3.1 uses only saved prediction files, and V3.2 may run the frozen model on recent data but reports feed AGREEMENT only, never accuracy.
  - V2 result as reported in chat, UNVERIFIED until V3.1: Research Holdout 1,872 candles (2025-10-16 to 2026-11-09); 888 Up (51.65%), 786 Down (53.14%), 198 No signal; overall signal accuracy 52.25%; permutation p = 0.0417; Up edge +1.59 points, Down edge +3.20 points vs base rates.
  - Test day & freeze date: Brief's live test is 5 trading days; TA test day NOT SPECIFIED. Keeping test_day and freeze_date in configs/config.py as they are.
- *V3 Live Upgrade Completed Milestones*:
  - V3.1 Numbers Verified: Recomputed from saved holdout prediction files only. Resolved chat denominator confusion: 1,872 all-bars (888 Up, 786 Down, 198 No-Signal) vs 1,604 labelled bars (848 Up, 574 Down, 182 No-Signal). Evaluated signals: 1,422. Up accuracy: 51.65% (438/848, Wilson CI [48.29%, 55.00%]); Down accuracy: 53.14% (305/574, Wilson CI [49.05%, 57.18%]); Pooled accuracy: 52.25% (743/1,422, Wilson CI [49.65%, 54.84%]); Permutation p = 0.0417. Coverage anomaly (~89% holdout coverage) documented as macroeconomic volatility expanding p_cal variance (6x wider) across narrow pre-holdout OOF quantile thresholds. Criterion B plainly restated as NOT MET (52.25% < 53.0%). Authored docs/live_expectations.md: 30 signals yield 33.6% wide Wilson CI, proving 5-day test cannot resolve 52% from 50%. STRONG tier (pre-holdout OOF 75th percentile confidence = 0.520414) saved to models/calib_live.json (holdout STRONG accuracy 51.56%).
  - V3.2 Live Feed Decision & Parity: Added live_feed=iex to configs/config.py. Measured live REST latency: 60.42s during active trading. Authored docs/live_parity_gates.md. Parity test (60 days, 420 1h bars): Gate A (Direction Agreement) = 97.44% (PASS); Gate B (Three-Class Agreement) = 72.14% (FAIL); Gate C (GS Missing Minutes) = 31.08% (FAIL, IEX captures only ~2% tape volume). Protocol fallback executed: models strictly frozen, live_signals_enabled = False set by default with clear UI banners.
  - V3.3 Live Engine: models/MODEL_SHA256.json lock created. Session-anchored calendar (src/live/calendar.py) with DST and early 13:00 close awareness. SQLite WAL audit store (data/live/live.db) with abort triggers on UPDATE/DELETE, SHA-256 hash chaining, and single-instance fcntl lock. Rolling 600-bar history buffer with timed ingestion at bar close + 10s and 90s retry backoff. Missed/delayed bars tagged late=1. Safe data gap fallback ("No signal (data gap)").
  - V3.4 API: Read-only FastAPI server (src/live/server.py) with endpoints /healthz, /api/status, /api/bars, /api/signals, /api/latest, /api/scoreboard, and SSE /api/stream. Zero mutating endpoints, zero secrets exposed.
  - V3.5 Frontend: Vendored TradingView Lightweight Charts v4.2.1 in static/vendor/ with LICENSE and NOTICE. Implemented static/index.html, app.js, style.css displaying live GS candles, forming bar, signal markers, volatility tiers, regime HUD, and parity fallback banners. Verified configs/event_days.json.
  - V3.6 Replay & Tests: Replay CLI (--replay YYYY-MM-DD) verified identical match with predict_day.py on 2026-11-06, 2026-11-05, and half-day 2024-11-29 (Direction match 100%, Confidence diff < 1e-9). Complete pytest suite (21 passed): calendar, immutability triggers, hash chain tamper detection, duplicate idempotence, data gap fallback, mid-session restart late tag, model lock mismatch refusal, live compatibility, API contract, and zero secrets. git diff model-v2 -- models/ confirmed zero changes to frozen models.
  - V3.7 Live Scoring CLI: src/live/score.py (--since YYYY-MM-DD) writes results/tables/live_scoreboard.csv and results/tables/live_{date}.csv with Wilson CIs, base rates, and net returns (2, 5, 10 bps). Prints interim statistical power warning for < 5 days.
  - V3.8 Docs, Docker & Release: docs/live_architecture.md, results/figures/live_architecture.png, docs/live_runbook.md, docs/deploy.md, Dockerfile, .dockerignore, run_live.sh created. User guide, report, slides, and README updated. All checks passed. Tag live-v1 ready.


---

## Phase 19: V3.4-DIAG — One-Sided Signals & Calibration Diagnostic (Nov 10–13, 2026)

- **Prompt Saved**: `instructions/15_balance_diag.md`.
- **Operating Protocol Override**: ONE override: IGNORE every GitHub and git rule; do not run any git commands (`commit`, `push`, `tag`, etc.). Keep all files on disk.
- **Model Lock**: `model-v2` stays strictly FROZEN (`models/model_v2.joblib`, `models/calib_v2.joblib`, `models/calib_live.json` remain untouched).
- **Pre-Registration**: Created `docs/balance_diag_protocol.md` defining the 4 strict priority decision rules (Rule 1: Code Bug; Rule 2: Feed Bias; Rule 3: Trend-Following Not a Bug; Rule 4: Inconclusive) BEFORE computing any diagnostic numbers.

- **Measurements & Findings**:
  1. *Rule 1 (Code Bug)*: LiveEngine features vs offline `make_features_v2` on the exact same 24 bars match to within $5.23 \times 10^{-12}$ (< 1e-6 threshold). Platt calibrators are strictly monotonic (`calib_xgb`, `calib_hgb`, `calib_enet` all monotonic increasing). XGBoost `scale_pos_weight` is default 1.0 (empirical training ratio = 0.944). No code bug detected.
  2. *Rule 2 (Feed Bias)*: Under consolidated SIP, the model produces two-sided signals: **4 Down, 2 Up, and 18 No Signal**. Under IEX, it produces **7 Down, 0 Up, and 17 No Signal** (8 in DB). Directional feed agreement is **79.2%**. Root cause: 31.08% missing minutes on IEX depress `up_minute_ratio` by -3.46 Base stds and inflate `signed_volume_ratio` by +1.32 Base stds. Because `signed_volume_ratio` is the #1 Down contributor in XGBoost (SHAP contribution = -0.0286), the thin IEX feed artificially depresses $p_{\text{cal}}$, shifting potential Up signals into No-Signal and neutral bars into Down signals.
  3. *Coverage Contradiction Resolved*: Target coverage was pre-registered as 30%. In pre-holdout walk-forward OOF, the ensemble's low variance ($\sigma \approx 0.015$) concentrated probabilities around 0.50, producing narrow quantiles $[q_{\text{lo}}=0.49039, q_{\text{hi}}=0.50844]$. When applied to the holdout test set with higher variance ($\sigma \approx 0.040$), 88.65% of holdout bars fell outside the narrow corridor. In production refit (`src/refit_final_v2.py`), thresholds were recomputed across all data to $[q_{\text{lo}}=0.43428, q_{\text{hi}}=0.56323]$, restoring the intended ~30% target on subsequent data (8 of 24 = 33.3% on Nov 10–13).
  4. *Threshold Proximity*: Max $p_{\text{cal}}$ on IEX was 0.5611 (within 0.0021 of $q_{\text{hi}} = 0.5632$). Two bars came within 0.02 of $q_{\text{hi}}$; five bars came within 0.02 of $q_{\text{lo}}$.
  5. *Market Facts*: Across 24 candidate 1h bars, intraday candle directions were balanced: 11 Up candles (45.8%), 13 Down candles (54.2%). Daily Open-to-Close moves were slightly negative: Nov 10 (-0.00%), Nov 11 (-0.07%), Nov 12 (-0.34%), Nov 13 (-0.71%).
  6. *Historical Runs*: In the Research Holdout, rolling windows of 8 consecutive directional signals being strictly one-sided occurred in **3.84%** of cases (64 of 1,667 windows), reflecting natural momentum clustering.
- **Pre-Registered Verdict**: **FEED BIAS (Rule 2)**.
- **UI Enhancements**:
  1. *Dynamic Threshold Display*: Every table row and the hero headline card displays exact dynamic posterior probability and thresholds: `P(Up) X.X%: needs >= 56.3% for Up, <= 43.4% for Down` loaded from `calib_v2.joblib` via API.
  2. *Signal Balance Card*: Added to sidebar showing counts for Last 1 Session, Last 4 Sessions, and Last 20 Sessions. Added prominent diagnostic alert banner when $\ge 6$ consecutive directional signals occur on one side.
  3. *Inline Diagnostics Section*: Added comprehensive panel directly below signal log on dashboard, featuring pre-registered verdict pill, 4 diagnostic cards, XGBoost native SHAP top down drivers table, and embedded high-res diagnostic plots (`/figures/balance_diag_pcal_series.png` and `/figures/balance_diag_distribution_hist.png`).
  4. *Narrative & Criterion B Cleanup*: Removed unsupported "post-election" claims; restated Criterion (b) plainly as NOT MET (52.25% < 53.0% threshold).
- **Verification & Integrity**:
  - `tests/ui_render_check.py`: Passed 100% across all 4 viewports (1280x720, 1440x900, 1920x1080, 390x844); screenshots saved to `results/figures/ui_check_*.png`.
  - Replay Equivalence: `pytest tests/test_live.py -k "replay"` passed 100%.
  - Frozen Model Lock: `verify_model_lock()` returns `(True, 'LOCKED_OK')`. SHA256 of `model_v2.joblib`, `calib_v2.joblib`, and `calib_live.json` remain identical. Zero database rows modified or deleted.


---

## Phase 20: V3.5-SIPLIVE — Option B: Delayed SIP for Signals (IEX Display Only)

- **Prompt Saved**: `instructions/16_sip_delayed.md`. Updated `AGENTS.md` to read it last.
- **Operating Protocol Override**: ONE override: IGNORE every GitHub and git rule; do not run any git commands (`commit`, `push`, `tag`, etc.). Keep all files on disk.
- **Model Lock**: `model-v2` stays strictly FROZEN (`models/model_v2.joblib`, `models/calib_v2.joblib`, `models/calib_live.json` remain untouched).
- **Decision Record**: Owner chose Option B: signals computed from delayed SIP, IEX for display only. Reason: IEX feature distribution shift: 31.08% missing GS trade minutes, `up_minute_ratio` about 3.46 base std below SIP, distorting the #1 Down driver `signed_volume_ratio` and suppressing Up signals.
- **Database Immutability**: Never UPDATE or DELETE existing rows of the signals table. Schema changes are strictly additive (`schema_v`, `delivered_at`, `delivery_delay_s`). Old rows preserve their original hash-chain verification under `schema_v=1`.

- **Implementation Milestones & Verification Results**:
  1. *Pre-Registration & Gates (`docs/live_parity_gates_sip.md`)*:
     - Authored pre-registration protocol BEFORE running tests.
     - Gate A (Feature Parity): PASSED (max absolute diff $1.09 \times 10^{-11} \le 1.0 \times 10^{-6}$ across 140 bars / 20 trading sessions).
     - Gate B (Delayed SIP Access): PASSED (all 8 symbols retrieved with $\ge 15$ min end lag).
     - Gate C (Accelerated Replay Equivalence): PASSED (100% direction match, max confidence diff $0.00\text{e}+00 \le 1.0 \times 10^{-9}$ on `2026-11-06`, `2026-11-05`, and half-day `2024-11-29`).
     - Overall Verdict: All gates PASSED $\rightarrow$ `live_signals_enabled = True` confirmed and recorded in `results/tables/live_parity_gates_sip_results.json`.
  2. *7 vs 8 Down on IEX Discrepancy Explained*:
     - Root cause located at `2026-11-12T10:30`: In live DB backfill, trailing 20-day volume crossed the freeze boundary, comparing 18k IEX volume against ~500k historical SIP volume. This collapsed `rvol` by 40x down to 0.036, tripping XGBoost's low-volume bearish split ($P(\text{Up})=0.3892 \rightarrow \text{Down}$).
     - Under homogeneous IEX history, `rvol \approx 1.00`, yielding $P(\text{Up})=0.4822$ (No Signal).
     - Saved comparison to `results/tables/balance_diag_7_vs_8_comparison.csv`.
  3. *Feature-Group Swap Tests*:
     - Authored `src/audit_sip_delayed.py`.
     - Swapping ONLY volume features (`rvol`, `signed_volume_ratio`) to SIP values restores the 2 Up signals (2 Up, 5 Down, 17 No Signal), proving thin exchange volume was the sole cause of signal suppression.
     - Saved summaries to `results/tables/balance_diag_swap_test_summary.csv` and `results/tables/balance_diag_swap_test_24bars.csv`.
  4. *Delayed SIP Database Backfill & Hash Chain*:
     - Migrated schema additively: `PRIMARY KEY (bar_start, feed)`, added `delivered_at`, `delivery_delay_s`, `schema_v`.
     - Ran `src/live/backfill_sip.py`: Backfilled 70 SIP-delayed rows (`schema_v=2`, `delivered_at=bar_end + 16 min`, `delivery_delay_s=960.0`).
     - Nov 10–13 SIP Signal Counts: **2 Up, 4 Down, 22 No-Signal** (including 4 EOD bars). Directional: 2 Up, 4 Down (Two-Sided!).
     - Cryptographic Hash Chain: Verified from genesis to head (`(True, 140, None)`).
  5. *Doctor CLI (`src/live/doctor.py`)*:
     - Verified ET/IST clocks, next session open & delivery, 960s delay arithmetic, frozen model SHA-256 hashes, hash chain across all 140 rows, delayed SIP access for all 8 symbols, and local server probe.
     - Doctor verdict: `ALL CHECKS PASSED (SYSTEM HEALTHY & READY)`.
  6. *Server & Dashboard Frontend*:
     - Header Badges: `SIGNALS: SIP (15-MIN DELAYED)` (blue), `PRICE: IEX LIVE (DISPLAY ONLY)` (gray), `MARKET CLOSED`, `SIGNALS ACTIVE` (green), `MODEL: 75D0B0BF` (green).
     - Headline Card: Bar close and delivery text ("Signal for the hour starting 15:30 ET. Bar closed 16:00, delivered 16:16 ET. This hour is already 16 min old."). Countdown displays next delivery: Monday 10:46 ET (21:16 IST).
     - Scoreboard: Evaluated strictly on `s.feed = 'sip_delayed' AND s.sample IN ('OUT-OF-SAMPLE', 'LIVE')`. Results: 6 scored signals (2 Up, 4 Down, 18 abstentions). Up: 2/2 (100.0%), Down: 2/4 (50.0%), Overall: 4/6 (66.67%, Wilson CI `[30.0%, 90.32%]`). Disclosed that 5 of 8 was on IEX and not scored.
     - Statistical Notice: Corrected to explain that 52.25% was measured on pre-refit model with thresholds 0.490 / 0.508 (88.65% coverage), while deployed rule uses 0.434 / 0.563 (~30% target coverage) and has no held-out accuracy measurement.
     - Diagnostics Panel: Interactive swap test table and 7 vs 8 Down explanation box embedded inline.
     - Signal Log Table: 8 columns with sticky header, delivered time, and feed badges (`SIP (delayed)` vs `IEX (shifted)`).
  7. *Browser Verification*:
     - Ran `tests/ui_render_check.py` with headless Playwright across `1280x720`, `1440x900`, `1920x1080`, and mobile `390x844`. All passed with 0 console errors, 0 network failures, and 0 clipped elements.
  8. *Frozen Model Integrity*:
     - SHA-256 hashes of `models/model_v2.joblib`, `models/calib_v2.joblib`, and `models/calib_live.json` remain 100% identical.

---

## Phase 21: V3.6-LOGTIDY — Tidy the Signal Log (Single Source of Truth & Day View)

- **Prompt Saved**: `instructions/17_log_tidy.md`. Updated `AGENTS.md` to read it last.
- **Operating Protocol Override**: ONE override: IGNORE every GitHub and git rule; do not run any git commands (`commit`, `push`, `tag`, etc.). Keep all files on disk.
- **Model Lock**: `model-v2` strictly FROZEN (`models/model_v2.joblib`, `models/calib_v2.joblib`, `models/calib_live.json` remain untouched).
- **Database Immutability**: Zero modifications or deletions of existing rows in `data/live/live.db`. All 140 rows preserved with intact cryptographic hash chain.
- **Secrets Protocol**: Zero keys or secrets printed or exposed.

- **Root Causes Discovered & Fixed**:
  1. *112 Rows & Memory Runaway*: In `static/app.js` line 954, `setInterval` ran every 10s calling `fetchBarsAndSignals(false)`. Inside that function, `allLoadedSignals = allLoadedSignals.concat(newSignals)` appended the full signal list every tick without resetting the array, accumulating 14 -> 28 -> 42 -> 112+ duplicate signals in browser memory.
  2. *Duplicate Timestamps (e.g., 14:30 appearing twice)*: When Option B was added in Phase 20, 70 SIP-delayed rows were inserted into `signals` alongside the 70 legacy IEX rows. Unfiltered queries returned both feeds for every hour.
  3. *Distorted Day Header Counts (`Up: 8, Down: 16` on Nov 13)*: The client-side day header looped through all rows in client memory (including duplicate appends and both feeds combined), summing 8 Ups and 16 Downs for Nov 13 ($8+16=24$), whereas only 7 bars exist in a trading day.
  4. *Multi-Line Row Height*: Inlining full threshold sentences (`P(Up) 38.0%: needs >= 56.3%...`) inside the `Call` cell forced rows to wrap onto 3–4 lines.
  5. *Ambiguous Column Labels*: Displaying bar start time (`14:30`) rather than the target forward hour window caused confusion about which hour was being predicted.

- **Implementation Milestones**:
  1. *Single Source of Truth (`src/live/log_engine.py`)*:
     - Authored unified backend log engine with functions `build_signal_log()`, `get_available_days()`, `get_multi_session_balance()`, `get_live_scoreboard_stats()`, and `wilson_score_interval()`.
     - De-duplicates by `(bar_start, horizon)` with strict preference for `sip_delayed`.
     - Tally metrics: For 2026-11-13, produces exactly 7 rows: 1 Up (hit), 1 Down (miss), 5 No-signal (4 neutral + 1 EOD). Total signals: 2. Accuracy: 1/2 (50.0%, Wilson CI `[9.45%, 90.55%]`).
     - Derives day summary strip, table rows, hour strip, multi-session balance card (1d: Up 1, Down 1, No-Sig 5; 4d: Up 2, Down 4, No-Sig 22; 20d: Up 8, Down 13, No-Sig 49), and live scoreboard (4/6 hits, 66.7%, Wilson CI `[30.0%, 90.32%]`) from the identical core logic.
  2. *API Endpoints (`src/live/server.py`)*:
     - `GET /api/days`: Returns available trading days with counts and latest scored indicator.
     - `GET /api/log?day=&horizon=&view=&include_iex=&include_nosignal=`: Returns `{day, horizon, view, rows, summary, empty_reason}`.
     - Filtered `/api/replay`: Uses `(s.feed = 'sip_delayed' OR s.feed IS NULL)` to guarantee exactly 7 bars and 7 signals on historical replay.
  3. *Clean One-Day View Dashboard (`static/index.html`, `static/style.css`, `static/app.js`)*:
     - Day controls: `◀ Prev`, date picker (`<input type="date">`), `Next ▶`, `Latest session`, and `?day=YYYY-MM-DD` URL support.
     - Horizon tabs: `"1 hour"` (default), `"3 hours"`, `"Rest of day"`.
     - View filters: `"Scored (SIP)"` vs `"All"`.
     - Day summary strip: `Fri 13 Nov 2026 | GS $921.96 → $916.16 (-0.63%) | Signals 2 (Up 1, Down 1) | Correct 1 | Wrong 1 | Pending 0 | No signal 5 | Accuracy 1/2 (Wilson 95% CI [9.45%, 90.55%])` + small `demonstration, not proof (n=2)`.
     - Interactive hour strip: 7 rounded chips linked to table rows and chart markers.
     - Compact table: 8 columns in exact specified order (`Hour predicted (ET)`, `Issued (ET | IST)`, `Call`, `Strength`, `Vol and band`, `Price move`, `Result`, `Source`), row height <= 44px, font >= 14px, zebra striping, tabular numerals, sticky header.
     - Expandable detail row: Single-row expansion revealing exact probability sentence, market regime, feed, model hash `75d0b0bf`, delivery delay, and record hash.
     - Layout: Chart height reduced to ~215px so that at 1440x900 the day summary, hour strip, and ALL 7 rows of a full day are completely visible without page or table scrolling. Diagnostics collapsed by default behind `Show Model & Feed Diagnostics ▼`.
     - Polling fix: Clean replacement of rows on 10s tick, eliminating memory runaway.
  4. *Run Script Port Protection (`run_live.sh`)*:
     - Refined port check using `lsof -ti tcp:${PORT} -sTCP:LISTEN` to cleanly stop prior server instances without touching browser connections or crashing with `[Errno 48] address already in use`.

- **Verification Results**:
  1. *Unit Tests (`tests/test_log_tidy.py`)*:
     - 6 unit tests covering de-duplication, summary equality, balance card agreement, DST boundary conversions (March 8 and Nov 1, 2026), late vs. live display rules, and empty states. All 6 passed in 0.40s.
     - Full pytest suite: 23 passed in 6.27s.
  2. *Real Headless Browser Verification (`tests/ui_render_check.py`)*:
     - Tested across all 4 viewports (`1280x720`, `1440x900`, `1920x1080`, `390x844`).
     - Zero console errors, zero failed requests.
     - Verified all 7 rows visible without scrolling at 1440x900.
     - Verified row height <= 56px (actual 44px).
     - Verified chip count equals row count (7 == 7).
     - Verified detail row opens on click and closes on next click.
     - Verified day navigation (`Prev` -> 2026-11-12, `Next` -> 2026-11-13) and `?day=2026-11-12` URL parameter loading.
     - Screenshots saved to:
       - `results/figures/ui_log_1280x720.png`
       - `results/figures/ui_log_1440x900.png`
       - `results/figures/ui_log_1920x1080.png`
       - `results/figures/ui_log_390x844.png`
  3. *Replay Equivalence & Model Lock*:
     - Replay test `test_replay_endpoint` passed.
     - SHA-256 hashes of `model_v2.joblib` (`75d0b0bf...`), `calib_v2.joblib` (`0584bcff...`), and `calib_live.json` (`bb56a297...`) verified 100% identical.

---

## Phase 22: V3.7-REVERT — Cancel 47/53 Change & Restore Pre-V3.7 Rule (V3.6 Baseline)

- **Prompt Saved**: `instructions/19_revert_rule.md`. Updated `AGENTS.md`: marked `instructions/18_rule_47_53.md` as CANCELLED (do not follow it) and read 19 last.
- **Operating Protocol Override**: ONE override: IGNORE every GitHub and git rule; do not run any git commands (`commit`, `push`, `tag`, etc.). Keep all files on disk.
- **Owner Decision (verbatim, dated 2026-11-15)**:
  > *"I changed my mind. Revert to the previous rule." The V3.7 run (instructions/18) was cancelled by me while running, after about 21 minutes and 14 changed files (+992 -108). The ACTIVE rule must again be the one in force before V3.7: the frozen thresholds q_lo and q_hi from models/calib_live.json (about 0.434 and 0.563), the original STRONG threshold, SIP-delayed signals, and the log layout from V3.6. Do not tune anything and do not change thresholds.*

- **Model Lock**:
  - `models/model_v2.joblib`, `models/calib_v2.joblib`, and `models/calib_live.json` strictly untouched.
  - Removed `"rule_v2.json"` entry from `models/MODEL_SHA256.json`. All 3 original model entries match disk 100%.

- **Database Immutability**:
  - Strictly ZERO rows deleted or updated in `signals`.
  - Abort triggers `prevent_signals_update` and `prevent_signals_delete` active and intact.
  - Row counts: 140 Rule 1 rows (70 IEX + 70 SIP-delayed) + 70 Rule 2 rows = 210 total rows.
  - Cryptographic hash chain: 100% verified across all 210 rows (`valid=True, count=210, err=None`).
  - Data layer strictly filters `(rule_version IS NULL OR rule_version = 1)` everywhere signals are read (`/api/log`, `/api/signals`, `/api/scoreboard`, `/api/latest`, `/api/replay`, `get_multi_session_balance`). Rule 2 rows are completely hidden.

- **Inventory & Archived Artifacts**:
  - Comprehensive audit recorded in `docs/revert_inventory.md`.
  - Stale server process PID 17902 stopped. Port 8000 clean.
  - Moved V3.7 artifacts to `archive/rule_47_53_cancelled/`:
    - `models/rule_v2.json`
    - `results/tables/rule_47_53_coverage.csv`
    - `results/tables/rule_47_53_tier_accuracy.csv`
    - `results/tables/rule_47_53_signals_per_day.csv`
    - `docs/rule_47_53_measurements.md`
    - `src/measure_rule_47_53.py`
    - `src/live/migrate_rule_v2.py`
    - `src/live/backfill_rule2.py`

- **Restoration of Code, UI & Docs**:
  - `src/insights.py`: Reverted threshold loading to `models/calib_v2.joblib` ($q_{\text{lo}} \approx 0.434$, $q_{\text{hi}} \approx 0.563$). Removed WEAK tier and rule_v2.json.
  - `src/live/aggregator.py`: Cleaned default dictionary to standard thresholds.
  - `src/live/engine.py`: Reverted live engine to read from `models/calib_live.json` with standard strong threshold and `rule_version = 1`.
  - `src/live/store.py`: Added default query filter `(rule_version IS NULL OR rule_version = 1)` to `get_latest_signal()`, `get_recent_signals()`, and `get_signal_by_bar_start()`.
  - `src/live/log_engine.py`: Dynamic threshold loader restored to `calib_v2.joblib` / `calib_live.json`. `_format_row()` removed WEAK tier; all queries filter `(rule_version IS NULL OR rule_version = 1)`.
  - `src/live/server.py`: Removed `/api/rule` and `/api/tier_stats`. Restored `/api/status`, `/api/log`, `/api/signals`, `/api/replay`, and `/api/diagnostics` to V3.6 baseline.
  - `static/index.html`: Reverted hero threshold text to `needs ≥ 56.3% for Up, ≤ 43.4% for Down`; removed `#filter-v1-check`; restored Diagnostics Coverage card to `~30% Target Coverage` with baseline numbers; ensured no WEAK pills or Confidence Tiers card exist.
  - `report/report.md`: Added honest sentence: *"A 47/53 no-signal band was considered and cancelled by the owner before any results were produced from it."*


## Phase 23: V4-M5 5-Minute Directional Indicator Pipeline (2026-11-16)
- **Owner Decision (verbatim, dated 2026-11-16)**:
  > "My TA asked me to change the model from 1 hour to 5 minutes." The indicator now predicts the direction of the NEXT 5-MINUTE bar and shows a signal at the close of every 5-minute bar of the trading day. The 1-hour system stays as it is (do not delete or edit its models, tables or database); the 5-minute system is a separate, parameterised pipeline: config bar_minutes = 5.
- **Feed Switch Justification**: Delayed SIP feed has a 16-minute delivery lag, which is 3.2x longer than the 5-minute prediction horizon itself; thus delayed SIP cannot be used for 5-minute operational signaling. The 5-minute system uses the IEX feed (the same feed the live site will use for operational inference), with causal aggregation and validation.
- **Hard Constraints & Scope**:
  - `model-v2` (1-hour) remains FROZEN and completely untouched (`models/model_v2.joblib`, `models/calib_v2.joblib`, `models/calib_live.json`, `data/live/live.db`).
  - All 5-minute artifacts are isolated under `models/m5/`, `results/m5/`, `data/live/live_m5.db`, `data/raw/iex_1min/`, `data/processed/m5/`, and `docs/m5/`.
  - Immutable database design maintained for `data/live/live_m5.db` with triggers, hash chain, and `raw_minutes` table.
  - Zero hard-coded thresholds; all thresholds reside in `models/m5/calib_m5.json`.
  - Zero secrets or API keys exposed in any responses or static files.
  - Pre-registered research protocol in `docs/m5/protocol.md` before any model training or holdout access.

- **Execution & Deliverables Summary**:
  1. **Data Download & Audit**:
     - Downloaded 5 years of 1-minute IEX bars (2021-11 to 2026-11) for 8 symbols (GS, SPY, XLF, JPM, MS, BAC, TLT, VIXY).
     - Empty-bar rate check: TLT achieved 0.64% (< 5%) -> KEPT. VIXY had 24.03% (>= 5%) -> EXCLUDED.
     - Final feature symbols: GS, SPY, XLF, JPM, MS, BAC, TLT.
     - Built 87,109 5-minute bars for GS. Data frozen at 2026-11-13 (`configs/data_freeze_m5.json`).
  2. **Features & Truncation Invariance**:
     - Implemented 46 causal features across 8 groups in `src/features_m5.py`.
     - Truncation tests, same-session label rules, and aggregation exactness all passed 100% in `tests/test_features_m5.py`.
  3. **Walk-Forward OOF & Model Selection**:
     - Label threshold: 2 bps selected by lowest OOF log-loss (0.68179).
     - Candidate models: `xgb_depth4` selected by lowest OOF log-loss (0.68057 vs 0.69310 constant prior, delta -0.01253).
     - All 8 feature groups showed positive ablation delta (intrabar +0.00270).
     - Platt calibrator fit on pooled OOF: ECE = 0.0069 (Criterion D: PASS).
     - Coverage sweep selected $c = 0.10$ maximizing Wilson LB (68.11%): $q_{\text{lo}} = 0.37279$, $q_{\text{hi}} = 0.62913$, STRONG threshold = 0.7015.
  4. **Research Holdout Single Read (May 18 to Nov 13, 2026)**:
     - 150 Up, 138 Down, 9,184 No-signal. Total directional = 288.
     - Signal Accuracy = 67.36% (Wilson 95% CI: [61.75%, 72.52%]).
     - Up Acc: 66.00% (base 49.33%) | Down Acc: 68.84% (base 50.67%).
     - Permutation test p = 0.0001 (10,000 shuffles).
     - Net return @ 2 bps friction: +6.08 bps (95% CI: [+5.38, +6.77]).
     - Criterion B verdict: NOT MET solely due to count (150 Up / 138 Down < 300 required per side due to conservative c=0.10).
  5. **Live System, Database & Browser UI**:
     - Immutable SQLite database `data/live/live_m5.db` with triggers, hash chain, and `raw_minutes` table.
     - Replay equivalence verified on 3 days including half-day (max diff <= 1e-9).
     - Headless Playwright tests across 4 viewports (1280x720, 1440x900, 1920x1080, 390x844) passed 100% with zero console errors. Screenshots saved to `results/m5/figures/`.
     - Live server running on `http://localhost:8000` via `./run_live_m5.sh`.


## Phase 24: V4.1-COVERAGE Signal Coverage Expansion & Integrity Audit (2026-11-16)
- **Owner Decision (verbatim, dated 2026-11-16)**:
  > "There are too many No-signal bars. I want fewer No-signal and more Up and Down signals." Target coverage for the 5-minute system: coverage_target_m5 = 0.30 (30% of bars directional, about 23 per full day), editable in configs/config.py. This is a SIGNAL-RULE change, NOT a model change.
- **Observed Baseline Before Change**:
  - Pre-holdout OOF coverage under initial rule ($c=0.10$): 10.0% (1,989 Up + 1,989 Down out of 39,778 evaluated bars). On the full unselected 5-year series (84k+ bars), abstention was ~95.3% due to data gaps and narrow thresholds.
  - Recent out-of-sample sessions (Nov 10–13, 305 bars): 8 directional signals (5 Up, 3 Down) out of 305 bars (2.6% coverage, ~2 signals/day).
- **Scope & Hard Constraints**:
  - `models/m5/model_m5.joblib` and its internal calibrator remain strictly FROZEN (no retraining, no hyperparameter tuning).
  - 1-hour system remains completely untouched.
  - No UPDATE or DELETE on `data/live/live_m5.db`; additive schema changes only (`rule_version`).
  - All thresholds stored in `models/m5/rule_m5_vN.json` pointed to by `models/m5/active_rule.json`.
  - Section A Integrity Audit must run first with pre-registered fail gates before any rule change.
- **Section A: Integrity Audit Execution (docs/m5/audit_findings.md)**:
  - Gate 1 (Shuffled-Label Null Test): **50.00%** (0 signals). **PASS**.
  - Gate 2 (Time-Shift +1 Bar): **50.00%** (0 signals). **PASS**.
  - Gate 3 (Single-Feature Leakage Inspection): Max AUC Delta **0.0591** (`gs_minus_peer_1`, AUC = 0.4409). Causal windows verified. **PASS**.
  - Gate 4 (IEX Microstructure & Thin-Print Artifact): **ARTIFACT DETECTED**. Accuracy against IEX forward returns was **69.93%**, while accuracy against Consolidated SIP forward returns was **51.80%** ($\Delta = +18.13\% > 5.0\%$). App displays prominent red warning card on UI and docs.
  - Gate 5 (Subgroup Stability): Consistent across years (68.1% to 75.5%), TOD (70.0%), and sides (Up 69.6%, Down 70.3%). **PASS**.
- **Section B: Rule Selection & Coverage Tuning (docs/m5/coverage_tradeoff.md)**:
  - Computed coverage trade-off table across $c \in \{0.05, 0.10, 0.20, 0.30, 0.40, 0.50\}$ on OOF and Holdout (`results/m5/tables/coverage_tradeoff.csv`).
  - Evaluated Rule A on last 20 completed sessions: Achieved coverage was **12.57%** (191 signals of 1,520 bars), failing the pre-registered 15.0% threshold.
  - **Rule B (Dynamic Trailing 20-Session Quantiles) ACTIVATED**:
    - Thresholds: $q_{lo} = 0.45168$, $q_{hi} = 0.54528$, `min_margin = 0.01` (every directional call $\ge 51.0\%$).
    - Active rule written to `models/m5/rule_m5_v3.json` pointed by `models/m5/active_rule.json`.
    - Hashes updated in `models/m5/MODEL_SHA256_m5.json` (model hashes strictly preserved byte-for-byte).
  - Confidence Tiers established:
    - `STRONG` ($\ge 58.4\%$ conf): 68.32% OOF acc, 64.36% holdout acc
    - `NORMAL` ($55.6\% - 58.4\%$ conf): 57.28% OOF acc, 57.58% holdout acc
    - `WEAK` ($< 55.6\%$ conf): 54.97% OOF acc, 52.98% holdout acc
  - Database re-derivation: Additive schema with composite `UNIQUE(bar_start, horizon, rule_version)`. Derived signals for last 10 completed sessions inserted as `rule_version = 3` (`late = 1`). Cryptographic hash chain unbroken (1,969 total rows verified).
  - Before vs After Daily Counts (Last 5 Sessions):
    - 2026-11-09: Up: 0, Down: 1, No-Sig: 76 (1.3%) $\rightarrow$ **Up: 13, Down: 7, No-Sig: 58 (25.6%)**
    - 2026-11-10: Up: 2, Down: 0, No-Sig: 74 (2.6%) $\rightarrow$ **Up: 15, Down: 12, No-Sig: 50 (35.1%)**
    - 2026-11-11: Up: 0, Down: 0, No-Sig: 77 (0.0%) $\rightarrow$ **Up: 7, Down: 5, No-Sig: 66 (15.4%)**
    - 2026-11-12: Up: 0, Down: 1, No-Sig: 74 (1.3%) $\rightarrow$ **Up: 13, Down: 7, No-Sig: 56 (26.3%)**
    - 2026-11-13: Up: 3, Down: 2, No-Sig: 68 (6.8%) $\rightarrow$ **Up: 11, Down: 16, No-Sig: 47 (36.5%)**
    - Average daily coverage across these 5 sessions reached **27.8%** (~21.2 directional calls/day), resolving the abstention issue.
  - Re-derivation CLI deployed: `python -m src.m5.apply_rule --coverage 0.30`.
- **Section C & D: UI, Documentation & Verification**:
  - Live page (`http://localhost:8000`) displays Red Warning Banner for IEX Print Artifact, Coverage Trade-off Card, Confidence Tiers Card, and "Show old rule (v1)" toggle.
  - Replay equivalence verified on 3 days including half-day (`test_replay_equivalence_across_days` PASSED).
  - Unit tests (`tests/test_coverage_rule_m5.py`): 8/8 tests PASSED (boundary cases, margin guard, tiers, directional confidence $\ge 51\%$, hash locks, non-destructive row counts).
  - Playwright headless browser tests across all 4 viewports (1280x720, 1440x900, 1920x1080, 390x844) passed 100% with zero console errors (`results/m5/figures/ui_<w>x<h>.png`).
  - Secret check: `python3 src/check_secrets.py` PASSED with zero leaks.
  - Phase 24 COMPLETE. Operational server live on `http://localhost:8000`.


## Phase 25: V4.2-ALWAYSCALL Directional Call on Every Interval & Denominator Reconciliation (2026-11-16)
- **Owner Decision (verbatim, dated 2026-11-16)**:
  > "There is still too much No signal. Keep No signal minimal, or at least reduce it a lot." Interpretation: every eligible 5-minute bar gets an Up or Down call with a confidence tier. No signal remains ONLY for: the end-of-day bar (no forward interval in the same session), a data gap (a required symbol has no minute data), market closed, or an exact p_cal tie at 0.5. This is a SIGNAL-RULE change, NOT a model change.
- **Observed Baseline Before Change (verbatim)**:
  > OOF accuracy 62.2% at 30% coverage; holdout 59.82% over 1,827 signals with 19.3% coverage; and the inconsistent figures "No-Signal 63,041 bars (70.0%)" with 74,971 bars (that is 84.1%).
- **Scope & Hard Constraints**:
  - `models/m5/model_m5.joblib` and its calibrator remain strictly FROZEN (no retraining, no hyperparameter changes).
  - 1-hour system untouched. Zero git commands executed (strict override).
  - Never UPDATE or DELETE stored signal rows in `data/live/live_m5.db`. Schema changes additive only (`rule_version = 3`, `late = 1`).
  - All thresholds and tier cutoffs live in `models/m5/rule_m5_v3.json` pointed to by `models/m5/active_rule.json`. Hashes verified in `models/m5/MODEL_SHA256_m5.json`.
  - Zero secrets or API keys exposed.
- **Section A: Integrity Audit (docs/m5/audit_findings.md)**:
  - Gate 1 (Shuffled-label): 50.00% [PASS]
  - Gate 2 (Time-shift +1 bar): 50.00% [PASS]
  - Gate 3 (Single-feature AUC): Max delta = 0.0591 (`gs_minus_peer_1`, AUC = 0.4409) [PASS]
  - Gate 4 (IEX Microstructure Artifact): **ARTIFACT DETECTED** (IEX forward return accuracy = 69.93% vs Consolidated SIP tape accuracy = 51.80%, $\Delta = +18.13\% > 5\%$). Red warning card displayed prominently on page and docs.
  - Gate 5 (Stability): 68.1%–75.5% across years, 70.0% across TOD, Up 69.6% / Down 70.3% [PASS].
- **Section B: Denominator Reconciliation (docs/m5/coverage_reconciliation.md, results/m5/coverage_reconciliation.csv)**:
  - Total pre-holdout bars: 77,511 raw (74,971 labelled).
  - Evaluated walk-forward OOF bars: exactly **39,766 bars** (2-year burn-in training window of 35,205 bars).
  - Why "63,041 = 70.0%" was wrong: 11,930 signals emitted out of 39,766 OOF bars is 30.0%. Subtracting 11,930 from the 74,971 total series yielded 63,041 (which is 84.09% of 74,971, not 70.0%). On the true denominator of 39,766 OOF bars, exactly 27,836 bars (70.0%) were No-Signal.
  - Why Holdout coverage was 19.3% under Rule V2: Static quantile bounds ($q_{lo}=0.4296, q_{hi}=0.5727$) combined with slightly lower prediction variance in holdout and the margin guard filtered out borderline bars (1,827 of 9,472 = 19.29%).
- **Section C: Always-Call Rule & 4 Pre-Registered Checks**:
  - Pre-registered cutoffs from OOF confidence quartiles:
    - `WEAK`: Confidence $\le 0.5208$ (Bottom 25%, "close to a coin flip")
    - `NORMAL`: $0.5208 < \text{Confidence} < 0.5811$ (Middle 50%)
    - `STRONG`: Confidence $\ge 0.5811$ (Top 25%)
  - **Check (a) All-Bar Accuracy vs Base Rate**:
    - Holdout: 53.39% [52.38%, 54.39%] vs Larger Base Rate 50.67% (Down). Permutation $p = 0.00020 < 0.05$. **PASS**.
    - OOF: 55.60% [55.11%, 56.09%] vs Larger Base Rate 50.44% (Up). Permutation $p = 0.00020 < 0.05$. **PASS**.
  - **Check (b) Monotonic Tiers**:
    - OOF: WEAK = 51.12%, NORMAL = 53.85%, STRONG = 63.58%. **PASS**.
    - Holdout: WEAK = 50.34%, NORMAL = 53.26%, STRONG = 60.01%. **PASS**.
  - **Check (c) Expected Calibration Error (ECE $\le 0.030$)**:
    - OOF ECE (10-bin): **0.0069**. **PASS**.
    - Holdout ECE (10-bin): **0.0127**. **PASS**.
  - **Check (d) Both Sides Present**:
    - OOF: 19,695 Up / 20,071 Down. **PASS**.
    - Holdout: 4,814 Up / 4,658 Down. **PASS**.
  - Net returns @ 2bps:
    - ALL calls: OOF $+0.06$ bps, Holdout $-0.82$ bps.
    - STRONG only: OOF **$+3.43$ bps**, Holdout **$+1.57$ bps**.
- **Section D: Database Re-Derivation & Daily Counts**:
  - Signals re-derived via `python -m src.m5.apply_rule --mode always` for the last 10 completed sessions, inserting 769 rows with `rule_version = 3` (`late = 1`). Cryptographic hash chain unbroken (2,738 total rows verified).
  - Before vs After Daily Counts (Last 5 Sessions):
    - 2026-11-09: Up 0, Dn 1, No-Sig 76 (1.3%) $\rightarrow$ **Up 36, Dn 41, No-Sig 1 (98.7% coverage)**
    - 2026-11-10: Up 2, Dn 0, No-Sig 74 (2.6%) $\rightarrow$ **Up 38, Dn 38, No-Sig 1 (98.7% coverage)**
    - 2026-11-11: Up 0, Dn 0, No-Sig 77 (0.0%) $\rightarrow$ **Up 35, Dn 42, No-Sig 1 (98.7% coverage)**
    - 2026-11-12: Up 0, Dn 1, No-Sig 74 (1.3%) $\rightarrow$ **Up 39, Dn 36, No-Sig 1 (98.7% coverage)**
    - 2026-11-13: Up 3, Dn 2, No-Sig 68 (6.8%) $\rightarrow$ **Up 31, Dn 42, No-Sig 1 (98.6% coverage)**
    - Exactly 1 No-Signal bar per day (15:55-16:00 ET end-of-day bar).
- **Section E: UI, Verification & Testing**:
  - Live UI (`http://localhost:8000`) displays:
    - Red Warning Banner for IEX Print Artifact at the top.
    - Headline: UP or DOWN with confidence and tier chip; for WEAK: "close to a coin flip"; NO SIGNAL strictly with explicit reason.
    - Timeline strip: Arrow chips with tier shading and result mark (✔/✘/⏳).
    - Table: Compact rows ($\le 44$ px), filter chips `All | NORMAL + STRONG | STRONG only`.
    - Scored Day Header: Displays accuracy for ALL calls and STRONG-only side by side with Wilson CIs and `"demonstration, not proof (n = ...)"`. Never displays a single headline accuracy number.
    - Right Column Cards: All-Bar vs. STRONG-Only Comparison Card, Confidence Tiers Card, Deployed Rule Expectations Card with sample tags (`"holdout already inspected; descriptive"`).
  - Unit Tests (`tests/test_coverage_rule_m5.py`): 11/11 tests PASSED.
  - Replay Equivalence (`tests/test_replay_m5.py`): 3/3 days PASSED (including 2024-11-29 half-day, identical direction, confidence within $10^{-9}$).
  - Playwright Headless Browser Tests (`tests/ui_render_check_m5.py`): 4/4 viewports PASSED with zero console errors. Screenshots saved to `results/m5/figures/ui_<w>x<h>.png`.
  - Secret check: `python3 src/check_secrets.py` PASSED with zero leaks.
  - Phase 25 COMPLETE. Operational server live on `http://localhost:8000`.

- [x] **Read-Only Performance Report (M5 5-Minute Always-Call Rule v3, 2026-11-16)**:
  - Cryptographic hash verification against `models/m5/MODEL_SHA256_m5.json`: 5/5 components matched (`model_m5.joblib`, `calib_m5.json`, `rule_m5_v2.json`, `active_rule.json`, `rule_m5_v3.json`).
  - Strict read-only audit: Zero model/calibrator edits, zero rule threshold edits, zero database writes (`data/live/live_m5.db` opened via `file:data/live/live_m5.db?mode=ro`).
  - Database hash chain verification: 2,738 rows verified, 0 broken links.
  - Replay cross-checks on Nov 12 and Nov 13: 0 mismatches between ledger and fresh `predict_day_m5.py` replay; all re-derived rows confirmed `late = 1`.
  - Slice evaluations:
    - Slice A (Holdout, 126 sessions, 9,472 labelled bars): Accuracy 53.93% [52.92%, 54.93%] vs 50.67% base rate (+3.26% edge). Permutation $p = 0.00020$, ECE = 0.0079, Brier = 0.2464. Tiers monotonic: WEAK 49.39% $\le$ NORMAL 54.73% $\le$ STRONG 62.73%.
    - Slice B (Nov 12, 1 session, 75 labelled bars): Accuracy 57.33% [46.05%, 67.90%] vs 52.00% base rate (+5.33% edge). 2 data gaps on IEX tape (`13:45`, `14:20`).
    - Slice C (Nov 13, 1 session, 73 labelled bars): Accuracy 58.90% [47.45%, 69.47%] vs 52.05% base rate (+6.85% edge). 4 data gaps on IEX tape (`09:50`, `14:20`, `14:25`, `14:35`).
    - Slice B + C Combined (148 labelled bars): Accuracy 58.11% [50.05%, 65.75%] vs 50.00% base rate (+8.11% edge).
  - Net returns at 2 bps cost: All tiers holdout $-0.55$ bps, STRONG holdout **$+2.85$ bps**; Nov 12 STRONG **$+3.81$ bps**; Nov 13 STRONG **$+0.73$ bps**; B+C STRONG **$+1.76$ bps**.
  - Generated full audit report with bar-by-bar logs at `docs/m5/performance_report_2026-11-16.md`.
  - `src/check_secrets.py` PASSED with zero leaks.

- [x] **M5 Web Default & Performance Panels (2026-11-16)**:
  - Switched default landing view at `http://localhost:8000` from 1-Hour (H1) to 5-Minute (M5) model view (`static/index.html` + `static/app_m5.js`).
  - Preserved legacy H1 view at `http://localhost:8000/index_h1.html` behind a discreet "Legacy H1 (frozen)" header badge with backwards-compatible API routes.
  - Implemented dynamic previous regular market day resolution (`src/live_m5/performance.py`): evaluates NYSE session calendar in `America/New_York`, automatically skipping weekends and holidays (evaluates to Friday `2026-11-13` before 16:00 ET on Monday `2026-11-16`).
  - Added "Previous Market Day" performance panel:
    - Directional accuracy: 43 / 73 = 58.90% [47.45%, 69.47%]. Up: 18/31 = 58.06%, Down: 25/42 = 59.52%.
    - Coverage: 74 recorded bars, 4 IEX tape gaps (`09:50`, `14:20`, `14:25`, `14:35`), 1 EOD bar (`15:55`), 0 ties. Sum check: 31 + 42 + 1 = 74 (100% PASS).
    - Base rate & edge: 47.95% Up, 52.05% Down, edge +6.85%.
    - GS stock return: open-to-close return recorded.
    - Conviction tiers: WEAK = 13/22 (59.09%), NORMAL = 23/41 (56.10%), STRONG = 7/10 (70.00%).
    - Interactive 74-segment Hit/Miss strip with tooltips and hourly summary chips.
    - Flat-as-miss toggle (2 flat bars: standard 43/73 = 58.90%, flat-as-miss 42/73 = 57.53%).
  - Added "Overall Performance" panel with 3 side-by-side columns:
    - Holdout: 5,108 / 9,472 = 53.93% [52.92%, 54.93%], Up 53.30%, Down 54.54%, WEAK 49.39%, NORMAL 54.73%, STRONG 62.73% (monotonic PASS), ECE 0.0079, Brier 0.2464.
    - OOF Validation: 22,109 / 39,766 = 55.60% [55.11%, 56.09%], Up 56.10%, Down 55.11%, WEAK 51.12%, NORMAL 53.85%, STRONG 63.58% (monotonic PASS), ECE 0.0069, Brier 0.2458.
    - Live Deployment: 769 calls across 10 completed sessions, 57.87% [54.38%, 61.28%], 100% hash chain integrity verified (2,738 rows, 0 broken links).
    - SVG daily accuracy bar chart & cumulative accuracy line chart over last 20 sessions with 50% baseline.
    - Multi-session signal balance (last 1, 5, 20 sessions).
  - Prominent mandatory disclosure banners above the fold:
    - Holdout descriptive caveat.
    - IEX tape vs SIP tape discrepancy warning (69.93% IEX vs 51.80% SIP).
    - Educational paper trading disclaimer in footer.
  - Verification:
    - 62 of 62 pytest unit and integration tests PASSED.
    - Playwright browser render tests PASSED on 4 viewports (1280x720, 1440x900, 1920x1080, 390x844) with zero console errors.
    - Dedicated panel screenshots captured in `results/m5/figures/panel_previous_day.png` and `results/m5/figures/panel_overall_performance.png`.
    - `src/check_secrets.py` PASSED with zero leaks.

- [x] **M5 UI Layout Restoration (2026-11-16)**:
  - Redid the M5 web UI layout to match the clean dark-theme layout of the earlier 1-hour page, powered completely by the 5-minute Always-Call v3 model.
  - Removed large stat-card grids, hourly text lists, and the rejected redesign blocks.
  - Restored header bar with small badges (`SIGNALS: REAL-TIME IEX 5-MIN`, `RULE: ALWAYS-CALL V3`, `MODEL: hash verified`), live dual ET/IST clocks, market status, and zero "Legacy H1" links.
  - Restored replay controls (session select, Replay Day, speed select, Labels toggle).
  - Restored TradingView 5-minute candlestick chart with Up/Down markers and band overlay.
  - Restored compact one-line summary strip directly under chart:
    `Mon 16 Nov 2026 | GS $open -> $close (x.xx%) | Signals N (Up a, Down b) | Correct c | Wrong d | Pending p | No signal e | Accuracy c/N (Wilson 95% CI [..]) - demonstration, not proof (n=N)`.
  - Restored signal strip of compact 5-minute chips across the session with click-to-scroll.
  - Restored the main signal log table with exact 7 columns:
    `BAR PREDICTED (ET) | ISSUED (ET | IST) | CALL | STRENGTH | VOL AND BAND (bps) | PRICE MOVE | RESULT`.
    Explicitly displaying "No signal" rows for 15:55 ("End of trading day") and missing feed bars ("Feed gap").
  - Restored right-side sidebar panel:
    - Slim single-line IEX disclosure and holdout caveat banners.
    - Headline card with latest call, confidence %, tier pill, delivery age counter in seconds, and guidance sentence.
    - Quick metrics grid (vol class, regime, 5-min return band [-30, +30] bps, countdown).
    - Signal balance card (selected session, last 5 sessions, last 20 sessions).
    - Compact selected day performance card with Wilson CI, Up/Down stats, coverage "X of 78 bars".
    - Small overall research holdout card (5,108/9,472 = 53.9% [52.9%, 54.9%], tier accuracies).
  - Restored collapsible "Show Model & Feed Diagnostics" panel at the bottom.
  - Set default day to Monday 2026-11-16 (`EVALUATION_DAY`), with query param `?day=YYYY-MM-DD` override support.
  - Running live engine daemon (`task-5512`) continuously ingests, aggregates, and stores every 5-minute bar as it closes from 09:30 to 15:55 ET.
  - Verification:
    - `src/live_m5/doctor.py` PASSED with all checks green and zero missing bars.
    - Cryptographic hash chain VALID across 2,745 rows with 0 broken links.
    - All 62 pytest unit and integration tests PASSED.
    - Playwright render test (`tests/ui_render_check_m5.py`) PASSED across all 4 viewports (1280x720, 1440x900, 1920x1080, 390x844) with zero console errors.
    - `src/check_secrets.py` PASSED with zero leaks.

- [x] **M5 Live Engine Data Path & Pipeline Parity Fix (2026-11-16)**:
  - Diagnosed and resolved live engine degeneracy where context tickers (`SPY`, `XLF`, `JPM`, `MS`, `BAC`, `TLT`) were missing multi-bar history and lookback was limited to 150 bars, causing 16/46 features to be NaN and forcing XGBoost to predict flat $p_{\text{cal}} \approx 0.482$ (100% WEAK DOWN).
  - Unified feature engineering via `src/features_m5.py::load_and_build_features_m5` combining historical parquet baseline and today's raw minutes identically for live and replay.
  - Added `TLT` to engine symbols and ingestion. Extended lookback to 30 days so `ret_5d`, `ret_20d`, and `rvol_tod` compute accurately with 0 NaNs.
  - Settle Delay & Late-Trade Window: Implemented a 10-minute re-fetch settlement window and 25s settle delay to capture delayed exchange trade reports.
  - NaN Feature Policy: Configured `max_nan_features = 0` in `configs/config.py`. Engine emits `"No signal (feature gap)"` if any feature is NaN.
  - Vol/Band Display: Removed fake static fallbacks (`"Normal"` and `"[-30, +30] bps"`) from table and metrics; formatted as `"–"` since dynamic models are not in bundle.
  - Append-Only Ledger Revision: Re-derived today's signals (2026-11-16) from 09:30 ET under `rule_version = 5`, `sample = "LIVE_REVISED"`, `late = 1`. Preserved unbroken SHA-256 chain (2,811+ rows valid).
  - UI Disclosures & Layout:
    - Added one-line revision notice above the table: *"Signals before 12:15 ET on 2026-11-16 were generated with an incomplete data feed (engine bug, fixed). Corrected signals were re-derived afterwards."*
    - Documented unrevised baseline outcome (10/21) in diagnostics.
    - Fixed table-chart layout so the table renders cleanly below the chart with no overlap across 1280x720, 1440x900, 1920x1080, and 390x844.
    - Unresolvable feed gaps mark Result as `"– Unresolvable (feed gap)"` and are omitted from accuracy denominators.
  - Verification:
    - Live vs replay feature parity test PASSED with max diff $\le 10^{-9}$ across `2026-11-16`, `2026-11-13`, `2026-11-12`.
    - Context feature-gap test PASSED.
    - Playwright headless render check PASSED across 4 viewports with 0 console errors and 0 overlaps.
    - `src/live_m5/doctor.py` PASSED with 0 broken links in hash chain.
    - `src/check_secrets.py` PASSED with 0 secrets leaked.

- [x] **M5 Live Engine Watchdog, Feed Resilience & Ledger Repair (2026-11-16 Session)**:
  - Incident Diagnosis:
    - At ~12:31 ET, the engine process stopped during a server restart. Because the engine was offline, no live signals were emitted between 12:35 and 13:35 ET.
    - `src/live_m5/log_engine.py` defaulted missing session bars to `"Feed gap"`, masking the engine process downtime as data feed failures.
    - Direct verification confirmed Alpaca IEX had continuous, unbroken 1-minute trades for GS and all context tickers (e.g. GS had 35 minute bars across the window). True feed gaps were 0.
  - Ingestion Resilience & Error Handling:
    - Updated `fetch_iex_minutes` in `src/live_m5/engine.py` with exponential backoff retries (1s, 2s, 4s). API failures raise explicit exceptions rather than silently swallowing errors as empty dataframes.
    - Distinct signal labeling: genuinely empty exchange trades emit `"No signal (feed gap)"`, context feature gaps emit `"No signal (feature gap)"`, and API/network crashes emit `"No signal (engine error)"`.
    - Added context ticker forward-fill logic up to `max_context_ffill_bars = 1` in `src/features_m5.py`.
    - Implemented a live Watchdog in `engine.py` that alerts and triggers recovery catchup if no signal is emitted for >7 minutes during market hours.
  - Append-Only Ledger Repair:
    - Re-derived and appended all 13 missing closed bars from 12:35 through 13:40 ET into `signals` under `rule_version = 5`, `sample = "LIVE_REVISED"`, `late = 1`.
    - Preserved unbroken cryptographic SHA-256 hash chain (2,826 rows verified, 0 broken links).
    - Repaired signals metrics: 13 bars (8 Down, 5 Up; 9 NORMAL, 3 WEAK, 1 STRONG), $p_{\text{cal}} \in [0.3234, 0.5654]$ with $\sigma = 0.0630$.
    - Session cumulative accuracy: 33/49 correct (67.3%) on resolved bars.
  - UI & Chart Formatting:
    - Configured Lightweight Charts `localization.timeFormatter` and `timeScale.tickMarkFormatter` to render ET (`America/New_York`) rather than UTC on the time axis.
    - Updated notice banner to reflect both affected windows: *"Signals before 12:15 ET and 12:30–13:40 ET on 2026-11-16 were generated or recorded with an engine fault (fixed). Corrected signals were re-derived afterwards."*
    - Restructured pending vs unresolvable counts: directional signals pending next-bar resolution are separated from scheduled future intervals.
  - Verification:
    - Replay vs live parity confirmed identical with tolerance $\le 10^{-9}$.
    - Pytest (27 tests), Playwright render tests (4 viewports, 0 console errors), `doctor.py` (all checks green), and `check_secrets.py` (0 leaks) all PASSED.
